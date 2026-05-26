"""
Standalone attractions-finder for any city/country.

Pipeline per attraction (parallel across MAX_WORKERS threads):
  1. Claude (LLM) — names, local names, website guess, approx coords
  2. Wikidata SPARQL — official website (P856) + curated coords (P625)
  3. Nominatim free-text — OSM geocoding
  4. Nominatim structured — category-based OSM search
  5. Overpass API — direct OSM name search within city bounding box

Early bail-out: if Wikidata + Nominatim already agree (≤150 m), skip 3-5.

Refill loop: if after filtering + dedup we are below COUNT, ask the LLM
for more (excluding names already tried) up to 3 times.

Public entry points:
    find_attractions(city, country, count) -> AttractionsResult
    build_attractions_map(result) -> folium.Map
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import anthropic
import folium
import requests
from shapely.geometry import shape, Point

from core.config import ANTHROPIC_API_KEY, MODEL
from core.models import AttractionsResult, FoundAttraction


# ── Config ────────────────────────────────────────────────────────────────────
MAX_WORKERS = 4
AGREE_M     = 150

NOMINATIM       = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REV   = "https://nominatim.openstreetmap.org/reverse"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
OVERPASS        = "https://overpass-api.de/api/interpreter"
USER_AGENT      = "TripAdvisorSimplified/1.0 (attractions finder)"

OVERPASS_TAG = {
    "museum":   '["tourism"="museum"]',
    "church":   '["amenity"="place_of_worship"]',
    "park":     '["leisure"="park"]',
    "castle":   '["historic"~"castle|ruins"]',
    "monument": '["historic"~"monument|memorial"]',
    "zoo":      '["tourism"="zoo"]',
}

NOMINATIM_TAG = {
    "museum":  {"amenity": "museum"},
    "church":  {"amenity": "place_of_worship"},
    "park":    {"leisure": "park"},
    "castle":  {"historic": "castle"},
    "zoo":     {"tourism": "zoo"},
}


# ── Per-call context (replaces module globals so we can serve multiple cities) ─

@dataclass
class _CityContext:
    city: str
    country: str
    center: tuple                  # (lat, lon)
    bbox: str                      # "south,west,north,east" for Overpass
    viewbox: str                   # "left,top,right,bottom" for Nominatim
    polygon: object | None         # shapely Polygon or None


# ── Thread-local logger + per-API throttles (process-wide) ─────────────────────

_tls = threading.local()

def _log(msg: str = "") -> None:
    if hasattr(_tls, "lines"):
        _tls.lines.append(msg)
    else:
        print(msg)


class _Throttle:
    def __init__(self, interval: float):
        self.interval = interval
        self.lock     = threading.Lock()
        self.last     = 0.0

    def wait(self) -> None:
        with self.lock:
            elapsed = time.time() - self.last
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last = time.time()


NOMINATIM_T = _Throttle(1.0)
WIKIDATA_T  = _Throttle(0.5)
OVERPASS_T  = _Throttle(0.8)

_DEAD_URLS: set[str]   = set()
_DEAD_URLS_LOCK        = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_m(a: tuple, b: tuple) -> float:
    R = 6_371_000
    la1, lo1 = map(math.radians, a)
    la2, lo2 = map(math.radians, b)
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(math.sqrt(h))


def _within_city(coord: tuple, ctx: _CityContext) -> bool:
    lat, lon = coord
    if ctx.polygon is not None:
        return ctx.polygon.contains(Point(lon, lat)) or ctx.polygon.touches(Point(lon, lat))
    return _haversine_m(ctx.center, coord) / 1000 <= 10


def _city_stop_words(city: str) -> set[str]:
    base = city.lower()
    return {base, base + "ie", base + "a", base + "ą"}


def _short_name(name: str) -> str:
    return name.split(" im.")[0].split(" w ")[0].split(",")[0].strip()


def _http_get(url, timeout: int = 10, **kwargs):
    headers = {"User-Agent": USER_AGENT}
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    r = requests.get(url, headers=headers, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r


def _verify_url(url: str) -> bool:
    if not url:
        return False
    with _DEAD_URLS_LOCK:
        if url in _DEAD_URLS:
            return False
    try:
        r = requests.head(url, timeout=3, allow_redirects=True,
                          headers={"User-Agent": USER_AGENT})
        if r.status_code == 405:
            r = requests.get(url, timeout=3, allow_redirects=True,
                             headers={"User-Agent": USER_AGENT}, stream=True)
        ok = r.status_code < 400
    except Exception:
        ok = False
    if not ok:
        with _DEAD_URLS_LOCK:
            _DEAD_URLS.add(url)
    return ok


def _fetch_city_polygon(city: str, country: str) -> _CityContext:
    print(f"Fetching administrative polygon for {city}, {country}...")
    NOMINATIM_T.wait()
    r = _http_get(NOMINATIM, params={
        "q": f"{city}, {country}", "format": "json",
        "polygon_geojson": 1, "limit": 5,
    })
    for hit in r.json():
        gj = hit.get("geojson", {})
        if gj.get("type") in ("Polygon", "MultiPolygon"):
            poly = shape(gj)
            c    = poly.centroid
            minx, miny, maxx, maxy = poly.bounds
            ctx  = _CityContext(
                city    = city,
                country = country,
                center  = (c.y, c.x),
                bbox    = f"{miny},{minx},{maxy},{maxx}",
                viewbox = f"{minx},{maxy},{maxx},{miny}",
                polygon = poly,
            )
            print(f"  polygon   : {hit.get('display_name','')[:70]}")
            print(f"  center    : {ctx.center[0]:.4f}, {ctx.center[1]:.4f}")
            return ctx
    print("  WARNING: no polygon found — using 10km radius fallback")
    return _CityContext(city, country, (0.0, 0.0), "", "", None)


def _strip_llm_prefix(model: str) -> str:
    """LLM_MODEL env carries 'anthropic/' for CrewAI; raw Anthropic SDK wants bare ID."""
    return model.removeprefix("anthropic/")


# ── Step 1: LLM ───────────────────────────────────────────────────────────────

def _ask_llm(count: int, city: str, country: str,
             exclude: set[str] | None = None) -> list[dict]:
    exclude_block = ""
    if exclude:
        sample = sorted(exclude)[:40]
        exclude_block = (
            "\n\nDO NOT INCLUDE these attractions (already considered):\n- "
            + "\n- ".join(sample)
        )

    prompt = f"""Find {count} top tourist attractions physically located inside {city} city, {country}.

Return ONLY a valid JSON object — no markdown, no prose:
{{
  "attractions": [
    {{
      "name": "Official English name",
      "local_name": "Exact name as in OpenStreetMap/Wikipedia (with diacritics)",
      "category": "museum|church|monument|park|castle|zoo|beach|other",
      "short_description": "1-2 sentences on why it is worth visiting",
      "website": "Official website URL if confident it exists, else empty string",
      "lat": 0.0,
      "lon": 0.0
    }}
  ]
}}

RULES:
- Only places physically inside {city} city limits — not nearby villages or towns
- local_name must be the exact OSM/Wikipedia name with correct diacritics
- lat/lon: best-estimate decimal coordinates for the entrance
- website: real URL only — empty string if uncertain
- When a landmark building stands on or in a public square (e.g. a town hall
  on a market square), include the BUILDING only, not the square. Prefer
  specific named buildings (town halls, churches, palaces, museums) over the
  general public spaces they sit on.
- Return exactly {count} attractions{exclude_block}"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=_strip_llm_prefix(MODEL),
        max_tokens=max(2048, count * 250),
        temperature=0,
        system="You are a precise travel researcher. Return ONLY valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)["attractions"]


# ── Step 2: Wikidata ──────────────────────────────────────────────────────────

def _wikidata_variants(name: str, local_name: str, city: str) -> list[str]:
    variants = []
    for n in [local_name, name]:
        if not n:
            continue
        variants.append(f"{n} {city}")
        variants.append(f"{_short_name(n)} {city}")
        variants.append(_short_name(n))
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


def _wikidata_qid(name: str, local_name: str, city: str) -> str | None:
    for variant in _wikidata_variants(name, local_name, city)[:3]:
        for attempt in (0, 1):
            try:
                WIKIDATA_T.wait()
                r = _http_get(WIKIDATA_SEARCH, params={
                    "action": "wbsearchentities", "search": variant,
                    "language": "en", "format": "json", "limit": 1,
                })
                hits = r.json().get("search", [])
                if hits:
                    _log(f"    wikidata QID   : {hits[0]['id']}  (via '{variant[:45]}')")
                    return hits[0]["id"]
                break
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    time.sleep(3)
                    continue
                _log(f"    [wikidata-search error] {str(e)[:60]}")
                break
    return None


def _wikidata_facts(qid: str) -> dict:
    query = f"""
    SELECT ?website ?coord WHERE {{
      OPTIONAL {{ wd:{qid} wdt:P856 ?website. }}
      OPTIONAL {{ wd:{qid} wdt:P625 ?coord. }}
    }} LIMIT 1"""
    WIKIDATA_T.wait()
    r = _http_get(WIKIDATA_SPARQL,
                  params={"query": query, "format": "json"},
                  headers={"Accept": "application/sparql-results+json"})
    rows = r.json().get("results", {}).get("bindings", [])
    if not rows:
        return {}
    row = rows[0]
    out = {}
    if "website" in row:
        out["website"] = row["website"]["value"]
    if "coord" in row:
        raw = row["coord"]["value"].removeprefix("Point(").rstrip(")")
        lon, lat = raw.split()
        out["lat"], out["lon"] = float(lat), float(lon)
    return out


# ── Step 3: Nominatim free-text ───────────────────────────────────────────────

def _nominatim_freetext(name: str, local_name: str, ctx: _CityContext) -> tuple | None:
    queries = [f"{local_name}, {ctx.city}", f"{_short_name(local_name)}, {ctx.city}"]
    if name and name != local_name:
        queries.append(f"{name}, {ctx.city}")
    for q in dict.fromkeys(queries):
        NOMINATIM_T.wait()
        try:
            r = _http_get(NOMINATIM, params={
                "q": q, "format": "json", "limit": 1, "viewbox": ctx.viewbox,
            })
            hits = r.json()
            if hits:
                coord = (float(hits[0]["lat"]), float(hits[0]["lon"]))
                if _within_city(coord, ctx):
                    _log(f"    nominatim-text : '{q[:50]}'  {coord}")
                    return coord
        except Exception as e:
            _log(f"    [nominatim-text error] {str(e)[:60]}")
    return None


# ── Step 4: Nominatim structured ─────────────────────────────────────────────

def _nominatim_structured(local_name: str, category: str, ctx: _CityContext) -> tuple | None:
    tag = NOMINATIM_TAG.get(category)
    if not tag:
        return None
    NOMINATIM_T.wait()
    key, val = next(iter(tag.items()))
    try:
        r = _http_get(NOMINATIM, params={
            key: val, "city": ctx.city, "country": ctx.country,
            "format": "json", "limit": 5,
        })
        hits = r.json()
    except Exception as e:
        _log(f"    [nominatim-struct error] {str(e)[:60]}")
        return None
    if not hits:
        return None
    local_words = {w for w in local_name.lower().split() if len(w) > 3}
    best, best_score = None, 0
    for h in hits:
        dn = h.get("display_name", "").lower()
        score = sum(1 for w in local_words if w in dn)
        if score > best_score:
            best, best_score = h, score
    if best and best_score > 0:
        coord = (float(best["lat"]), float(best["lon"]))
        if _within_city(coord, ctx):
            _log(f"    nominatim-struct: '{best['display_name'][:60]}'  {coord}")
            return coord
    return None


# ── Step 5: Overpass ──────────────────────────────────────────────────────────

def _name_overlap(a: str, b: str, stop_words: set) -> float:
    def tokens(s):
        return {w.lower() for w in re.split(r"[\s,.\-/()]+", s)
                if len(w) > 3 and w.lower() not in stop_words}
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _overpass_search(name: str, local_name: str, category: str,
                     ctx: _CityContext) -> tuple | None:
    """Returns (coord, overlap_score) or None. Uses local_name only."""
    tag_filter  = OVERPASS_TAG.get(category, "")
    stop_words  = _city_stop_words(ctx.city)
    MIN_OVERLAP = 0.40

    source_name = local_name or name
    words = [w for w in re.split(r"[\s,.\-]+", source_name)
             if len(w) > 3 and w.lower() not in stop_words]
    if not words:
        return None
    pattern = "|".join(re.escape(w) for w in words[:3])
    query = f"""
[out:json][timeout:8];
(
  node{tag_filter}["name"~"{pattern}",i]({ctx.bbox});
  way{tag_filter}["name"~"{pattern}",i]({ctx.bbox});
);
out center 5;
"""
    try:
        OVERPASS_T.wait()
        r = _http_get(OVERPASS, timeout=8, params={"data": query})
        best_coord, best_score = None, 0.0
        for el in r.json().get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if not (lat and lon):
                continue
            coord = (float(lat), float(lon))
            if not _within_city(coord, ctx):
                continue
            el_name = el.get("tags", {}).get("name", "")
            score   = max(
                _name_overlap(source_name, el_name, stop_words),
                _name_overlap(source_name, el.get("tags", {}).get("name:en", ""), stop_words),
            )
            if score >= MIN_OVERLAP and score > best_score:
                best_coord, best_score = coord, score
                _log(f"    overpass       : '{el_name}'  {coord}  (overlap={score:.2f})")
        if best_coord:
            return best_coord, best_score
    except Exception as e:
        _log(f"    [overpass error] {str(e)[:60]}")
    return None


# ── Reconcile ─────────────────────────────────────────────────────────────────

def _reconcile(sources: dict, llm_coord, llm_website: str, wd_website: str,
               source_weights: dict | None = None):
    website = wd_website or llm_website
    weights = {"nominatim": 0.5, "nominatim-structured": 0.4, "overpass": 0.5}
    if source_weights:
        weights.update(source_weights)

    if not sources:
        if llm_coord:
            return *llm_coord, "llm-estimate", "llm-estimate", website
        return None, None, "low", "none", website

    wd = sources.get("wikidata")
    if wd:
        others   = {k: v for k, v in sources.items() if k != "wikidata"}
        agreeing = [k for k, v in others.items() if _haversine_m(wd, v) <= AGREE_M]
        conf     = "high" if agreeing else "medium"
        src      = "wikidata" + (f" + {', '.join(agreeing)} agree" if agreeing else " only")
        return *wd, conf, src, website

    items = list(sources.items())
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            n1, c1 = items[i]
            n2, c2 = items[j]
            if _haversine_m(c1, c2) <= AGREE_M:
                return *c1, "high", f"{n1} + {n2} agree", website

    best_name, best_coord = max(sources.items(), key=lambda kv: weights.get(kv[0], 0.5))
    return *best_coord, "medium", best_name, website


# ── Enrich one attraction ─────────────────────────────────────────────────────

def _enrich(a: dict, ctx: _CityContext) -> tuple[dict, list[str]]:
    _tls.lines = []

    llm_lat = a.get("lat") or None
    llm_lon = a.get("lon") or None
    llm_web = a.get("website") or ""

    if llm_lat and not _within_city((llm_lat, llm_lon), ctx):
        _log(f"    [llm coord rejected] {llm_lat},{llm_lon} too far from {ctx.city}")
        llm_lat = llm_lon = None
    llm_coord = (llm_lat, llm_lon) if llm_lat else None

    sources    = {}
    wd_website = ""

    try:
        qid = _wikidata_qid(a["name"], a["local_name"], ctx.city)
        if qid:
            facts = _wikidata_facts(qid)
            if "lat" in facts:
                cand = (facts["lat"], facts["lon"])
                if _within_city(cand, ctx):
                    sources["wikidata"] = cand
                    wd_website = facts.get("website", "")
                    _log(f"    wikidata coord : {cand}")
                else:
                    _log(f"    [wikidata entity rejected] coords {_haversine_m(ctx.center,cand)/1000:.1f} km away")
        else:
            _log(f"    wikidata QID   : not found")
    except Exception as e:
        _log(f"    [wikidata error] {str(e)[:60]}")

    nm = _nominatim_freetext(a["name"], a["local_name"], ctx)
    if nm:
        sources["nominatim"] = nm

    have_agreement = (
        "wikidata" in sources and "nominatim" in sources
        and _haversine_m(sources["wikidata"], sources["nominatim"]) <= AGREE_M
    )

    op_score = 0.0
    if not have_agreement:
        if "nominatim" not in sources:
            nm_s = _nominatim_structured(a["local_name"], a["category"], ctx)
            if nm_s:
                sources["nominatim-structured"] = nm_s
        op_result = _overpass_search(a["name"], a["local_name"], a["category"], ctx)
        if op_result:
            op_coord, op_score = op_result
            sources["overpass"] = op_coord

    lat, lon, confidence, coord_source, website = _reconcile(
        sources, llm_coord, llm_web, wd_website,
        source_weights={"overpass": op_score} if op_score else None,
    )

    if website:
        if _verify_url(website):
            _log(f"    website OK     : {website}")
        else:
            _log(f"    [website dead] : {website} — removed")
            website = ""

    result = {**a, "lat": lat, "lon": lon, "website": website,
              "confidence": confidence, "coord_source": coord_source, "address": ""}

    if lat is not None and confidence not in ("llm-estimate", "low"):
        try:
            NOMINATIM_T.wait()
            r = _http_get(NOMINATIM_REV, params={
                "lat": lat, "lon": lon, "format": "json", "zoom": 17,
            })
            result["address"] = r.json().get("display_name", "")
        except Exception:
            pass

    return result, list(_tls.lines)


def _parallel_enrich(attractions: list[dict], ctx: _CityContext) -> list[dict]:
    if not attractions:
        return []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(lambda a: _enrich(a, ctx), attractions))

    out = []
    for a, (result, lines) in zip(attractions, results):
        print(f"\n  → {a['name']}")
        for line in lines:
            print(line)
        print(f"     coords  : {result['lat']}, {result['lon']}  [{result['confidence']}]")
        out.append(result)
    return out


# ── Deduplicate ───────────────────────────────────────────────────────────────

def _deduplicate_nearby(attractions: list[dict],
                        threshold_m: float = 150) -> tuple[list[dict], list[tuple]]:
    conf_rank = {"high": 3, "medium": 2, "llm-estimate": 1, "low": 0}
    kept: list[dict]     = []
    dropped: list[tuple] = []

    for a in attractions:
        if a.get("lat") is None:
            kept.append(a)
            continue
        merged = False
        for i, k in enumerate(kept):
            if k.get("lat") is None:
                continue
            dist = _haversine_m((a["lat"], a["lon"]), (k["lat"], k["lon"]))
            if dist <= threshold_m:
                a_score = (conf_rank.get(a["confidence"], 0), bool(a.get("website")))
                k_score = (conf_rank.get(k["confidence"], 0), bool(k.get("website")))
                if a_score > k_score:
                    dropped.append((k["name"], a["name"], dist))
                    kept[i] = a
                else:
                    dropped.append((a["name"], k["name"], dist))
                merged = True
                break
        if not merged:
            kept.append(a)
    return kept, dropped


# ── Public entry point ────────────────────────────────────────────────────────

def find_attractions(city: str, country: str, count: int = 10) -> AttractionsResult:
    """Discover and ground `count` verified tourist attractions for a city.

    Returns an AttractionsResult with the verified list (LLM hallucinations
    and low-confidence results dropped). May return fewer than `count` for
    small cities where the LLM hallucination rate is high.
    """
    t_start = time.time()
    ctx     = _fetch_city_polygon(city, country)

    fetch_count = max(count + 10, int(count * 2.0))

    print(f"\n=== Step 1: LLM — {fetch_count} attractions in {city}, {country} ===")
    attractions = _ask_llm(fetch_count, city, country)
    for a in attractions:
        print(f"  • {a['name']}  /  {a['local_name']}")

    print(f"\n=== Steps 2-5: parallel enrichment ({MAX_WORKERS} workers) ===")
    enriched = _parallel_enrich(attractions, ctx)
    enriched = [e for e in enriched if e["confidence"] not in ("llm-estimate", "low")]
    enriched, _ = _deduplicate_nearby(enriched, threshold_m=150)

    tried_names = {a["name"].lower() for a in attractions}
    tries       = 0
    while len(enriched) < count and tries < 3:
        tries += 1
        needed = (count - len(enriched)) + 3
        print(f"\n=== Refill {tries}: requesting {needed} more (have {len(enriched)}/{count}) ===")
        try:
            more = _ask_llm(needed, city, country, exclude=tried_names)
        except Exception as e:
            print(f"  [refill failed] {e}")
            break
        for a in more:
            print(f"  • {a['name']}  /  {a['local_name']}")
        tried_names.update(m["name"].lower() for m in more)
        more_enriched = _parallel_enrich(more, ctx)
        more_enriched = [e for e in more_enriched if e["confidence"] not in ("llm-estimate", "low")]
        combined, _ = _deduplicate_nearby(enriched + more_enriched, threshold_m=150)
        enriched = combined

    enriched = enriched[:count]
    elapsed  = time.time() - t_start

    print(f"\n=== Done: {len(enriched)} verified in {elapsed:.0f}s ===")

    return AttractionsResult(
        city            = city,
        country         = country,
        attractions     = [FoundAttraction(**e) for e in enriched],
        elapsed_seconds = elapsed,
    )


# ── Format helper for the CrewAI optimizer ────────────────────────────────────

CATEGORY_DURATION_MIN = {
    "museum":   90,
    "church":   30,
    "park":     60,
    "castle":   90,
    "monument": 20,
    "zoo":      120,
    "beach":    60,
    "other":    60,
}


def format_for_optimizer(result: AttractionsResult) -> str:
    """Convert AttractionsResult into the text format the optimizer task expects.

    The optimizer needs: name, city, lat/lon, duration_min, source URL.
    Duration is inferred from category. Source is the website (preferred) or
    an OpenStreetMap permalink based on coords.
    """
    lines = []
    for a in result.attractions:
        if a.lat is None or a.lon is None:
            continue
        duration = CATEGORY_DURATION_MIN.get(a.category, 60)
        source   = a.website or f"https://www.openstreetmap.org/?mlat={a.lat}&mlon={a.lon}&zoom=17"
        lines.append(
            f"- {a.name} ({result.city}, {result.country}) — "
            f"lat: {a.lat:.5f}, lon: {a.lon:.5f}, "
            f"duration_min: {duration}, source: {source}"
        )
    return f"Verified attractions ({len(lines)} items):\n" + "\n".join(lines)


# ── Map renderer (returns folium.Map; caller decides what to do with it) ──────

def build_attractions_map(result: AttractionsResult) -> folium.Map:
    """Return a Folium map with colour-coded pins for each attraction."""
    pts = [(a.lat, a.lon) for a in result.attractions if a.lat is not None]
    center = (sum(p[0] for p in pts)/len(pts),
              sum(p[1] for p in pts)/len(pts)) if pts else (0.0, 0.0)

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    pin_color = {"high": "green", "medium": "blue",
                 "llm-estimate": "orange", "low": "red"}

    for a in result.attractions:
        if a.lat is None:
            continue
        popup_html = (
            f"<b>{a.name}</b><br>"
            f"<i>{a.category}</i><br>"
            f"{a.short_description}<br>"
            + (f"<small>{a.address[:80]}</small><br>" if a.address else "")
            + (f"<a href='{a.website}' target='_blank'>Website</a>"
               if a.website else "<em>no website</em>")
            + f"<br><small>conf: {a.confidence} | {a.coord_source}</small>"
        )
        folium.Marker(
            location=[a.lat, a.lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{a.name} [{a.confidence}]",
            icon=folium.Icon(color=pin_color.get(a.confidence, "gray"),
                             icon="info-sign"),
        ).add_to(m)
    return m
