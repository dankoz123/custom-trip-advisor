"""
Attractions grounding module (free stack, polygon-bounded).

Converts LLM-curated attraction NAMES into verified attractions with accurate
lat/lon, official website, and reverse-geocoded address.

Public API:
    generate_attractions(city, country, attractions) -> dict
    save_outputs(payload, slug=None, output_dir=".") -> (json_path, html_path)
    print_summary(payload) -> None
    fetch_city_bounds(city, country) -> dict

Pipeline per attraction:
  1. LLM-curated names + descriptions (caller-supplied)
  2. Wikidata SPARQL  -> official website + candidate coordinates + Wikipedia
  3. Nominatim search -> candidate coordinates (always, for cross-validation)
  4. Reconcile WD vs OSM coords (Haversine), reject anything outside the city polygon
  5. Nominatim reverse-geocode -> human-readable address (street-level)
  6. Folium HTML map with pins + city outline

All sources are free and key-less:
  - Wikidata Query Service (SPARQL)
  - OSM Nominatim
  - OSM tile server (via Folium / Leaflet)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import requests
from shapely.geometry import shape, Point
from shapely.geometry.base import BaseGeometry

from core.models import ResearcherOutput

USER_AGENT = "TripAdvisorSimplified/Grounder (multi-agent pipeline)"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"

COORD_DISAGREE_THRESHOLD_M = 150     # WD vs OSM disagreement → prefer OSM
NOMINATIM_POLITENESS_S = 1.1         # min interval between Nominatim requests
WIKIDATA_POLITENESS_S = 1.2          # min interval between Wikidata requests


# ----------------------------- helpers ---------------------------------------

def haversine_m(a: tuple, b: tuple) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000
    la1, lo1 = map(math.radians, a)
    la2, lo2 = map(math.radians, b)
    dla, dlo = la2 - la1, lo2 - lo1
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _get_with_retry(url: str, **kw) -> requests.Response:
    """GET with exponential backoff on transient 4xx/5xx + network errors."""
    delay = 2.0
    timeout = kw.pop("timeout", 20)
    last_exc = None
    for _ in range(5):
        try:
            r = requests.get(url, timeout=timeout, **kw)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            print(f"   [retry] {type(e).__name__}, sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After", delay))
            print(f"   [retry] {r.status_code} on {url[:60]}…, sleeping {wait:.1f}s")
            time.sleep(wait)
            delay = min(delay * 2, 30)
            continue
        r.raise_for_status()
        return r
    if last_exc:
        raise last_exc
    r.raise_for_status()
    return r


# ----------------------------- city bounds -----------------------------------

def fetch_city_bounds(city: str, country: str) -> dict:
    """Resolve a city to its centroid + administrative polygon.

    Returns:
      {
        "centroid": (lat, lon),
        "polygon":  shapely geometry (Polygon or MultiPolygon),
        "polygon_geojson": dict (GeoJSON for map rendering / serialization),
        "max_extent_km": furthest distance from centroid to polygon vertex,
        "display_name": OSM display_name of the resolved area,
      }
    """
    r = _get_with_retry(NOMINATIM,
                       params={"q": f"{city}, {country}",
                               "format": "json",
                               "polygon_geojson": 1,
                               "limit": 5},
                       headers={"User-Agent": USER_AGENT}, timeout=20)
    hits = r.json()
    if not hits:
        raise RuntimeError(f"Could not geocode city: {city}, {country}")

    chosen = None
    for hit in hits:
        gj = hit.get("geojson")
        if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
            chosen = hit
            break
    if chosen is None:
        raise RuntimeError(
            f"No administrative polygon returned by Nominatim for {city}, {country}. "
            "Try a more specific query (e.g. include the region)."
        )

    centroid = (float(chosen["lat"]), float(chosen["lon"]))
    polygon = shape(chosen["geojson"])

    # Walk every polygon vertex and compute the furthest haversine distance.
    max_m = 0.0
    geoms = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
    for g in geoms:
        for lon, lat in g.exterior.coords:
            d = haversine_m(centroid, (lat, lon))
            if d > max_m:
                max_m = d

    return {
        "centroid": centroid,
        "polygon": polygon,
        "polygon_geojson": chosen["geojson"],
        "max_extent_km": round(max_m / 1000, 1),
        "display_name": chosen.get("display_name", ""),
    }


def in_city(coord: tuple, polygon: BaseGeometry) -> bool:
    """Point-in-polygon test. coord = (lat, lon); shapely uses (x, y) = (lon, lat)."""
    lat, lon = coord
    p = Point(lon, lat)
    return polygon.contains(p) or polygon.touches(p)


# ----------------------------- wikidata --------------------------------------

def find_wikidata_id(name: str) -> str | None:
    """Resolve a name to a Wikidata QID via the search API."""
    r = _get_with_retry(WIKIDATA_SEARCH,
                       params={"action": "wbsearchentities",
                               "search": name, "language": "en",
                               "format": "json", "limit": 1},
                       headers={"User-Agent": USER_AGENT}, timeout=15)
    hits = r.json().get("search", [])
    return hits[0]["id"] if hits else None


def wikidata_facts(qid: str) -> dict:
    """SPARQL: official website (P856), coordinates (P625), English Wikipedia link."""
    query = f"""
    SELECT ?website ?coord ?article WHERE {{
      OPTIONAL {{ wd:{qid} wdt:P856 ?website. }}
      OPTIONAL {{ wd:{qid} wdt:P625 ?coord. }}
      OPTIONAL {{
        ?article schema:about wd:{qid};
                 schema:isPartOf <https://en.wikipedia.org/>.
      }}
    }} LIMIT 1
    """
    r = _get_with_retry(WIKIDATA_SPARQL,
                       params={"query": query, "format": "json"},
                       headers={"User-Agent": USER_AGENT,
                                "Accept": "application/sparql-results+json"},
                       timeout=20)
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
        out["latitude"] = float(lat)
        out["longitude"] = float(lon)
    if "article" in row:
        out["wikipedia"] = row["article"]["value"]
    return out


# ----------------------------- nominatim -------------------------------------

def nominatim_geocode_raw(q: str) -> dict:
    r = _get_with_retry(NOMINATIM,
                       params={"q": q, "format": "json", "limit": 1},
                       headers={"User-Agent": USER_AGENT}, timeout=15)
    hits = r.json()
    if not hits:
        return {}
    return {"latitude": float(hits[0]["lat"]),
            "longitude": float(hits[0]["lon"])}


def nominatim_reverse(lat: float, lon: float, lang: str = "en") -> str:
    """Reverse-geocode coordinates to a human-readable address (street-level)."""
    try:
        r = _get_with_retry(NOMINATIM_REVERSE,
                            params={"lat": lat, "lon": lon, "format": "json",
                                    "addressdetails": 1, "zoom": 17,
                                    "accept-language": lang},
                            headers={"User-Agent": USER_AGENT}, timeout=15)
        return r.json().get("display_name", "")
    except Exception:
        return ""


# ----------------------------- enrichment ------------------------------------

def enrich(attraction: dict, city: str, country: str, polygon: BaseGeometry) -> dict:
    out = dict(attraction)
    out.update({"website": "", "latitude": None, "longitude": None,
                "wikipedia": "", "confidence": "low",
                "source": [], "address": ""})

    # --- Wikidata (website + candidate coords) — isolated, never aborts ---
    wd_coord = None
    try:
        qid = (find_wikidata_id(attraction["name"])
               or find_wikidata_id(attraction["local_name"]))
        if qid:
            out["wikidata_id"] = qid
            facts = wikidata_facts(qid)
            if facts.get("website"):
                out["website"] = facts["website"]
            if facts.get("wikipedia"):
                out["wikipedia"] = facts["wikipedia"]
            if "latitude" in facts:
                cand = (facts["latitude"], facts["longitude"])
                if in_city(cand, polygon):
                    wd_coord = cand
                else:
                    print(f"   [reject] WD coord outside city polygon, ignoring")
                    out["wikidata_out_of_city"] = True
            out["source"].append("wikidata")
    except Exception as e:
        print(f"   [wikidata-fail] {type(e).__name__}: {e}")
        out["source"].append("wikidata-error")

    # --- Nominatim (always, for cross-validation), city-scoped queries ---
    nm_coord = None
    local_short = " ".join(attraction["local_name"].split()[:3])
    queries = [
        f"{attraction['local_name']}, {city}, {country}",
        f"{attraction['name']}, {city}, {country}",
        f"{attraction['local_name'].split(',')[0].split(' im.')[0]}, {city}",
        f"{local_short} {city}",
    ]
    for q in queries:
        time.sleep(NOMINATIM_POLITENESS_S)
        geo = nominatim_geocode_raw(q)
        if not geo:
            continue
        cand = (geo["latitude"], geo["longitude"])
        if in_city(cand, polygon):
            nm_coord = cand
            out["source"].append(f"nominatim:{q[:40]}")
            break

    # --- Reconcile coordinates ---
    if wd_coord and nm_coord:
        dist = haversine_m(wd_coord, nm_coord)
        out["coord_disagreement_m"] = round(dist, 1)
        if dist <= COORD_DISAGREE_THRESHOLD_M:
            out["latitude"], out["longitude"] = wd_coord
            out["coord_source"] = "wikidata (agrees with nominatim)"
            out["confidence"] = "high"
        else:
            out["latitude"], out["longitude"] = nm_coord
            out["coord_source"] = "nominatim (wikidata disagreed)"
            out["confidence"] = "medium"
    elif wd_coord:
        out["latitude"], out["longitude"] = wd_coord
        out["coord_source"] = "wikidata only"
        out["confidence"] = "medium"
    elif nm_coord:
        out["latitude"], out["longitude"] = nm_coord
        out["coord_source"] = "nominatim only"
        out["confidence"] = "medium"

    # --- Reverse geocode for full address ---
    if out["latitude"] is not None:
        time.sleep(NOMINATIM_POLITENESS_S)
        out["address"] = nominatim_reverse(out["latitude"], out["longitude"])

    return out


# ----------------------------- map rendering ---------------------------------

def build_map(attractions: list[dict],
              polygon_geojson: dict | None,
              centroid: tuple,
              out_path: str) -> None:
    import folium
    pts = [(a["latitude"], a["longitude"]) for a in attractions
           if a["latitude"] is not None]
    center = (sum(p[0] for p in pts) / len(pts),
              sum(p[1] for p in pts) / len(pts)) if pts else centroid

    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # City boundary outline (faint, no fill)
    if polygon_geojson is not None:
        folium.GeoJson(
            polygon_geojson,
            name="city boundary",
            style_function=lambda _: {"color": "#3366cc", "weight": 2,
                                       "fillOpacity": 0.05},
        ).add_to(m)

    for a in attractions:
        if a["latitude"] is None:
            continue
        popup = (f"<b>{a['name']}</b><br>"
                 f"<i>{a['category']}</i><br>"
                 f"{a['short_description']}<br>"
                 + (f"<small>{a.get('address','')}</small><br>"
                    if a.get("address") else "")
                 + (f"<a href='{a['website']}' target='_blank'>Website</a>"
                    if a["website"] else "<em>no website</em>"))
        folium.Marker(
            location=[a["latitude"], a["longitude"]],
            popup=folium.Popup(popup, max_width=300),
            tooltip=a["name"],
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)
    m.save(out_path)


# ----------------------------- public API ------------------------------------

def generate_attractions(city: str,
                         country: str,
                         attractions: list[dict]) -> dict:
    """Ground a list of LLM-curated attractions for a given city.

    Args:
      city: city name (e.g. "Koszalin", "Ningbo")
      country: country name (e.g. "Poland", "China")
      attractions: list of dicts with keys: name, local_name, category, short_description

    Returns:
      Payload dict with city metadata + enriched attractions.
    """
    print(f"Resolving city polygon for {city}, {country}...")
    bounds = fetch_city_bounds(city, country)
    centroid = bounds["centroid"]
    polygon = bounds["polygon"]
    print(f"  centroid: {centroid[0]:.5f}, {centroid[1]:.5f}")
    print(f"  max extent: {bounds['max_extent_km']} km from centroid")
    print(f"  OSM area: {bounds['display_name'][:80]}\n")

    enriched = []
    for a in attractions:
        print(f"-> {a['name']}")
        try:
            enriched.append(enrich(a, city, country, polygon))
        except Exception as e:
            print(f"   [error] {type(e).__name__}: {e}")
            partial = dict(a)
            partial.update({"website": "", "latitude": None, "longitude": None,
                            "wikipedia": "", "confidence": "low",
                            "source": ["error"], "address": "", "error": str(e)})
            enriched.append(partial)
        time.sleep(WIKIDATA_POLITENESS_S)

    return {
        "location_query": city,
        "country": country,
        "centroid": {"latitude": centroid[0], "longitude": centroid[1]},
        "max_extent_km": bounds["max_extent_km"],
        "polygon_geojson": bounds["polygon_geojson"],
        "attractions": enriched,
    }


def save_outputs(payload: dict,
                 slug: str | None = None,
                 output_dir: str | Path = ".") -> tuple[str, str]:
    """Save JSON + Folium HTML to disk. Returns (json_path, html_path)."""
    if slug is None:
        slug = payload["location_query"].lower().replace(" ", "_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"attractions_{slug}.json"
    html_path = output_dir / f"attractions_{slug}.html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    build_map(payload["attractions"],
              payload.get("polygon_geojson"),
              (payload["centroid"]["latitude"], payload["centroid"]["longitude"]),
              str(html_path))
    return str(json_path), str(html_path)


def print_summary(payload: dict) -> None:
    print("\n--- Summary ---")
    print(f"City: {payload['location_query']}, {payload['country']}  "
          f"(max extent {payload['max_extent_km']} km)")
    for a in payload["attractions"]:
        coord = (f"{a['latitude']:.5f}, {a['longitude']:.5f}"
                 if a["latitude"] is not None else "NO COORDS")
        web = a["website"] or "(no website)"
        disagree = (f"  [WD vs OSM diff: {a['coord_disagreement_m']} m]"
                    if a.get("coord_disagreement_m") is not None else "")
        print(f"  {a['name']}\n    {coord}  conf={a['confidence']}{disagree}")
        print(f"    src: {a.get('coord_source','?')}")
        print(f"    addr: {a.get('address','') or '-'}")
        print(f"    {web}")


# ----------------------------- crew integration ------------------------------

def ground_research_output(research: ResearcherOutput,
                           city: str,
                           country: str,
                           filter_unverified: bool = True) -> list[dict]:
    """Bridge from Researcher's pydantic output to grounded attractions.

    Behavior:
      - Tries to fully ground each attraction via generate_attractions().
      - If the city polygon can't be fetched, falls back entirely to the LLM's
        estimated lat/lon for every attraction (confidence="low").
      - If grounding succeeds but a specific attraction couldn't be located,
        keeps that one's LLM-estimated lat/lon (confidence="low").
      - duration_min flows through from Researcher unchanged.

    Returns a list of dicts (same shape as generate_attractions()['attractions'])
    with duration_min added on each item.
    """
    attractions_for_grounder = [
        {"name": a.name,
         "local_name": a.local_name or a.name,
         "category": a.category,
         "short_description": a.short_description}
        for a in research.attractions
    ]

    try:
        payload = generate_attractions(city, country, attractions_for_grounder)
        grounded = payload["attractions"]
    except Exception as e:
        print(f"   [city-bounds-fail] {type(e).__name__}: {e} — using LLM estimates")
        grounded = []
        for a in research.attractions:
            grounded.append({
                "name": a.name,
                "local_name": a.local_name,
                "category": a.category,
                "short_description": a.short_description,
                "latitude": a.lat,
                "longitude": a.lon,
                "address": "",
                "website": "",
                "wikipedia": "",
                "confidence": "low",
                "coord_source": "llm-estimate",
                "source": ["llm-estimate"],
            })

    # Per-attraction fallback + carry through duration_min + assign id
    for idx, (grounded_attr, researcher_attr) in enumerate(zip(grounded, research.attractions)):
        if grounded_attr.get("latitude") is None and researcher_attr.lat is not None:
            grounded_attr["latitude"] = researcher_attr.lat
            grounded_attr["longitude"] = researcher_attr.lon
            grounded_attr["confidence"] = "low"
            grounded_attr["coord_source"] = "llm-estimate"
        grounded_attr["duration_min"] = researcher_attr.duration_min
        grounded_attr["id"] = idx

    if filter_unverified:
        before = len(grounded)
        grounded = [g for g in grounded if g.get("confidence") in ("high", "medium")]
        print(f"   [filter] kept {len(grounded)}/{before} verified attractions")

    return grounded
