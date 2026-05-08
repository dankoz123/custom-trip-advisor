import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import requests
import folium
from streamlit_folium import st_folium
import streamlit as st
import streamlit.components.v1 as components
from crewai import Crew, Process

import time as _time

from crewai import LLM
from core.config import MODEL, ANTHROPIC_API_KEY
from core.email_sender import send_itinerary_email
from core.sms_sender import send_itinerary_sms
from core.guard import validate_destination


def show_banner(message: str, level: str = "success") -> None:
    st.session_state["banner"] = {"message": message, "level": level, "ts": _time.time()}
from core.agents import make_researcher, make_optimizer, make_planner
from core.tasks import make_research_task, make_optimization_task, make_planning_task
from core.models import Itinerary
from crewai.types.usage_metrics import UsageMetrics

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Custom Trip Advisor", page_icon="✈️", layout="wide")
st.markdown("""
<style>
/* Open Trip Map link text */
a[href="/app/static/map.html"],
a[href="/app/static/map.html"] div,
a[href="/app/static/map.html"]:visited {
    color: white !important;
}
/* Target primary buttons across Streamlit versions */
button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
    background-color: #2563EB !important;
    border-color:     #2563EB !important;
    color:            white   !important;
}
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color:     #1d4ed8 !important;
}
</style>
""", unsafe_allow_html=True)
st.title("✈️ Custom Trip Advisor")


# ── Sidebar — inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Plan your trip")

    destination = st.text_input("Destination", placeholder="e.g. Japan, Italy, Peru")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date")
    with col2:
        end_date = st.date_input("End date")

    if start_date and end_date and end_date >= start_date:
        total_days = (end_date - start_date).days + 1
        if total_days > 1:
            explore_days = st.slider("Days to explore", min_value=1, max_value=total_days, value=total_days)
        else:
            explore_days = 1
            st.info("1-day trip selected.")
    else:
        explore_days = None

    hours_per_day = st.slider(
        "Hours to explore per day",
        min_value=1, max_value=12, value=8, step=1,
        help="Visits per day will be between 90% and 100% of this limit. Driving time is excluded.",
    )

    generate = st.button("Generate Itinerary", type="primary", use_container_width=True)

    # ── Load a previously saved itinerary ─────────────────────────────────────
    st.divider()
    st.subheader("Load saved itinerary")
    itinerary_dir = Path(__file__).parent.parent / "itinerary"
    def _sort_key(p: Path):
        try:
            import json
            d = json.loads(p.read_text())
            return (d.get("destination", "").upper(), d.get("start_date", ""), d.get("total_days", 0))
        except Exception:
            return (p.name.upper(), "", 0)
    saved_files = sorted(itinerary_dir.glob("*.json"), key=_sort_key) if itinerary_dir.exists() else []
    if saved_files:
        def _file_label(p: Path) -> str:
            try:
                import json
                d = json.loads(p.read_text())
                return f"{d['destination'].upper()}  ·  {d['start_date']}  ·  {d['total_days']} days"
            except Exception:
                return p.name

        selected_file = st.selectbox(
            "Select itinerary",
            options=saved_files,
            format_func=_file_label,
        )
        load_btn = st.button("Load", use_container_width=True)
    else:
        st.caption("No saved itineraries found.")
        load_btn = False
        selected_file = None


# ── Day colour palette ─────────────────────────────────────────────────────────
_DAY_COLOURS = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#9B59B6",  # purple
    "#E67E22",  # orange
    "#1ABC9C",  # teal
    "#E91E63",  # pink
    "#34495E",  # dark slate
    "#F39C12",  # amber
    "#00BCD4",  # cyan
]

def _day_label(day_number: int) -> str:
    if day_number <= 9:
        return str(day_number)
    return chr(ord("A") + day_number - 10)


# ── OSRM road routing ──────────────────────────────────────────────────────────

def fetch_day_route(attractions) -> dict | None:
    """Fetch real driving route from OSRM for a day's attractions."""
    located = [a for a in attractions if a.lat is not None and a.lon is not None]
    if len(located) < 2:
        return None
    coords = ";".join(f"{a.lon},{a.lat}" for a in located)
    try:
        resp = requests.get(
            f"http://router.project-osrm.org/route/v1/driving/{coords}",
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != "Ok":
            return None
        route = data["routes"][0]
        return {
            "geometry": route["geometry"]["coordinates"],
            "legs": [
                {"distance_m": leg["distance"], "duration_s": leg["duration"]}
                for leg in route["legs"]
            ],
        }
    except Exception:
        return None


def trim_to_visit_budget(itinerary: Itinerary, hours_per_day: float) -> Itinerary:
    """Remove trailing attractions from any day whose visit time exceeds the budget."""
    max_min = int(hours_per_day * 60)
    for day in itinerary.days:
        while len(day.attractions) > 1:
            if sum(a.duration_min for a in day.attractions) <= max_min:
                break
            day.attractions.pop()
    return itinerary


def fetch_all_routes(itinerary: Itinerary) -> dict:
    """Return {day_number: route_data} for every day with ≥2 attractions."""
    routes = {}
    for day in itinerary.days:
        route = fetch_day_route(day.attractions)
        if route:
            routes[day.day_number] = route
    return routes


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(itinerary: Itinerary, routes: dict) -> folium.Map:
    all_lats = [a.lat for d in itinerary.days for a in d.attractions if a.lat is not None]
    all_lons = [a.lon for d in itinerary.days for a in d.attractions if a.lon is not None]
    centre = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)) if all_lats else (20, 0)

    m = folium.Map(location=centre, zoom_start=7, control_scale=True)

    for day in itinerary.days:
        colour = _DAY_COLOURS[(day.day_number - 1) % len(_DAY_COLOURS)]
        label  = _day_label(day.day_number)

        if day.day_number in routes:
            road_coords = [(lat, lon) for lon, lat in routes[day.day_number]["geometry"]]
            total_drive_s = sum(l["duration_s"] for l in routes[day.day_number]["legs"])
            total_drive_min = round(total_drive_s / 60)
            folium.PolyLine(
                road_coords,
                color=colour,
                weight=4,
                opacity=0.8,
                tooltip=f"Day {day.day_number}: {day.title} · {total_drive_min} min driving",
            ).add_to(m)
        else:
            coords = [(a.lat, a.lon) for a in day.attractions if a.lat is not None and a.lon is not None]
            if len(coords) > 1:
                folium.PolyLine(coords, color=colour, weight=3, opacity=0.6,
                                tooltip=f"Day {day.day_number}: {day.title}").add_to(m)

        for attr in day.attractions:
            if attr.lat is None or attr.lon is None:
                continue
            folium.Marker(
                location=(attr.lat, attr.lon),
                popup=folium.Popup(
                    f"<b>{attr.place}</b><br>Day {day.day_number} · {attr.time}<br>{attr.duration_min} min",
                    max_width=220,
                ),
                tooltip=attr.place,
                icon=folium.DivIcon(
                    html=(
                        f'<div style="background:{colour};color:white;border-radius:50%;'
                        f'width:28px;height:28px;display:flex;align-items:center;'
                        f'justify-content:center;font-weight:bold;font-size:13px;'
                        f'border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,.4);">'
                        f'{label}</div>'
                    ),
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
            ).add_to(m)

    return m


def save_map(itinerary: Itinerary, routes: dict) -> None:
    """Build the map and save it as a standalone HTML file in the static folder."""
    m = build_map(itinerary, routes)
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    m.save(str(static_dir / "map.html"))


# ── Itinerary renderer ─────────────────────────────────────────────────────────

def render_itinerary(itinerary: Itinerary, routes: dict) -> None:
    st.markdown(
        f"### {itinerary.destination} &nbsp;·&nbsp; "
        f"{itinerary.start_date} → {itinerary.end_date} &nbsp;·&nbsp; "
        f"{itinerary.total_days} days"
    )
    st.divider()

    for day in itinerary.days:
        st.markdown(f"#### Day {day.day_number} &nbsp;·&nbsp; {day.date} &nbsp;·&nbsp; {day.title}")
        st.caption(day.description)

        day_route = routes.get(day.day_number)
        header = st.columns([1, 4, 1.5, 2])
        header[0].markdown("**Time**")
        header[1].markdown("**Place**")
        header[2].markdown("**Duration**")
        header[3].markdown("**Source**")

        total_visit_min = 0
        total_drive_min = 0

        for i, attr in enumerate(day.attractions):
            col_time, col_place, col_dur, col_src = st.columns([1, 4, 1.5, 2])
            col_time.write(attr.time)
            col_place.write(attr.place)
            col_dur.write(f"{attr.duration_min} min")
            col_src.markdown(f"[link]({attr.source})")
            total_visit_min += attr.duration_min

            if day_route and i < len(day.attractions) - 1:
                leg = day_route["legs"][i]
                drive_min = round(leg["duration_s"] / 60)
                drive_km  = leg["distance_m"] / 1000
                total_drive_min += drive_min
                st.markdown(
                    f'<div style="color:#888;font-size:12px;padding:2px 0 2px 8px;">'
                    f'🚗 &nbsp;{drive_min} min &nbsp;·&nbsp; {drive_km:.1f} km</div>',
                    unsafe_allow_html=True,
                )

        drive_part = (
            f" &nbsp;+&nbsp; 🚗 **{total_drive_min} min driving** (additional)"
            if total_drive_min else ""
        )
        st.caption(
            f"⏱ Exploring: **{total_visit_min // 60}h {total_visit_min % 60:02d}min** "
            f"({len(day.attractions)} attractions)"
            f"{drive_part}"
        )
        st.write("")

    st.divider()
    col_map, col_email, col_sms = st.columns(3)
    with col_map:
        st.markdown(
            '<style>.map-btn-wrap p{margin:0!important}</style>'
            '<div class="map-btn-wrap">'
            '<a href="/app/static/map.html" target="_blank" '
            'style="text-decoration:none;color:white!important;display:block;">'
            '<div style="background:#2563EB;color:white!important;border-radius:8px;'
            'padding:0.55rem 0.75rem;font-size:1rem;font-weight:400;cursor:pointer;'
            'text-align:center;line-height:1.6;width:100%;">🗺️ &nbsp;Open Trip Map</div>'
            '</a></div>',
            unsafe_allow_html=True,
        )
    with col_email:
        send_email_btn = st.button("📧 Send Email", use_container_width=True, type="primary")
        email_to = st.text_input(
            " ",
            placeholder="name@example.com",
            key="email_to",
            label_visibility="visible",
            help="Provide an email address and select the 'Send Email' button.",
        )
    with col_sms:
        send_sms_btn = st.button("💬 Send SMS", use_container_width=True, type="primary")
        st.text_input(" ", value="IN DEVELOPMENT", disabled=True, label_visibility="visible")

    if send_sms_btn:
        if not phone_to:
            show_banner("Enter a recipient phone number.", "warning")
        else:
            with st.spinner("Sending SMS…"):
                ok, err = send_itinerary_sms(phone_to, itinerary)
            if ok:
                show_banner("SMS sent")
            else:
                show_banner(f"Failed to send SMS: {err}", "error")
        st.rerun()

    if send_email_btn:
        if not email_to:
            show_banner("Enter a recipient email address.", "warning")
        else:
            with st.spinner("Sending email…"):
                ok, err = send_itinerary_email(email_to, itinerary, routes)
            if ok:
                show_banner("Email sent")
            else:
                show_banner(f"Failed to send email: {err}", "error")
        st.rerun()


# ── Usage renderer ─────────────────────────────────────────────────────────────

def render_usage(usage: UsageMetrics, model: str) -> None:
    uncached = usage.prompt_tokens - usage.cached_prompt_tokens
    cost_in  = (uncached                   / 1_000_000) * 0.80
    cost_ca  = (usage.cached_prompt_tokens / 1_000_000) * 0.08
    cost_out = (usage.completion_tokens    / 1_000_000) * 4.00
    total    = cost_in + cost_ca + cost_out

    with st.expander("Usage & Cost", expanded=True):
        st.caption(f"Model: `{model}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prompt tokens",     f"{usage.prompt_tokens:,}")
        c2.metric("Completion tokens", f"{usage.completion_tokens:,}")
        c3.metric("Total tokens",      f"{usage.total_tokens:,}")
        c4.metric("Total cost (USD)",  f"${total:.6f}")


# ── Crew runner ────────────────────────────────────────────────────────────────

def run_crew(destination: str, start_dt: datetime, end_dt: datetime,
             explore: int, hours_per_day: float) -> tuple:
    START_DATE = start_dt.strftime("%Y-%m-%d")
    END_DATE   = end_dt.strftime("%Y-%m-%d")
    date_range = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(explore)]

    llm = LLM(model=MODEL, temperature=0.3, max_tokens=16000)
    researcher        = make_researcher(destination, llm)
    optimizer         = make_optimizer(destination, explore, hours_per_day, llm)
    planner           = make_planner(destination, explore, START_DATE, hours_per_day, llm)
    research_task     = make_research_task(researcher, destination, explore, date_range, hours_per_day)
    optimization_task = make_optimization_task(optimizer, research_task, destination, explore, hours_per_day)
    planning_task     = make_planning_task(
        planner, optimization_task, destination,
        START_DATE, END_DATE, explore, date_range, hours_per_day,
    )
    crew = Crew(
        agents=[researcher, optimizer, planner],
        tasks=[research_task, optimization_task, planning_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return result.pydantic, result.token_usage


# ── Main ───────────────────────────────────────────────────────────────────────
if load_btn and selected_file:
    try:
        itinerary = Itinerary.model_validate_json(selected_file.read_text())
        routes = fetch_all_routes(itinerary)
        save_map(itinerary, routes)
        st.session_state["itinerary"] = itinerary
        st.session_state["usage"]     = None
        st.session_state["routes"]    = routes
    except Exception as e:
        st.error(f"Could not load itinerary: {e}")

if generate:
    if not destination:
        st.error("Please enter a destination.")
    elif end_date < start_date:
        st.error("End date must be on or after start date.")
    elif explore_days is None:
        st.error("Please select valid dates first.")
    else:
        with st.spinner("Validating destination…"):
            is_valid, reason = validate_destination(destination)
        if not is_valid:
            st.error(f'"{destination}" is not a valid travel destination: {reason}')
        else:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt   = datetime.combine(end_date,   datetime.min.time())

            st.session_state.pop("usage", None)

            with st.spinner(f"Planning your {explore_days}-day trip to {destination}…"):
                last_error = None
                for attempt in range(1, 4):
                    try:
                        itinerary, usage = run_crew(destination, start_dt, end_dt, explore_days, hours_per_day)
                        itinerary = trim_to_visit_budget(itinerary, hours_per_day)
                        routes = fetch_all_routes(itinerary)
                        save_map(itinerary, routes)
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < 3:
                            st.toast(f"Attempt {attempt} failed, retrying…")
                if last_error:
                    st.error(f"Could not generate itinerary after 3 attempts: {last_error}")
                    st.stop()

            st.session_state["itinerary"] = itinerary
            st.session_state["usage"]     = usage
            st.session_state["routes"]    = routes

            output_file = f"itinerary_{destination.replace(' ', '_')}_{start_date}.json"
            itinerary_dir = Path(__file__).parent.parent / "itinerary"
            itinerary_dir.mkdir(exist_ok=True)
            (itinerary_dir / output_file).write_text(itinerary.model_dump_json(indent=2))

# ── Notification banner ────────────────────────────────────────────────────────
_banner_slot = st.empty()

if "banner" in st.session_state:
    b = st.session_state["banner"]
    fn = {"success": _banner_slot.success,
          "error":   _banner_slot.error,
          "warning": _banner_slot.warning}.get(b["level"], _banner_slot.success)
    fn(b["message"])

@st.fragment(run_every=1)
def _banner_watchdog():
    b = st.session_state.get("banner")
    if b and _time.time() - b["ts"] >= 10:
        del st.session_state["banner"]
        st.rerun()

_banner_watchdog()

if "itinerary" in st.session_state:
    itinerary = st.session_state["itinerary"]
    usage     = st.session_state["usage"]
    routes    = st.session_state.get("routes", {})

    st.subheader("📅 Itinerary")
    render_itinerary(itinerary, routes)

    if usage:
        render_usage(usage, MODEL)
