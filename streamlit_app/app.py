import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import folium
from streamlit_folium import st_folium
import streamlit as st
import streamlit.components.v1 as components

import time as _time

from core.config import MODEL
from core.email_sender import send_itinerary_email
from core.sms_sender import send_itinerary_sms
from core.guard import validate_destination
from core.routing import fetch_day_route, fetch_all_routes, fetch_intercity_routes
from core.map_builder import DAY_COLOURS, day_label, build_map, save_map
from core.crew_runner import run_crew, run_crew_with_attractions, trim_to_visit_budget
from core.attractions_finder import find_attractions, build_attractions_map, format_for_optimizer


def show_banner(message: str, level: str = "success") -> None:
    st.session_state["banner"] = {"message": message, "level": level, "ts": _time.time()}
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

    def _capitalize_destination():
        val = st.session_state.get("destination_input", "")
        if val:
            st.session_state["destination_input"] = val[0].upper() + val[1:]

    destination = st.text_input(
        "Destination",
        placeholder="e.g. Koszalin, Poland",
        help="Enter the destination as 'City, Country'.",
        key="destination_input",
        on_change=_capitalize_destination,
    )

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


_STATIC_DIR = Path(__file__).parent / "static"


# ── Itinerary renderer ─────────────────────────────────────────────────────────

def render_itinerary(itinerary: Itinerary, routes: dict,
                     intercity_routes: dict | None = None) -> None:
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

        ic = (intercity_routes or {}).get(day.day_number)
        if ic:
            ic_min = round(ic["legs"][0]["duration_s"] / 60)
            ic_km  = round(ic["legs"][0]["distance_m"] / 1000, 1)
            st.markdown(
                f'<div style="background:#f1f5f9;border-left:4px solid #64748b;'
                f'border-radius:0 6px 6px 0;padding:0.5rem 1rem;'
                f'font-size:0.875rem;color:#475569;margin:0.5rem 0 0.75rem 0;">'
                f'🚗 Travel to next city &nbsp;·&nbsp; '
                f'<strong>{ic["from_place"]}</strong> → <strong>{ic["to_place"]}</strong>'
                f'&nbsp;·&nbsp; {ic_km} km &nbsp;·&nbsp; ~{ic_min} min'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
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



# ── Main ───────────────────────────────────────────────────────────────────────
if load_btn and selected_file:
    try:
        itinerary        = Itinerary.model_validate_json(selected_file.read_text())
        routes           = fetch_all_routes(itinerary)
        intercity_routes = fetch_intercity_routes(itinerary)
        save_map(itinerary, routes, _STATIC_DIR, intercity_routes)
        st.session_state["itinerary"]        = itinerary
        st.session_state["usage"]            = None
        st.session_state["routes"]           = routes
        st.session_state["intercity_routes"] = intercity_routes
        st.session_state.pop("attractions_result", None)
    except Exception as e:
        st.error(f"Could not load itinerary: {e}")

if generate:
    if not destination:
        st.error("Please enter a destination as 'City, Country'.")
    elif "," not in destination:
        st.error("Destination must be in 'City, Country' format (e.g. 'Koszalin, Poland').")
    elif end_date < start_date:
        st.error("End date must be on or after start date.")
    elif explore_days is None:
        st.error("Please select valid dates first.")
    else:
        city, country = [s.strip() for s in destination.split(",", 1)]
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt   = datetime.combine(end_date,   datetime.min.time())

        # Count = total exploration hours (≈ 1 attraction per hour average)
        attractions_count = explore_days * hours_per_day

        st.session_state.pop("usage", None)

        # ── Step 1: find verified attractions ─────────────────────────────────
        with st.spinner(f"Step 1/2: Finding {attractions_count} attractions in {city}, {country}… (typically 3–5 min)"):
            try:
                attractions_result = find_attractions(city, country, attractions_count)
            except Exception as e:
                st.error(f"Attractions finder failed: {e}")
                st.stop()

        if not attractions_result.attractions:
            st.error(f"No verifiable attractions found in {city}, {country}.")
            st.stop()

        attractions_text = format_for_optimizer(attractions_result)

        # ── Step 2: optimize + plan into a multi-day itinerary ────────────────
        with st.spinner(f"Step 2/2: Planning {explore_days}-day itinerary…"):
            last_error = None
            for attempt in range(1, 4):
                try:
                    itinerary, usage = run_crew_with_attractions(
                        attractions_text, destination, start_dt, end_dt,
                        explore_days, hours_per_day,
                    )
                    itinerary        = trim_to_visit_budget(itinerary, hours_per_day)
                    routes           = fetch_all_routes(itinerary)
                    intercity_routes = fetch_intercity_routes(itinerary)
                    save_map(itinerary, routes, _STATIC_DIR, intercity_routes)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < 3:
                        st.toast(f"Attempt {attempt} failed, retrying…")
            if last_error:
                st.error(f"Could not plan itinerary after 3 attempts: {last_error}")
                st.stop()

        st.session_state["itinerary"]          = itinerary
        st.session_state["usage"]              = usage
        st.session_state["routes"]             = routes
        st.session_state["intercity_routes"]   = intercity_routes
        st.session_state["attractions_result"] = attractions_result

        output_file = f"itinerary_{destination.replace(' ', '_').replace(',', '')}_{start_date}.json"
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
    itinerary        = st.session_state["itinerary"]
    usage            = st.session_state["usage"]
    routes           = st.session_state.get("routes", {})
    intercity_routes = st.session_state.get("intercity_routes", {})

    st.subheader("📅 Itinerary")
    render_itinerary(itinerary, routes, intercity_routes)

    if usage:
        render_usage(usage, MODEL)

