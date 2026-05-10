import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.guard import validate_destination
from core.models import Itinerary
from core.email_sender import send_itinerary_email
from core.routing import fetch_all_routes
from core.map_builder import DAY_COLOURS, save_map
from core.crew_runner import run_crew, trim_to_visit_budget
from core.config import MODEL

# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
STATIC_DIR    = BASE_DIR / "static"
ITINERARY_DIR = BASE_DIR.parent / "itinerary"

app = FastAPI(title="Custom Trip Advisor")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _list_saved() -> list[dict]:
    if not ITINERARY_DIR.exists():
        return []
    result = []
    for f in sorted(ITINERARY_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            result.append({
                "filename":    f.name,
                "destination": d.get("destination", ""),
                "start_date":  d.get("start_date", ""),
                "total_days":  d.get("total_days", 0),
            })
        except Exception:
            pass
    return sorted(result, key=lambda x: (x["destination"].upper(), x["start_date"], x["total_days"]))


def _save_itinerary(itinerary: Itinerary) -> str:
    ITINERARY_DIR.mkdir(exist_ok=True)
    filename = f"itinerary_{itinerary.destination.replace(' ', '_')}_{itinerary.start_date}.json"
    (ITINERARY_DIR / filename).write_text(itinerary.model_dump_json(indent=2))
    return filename


def _build_day_data(itinerary: Itinerary, routes: dict) -> list[dict]:
    days = []
    for day in itinerary.days:
        day_route = routes.get(day.day_number)
        total_visit = 0
        total_drive = 0
        rows = []
        for i, attr in enumerate(day.attractions):
            total_visit += attr.duration_min
            rows.append({"type": "attraction", "attr": attr})
            if day_route and i < len(day.attractions) - 1:
                leg = day_route["legs"][i]
                drive_min = round(leg["duration_s"] / 60)
                drive_km  = leg["distance_m"] / 1000
                total_drive += drive_min
                rows.append({"type": "drive", "drive_min": drive_min, "drive_km": round(drive_km, 1)})
        days.append({
            "day":         day,
            "rows":        rows,
            "total_visit": total_visit,
            "total_drive": total_drive,
            "colour":      DAY_COLOURS[(day.day_number - 1) % len(DAY_COLOURS)],
        })
    return days


def _calc_usage(usage) -> dict | None:
    if not usage:
        return None
    uncached   = usage.prompt_tokens - usage.cached_prompt_tokens
    cost_in    = (uncached / 1_000_000) * 0.80
    cost_ca    = (usage.cached_prompt_tokens / 1_000_000) * 0.08
    cost_out   = (usage.completion_tokens / 1_000_000) * 4.00
    return {
        "model":             MODEL,
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens":      usage.total_tokens,
        "total_cost":        cost_in + cost_ca + cost_out,
    }


def _render_itinerary(request: Request, itinerary: Itinerary, routes: dict,
                      filename: str, usage=None) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/itinerary.html", {
        "itinerary": itinerary,
        "days":      _build_day_data(itinerary, routes),
        "filename":  filename,
        "usage":     _calc_usage(usage),
        "error":     None,
    })


def _render_error(request: Request, message: str) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/itinerary.html", {
        "error": message,
    })


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "saved": _list_saved(),
    })


@app.post("/generate", response_class=HTMLResponse)
def generate(
    request:       Request,
    destination:   str = Form(...),
    start_date:    str = Form(...),
    end_date:      str = Form(...),
    explore_days:  int = Form(1),
    hours_per_day: int = Form(8),
):
    if not destination.strip():
        return _render_error(request, "Please enter a destination.")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    if end_dt < start_dt:
        return _render_error(request, "End date must be on or after start date.")

    is_valid, reason = validate_destination(destination)
    if not is_valid:
        return _render_error(request, f'"{destination}" is not a valid travel destination: {reason}')

    last_error = None
    for attempt in range(1, 4):
        try:
            itinerary, usage = run_crew(destination, start_dt, end_dt, explore_days, hours_per_day)
            itinerary = trim_to_visit_budget(itinerary, hours_per_day)
            routes    = fetch_all_routes(itinerary)
            save_map(itinerary, routes, STATIC_DIR)
            filename  = _save_itinerary(itinerary)
            return _render_itinerary(request, itinerary, routes, filename, usage)
        except Exception as e:
            last_error = e

    return _render_error(request, f"Could not generate itinerary after 3 attempts: {last_error}")


@app.post("/load", response_class=HTMLResponse)
def load(request: Request, filename: str = Form(...)):
    path = ITINERARY_DIR / filename
    if not path.exists():
        return _render_error(request, "Itinerary file not found.")
    try:
        itinerary = Itinerary.model_validate_json(path.read_text())
        routes    = fetch_all_routes(itinerary)
        save_map(itinerary, routes, STATIC_DIR)
        return _render_itinerary(request, itinerary, routes, filename)
    except Exception as e:
        return _render_error(request, f"Could not load itinerary: {e}")



@app.post("/email", response_class=HTMLResponse)
def send_email(request: Request, filename: str = Form(...), to_address: str = Form(...)):
    path = ITINERARY_DIR / filename
    if not path.exists():
        return templates.TemplateResponse(request, "partials/email_result.html", {
            "success": False, "error": "Itinerary not found.", "to_address": to_address,
        })
    itinerary = Itinerary.model_validate_json(path.read_text())
    routes    = fetch_all_routes(itinerary)
    success, error = send_itinerary_email(to_address, itinerary, routes)
    return templates.TemplateResponse(request, "partials/email_result.html", {
        "success":    success,
        "error":      error,
        "to_address": to_address,
    })
