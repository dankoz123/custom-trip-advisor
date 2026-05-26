import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File

from core.guard import validate_destination
from core.transcriber import transcribe
from core.models import Itinerary, AttractionsResult
from core.email_sender import send_itinerary_email
from core.crew_runner import run_crew, run_crew_with_attractions, trim_to_visit_budget
from core.attractions_finder import find_attractions, format_for_optimizer
from fastapi_app.schemas import (
    ValidateRequest, ValidateResponse,
    GenerateRequest, EmailRequest,
    FindAttractionsRequest,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Custom Trip Advisor API",
    description="Multi-agent AI travel planner — REST API",
    version="1.0.0",
)

ITINERARY_DIR = Path(__file__).parent.parent / "itinerary"


def _save(itinerary: Itinerary) -> Path:
    ITINERARY_DIR.mkdir(exist_ok=True)
    filename = f"itinerary_{itinerary.destination.replace(' ', '_')}_{itinerary.start_date}.json"
    path = ITINERARY_DIR / filename
    path.write_text(itinerary.model_dump_json(indent=2))
    return path


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/validate", response_model=ValidateResponse, summary="Validate a destination")
def validate(body: ValidateRequest):
    is_valid, reason = validate_destination(body.destination)
    return ValidateResponse(is_valid=is_valid, reason=reason)


@app.post("/generate", response_model=Itinerary, summary="Generate an itinerary")
def generate(body: GenerateRequest):
    if body.end_date < body.start_date:
        raise HTTPException(status_code=400, detail="end_date must be on or after start_date")

    if "," not in body.destination:
        raise HTTPException(
            status_code=400,
            detail="destination must be in 'City, Country' format (e.g. 'Koszalin, Poland')",
        )

    city, country     = [s.strip() for s in body.destination.split(",", 1)]
    start_dt          = datetime.combine(body.start_date, datetime.min.time())
    end_dt            = datetime.combine(body.end_date,   datetime.min.time())
    total_days        = (end_dt - start_dt).days + 1
    attractions_count = total_days * body.hours_per_day

    # Step 1 — find verified attractions (Wikidata + Nominatim + Overpass grounding)
    try:
        attractions_result = find_attractions(city, country, attractions_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attractions finder failed: {e}")

    if not attractions_result.attractions:
        raise HTTPException(
            status_code=404,
            detail=f"No verifiable attractions found in {city}, {country}",
        )

    attractions_text = format_for_optimizer(attractions_result)

    # Step 2 — optimize + plan the multi-day itinerary
    last_error = None
    for attempt in range(1, 4):
        try:
            itinerary, _ = run_crew_with_attractions(
                attractions_text, body.destination, start_dt, end_dt,
                total_days, body.hours_per_day,
            )
            itinerary = trim_to_visit_budget(itinerary, body.hours_per_day)
            _save(itinerary)
            return itinerary
        except Exception as e:
            last_error = e

    raise HTTPException(status_code=500, detail=f"Failed after 3 attempts: {last_error}")


@app.get("/itineraries", summary="List saved itineraries")
def list_itineraries():
    if not ITINERARY_DIR.exists():
        return []
    files = sorted(ITINERARY_DIR.glob("*.json"))
    result = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            result.append({
                "filename": f.name,
                "destination": d.get("destination"),
                "start_date": d.get("start_date"),
                "total_days": d.get("total_days"),
            })
        except Exception:
            pass
    return result


@app.get("/itineraries/{filename}", response_model=Itinerary, summary="Load a saved itinerary")
def get_itinerary(filename: str):
    path = ITINERARY_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return Itinerary.model_validate_json(path.read_text())


@app.post("/itineraries/{filename}/email", summary="Email a saved itinerary")
def email_itinerary(filename: str, body: EmailRequest):
    path = ITINERARY_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Itinerary not found")
    itinerary = Itinerary.model_validate_json(path.read_text())
    success, error = send_itinerary_email(body.to_address, itinerary, routes={})
    if not success:
        raise HTTPException(status_code=500, detail=error)
    return {"message": f"Email sent to {body.to_address}"}


@app.post("/attractions/find", response_model=AttractionsResult,
          summary="Find verified attractions for a city (standalone)")
def find_attractions_endpoint(body: FindAttractionsRequest):
    if body.count < 1 or body.count > 30:
        raise HTTPException(status_code=400, detail="count must be between 1 and 30")
    try:
        return find_attractions(body.city, body.country, body.count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attractions finder failed: {e}")


@app.post("/transcribe", summary="Transcribe audio to text")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    audio_bytes = await file.read()
    try:
        text = transcribe(audio_bytes, suffix=suffix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    return {"text": text}
