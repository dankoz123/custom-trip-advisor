import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File

from core.guard import validate_destination
from core.transcriber import transcribe
from core.models import Itinerary
from core.email_sender import send_itinerary_email
from fastapi_app.schemas import (
    ValidateRequest, ValidateResponse,
    GenerateRequest, EmailRequest,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Custom Trip Advisor API",
    description="Multi-agent AI travel planner — REST API",
    version="1.0.0",
)

ITINERARY_DIR = Path(__file__).parent.parent / "itinerary"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _run_crew(destination: str, start_date, end_date, hours_per_day: int) -> Itinerary:
    from crewai import Crew, Process, LLM
    from datetime import timedelta
    from core.config import MODEL
    from core.agents import make_researcher, make_optimizer, make_planner
    from core.tasks import make_research_task, make_optimization_task, make_planning_task

    start_dt   = datetime.combine(start_date, datetime.min.time())
    end_dt     = datetime.combine(end_date,   datetime.min.time())
    total_days = (end_dt - start_dt).days + 1
    START_DATE = start_dt.strftime("%Y-%m-%d")
    END_DATE   = end_dt.strftime("%Y-%m-%d")
    date_range = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days)]

    llm        = LLM(model=MODEL, temperature=0.3, max_tokens=16000)
    researcher = make_researcher(destination, llm)
    optimizer  = make_optimizer(destination, total_days, hours_per_day, llm)
    planner    = make_planner(destination, total_days, START_DATE, hours_per_day, llm)

    research_task     = make_research_task(researcher, destination, total_days, date_range, hours_per_day)
    optimization_task = make_optimization_task(optimizer, research_task, destination, total_days, hours_per_day)
    planning_task     = make_planning_task(
        planner, optimization_task, destination,
        START_DATE, END_DATE, total_days, date_range, hours_per_day,
    )
    crew   = Crew(
        agents=[researcher, optimizer, planner],
        tasks=[research_task, optimization_task, planning_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return result.pydantic


def _trim(itinerary: Itinerary, hours_per_day: int) -> Itinerary:
    max_min = hours_per_day * 60
    for day in itinerary.days:
        total = 0
        kept  = []
        for a in day.attractions:
            if total + a.duration_min <= max_min:
                kept.append(a)
                total += a.duration_min
        day.attractions = kept
    return itinerary


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

    is_valid, reason = validate_destination(body.destination)
    if not is_valid:
        raise HTTPException(status_code=422, detail=f'Invalid destination: {reason}')

    last_error = None
    for attempt in range(1, 4):
        try:
            itinerary = _run_crew(body.destination, body.start_date, body.end_date, body.hours_per_day)
            itinerary = _trim(itinerary, body.hours_per_day)
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
            import json
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


@app.post("/transcribe", summary="Transcribe audio to text")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    audio_bytes = await file.read()
    try:
        text = transcribe(audio_bytes, suffix=suffix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    return {"text": text}
