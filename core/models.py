from pydantic import BaseModel


class Attraction(BaseModel):
    time: str
    place: str
    duration_min: int
    source: str
    lat: float | None = None
    lon: float | None = None


class Day(BaseModel):
    day_number: int
    date: str
    title: str
    description: str
    attractions: list[Attraction]


class Itinerary(BaseModel):
    destination: str
    start_date: str
    end_date: str
    total_days: int
    days: list[Day]


# ── Attractions Finder (standalone feature; independent of the itinerary flow) ──

class FoundAttraction(BaseModel):
    """A tourist attraction discovered and grounded via Wikidata/Nominatim/Overpass."""
    name: str
    local_name: str
    category: str
    short_description: str
    website: str = ""
    lat: float | None = None
    lon: float | None = None
    confidence: str        # "high" | "medium" | "llm-estimate" | "low"
    coord_source: str
    address: str = ""


class AttractionsResult(BaseModel):
    city: str
    country: str
    attractions: list[FoundAttraction]
    elapsed_seconds: float
