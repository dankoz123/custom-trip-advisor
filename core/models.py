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
