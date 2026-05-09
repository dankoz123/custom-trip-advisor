from pydantic import BaseModel
from datetime import date


class ValidateRequest(BaseModel):
    destination: str


class ValidateResponse(BaseModel):
    is_valid: bool
    reason: str


class GenerateRequest(BaseModel):
    destination: str
    start_date: date
    end_date: date
    hours_per_day: int = 8


class EmailRequest(BaseModel):
    to_address: str
