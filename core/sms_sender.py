from twilio.rest import Client

from core.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
from core.models import Itinerary


def _build_message(itinerary: Itinerary) -> str:
    return (
        f"Custom Trip Advisor\n"
        f"Destination: {itinerary.destination}\n"
        f"Start date: {itinerary.start_date}\n"
        f"Duration: {itinerary.total_days} days"
    )


def send_itinerary_sms(to_number: str, itinerary: Itinerary) -> tuple[bool, str]:
    """Send itinerary summary via Twilio SMS. Returns (success, error_message)."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_FROM_NUMBER:
        return False, "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER must be set in .env"

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=_build_message(itinerary),
            from_=TWILIO_FROM_NUMBER,
            to=to_number,
        )
        return True, ""
    except Exception as e:
        return False, str(e)
