import anthropic
from core.config import ANTHROPIC_API_KEY, MODEL


def validate_destination(destination: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message). error_message is empty when valid."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    model_id = MODEL.replace("anthropic/", "")
    message = client.messages.create(
        model=model_id,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Is "{destination}" a real geographic location (country, city, region, or area) '
                    f"that a person could travel to? "
                    f'Reply with exactly one word: YES or NO. '
                    f"If NO, add a short reason after a colon, e.g. NO: not a real place."
                ),
            }
        ],
    )

    reply = message.content[0].text.strip()
    if reply.upper().startswith("YES"):
        return True, ""
    reason = reply.split(":", 1)[1].strip() if ":" in reply else "not a recognised travel destination"
    return False, reason
