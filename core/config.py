import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import LLM

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

MODEL = os.getenv("LLM_MODEL")
if not MODEL:
    raise EnvironmentError("LLM_MODEL not found in .env")

RESEND_API_KEY      = os.getenv("RESEND_API_KEY", "")
EMAIL_SENDER        = os.getenv("EMAIL_SENDER", "")

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER", "")

llm = LLM(
    model=MODEL,
    temperature=0.3,
    max_tokens=16000,
)
