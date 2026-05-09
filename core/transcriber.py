import tempfile
from pathlib import Path
from faster_whisper import WhisperModel

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Transcribe audio bytes and return the text."""
    model = _get_model()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path)
        return " ".join(s.text for s in segments).strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
