"""Text preprocessing shared by training and serving (one code path = no train/serve skew)."""
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s-]{7,}\d")
# Reference codes such as SIMQ-V9LU2KD or #48213 carry no class signal.
_REFCODE = re.compile(r"\b(?=[A-Z0-9-]*\d)[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})+\b|#\d+")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize one string while preserving Sinhala/Tamil combining marks (NFC, not NFKD)."""
    # Replace lone surrogates (invalid Unicode that JSON can still carry) so nothing downstream crashes.
    text = (text or "").encode("utf-8", "replace").decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _REFCODE.sub(" reftoken ", text)
    text = text.lower()
    text = _NUM.sub("0", text)
    return _WS.sub(" ", text).strip()


def encoder_input(subject: str | None, text: str) -> str:
    """Input for the sentence encoder: light cleanup only (case and punctuation carry meaning)."""
    raw = f"{subject or ''}\n{text or ''}".encode("utf-8", "replace").decode("utf-8")
    raw = unicodedata.normalize("NFC", raw)
    return "query: " + _WS.sub(" ", raw).strip()


def build_input(channel: str, subject: str | None, text: str) -> str:
    """Combine the three model inputs. The channel becomes a token the vectorizer can use."""
    subject = normalize(subject or "")
    body = normalize(text)
    parts = [f"chan_{channel}"]
    if subject:
        parts.append(f"{subject} |")
    parts.append(body)
    return " ".join(parts)
