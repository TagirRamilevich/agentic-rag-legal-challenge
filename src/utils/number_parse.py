import re
from typing import Optional

_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "t": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
}

_NUM_RE = re.compile(
    r"[-+]?\(?\d[\d,]*\.?\d*\)?"
    r"\s*(?:million|billion|trillion|thousand|[kmbt]n?)\b"
    r"|[-+]?\(?\d[\d,]*\.?\d*\)?",
    re.IGNORECASE,
)
_MULT_RE = re.compile(r"\b(million|billion|trillion|thousand|[kmbt]n?)\b", re.IGNORECASE)


def parse_number(text: str) -> Optional[float]:
    text = re.sub(r"[$€£¥₹\s]", " ", text).strip()

    m = _NUM_RE.search(text)
    if not m:
        return None

    span = m.group(0)
    negative = span.startswith("(") and span.endswith(")")
    clean = re.sub(r"[(),]", "", span).strip()

    mult_match = _MULT_RE.search(clean)
    multiplier = 1
    if mult_match:
        multiplier = _MULTIPLIERS.get(mult_match.group(1).lower(), 1)
        clean = clean[: mult_match.start()].strip()

    try:
        value = float(clean) * multiplier
    except ValueError:
        return None

    if negative:
        value = -value

    return int(value) if value == int(value) else value
