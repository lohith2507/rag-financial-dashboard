import re

_NUMBER = re.compile(r"\$?(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?")


def extract_number(text: str) -> float | None:
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    whole, frac = matches[-1]
    value = float(whole.replace(",", ""))
    if frac:
        value += float(f"0.{frac}")
    return value


def is_correct(answer_text: str, expected: float) -> bool:
    """Return True if the last number in ``answer_text`` is close to ``expected``.

    The allowed error is the larger of $0.51 or 1% of ``expected``, so small
    rounding differences pass. Returns False when no number can be extracted.
    """
    tolerance = max(0.51, expected * 0.01)
    value = extract_number(answer_text)
    return value is not None and abs(value - expected) <= tolerance
