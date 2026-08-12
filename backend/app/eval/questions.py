from collections import defaultdict

from pydantic import BaseModel

from app.data.generator import GeneratedTxn

SPEND_EXCLUDED = ("Salary",)

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class EvalQuestion(BaseModel):
    question: str
    expected: float


def build_eval_set(txns: list[GeneratedTxn]) -> list[EvalQuestion]:
    sums: dict[tuple[int, int, str], float] = defaultdict(float)
    for t in txns:
        if t.category in SPEND_EXCLUDED:
            continue
        sums[(t.date.year, t.date.month, t.category)] += t.amount

    questions = [
        EvalQuestion(
            question=(
                f"How much did I spend on {category} in "
                f"{MONTH_NAMES[month - 1]} {year}?"
            ),
            expected=round(total, 2),
        )
        for (year, month, category), total in sorted(sums.items())
        if total > 0
    ]
    return questions
