import random
from datetime import date, timedelta

from pydantic import BaseModel

CATEGORY_PROFILES = {
    "Groceries": (40, 120, ["Whole Foods", "Trader Joe's", "Safeway"]),
    "Dining": (12, 60, ["Chipotle", "Olive Garden", "Local Cafe"]),
    "Transport": (5, 40, ["Uber", "Shell", "Metro Transit"]),
    "Subscriptions": (5, 20, ["Netflix", "Spotify", "iCloud"]),
    "Utilities": (60, 200, ["City Power", "Water Dept", "Comcast"]),
}
MONTHLY_RENT = ("Rent", 1800.0, "Greenfield Apartments")
MONTHLY_SALARY = ("Salary", 5200.0, "ACME Corp Payroll")


class GeneratedTxn(BaseModel):
    date: date
    merchant: str
    amount: float
    category: str
    description: str
    is_anomaly: bool = False


def generate_transactions(months: int = 6, seed: int = 42) -> list[GeneratedTxn]:
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    txns: list[GeneratedTxn] = []

    for m in range(months):
        month_start = _add_months(start, m)
        # Fixed monthly income + rent
        cat, amt, merch = MONTHLY_SALARY
        txns.append(GeneratedTxn(date=month_start, merchant=merch, amount=amt,
                                 category=cat, description="monthly salary"))
        cat, amt, merch = MONTHLY_RENT
        txns.append(GeneratedTxn(date=month_start + timedelta(days=1), merchant=merch,
                                 amount=amt, category=cat, description="monthly rent"))
        # Variable spend
        for _ in range(rng.randint(20, 35)):
            category = rng.choice(list(CATEGORY_PROFILES))
            low, high, merchants = CATEGORY_PROFILES[category]
            amount = round(rng.uniform(low, high), 2)
            day = month_start + timedelta(days=rng.randint(0, 27))
            merchant = rng.choice(merchants)
            txns.append(GeneratedTxn(date=day, merchant=merchant, amount=amount,
                                     category=category, description=f"{category.lower()} at {merchant}"))

    _plant_anomalies(txns, rng)
    txns.sort(key=lambda t: t.date)
    return txns


def _plant_anomalies(txns: list[GeneratedTxn], rng: random.Random) -> None:
    # Baseline is the max across ALL normal transactions (incl. salary), so
    # planted anomalies exceed every non-anomalous amount — the test relies on this.
    all_amounts = [t.amount for t in txns]
    normal_max = max(all_amounts)
    victims_pool = [t for t in txns if t.category in CATEGORY_PROFILES]
    count = min(rng.randint(2, 4), len(victims_pool))
    victims = rng.sample(victims_pool, count)
    for t in victims:
        t.amount = round(normal_max * rng.uniform(1.5, 2.5), 2)
        t.is_anomaly = True
        t.description = f"UNUSUAL: large {t.category.lower()} charge at {t.merchant}"


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    return date(year, month % 12 + 1, 1)
