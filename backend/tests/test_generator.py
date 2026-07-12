from app.data.generator import generate_transactions


def test_generator_is_deterministic():
    a = generate_transactions(months=3, seed=7)
    b = generate_transactions(months=3, seed=7)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]


def test_generator_plants_anomalies():
    txns = generate_transactions(months=6, seed=42)
    anomalies = [t for t in txns if t.is_anomaly]
    assert 1 <= len(anomalies) <= 15
    # anomalies are unusually large
    normal_max = max(t.amount for t in txns if not t.is_anomaly)
    assert all(t.amount > normal_max for t in anomalies)


def test_categories_are_known():
    txns = generate_transactions(months=2, seed=1)
    known = {"Groceries", "Rent", "Salary", "Dining", "Subscriptions", "Transport", "Utilities"}
    assert {t.category for t in txns}.issubset(known)
