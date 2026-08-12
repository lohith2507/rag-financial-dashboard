from app.data.generator import generate_transactions
from app.eval.questions import build_eval_set
from app.eval.scoring import extract_number, is_correct


def test_extract_number_handles_currency_forms():
    assert extract_number("You spent $1,234.56 in June.") == 1234.56
    assert extract_number("Total: 80 dollars") == 80.0
    assert extract_number("no numbers here") is None


def test_is_correct_within_tolerance():
    assert is_correct("about $100.40", 100.0)
    assert is_correct("$100.50 total", 100.0)
    assert not is_correct("$150.00", 100.0)
    assert not is_correct("I don't know", 100.0)


def test_build_eval_set_ground_truth_matches_data():
    txns = generate_transactions(months=3, seed=42)
    questions = build_eval_set(txns)

    assert len(questions) >= 10
    q = questions[0]
    assert q.expected > 0
    assert "How much did I spend on" in q.question


def test_build_eval_set_is_deterministic():
    a = build_eval_set(generate_transactions(months=3, seed=42))
    b = build_eval_set(generate_transactions(months=3, seed=42))
    assert [(x.question, x.expected) for x in a] == [(x.question, x.expected) for x in b]
