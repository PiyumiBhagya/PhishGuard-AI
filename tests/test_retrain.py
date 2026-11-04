import pandas as pd

from src.data import registered_domain
from src.retrain import next_version, should_promote, split_logs


def test_next_version():
    assert next_version("v1") == "v2"
    assert next_version("v9") == "v10"
    assert next_version("unknown") == "v2"


def test_promote_when_clearly_better_and_no_regression():
    ok, reason = should_promote(0.80, 0.79, 0.40, 0.70)
    assert ok is True
    assert reason.startswith("promoted")


def test_reject_when_holdout_regresses():
    ok, reason = should_promote(0.80, 0.70, 0.40, 0.90)
    assert ok is False
    assert "holdout" in reason


def test_reject_when_no_gain_on_recent_traffic():
    ok, reason = should_promote(0.80, 0.80, 0.50, 0.51)
    assert ok is False
    assert "recent" in reason


def test_split_logs_never_shares_a_domain():
    urls = [f"https://sub{i}.host{i % 20}.com/page" for i in range(200)]
    logs = pd.DataFrame({"url": urls, "true_label": [i % 2 for i in range(200)]})
    train, held_out = split_logs(logs, eval_share=0.4, seed=1)
    assert len(train) + len(held_out) == 200
    assert not set(train["url"].map(registered_domain)) & set(held_out["url"].map(registered_domain))