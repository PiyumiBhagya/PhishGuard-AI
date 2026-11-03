import numpy as np
import pandas as pd

from simulator.traffic import (
    NEW_TLDS,
    TRUSTED_TLDS,
    draw_batch,
    mutate_phishing,
    mutate_phishing_evasive,
)


def _pool():
    return pd.DataFrame({
        "URL": [f"http://phish{i}.com/x" for i in range(20)] + [f"https://www.legit{i}.org" for i in range(20)],
        "is_phishing": [1] * 20 + [0] * 20,
    })


def test_mutate_changes_tld_and_keeps_path():
    out = mutate_phishing("http://paypal-login.com/a/b", np.random.default_rng(0))
    host = out.split("/")[2]
    assert out.startswith("https://")
    assert out.endswith("/a/b")
    assert host.rsplit(".", 1)[-1] in NEW_TLDS


def test_draw_batch_size_and_share():
    urls, labels = draw_batch(_pool(), 40, np.random.default_rng(1), phishing_share=0.5)
    assert len(urls) == len(labels) == 40
    assert sum(labels) == 20


def test_drift_batch_mutates_all_phishing_urls():
    urls, labels = draw_batch(_pool(), 20, np.random.default_rng(2), drift=True, phishing_share=1.0)
    assert all(labels)
    assert all(u.split("/")[2].rsplit(".", 1)[-1] in NEW_TLDS for u in urls)


def test_evasive_mutation_looks_clean():
    out = mutate_phishing_evasive("http://paypal-login.com/a", np.random.default_rng(0))
    host = out.split("/")[2]
    assert not any(ch.isdigit() or ch == "-" for ch in host)
    assert host.rsplit(".", 1)[-1] in TRUSTED_TLDS


def test_draw_batch_evasive_style():
    urls, _ = draw_batch(_pool(), 20, np.random.default_rng(3), drift=True,
                         phishing_share=1.0, style="evasive")
    assert all(u.split("/")[2].rsplit(".", 1)[-1] in TRUSTED_TLDS for u in urls)