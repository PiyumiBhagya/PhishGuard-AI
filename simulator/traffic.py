"""Build realistic traffic for the PhishGuard API, with optional drift scenarios."""
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract

from src.data import load_clean, split
from src.features import canonical_host

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# "obvious" drift: lure words, digits, unusual TLDs
NEW_TLDS = ["xyz", "cfd", "click", "icu", "buzz", "lol"]
LURE_WORDS = ["secure", "verify", "login", "account", "update", "support"]

# "evasive" drift: clean-looking pronounceable domains on common TLDs
TRUSTED_TLDS = ["com", "org", "net", "info"]
SYLLABLES = ["ka", "lo", "mi", "ra", "ten", "vo", "sa", "ni", "do", "pa",
             "lu", "be", "zor", "fin", "mar", "tel", "con", "gri", "por", "sen"]


def build_pool(sample: bool = False, seed: int = 42) -> pd.DataFrame:
    """URLs the model never trained on: the test part of the domain-grouped split."""
    df = load_clean("data/sample.csv" if sample else "data/raw/phiusil.csv")
    _, _, test = split(df, seed=seed)
    return test[["URL", "is_phishing"]].reset_index(drop=True)


def mutate_phishing(url: str, rng: np.random.Generator) -> str:
    """Obvious drift: lure word + digits in the domain name, and a TLD the model has not seen."""
    u = url.strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    ext = _EXTRACT(canonical_host(u))
    base = ext.domain or "site"
    label = f"{base}-{rng.choice(LURE_WORDS)}{int(rng.integers(100, 9999))}"
    host = f"{label}.{rng.choice(NEW_TLDS)}"
    return f"https://{host}{parsed.path}"


def mutate_phishing_evasive(url: str, rng: np.random.Generator) -> str:
    """Evasive drift: attackers copy what legitimate domains look like
    (short pronounceable name, common TLD, no digits, no hyphens, no lure words)."""
    parsed = urlparse(url.strip() if "://" in url else "http://" + url.strip())
    name = "".join(rng.choice(SYLLABLES, size=int(rng.integers(3, 5))))
    host = f"{name}.{rng.choice(TRUSTED_TLDS)}"
    return f"https://{host}{parsed.path}"


def draw_batch(pool: pd.DataFrame, size: int, rng: np.random.Generator,
               drift: bool = False, phishing_share: float | None = None,
               style: str = "obvious"):
    """Return (urls, labels). In drift mode the phishing URLs are mutated."""
    share = float(pool["is_phishing"].mean()) if phishing_share is None else phishing_share
    n_phish = int(round(size * share))

    def pick(label: int, n: int) -> list[str]:
        part = pool[pool["is_phishing"] == label]
        seed = int(rng.integers(0, 2**31 - 1))
        return part["URL"].sample(n, random_state=seed, replace=True).tolist()

    phish = pick(1, n_phish)
    if drift:
        mutate = mutate_phishing_evasive if style == "evasive" else mutate_phishing
        phish = [mutate(u, rng) for u in phish]
    legit = pick(0, size - n_phish)

    urls = phish + legit
    labels = [1] * len(phish) + [0] * len(legit)
    order = rng.permutation(len(urls))
    return [urls[i] for i in order], [labels[i] for i in order]