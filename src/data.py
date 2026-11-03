"""Load, clean and split the PhiUSIIL data (grouped by domain)."""
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

from src.features import canonical_host

RAW_PATH = "data/raw/phiusil.csv"
SEED = 42

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def registered_domain(url: str) -> str:
    """e.g. abc.weeblysite.com -> weeblysite.com"""
    host = canonical_host(url)
    ext = _EXTRACT(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return host


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["URL", "label"])
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    # In the dataset 1 = legitimate, 0 = phishing. We flip it so that
    # is_phishing = 1 means phishing (the class we want to catch).
    df["is_phishing"] = 1 - df["label"]
    df["domain"] = df["URL"].map(registered_domain)
    return df


def split(df: pd.DataFrame, seed: int = SEED):
    """70% train / 15% validation / 15% test, never splitting a domain."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(outer.split(df, groups=df["domain"]))
    train, rest = df.iloc[train_idx], df.iloc[rest_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(inner.split(rest, groups=rest["domain"]))
    return train, rest.iloc[val_idx], rest.iloc[test_idx]