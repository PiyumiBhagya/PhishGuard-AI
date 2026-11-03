"""Create a small balanced sample (URL + label) for CI tests."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/phiusil.csv")
OUT = Path("data/sample.csv")

df = pd.read_csv(RAW, usecols=["URL", "label"])
sample = df.groupby("label").sample(n=10_000, random_state=42)
sample = sample.sample(frac=1, random_state=42)  # shuffle rows
sample.to_csv(OUT, index=False)

print(sample.shape, sample["label"].value_counts().to_dict())