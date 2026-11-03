from urllib.parse import urlparse

import pandas as pd

df = pd.read_csv("data/raw/phiusil.csv", usecols=["URL", "label"])
df = df.drop_duplicates(subset="URL")

parsed = df["URL"].map(urlparse)
df["has_path"] = parsed.map(lambda p: len(p.path.strip("/")) > 0)
df["has_query"] = parsed.map(lambda p: len(p.query) > 0)

print("share of URLs with a path / query, by label:")
print(df.groupby("label")[["has_path", "has_query"]].mean().round(3))

rule = df["URL"].str.startswith("https://www.").astype(int)
print("\naccuracy of the trivial rule 'starts with https://www.':",
      round((rule == df["label"]).mean(), 3))

print("\n10 legitimate URLs:")
for u in df[df.label == 1]["URL"].sample(10, random_state=1):
    print("  ", u)
print("\n10 phishing URLs:")
for u in df[df.label == 0]["URL"].sample(10, random_state=1):
    print("  ", u[:120])