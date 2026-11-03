import pandas as pd

df = pd.read_csv(
    "data/raw/phiusil.csv",
    usecols=["URL", "label", "URLLength", "TLD"],
)

print("missing values:", int(df.isna().sum().sum()))
print("duplicate URLs:", int(df["URL"].duplicated().sum()))
conflict = df.groupby("URL")["label"].nunique().gt(1).sum()
print("URLs with conflicting labels:", int(conflict))

# artifact check: how do the two classes differ in simple URL patterns?
df["starts_https"] = df["URL"].str.startswith("https://")
df["has_www"] = df["URL"].str.contains("://www.", regex=False)
print("\nshare of URLs by label (0 = phishing, 1 = legitimate):")
print(df.groupby("label")[["starts_https", "has_www"]].mean().round(3))

print("\nURL length by label:")
print(df.groupby("label")["URLLength"].describe().round(1))

print("\ntop 10 TLDs per label:")
for lab, g in df.groupby("label"):
    print(lab, g["TLD"].value_counts().head(10).to_dict())