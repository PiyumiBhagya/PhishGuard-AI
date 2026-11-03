import pandas as pd

from src.features import extract_features

PAIRS = {
    "url_length": "URLLength",
    "domain_length": "DomainLength",
    "is_ip": "IsDomainIP",
    "num_subdomains": "NoOfSubDomain",
    "tld_length": "TLDLength",
    "is_https": "IsHTTPS",
    "num_digits": "NoOfDegitsInURL",
    "num_letters": "NoOfLettersInURL",
    "num_equals": "NoOfEqualsInURL",
    "num_qmark": "NoOfQMarkInURL",
    "num_amp": "NoOfAmpersandInURL",
    "num_other_special": "NoOfOtherSpecialCharsInURL",
}

df = pd.read_csv("data/raw/phiusil.csv", usecols=["URL", *PAIRS.values()])
df = df.sample(5000, random_state=42)
mine = pd.DataFrame([extract_features(u) for u in df["URL"]], index=df.index)

for m, theirs in PAIRS.items():
    rate = (mine[m] == df[theirs]).mean()
    print(f"{m:20s} vs {theirs:28s} exact match: {rate:.3f}")