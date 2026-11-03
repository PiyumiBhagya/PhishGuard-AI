import pandas as pd

from src.data import load_clean
from src.features import extract_host_features

df = load_clean().sample(30000, random_state=1)
f = pd.DataFrame([extract_host_features(u) for u in df["URL"]], index=df.index)
f["is_phishing"] = df["is_phishing"]

print("share of hosts that have a subdomain (0 = legit, 1 = phishing):")
print((f["num_subdomains"] > 0).groupby(f["is_phishing"]).mean().round(3))