from src.data import load_clean, split

df = load_clean()
train, val, test = split(df)

print("rows after removing duplicates:", len(df))
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(
        f"{name:5s} rows={len(part):7d}  "
        f"phishing_share={part['is_phishing'].mean():.3f}  "
        f"domains={part['domain'].nunique()}"
    )
print("domains shared by train and test:", len(set(train.domain) & set(test.domain)))
print("domains shared by train and val: ", len(set(train.domain) & set(val.domain)))