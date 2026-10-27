import pandas as pd

df = pd.read_csv("bearing_features.csv")

train_bearings = [
    "K001", "K002",
    "KA01", "KA03",
    "KI01", "KI03"
]

test_bearings = ["K003", "KA05", "KI05"]

train = df[df["bearing_id"].isin(train_bearings)]
test = df[df["bearing_id"].isin(test_bearings)]

features = ["rms", "kurtosis", "crest_factor"]

print("\nTRAINING FEATURE RANGES")
print(train[features].agg(["min", "max", "mean", "std"]))

print("\nUNSEEN BEARING FEATURE RANGES")
print(test[features].agg(["min", "max", "mean", "std"]))

print("\nFEATURE MEANS BY BEARING")
print(df.groupby(["bearing_id", "label"])[features].mean())