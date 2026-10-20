import pandas as pd

df = pd.read_csv("bearing_features.csv")

summary = df.groupby(
    ["bearing_id", "label"]
)[["rms", "peak", "kurtosis"]].mean()

print("\nAverage features by bearing:")
print(summary.round(4)) 