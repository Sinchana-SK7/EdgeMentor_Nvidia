import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("bearing_features.csv")

features = [
    "rms",
    "peak",
    "kurtosis",
    "crest_factor",
    "skewness",
    "energy_0_1k",
    "energy_1_3k",
    "energy_3_6k",
    "energy_6_10k"
]

train_bearings = [
    "K001", "K002",
    "KA01", "KA03",
    "KI01", "KI03"
]

test_bearings = ["K003", "KA05", "KI05"]

train = df[df["bearing_id"].isin(train_bearings)]
test = df[df["bearing_id"].isin(test_bearings)]

# Fit preprocessing using training data only
scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])

# Learn the distribution of training examples
detector = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42
)

detector.fit(X_train)

# -1 = anomaly, +1 = familiar
test_predictions = detector.predict(X_test)

results = test[["bearing_id", "label"]].copy()
results["anomaly"] = test_predictions

print("\nANOMALY DETECTION RESULTS")
print("-------------------------")

print(
    results.groupby("bearing_id")["anomaly"]
    .apply(lambda x: (x == -1).sum())
)

print("\nTotal flagged as anomalies:")
print((results["anomaly"] == -1).sum())

print("\nTotal test samples:")
print(len(results))