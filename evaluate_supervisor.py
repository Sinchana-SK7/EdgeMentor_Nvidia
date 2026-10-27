import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("bearing_features.csv")

features = [
    "rms", "peak", "kurtosis",
    "crest_factor", "skewness",
    "energy_0_1k", "energy_1_3k",
    "energy_3_6k", "energy_6_10k"
]

train_ids = ["K001", "K002", "KA01", "KA03", "KI01", "KI03"]
test_ids = ["K003", "KA05", "KI05"]

train = df[df["bearing_id"].isin(train_ids)]
test = df[df["bearing_id"].isin(test_ids)]

X_train = train[features]
X_test = test[features]
y_train = train["label"]
y_test = test["label"].to_numpy()

# Local fault classifier
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
confidence = classifier.predict_proba(X_test).max(axis=1)

# Local anomaly detector
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

detector = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42
)

detector.fit(X_train_scaled)
anomaly_flags = detector.predict(X_test_scaled) == -1

# Compare two reliability strategies
confidence_only = confidence >= 0.90
combined = (confidence >= 0.90) & (~anomaly_flags)

def evaluate(name, accepted):
    accepted_count = np.sum(accepted)
    total = len(predictions)

    errors = np.sum(
        predictions[accepted] != y_test[accepted]
    )

    coverage = accepted_count / total

    risk = (
        errors / accepted_count
        if accepted_count > 0
        else float("nan")
    )

    print(f"\n{name}")
    print(f"Accepted: {accepted_count}/{total}")
    print(f"Escalated: {total - accepted_count}")
    print(f"Wrong accepted: {errors}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Selective risk: {risk:.2%}")

evaluate("CONFIDENCE ONLY", confidence_only)
evaluate("CONFIDENCE + ANOMALY", combined)

print("\nCOMBINED SUPERVISOR BY BEARING")

results = test[["bearing_id", "label"]].copy()
results["predicted"] = predictions
results["confidence"] = confidence
results["anomaly"] = anomaly_flags
results["accepted"] = combined

print(
    results.groupby("bearing_id").agg(
        accepted=("accepted", "sum"),
        anomalies=("anomaly", "sum")
    )
)