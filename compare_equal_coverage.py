import numpy as np
import pandas as pd

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

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
confidence = classifier.predict_proba(X_test).max(axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

detector = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42
)
detector.fit(X_train_scaled)

anomaly = detector.predict(X_test_scaled) == -1

# Combined supervisor from our previous experiment
combined = (confidence >= 0.90) & (~anomaly)
target_count = int(combined.sum())

# Confidence-only: accept the same number of predictions
# Tie-breaking is deterministic, but arbitrary.
ranked = np.argsort(-confidence, kind="stable")

confidence_equal = np.zeros(len(test), dtype=bool)
confidence_equal[ranked[:target_count]] = True

def report(name, accepted):
    count = int(accepted.sum())
    wrong = int((predictions[accepted] != y_test[accepted]).sum())

    print(f"\n{name}")
    print(f"Accepted: {count}")
    print(f"Wrong accepted: {wrong}")
    print(f"Selective risk: {wrong/count:.2%}" if count else "Risk: N/A")

report("CONFIDENCE ONLY — EQUAL COVERAGE", confidence_equal)
report("COMBINED SUPERVISOR", combined)

print("\nCONFIDENCE-ONLY ACCEPTED BY BEARING")
print(test.loc[confidence_equal, "bearing_id"].value_counts())