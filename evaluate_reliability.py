import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the extracted bearing features
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

# Keep the same bearing-level split
train_bearings = [
    "K001", "K002",
    "KA01", "KA03",
    "KI01", "KI03"
]

test_bearings = ["K003", "KA05", "KI05"]

train = df[df["bearing_id"].isin(train_bearings)]
test = df[df["bearing_id"].isin(test_bearings)]

X_train = train[features]
y_train = train["label"]

X_test = test[features]
y_test = test["label"]

# Train the same Random Forest baseline
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Generate predictions and confidence scores
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
confidence = probabilities.max(axis=1)

print("\nEDGEMENTOR RELIABILITY EXPERIMENT")
print("--------------------------------")

# Evaluate several confidence thresholds
thresholds = [0.50, 0.70, 0.90, 0.95, 0.99]

for threshold in thresholds:

    # Accept only predictions above the threshold
    accepted = confidence >= threshold

    total = len(predictions)
    accepted_count = accepted.sum()

    coverage = accepted_count / total

    if accepted_count > 0:
        errors = (
            predictions[accepted] != y_test.to_numpy()[accepted]
        ).sum()

        selective_risk = errors / accepted_count

        print(
            f"Threshold: {threshold:.2f} | "
            f"Coverage: {coverage:.2%} | "
            f"Selective Risk: {selective_risk:.2%} | "
            f"Accepted: {accepted_count}/{total}"
        )
    else:
        print(
            f"Threshold: {threshold:.2f} | "
            f"Coverage: 0.00% | "
            f"Selective Risk: N/A | "
            f"Accepted: 0/{total}"
        )