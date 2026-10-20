import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load our prepared dataset
df = pd.read_csv("bearing_features.csv")

# Select numerical features
features = [
    "rms",
    "peak",
    "kurtosis",
    "crest_factor",
    "skewness"
]

# Split by physical bearing identity
train_bearings = [
    "K001", "K002",
    "KA01", "KA03",
    "KI01", "KI03"
]

test_bearings = [
    "K003",
    "KA05",
    "KI05"
]

train = df[df["bearing_id"].isin(train_bearings)]
test = df[df["bearing_id"].isin(test_bearings)]

X_train = train[features]
y_train = train["label"]

X_test = test[features]
y_test = test["label"]

# Scale features and train classifier
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# Predict unseen bearing conditions
predictions = model.predict(X_test)

print("Training recordings:", len(train))
print("Testing recordings:", len(test))

print("\nTest Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Healthy", "Outer Damage", "Inner Damage"],
        zero_division=0
    )
)

# Examine predictions and confidence
probabilities = model.predict_proba(X_test)

results = test[["bearing_id", "label"]].copy()

results["predicted"] = predictions
results["confidence"] = probabilities.max(axis=1)

print("\nPredictions by bearing:")
print(
    results.groupby(
        ["bearing_id", "label", "predicted"]
    ).size()
)

print("\nAverage confidence by bearing:")
print(
    results.groupby("bearing_id")["confidence"].mean()
)