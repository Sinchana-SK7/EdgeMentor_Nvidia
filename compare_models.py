import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load our dataset
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

# Same bearing-level split as our first experiment
train_bearings = [
    "K001", "K002",
    "KA01", "KA03",
    "KI01", "KI03"
]

test_bearings = [
    "K003", "KA05", "KI05"
]

train = df[df["bearing_id"].isin(train_bearings)]
test = df[df["bearing_id"].isin(test_bearings)]

X_train = train[features]
y_train = train["label"]

X_test = test[features]
y_test = test["label"]

# Define two classifiers
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        random_state=42
    )
}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(f"\n========== {name} ==========")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "Healthy",
                "Outer Damage",
                "Inner Damage"
            ],
            zero_division=0
        )
    )

    results = test[["bearing_id", "label"]].copy()

    results["predicted"] = predictions
    results["confidence"] = probabilities.max(axis=1)

    print("Predictions by bearing:")
    print(
        results.groupby(
            ["bearing_id", "label", "predicted"]
        ).size()
    )

    print("\nAverage confidence by bearing:")
    print(
        results.groupby("bearing_id")["confidence"].mean()
    )