
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

from build_dataset import load_vibration, extract_features
from sensor_health import check_sensor
from teacher_ai import ask_teacher


FEATURE_NAMES = [
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

TRAIN_BEARINGS = [
    "K001", "K002",
    "KA01", "KA03",
    "KI01", "KI03"
]

LABELS = {
    0: "Healthy",
    1: "Outer Damage",
    2: "Inner Damage"
}


def train_models():
    """Train the classifier and anomaly detector."""

    df = pd.read_csv("bearing_features.csv")

    train = df[df["bearing_id"].isin(TRAIN_BEARINGS)]

    X_train = train[FEATURE_NAMES]
    y_train = train["label"]

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    classifier.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42
    )
    anomaly_detector.fit(X_train_scaled)

    return classifier, scaler, anomaly_detector


def analyze_bearing(
    bearing,
    recording=1,
    use_teacher=False,
    sensor_fault=None
):
    """Analyze one bearing recording."""

    # Step 1: Load vibration data
    file_path = Path(
        f"data/{bearing}/N09_M07_F10_{bearing}_{recording}.mat"
    )

    signal = load_vibration(file_path)

    # Simulate sensor malfunction for demonstration
    if sensor_fault == "Stuck Sensor":
        signal = np.full_like(signal, 0.15)

    elif sensor_fault == "Missing Readings":
        signal = signal.copy()
        signal[100] = np.nan

    # Check sensor health
    sensor_status = check_sensor(signal)

    result = {
        "bearing": bearing,
        "recording": recording,
        "sensor_status": sensor_status,
        "prediction": None,
        "confidence": None,
        "low_confidence": None,
        "anomaly_flag": None,
        "decision": "ABSTAIN",
        "teacher_response": None
    }

    # Stop processing if the sensor is invalid
    if sensor_status.startswith("INVALID"):
        return result

    # Step 2: Extract features
    features = extract_features(signal)

    input_features = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )

    # Step 3: Run trained models
    classifier, scaler, anomaly_detector = train_models()

    prediction = int(
        classifier.predict(input_features)[0]
    )

    probabilities = classifier.predict_proba(
        input_features
    )[0]

    confidence = float(max(probabilities))

    # Step 4: Reliability supervisor
    low_confidence = confidence < 0.90

    input_scaled = scaler.transform(input_features)

    anomaly_flag = bool(
        anomaly_detector.predict(input_scaled)[0] == -1
    )

    decision = (
        "ESCALATE"
        if low_confidence or anomaly_flag
        else "ACCEPT"
    )

    result.update({
        "prediction": LABELS[prediction],
        "confidence": confidence,
        "low_confidence": low_confidence,
        "anomaly_flag": anomaly_flag,
        "decision": decision,
        "features": features
    })

    # Step 5: Optional NVIDIA teacher escalation
    if decision == "ESCALATE" and use_teacher:
        result["teacher_response"] = ask_teacher(
            prediction=LABELS[prediction],
            confidence=confidence,
            low_confidence=low_confidence,
            anomaly_flag=anomaly_flag
        )

    return result


if __name__ == "__main__":

    result = analyze_bearing(
        bearing="KA05",
        recording=1,
        use_teacher=False,
        sensor_fault="Stuck Sensor"
    )

    print("\n========== EdgeMentor ==========")

    for key, value in result.items():
        if key == "features":
            continue

        print(f"{key}: {value}")
