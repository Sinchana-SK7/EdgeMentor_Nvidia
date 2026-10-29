
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)
from sklearn.preprocessing import StandardScaler

from scipy.signal import butter, sosfiltfilt, hilbert

from build_dataset import load_vibration
from sensor_health import check_sensor
from teacher_ai import ask_teacher


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

FS = 64000

# Approximate theoretical defect frequencies at 900 RPM
BPFO = 45.82
BPFI = 74.15

FEATURE_NAMES = [
    "bpfo_h1",
    "bpfo_h2",
    "bpfo_h3",
    "bpfi_h1",
    "bpfi_h2",
    "bpfi_h3"
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

CONFIDENCE_THRESHOLD = 0.90

# Same filter used in our successful experiment
SOS = butter(
    4,
    [3000, 10000],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# --------------------------------------------------
# 2. Physics-based feature extraction
# --------------------------------------------------

def extract_fault_features(signal):
    """
    Extract normalized envelope-spectrum amplitudes
    near BPFO and BPFI harmonics.
    """

    signal = np.asarray(signal, dtype=float).ravel()

    # Bandpass vibration between 3 kHz and 10 kHz
    filtered = sosfiltfilt(SOS, signal)

    # Extract the amplitude envelope
    envelope = np.abs(hilbert(filtered))

    # Remove the envelope's DC component
    envelope = envelope - np.mean(envelope)

    # Compute envelope frequency spectrum
    spectrum = np.abs(np.fft.rfft(envelope))

    frequencies = np.fft.rfftfreq(
        len(envelope),
        d=1 / FS
    )

    # Normalize by the RMS spectrum amplitude
    # within the 20-1000 Hz reference region
    reference_mask = (
        (frequencies >= 20) &
        (frequencies <= 1000)
    )

    reference = (
        np.sqrt(
            np.mean(spectrum[reference_mask] ** 2)
        ) + 1e-12
    )

    features = {}

    for name, base_frequency in [
        ("bpfo", BPFO),
        ("bpfi", BPFI)
    ]:

        for harmonic in [1, 2, 3]:

            target = base_frequency * harmonic

            # Search within +/- 2 Hz
            mask = (
                np.abs(frequencies - target) <= 2.0
            )

            amplitude = float(np.max(spectrum[mask]))

            features[f"{name}_h{harmonic}"] = (
                amplitude / reference
            )

    return features


# --------------------------------------------------
# 3. Train classifier and anomaly detector
# --------------------------------------------------

@lru_cache(maxsize=1)
def train_models():
    """
    Train models using physics-based features.

    The cache prevents retraining for every dashboard
    interaction during the same Python process.
    """

    dataset_path = Path("bearing_fault_features.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(
            "bearing_fault_features.csv not found. "
            "Run python evaluate_fault_features.py first."
        )

    df = pd.read_csv(dataset_path)

    train = df[
        df["bearing_id"].isin(TRAIN_BEARINGS)
    ].copy()

    if train.empty:
        raise ValueError("No training recordings found.")

    X_train = train[FEATURE_NAMES]
    y_train = train["label"]

    # Three-class bearing health classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    # Scale features for anomaly detection
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # Detect unusual feature patterns
    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42
    )

    anomaly_detector.fit(X_train_scaled)

    return classifier, scaler, anomaly_detector


# --------------------------------------------------
# 4. Analyze a bearing
# --------------------------------------------------

def analyze_bearing(
    bearing,
    recording=1,
    use_teacher=False,
    sensor_fault=None
):
    """
    Analyze one bearing recording.

    Returns prediction, confidence, sensor status,
    anomaly flag, and reliability decision.
    """

    # Load vibration signal
    file_path = (
        Path("data")
        / bearing
        / f"N09_M07_F10_{bearing}_{recording}.mat"
    )

    signal = load_vibration(file_path)

    # Simulated sensor faults for the hackathon demo
    if sensor_fault == "Stuck Sensor":

        signal = np.full_like(signal, 0.15)

    elif sensor_fault == "Missing Readings":

        signal = signal.copy()
        signal[100] = np.nan

    # Check sensor health before classification
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

    # Never classify an invalid sensor signal
    if sensor_status.startswith("INVALID"):
        return result

    # Extract the same features used during training
    features = extract_fault_features(signal)

    input_features = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )

    # Load trained models
    classifier, scaler, anomaly_detector = (
        train_models()
    )

    # Predict bearing health
    prediction = int(
        classifier.predict(input_features)[0]
    )

    probabilities = classifier.predict_proba(
        input_features
    )[0]

    confidence = float(np.max(probabilities))

    # --------------------------------------------------
    # 5. Reliability supervisor
    # --------------------------------------------------

    low_confidence = (
        confidence < CONFIDENCE_THRESHOLD
    )

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

    # --------------------------------------------------
    # 6. NVIDIA Teacher AI
    # --------------------------------------------------

    if decision == "ESCALATE" and use_teacher:

        result["teacher_response"] = ask_teacher(
            prediction=LABELS[prediction],
            confidence=confidence,
            low_confidence=low_confidence,
            anomaly_flag=anomaly_flag
        )

    return result


# --------------------------------------------------
# 7. Quick local test
# --------------------------------------------------

if __name__ == "__main__":

    for bearing in ["K003", "KA05", "KI05"]:

        result = analyze_bearing(
            bearing=bearing,
            recording=1,
            use_teacher=False,
            sensor_fault=None
        )

        print("\n========== EdgeMentor ==========")

        for key, value in result.items():

            if key == "features":
                continue

            print(f"{key}: {value}")
