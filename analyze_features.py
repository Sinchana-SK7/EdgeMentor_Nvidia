from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis

def load_vibration(file_path):
    data = loadmat(file_path)

    key = next(k for k in data if not k.startswith("__"))
    record = data[key][0, 0]

    return record["Y"][0, 6]["Data"].flatten()


def calculate_features(signal):
    return {
        "rms": np.sqrt(np.mean(signal ** 2)),
        "peak": np.max(np.abs(signal)),
        "kurtosis": kurtosis(signal, fisher=False)
    }


for bearing in ["K001", "KA01", "KI01"]:
    folder = Path("data") / bearing

    files = sorted(folder.glob("N09_M07_F10_*.mat"))

    print(f"\nBearing: {bearing}")
    print("Number of recordings:", len(files))

    results = []

    for file in files:
        signal = load_vibration(file)
        features = calculate_features(signal)
        results.append(features)

    if not results:
        print("No matching recordings found.")
        continue

    for feature in ["rms", "peak", "kurtosis"]:
        values = [result[feature] for result in results]

        print(
            f"{feature}: "
            f"mean={np.mean(values):.4f}, "
            f"std={np.std(values):.4f}"
        )