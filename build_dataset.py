from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.io import loadmat
from scipy.stats import kurtosis

# Bearing identities and provisional class labels
bearings = {
    "K001": 0, "K002": 0, "K003": 0,
    "KA01": 1, "KA03": 1, "KA05": 1,
    "KI01": 2, "KI03": 2, "KI05": 2
}

def load_vibration(file_path):
    data = loadmat(file_path)
    key = next(k for k in data if not k.startswith("__"))
    record = data[key][0, 0]

    channels = record["Y"]
    for channel in channels.flat:
        name = str(channel["Name"].flat[0])
        if name == "vibration_1":
            return np.asarray(channel["Data"], dtype=float).ravel()

    raise ValueError(f"Vibration channel not found: {file_path}")


def extract_features(signal):
    # Time-domain features
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    # Frequency-domain features
    sampling_rate = 64000

    centered_signal = signal - np.mean(signal)

    fft = np.fft.rfft(centered_signal)

    frequencies = np.fft.rfftfreq(
        len(centered_signal),
        d=1 / sampling_rate
    )

    power = np.abs(fft) ** 2

    total_power = np.sum(power)

    def band_energy(low, high):
        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )

        if total_power == 0:
            return 0.0

        return np.sum(power[mask]) / total_power

    return {
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis(signal, fisher=False),
        "crest_factor": peak / rms if rms > 0 else 0,
        "skewness": skew(signal),

        "energy_0_1k": band_energy(0, 1000),
        "energy_1_3k": band_energy(1000, 3000),
        "energy_3_6k": band_energy(3000, 6000),
        "energy_6_10k": band_energy(6000, 10000)
    }


if __name__ == "__main__":
    rows = []

    for bearing, label in bearings.items():
        folder = Path("data") / bearing
        files = sorted(folder.glob(f"N09_M07_F10_{bearing}_*.mat"))

        print(f"{bearing}: {len(files)} recordings")
                # Validate that all 20 recordings are available
        expected_files = {
            f"N09_M07_F10_{bearing}_{i}.mat"
            for i in range(1, 21)
        }

        actual_files = {file.name for file in files}
        missing_files = expected_files - actual_files

        if missing_files:
            raise FileNotFoundError(
                f"{bearing}: Missing recordings: "
                f"{sorted(missing_files)}"
            )

        for file in files:
            signal = load_vibration(file)
            features = extract_features(signal)

            rows.append({
                "bearing_id": bearing,
                "recording": file.name,
                "label": label,
                **features
            })

    df = pd.DataFrame(rows)

    df.to_csv("bearing_features.csv", index=False)

    print("\nTotal recordings:", len(df))
    print("\nRecordings per class:")
    print(df.groupby("label").size())
    print("\nFirst five rows:")
    print(df.head())