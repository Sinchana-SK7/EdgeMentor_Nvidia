from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_vibration(file_path):
    # Load MATLAB file
    data = loadmat(file_path)

    # Get the recording variable
    key = next(k for k in data if not k.startswith("__"))
    recording = data[key][0, 0]

    # Select vibration channel
    vibration = recording["Y"][0, 6]["Data"]

    # Convert to 1D array
    return vibration.flatten()


# Load healthy and damaged recordings
healthy = load_vibration(
    "data/K001/N09_M07_F10_K001_1.mat"
)

damaged = load_vibration(
    "data/KA01/N09_M07_F10_KA01_1.mat"
)

import numpy as np

def calculate_features(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    return rms, peak


healthy_rms, healthy_peak = calculate_features(healthy)
damaged_rms, damaged_peak = calculate_features(damaged)

print("\nHealthy Bearing:")
print("RMS:", healthy_rms)
print("Peak:", healthy_peak)

print("\nDamaged Bearing:")
print("RMS:", damaged_rms)
print("Peak:", damaged_peak)

from scipy.stats import kurtosis

healthy_kurtosis = kurtosis(healthy, fisher=False)
damaged_kurtosis = kurtosis(damaged, fisher=False)

print("\nKurtosis Comparison:")
print("Healthy:", healthy_kurtosis)
print("Damaged:", damaged_kurtosis)
# Display first 5000 samples
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(healthy[:5000])
plt.title("Healthy Bearing - K001")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(damaged[:5000])
plt.title("Damaged Bearing - KA01")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()