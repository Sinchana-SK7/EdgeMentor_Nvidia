from inspect_fft import load_vibration
from sensor_health import check_sensor
import numpy as np

# Load one real Paderborn bearing recording
signal = load_vibration("K003")

print("Bearing: K003")
print("Number of samples:", len(signal))
print("Minimum:", np.min(signal))
print("Maximum:", np.max(signal))
print("Standard deviation:", np.std(signal))

# Check the original signal
print("Original signal:", check_sensor(signal))

# Simulate a stuck sensor
stuck_signal = np.full_like(signal, 0.15)

print("Stuck signal:", check_sensor(stuck_signal))

# Simulate missing readings
missing_signal = signal.copy()
missing_signal[100:200] = np.nan

print("Missing readings:", check_sensor(missing_signal))
# Simulate a sensor limited to -0.5 and +0.5
sensor_limit = 0.5

clipped_signal = np.clip(
    signal,
    -sensor_limit,
    sensor_limit
)

print("\nClipping experiment")
print("Original maximum:", np.max(signal))
print("Clipped maximum:", np.max(clipped_signal))

clipped_count = np.sum(
    np.abs(clipped_signal) >= sensor_limit
)

clipped_ratio = clipped_count / len(clipped_signal)

print("Clipped percentage:", clipped_ratio * 100)

print(
    "Clipped signal:",
    check_sensor(clipped_signal, sensor_limit=sensor_limit)
)