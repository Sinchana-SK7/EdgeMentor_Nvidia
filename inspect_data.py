from scipy.io import loadmat

file_path = "data/K001/N09_M07_F10_K001_1.mat"

data = loadmat(file_path)

print("Variables inside MATLAB file:")
for key in data.keys():
    print(key)


recording = data["N09_M07_F10_K001_1"]

print("\nRecording type:", type(recording))
print("Recording shape:", recording.shape)
print("Recording fields:", recording.dtype.names)


record = recording[0, 0]

for field in recording.dtype.names:
    value = record[field]

    print(f"\nField: {field}")
    print("Shape:", value.shape)
    print("Data type:", value.dtype)

    if value.dtype.names:
        print("Nested fields:", value.dtype.names)


channels = record["Y"]

for i in range(channels.shape[1]):
    channel = channels[0, i]

    name = channel["Name"]
    unit = channel["Unit"]

    print(f"\nChannel {i}")
    print("Name:", name)
    print("Unit:", unit)

import numpy as np

# Select vibration sensor
vibration_channel = channels[0, 6]

# Access its measurement data
vibration_data = vibration_channel["Data"]

print("\nVibration data type:", type(vibration_data))
print("Vibration data shape:", vibration_data.shape)
print("Vibration data dtype:", vibration_data.dtype)

# Inspect the first few values
print("First values:", vibration_data.flat[:5])

import matplotlib.pyplot as plt

# Convert the 2D array into a 1D signal
vibration = vibration_data.flatten()

# Plot the first 5000 samples
plt.figure(figsize=(12, 4))

plt.plot(vibration[:5000])

plt.title("Healthy Bearing - Vibration Signal")
plt.xlabel("Sample Number")
plt.ylabel("Vibration Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()