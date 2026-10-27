import numpy as np


def check_sensor(signal, sensor_limit=None):
    signal = np.asarray(signal)

    if signal.size == 0:
        return "INVALID: Empty signal"

    if not np.all(np.isfinite(signal)):
        return "INVALID: Missing or infinite readings"

    variation = np.std(signal)

    if variation < 1e-8:
        return "INVALID: Sensor may be stuck"

    # Only check clipping when the sensor limit is known
    if sensor_limit is not None:
        clipped = np.abs(signal) >= sensor_limit
        clipped_ratio = np.mean(clipped)

        if clipped_ratio > 0.10:
            return "INVALID: Possible sensor clipping"

    return "VALID: No basic sensor fault detected"


# Run these tests only when executing sensor_health.py directly
if __name__ == "__main__":

    # Test 1: Functioning sensor
    normal_signal = np.array([0.12, 0.18, -0.09, 0.25, -0.14])

    # Test 2: Stuck sensor
    stuck_signal = np.array([0.15, 0.15, 0.15, 0.15, 0.15])

    # Test 3: Missing reading
    missing_signal = np.array([0.12, 0.18, np.nan, 0.25, -0.14])

    # Test 4: Simulated clipped sensor
    clipped_signal = np.array([
        0.2, 0.5, 1.0, 1.0, 1.0,
        -0.4, -1.0, -1.0, 0.3, 0.1
    ])

    print("Normal sensor:", check_sensor(normal_signal))
    print("Stuck sensor:", check_sensor(stuck_signal))
    print("Missing reading:", check_sensor(missing_signal))
    print(
        "Clipped sensor:",
        check_sensor(clipped_signal, sensor_limit=1.0)
    )