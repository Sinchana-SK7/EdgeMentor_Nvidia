import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_vibration(bearing):
    path = (
        f"data/{bearing}/"
        f"N09_M07_F10_{bearing}_1.mat"
    )

    data = loadmat(path)

    measurement = data[
        f"N09_M07_F10_{bearing}_1"
    ]

    channels = measurement["Y"][0, 0]

    for channel in channels[0]:
        name = channel["Name"][0]

        if name == "vibration_1":
            return np.asarray(channel["Data"]).flatten()

    raise ValueError("Vibration channel not found")

if __name__ == "__main__":
    bearings = ["K003", "KA05", "KI05"]

    sampling_rate = 64000

    plt.figure(figsize=(12, 8))

    for bearing in bearings:
        signal = load_vibration(bearing)
        print(
        bearing,
        "samples:", len(signal),
        "standard deviation:", np.std(signal)
        )
        print(f"\nBearing: {bearing}")
        print("Signal shape:", signal.shape)
        print("First 10 samples:", signal[:10])
        print("Minimum:", np.min(signal))
        print("Maximum:", np.max(signal))
        print("Standard deviation:", np.std(signal))

        signal = signal - np.mean(signal)

        fft = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1 / sampling_rate
        )

        magnitude = np.abs(fft) / len(signal)
        print("Maximum FFT magnitude:", np.max(magnitude))
        print("NaN values:", np.isnan(magnitude).sum())
        plt.plot(
            frequencies,
            magnitude,
            label=bearing,
            alpha=0.8
        )

    plt.xlim(0, 10000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT Magnitude")
    plt.title("Bearing Vibration Frequency Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()