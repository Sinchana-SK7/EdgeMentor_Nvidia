
"""
EdgeMentor diagnostic tools.

These functions perform deterministic analysis on recorded
bearing vibration data. NVIDIA Nemotron may request them,
but cannot execute arbitrary Python code.
"""

from pathlib import Path

import numpy as np
from scipy.signal import hilbert, sosfiltfilt

from build_dataset import load_vibration
from edgementor import (
    FS,
    BPFO,
    BPFI,
    SOS,
    extract_fault_features
)


ALLOWED_BEARINGS = {
    "K003", "KA05", "KI05"
}


def load_recording(bearing_id, recording=1):
    if bearing_id not in ALLOWED_BEARINGS:
        raise ValueError("Bearing ID not permitted.")

    if not isinstance(recording, int) or not 1 <= recording <= 20:
        raise ValueError("Recording must be between 1 and 20.")

    path = (
        Path("data")
        / bearing_id
        / f"N09_M07_F10_{bearing_id}_{recording}.mat"
    )

    return np.asarray(
        load_vibration(path),
        dtype=float
    ).ravel()


def inspect_bearing_spectrum(bearing_id, recording=1):
    """
    Return measured BPFO/BPFI harmonic evidence.
    """

    signal = load_recording(bearing_id, recording)

    features = extract_fault_features(signal)

    filtered = sosfiltfilt(SOS, signal)
    envelope = np.abs(hilbert(filtered))
    envelope -= envelope.mean()

    spectrum = np.abs(np.fft.rfft(envelope))

    frequencies = np.fft.rfftfreq(
        len(envelope),
        d=1 / FS
    )

    reference_mask = (
        (frequencies >= 20) &
        (frequencies <= 1000)
    )

    reference = (
        np.sqrt(
            np.mean(spectrum[reference_mask] ** 2)
        )
        + 1e-12
    )

    evidence = []

    for prefix, fundamental in [
        ("BPFO", BPFO),
        ("BPFI", BPFI)
    ]:
        for harmonic in range(1, 4):
            target = fundamental * harmonic

            mask = (
                np.abs(frequencies - target) <= 2
            )

            indices = np.where(mask)[0]

            peak_index = indices[
                np.argmax(spectrum[indices])
            ]

            evidence.append({
                "feature": f"{prefix.lower()}_h{harmonic}",
                "target_hz": round(target, 2),
                "measured_peak_hz": round(
                    float(frequencies[peak_index]), 2
                ),
                "normalized_amplitude": round(
                    float(spectrum[peak_index] / reference),
                    4
                )
            })

    return {
        "tool": "inspect_bearing_spectrum",
        "bearing_id": bearing_id,
        "recording": recording,
        "sampling_rate_hz": FS,
        "method": (
            "3-10 kHz bandpass, Hilbert envelope, "
            "FFT, +/-2 Hz harmonic peak search"
        ),
        "harmonics": evidence,
        "features": {
            key: round(float(value), 4)
            for key, value in features.items()
        },
        "interpretation_limit": (
            "Harmonic amplitudes are diagnostic features, "
            "not proof of a specific bearing defect."
        )
    }


def compare_fault_evidence(bearing_id, recording=1):
    """
    Summarize measured outer-race and inner-race
    harmonic amplitudes without declaring a diagnosis.
    """

    signal = load_recording(bearing_id, recording)

    features = extract_fault_features(signal)

    bpfo_values = [
        features[f"bpfo_h{i}"]
        for i in range(1, 4)
    ]

    bpfi_values = [
        features[f"bpfi_h{i}"]
        for i in range(1, 4)
    ]

    bpfo_mean = float(np.mean(bpfo_values))
    bpfi_mean = float(np.mean(bpfi_values))

    return {
        "tool": "compare_fault_evidence",
        "bearing_id": bearing_id,
        "recording": recording,
        "bpfo_mean_normalized_amplitude": round(
            bpfo_mean, 4
        ),
        "bpfi_mean_normalized_amplitude": round(
            bpfi_mean, 4
        ),
        "bpfo_to_bpfi_ratio": round(
            bpfo_mean / (bpfi_mean + 1e-12),
            4
        ),
        "caution": (
            "The ratio is descriptive only. "
            "It is not a calibrated fault probability "
            "or a validated decision threshold."
        )
    }


def inspect_signal_quality(bearing_id, recording=1):
    """
    Inspect basic numerical signal characteristics.
    """

    signal = load_recording(bearing_id, recording)

    finite_mask = np.isfinite(signal)
    finite_values = signal[finite_mask]

    if len(finite_values) == 0:
        return {
            "tool": "inspect_signal_quality",
            "status": "INVALID",
            "reason": "No finite measurements."
        }

    rms = float(
        np.sqrt(np.mean(finite_values ** 2))
    )

    peak = float(
        np.max(np.abs(finite_values))
    )

    return {
        "tool": "inspect_signal_quality",
        "bearing_id": bearing_id,
        "recording": recording,
        "sample_count": int(len(signal)),
        "non_finite_samples": int(
            np.sum(~finite_mask)
        ),
        "mean": round(
            float(np.mean(finite_values)), 6
        ),
        "standard_deviation": round(
            float(np.std(finite_values)), 6
        ),
        "rms": round(rms, 6),
        "absolute_peak": round(peak, 6),
        "crest_factor": round(
            peak / (rms + 1e-12), 4
        ),
        "caution": (
            "These are basic numerical checks, "
            "not a complete sensor-health assessment."
        )
    }


DIAGNOSTIC_TOOLS = {
    "inspect_bearing_spectrum": inspect_bearing_spectrum,
    "compare_fault_evidence": compare_fault_evidence,
    "inspect_signal_quality": inspect_signal_quality
}


if __name__ == "__main__":
    import json

    for name, tool in DIAGNOSTIC_TOOLS.items():
        print(f"\n===== {name} =====")

        result = tool(
            bearing_id="KA05",
            recording=1
        )

        print(json.dumps(result, indent=2))
