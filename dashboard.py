import streamlit as st
from edgementor import analyze_bearing
from teacher_ai import ask_teacher
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from build_dataset import load_vibration

st.set_page_config(
    page_title="EdgeMentor",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ EdgeMentor")
st.subheader("Self-Aware Edge AI for Industrial Predictive Maintenance")

st.write(
    "EdgeMentor monitors bearing health, detects unreliable "
    "predictions, and escalates suspicious cases for human review."
)

st.divider()

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

if "teacher_response" not in st.session_state:
    st.session_state.teacher_response = None

# Bearing selection
bearing = st.selectbox(
    "Select Bearing",
    ["K003", "KA05", "KI05"]
)

recording = st.number_input(
    "Recording Number",
    min_value=1,
    max_value=20,
    value=1
)
sensor_scenario = st.selectbox(
    "Sensor Scenario",
    [
        "Normal",
        "Stuck Sensor",
        "Missing Readings"
    ]
)

# Run analysis
if st.button("Analyze Bearing", type="primary"):

    st.session_state.teacher_response = None
    st.session_state.analyzed_scenario = sensor_scenario

    with st.spinner("Analyzing vibration data..."):
        st.session_state.result = analyze_bearing(
            bearing=bearing,
            recording=recording,
            use_teacher=False,
            sensor_fault=(
                None if sensor_scenario == "Normal"
                else sensor_scenario
    )
)

# Display saved analysis results
result = st.session_state.result
if result is not None:
    st.subheader("Vibration Signal")

    file_path = Path(
        f"data/{result['bearing']}/"
        f"N09_M07_F10_{result['bearing']}_{result['recording']}.mat"
    )

    signal = load_vibration(file_path)

    # Apply the selected simulation to the displayed signal
    if st.session_state.get("analyzed_scenario") == "Stuck Sensor":
        signal = np.full_like(signal, 0.15)

    elif st.session_state.get("analyzed_scenario") == "Missing Readings":
        signal = signal.copy()
        signal[100] = np.nan

    # Display the first 2,000 samples
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal[:2000], linewidth=0.8)
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Vibration Amplitude")
    ax.set_title("Bearing Vibration Waveform")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

if result is not None:

    st.subheader("Analysis Results")

    st.write("**Sensor Status:**", result["sensor_status"])

    if result["decision"] == "ABSTAIN":
        st.error("Sensor data is invalid. Prediction withheld.")

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Condition",
                result["prediction"]
            )

        with col2:
            st.metric(
                "Model Confidence",
                f"{result['confidence']:.1%}"
            )

        with col3:
            st.metric(
                "Supervisor Decision",
                result["decision"]
            )

        st.subheader("Reliability Checks")

        st.write(
            "Low Confidence:",
            result["low_confidence"]
        )

        st.write(
            "Distribution Anomaly:",
            result["anomaly_flag"]
        )

        if result["decision"] == "ESCALATE":

            st.warning(
                "Prediction requires further review."
            )

            if st.button("Ask NVIDIA Teacher AI"):

                with st.spinner("Consulting NVIDIA Nemotron..."):

                    st.session_state.teacher_response = ask_teacher(
                        prediction=result["prediction"],
                        confidence=result["confidence"],
                        low_confidence=result["low_confidence"],
                        anomaly_flag=result["anomaly_flag"]
                    )

            if st.session_state.teacher_response is not None:

                st.subheader("NVIDIA Nemotron Guidance")

                st.markdown(
                    st.session_state.teacher_response
                )

        else:
            st.success(
                "Prediction passed configured reliability checks."
            )