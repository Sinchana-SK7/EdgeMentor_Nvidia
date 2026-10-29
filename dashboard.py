
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert

from edgementor import analyze_bearing
from teacher_ai import investigate_bearing
from build_dataset import load_vibration


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EdgeMentor | Industrial Edge Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

FS = 64000
BPFO = 45.82
BPFI = 74.15
CONFIDENCE_THRESHOLD = 0.90

CYAN = "#36ddf7"
AMBER = "#fbbf24"
GREEN = "#35d8ab"
RED = "#fb7185"
PANEL = "#122238"
BACKGROUND = "#08111f"
GRID = "#294158"

FEATURE_NAMES = [
    "bpfo_h1", "bpfo_h2", "bpfo_h3",
    "bpfi_h1", "bpfi_h2", "bpfi_h3"
]


# ============================================================
# CUSTOM THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            ellipse at top right,
            rgba(0, 190, 255, 0.08),
            transparent 45%
        ),
        #08111f;
    color: #eaf2ff;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #0d1b2e;
    border-right: 1px solid #233b55;
}

h1, h2, h3 {
    color: #f2f7ff !important;
}

[data-testid="stMetric"] {
    background: #122238;
    padding: 17px;
    border-radius: 14px;
    border: 1px solid #25405b;
}

[data-testid="stMetricLabel"] {
    color: #9eb6d1;
}

[data-testid="stMetricValue"] {
    color: #f2f7ff;
}

.hero {
    padding: 20px 25px;
    background: linear-gradient(
        110deg,
        #102a43,
        #102238 65%,
        #0d3341
    );
    border: 1px solid #28526c;
    border-radius: 16px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    color: #f5fbff;
    letter-spacing: 1px;
}

.hero-subtitle {
    color: #9fbad2;
    font-size: 13px;
    margin-top: 5px;
}

.eyebrow {
    color: #53d9f7;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
}

.status-card {
    border-radius: 13px;
    padding: 16px 20px;
    margin: 12px 0 16px 0;
    border: 1px solid;
}

.status-accept {
    background: #10372e;
    border-color: #218c6c;
    color: #a3f3d1;
}

.status-escalate {
    background: #3a2b12;
    border-color: #a87924;
    color: #ffe0a1;
}

.status-abstain {
    background: #3c1d28;
    border-color: #a94b66;
    color: #ffc2d0;
}

.status-title {
    font-size: 17px;
    font-weight: 750;
}

.status-description {
    font-size: 13px;
    margin-top: 4px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #00b8d9,
        #176fe0
    );
    color: white;
    border: none;
    font-weight: 700;
    border-radius: 10px;
}

div.stButton > button {
    border-radius: 10px;
}

hr {
    border-color: #24415c;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "teacher_response" not in st.session_state:
    st.session_state.teacher_response = None

if "analyzed_scenario" not in st.session_state:
    st.session_state.analyzed_scenario = "Normal"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def get_signal(bearing, recording):
    path = Path(
        f"data/{bearing}/"
        f"N09_M07_F10_{bearing}_{recording}.mat"
    )

    return np.asarray(
        load_vibration(path),
        dtype=float
    ).ravel()


def apply_sensor_scenario(signal, scenario):
    signal = signal.copy()

    if scenario == "Stuck Sensor":
        signal[:] = 0.15

    elif scenario == "Missing Readings":
        signal[100] = np.nan

    return signal


# ============================================================
# SIGNAL PROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def compute_envelope_spectrum(signal):
    """
    Match the physics-based feature extraction approach:
    - 64 kHz sampling
    - 3–10 kHz bandpass
    - Hilbert envelope
    - FFT of demeaned envelope
    """

    sos = butter(
        4,
        [3000, 10000],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal)

    envelope = np.abs(hilbert(filtered))
    envelope = envelope - envelope.mean()

    spectrum = np.abs(np.fft.rfft(envelope))
    frequencies = np.fft.rfftfreq(
        len(envelope),
        d=1 / FS
    )

    background_mask = (
        (frequencies >= 20) &
        (frequencies <= 1000)
    )

    scale = (
        np.sqrt(
            np.mean(
                spectrum[background_mask] ** 2
            )
        )
        + 1e-12
    )

    normalized_spectrum = spectrum / scale

    return frequencies, normalized_spectrum


def extract_display_features(frequencies, spectrum):
    """
    Six normalized harmonic amplitudes.

    Use the maximum amplitude within +/-2 Hz
    of each theoretical harmonic.
    """

    features = {}

    for prefix, fundamental in [
        ("bpfo", BPFO),
        ("bpfi", BPFI)
    ]:

        for harmonic in range(1, 4):
            target = fundamental * harmonic

            mask = (
                np.abs(frequencies - target) <= 2.0
            )

            value = (
                float(np.max(spectrum[mask]))
                if np.any(mask)
                else 0.0
            )

            features[f"{prefix}_h{harmonic}"] = value

    return features


# ============================================================
# PLOTLY CHART STYLING
# ============================================================

def style_chart(fig, height=310):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="Segoe UI, Arial",
            color="#dce9f8",
            size=12
        ),
        height=height,
        margin=dict(
            l=35,
            r=25,
            t=35,
            b=40
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.15,
            x=0
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False
    )

    return fig


# ============================================================
# INTERACTIVE WAVEFORM
# ============================================================

def render_waveform(signal):
    # Display a short recorded segment.
    # Plotly permits zooming and inspecting individual samples.
    samples = signal[:2000]

    time_ms = (
        np.arange(len(samples)) / FS * 1000
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_ms,
            y=samples,
            mode="lines",
            name="Vibration",
            line=dict(
                color=CYAN,
                width=1.3
            ),
            hovertemplate=(
                "Time: %{x:.3f} ms"
                "<br>Amplitude: %{y:.4f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        xaxis_title="Time (ms)",
        yaxis_title="Vibration amplitude",
        showlegend=False
    )

    style_chart(fig, height=310)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


# ============================================================
# ENVELOPE FREQUENCY SPECTRUM
# ============================================================

def render_spectrum(frequencies, spectrum):
    mask = (
        (frequencies >= 20) &
        (frequencies <= 260)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frequencies[mask],
            y=spectrum[mask],
            mode="lines",
            name="Envelope spectrum",
            line=dict(
                color=CYAN,
                width=1.5
            )
        )
    )

    for fundamental, label, color in [
        (BPFO, "BPFO", CYAN),
        (BPFI, "BPFI", AMBER)
    ]:

        for harmonic in range(1, 4):
            target = fundamental * harmonic

            fig.add_vline(
                x=target,
                line_color=color,
                line_width=1.5,
                line_dash="dash",
                opacity=0.8
            )

            if harmonic == 1:
                fig.add_annotation(
                    x=target,
                    y=1.0,
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(
                        color=color,
                        size=11
                    ),
                    yanchor="bottom"
                )

    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Normalized amplitude",
        showlegend=False
    )

    style_chart(fig, height=330)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.caption(
        "Dashed markers indicate theoretical bearing fault "
        "frequencies and their harmonics. Peaks are measured "
        "from the selected recording; markers are not "
        "confirmed fault detections."
    )


# ============================================================
# SIX PHYSICS-INFORMED FEATURES
# ============================================================

def render_feature_chart(features):
    names = FEATURE_NAMES

    values = [
        features[name]
        for name in names
    ]

    colors = [
        CYAN if name.startswith("bpfo")
        else AMBER
        for name in names
    ]

    labels = [
        name.upper().replace("_", " ")
        for name in names
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[
                f"{value:.2f}"
                for value in values
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Normalized amplitude: %{y:.3f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        xaxis_title="Fault-frequency feature",
        yaxis_title="Normalized amplitude",
        showlegend=False
    )

    style_chart(fig, height=330)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.caption(
        "Cyan: outer-race frequency harmonics (BPFO). "
        "Amber: inner-race frequency harmonics (BPFI). "
        "These are feature values, not class probabilities "
        "or feature-importance scores."
    )


# ============================================================
# SUPERVISOR VISUALIZATION
# ============================================================

def display_status(decision):
    if decision == "ACCEPT":
        css_class = "status-accept"
        title = "✓ PREDICTION ACCEPTED"
        description = (
            "The prediction passed configured reliability checks."
        )

    elif decision == "ESCALATE":
        css_class = "status-escalate"
        title = "⚠ HUMAN REVIEW REQUIRED"
        description = (
            "The supervisor flagged this prediction. "
            "NVIDIA Nemotron can provide diagnostic guidance."
        )

    else:
        css_class = "status-abstain"
        title = "✕ SENSOR INPUT REJECTED"
        description = (
            "Sensor validation failed. "
            "Classification was withheld."
        )

    st.markdown(
        f"""
<div class="status-card {css_class}">
    <div class="status-title">{title}</div>
    <div class="status-description">{description}</div>
</div>
        """,
        unsafe_allow_html=True
    )


def render_decision_pipeline(result):
    decision = result["decision"]

    if decision == "ABSTAIN":
        steps = [
            ("01", "SENSOR", "INVALID"),
            ("02", "FEATURES", "SKIPPED"),
            ("03", "CLASSIFIER", "WITHHELD"),
            ("04", "SUPERVISOR", "INPUT REJECTED"),
            ("05", "ACTION", "ABSTAIN")
        ]

    else:
        reasons = []

        if result["low_confidence"]:
            reasons.append("LOW CONFIDENCE")

        if result["anomaly_flag"]:
            reasons.append("ANOMALY")

        reason = (
            " + ".join(reasons)
            if reasons
            else "CHECKS PASSED"
        )

        steps = [
            ("01", "SENSOR", "VALID"),
            ("02", "FEATURES", "6 EXTRACTED"),
            ("03", "CLASSIFIER", result["prediction"]),
            ("04", "SUPERVISOR", reason),
            ("05", "ACTION", decision)
        ]

    columns = st.columns(5, gap="small")

    for column, (number, label, value) in zip(
        columns, steps
    ):
        with column:
            with st.container(border=True):
                st.caption(f"{number} · {label}")
                st.markdown(f"**{value}**")


def render_confidence_monitor(result):
    if result["decision"] == "ABSTAIN":
        st.info(
            "Request an independent second opinion from NVIDIA "
            "Nemotron. It can inspect approved diagnostic "
            "measurements and assess the evidence alongside "
            "the edge model's prediction. The investigation "
            "does not automatically change the edge decision."
        )
        return

    confidence = float(result["confidence"])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Confidence",
            f"{confidence:.1%}"
        )

    with col2:
        st.metric(
            "Acceptance Threshold",
            f"{CONFIDENCE_THRESHOLD:.0%}"
        )

    st.progress(
        min(max(confidence, 0.0), 1.0)
    )

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "Confidence below threshold. "
            "Review required."
        )
    else:
        st.success(
            "Confidence threshold passed. "
            "Other reliability checks must also pass."
        )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">INDUSTRIAL EDGE INTELLIGENCE</div>
    <div class="hero-title"> EDGEMENTOR</div>
    <div class="hero-subtitle">
        Self-Aware Edge AI for Industrial Predictive Maintenance
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Control Center")

    st.caption(
        "Configure a recorded bearing analysis scenario."
    )

    bearing = st.selectbox(
        "Bearing ID",
        ["K003", "KA05", "KI05"]
    )

    recording = st.number_input(
        "Recording Number",
        min_value=1,
        max_value=20,
        value=1,
        step=1
    )

    sensor_scenario = st.selectbox(
        "Sensor Scenario",
        [
            "Normal",
            "Stuck Sensor",
            "Missing Readings"
        ]
    )

    analyze_clicked = st.button(
        "⚡ Analyze Bearing",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.markdown("###  System Modules")

    st.caption("✓ Physics-informed classifier")
    st.caption("✓ Confidence monitoring")
    st.caption("✓ Distribution anomaly detection")
    st.caption("✓ Sensor health validation")
    st.caption("✓ NVIDIA Nemotron escalation")

    st.divider()

    st.caption(
        "Research prototype · Paderborn bearing dataset"
    )


# ============================================================
# SESSION ACTION
# ============================================================

if analyze_clicked:
    st.session_state.teacher_response = None
    st.session_state.analyzed_scenario = sensor_scenario

    with st.spinner("Analyzing bearing vibration..."):
        st.session_state.result = analyze_bearing(
            bearing=bearing,
            recording=int(recording),
            use_teacher=False,
            sensor_fault=(
                None
                if sensor_scenario == "Normal"
                else sensor_scenario
            )
        )


# ============================================================
# MAIN DASHBOARD
# ============================================================

result = st.session_state.result

if result is None:
    st.info(
        "Select a bearing and scenario in the Control Center, "
        "then click Analyze Bearing."
    )

else:
    decision = result["decision"]
    scenario = st.session_state.analyzed_scenario

    st.markdown(
        '<div class="eyebrow">ANALYSIS OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"### Bearing {result['bearing']} "
        f"· Recording {result['recording']}"
    )

    # TOP METRICS

    col1, col2, col3 = st.columns(3)

    if decision == "ABSTAIN":
        with col1:
            st.metric("Sensor Health", "INVALID")
        with col2:
            st.metric("Prediction", "WITHHELD")
        with col3:
            st.metric("Supervisor Decision", "ABSTAIN")
    else:
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
                decision
            )

    display_status(decision)

    # AI PIPELINE

    st.markdown("###  AI Reliability Decision Pipeline")
    render_decision_pipeline(result)

    # CONFIDENCE

    st.markdown("###  Confidence Threshold Monitor")
    render_confidence_monitor(result)

    # LOAD THE SELECTED RECORDED SIGNAL

    raw_signal = get_signal(
        result["bearing"],
        result["recording"]
    )

    display_signal = apply_sensor_scenario(
        raw_signal,
        scenario
    )

    # SIGNAL INTELLIGENCE

    st.divider()
    st.markdown("##  Signal Intelligence")

    st.caption(
        "Recorded vibration replay and physics-informed "
        "signal diagnostics."
    )

    st.markdown("### Interactive Vibration Waveform")
    render_waveform(display_signal)

    if decision == "ABSTAIN":
        st.error(
            "Sensor input is invalid. "
            "Frequency analysis and feature extraction "
            "are withheld."
        )

    else:
        with st.spinner("Computing envelope spectrum..."):
            frequencies, spectrum = (
                compute_envelope_spectrum(raw_signal)
            )

            display_features = extract_display_features(
                frequencies,
                spectrum
            )

        left, right = st.columns(
            2,
            gap="large"
        )

        with left:
            st.markdown("### Envelope Frequency Spectrum")
            render_spectrum(
                frequencies,
                spectrum
            )

        with right:
            st.markdown("### Physics-Informed Features")
            render_feature_chart(
                display_features
            )

        with st.expander("View extracted feature values"):
            feature_table = pd.DataFrame(
                [
                    {
                        "Feature": name,
                        "Normalized amplitude": round(
                            display_features[name], 4
                        )
                    }
                    for name in FEATURE_NAMES
                ]
            )

            st.dataframe(
                feature_table,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Feature visualization uses the same nominal "
            "fault-frequency targets and signal-processing "
            "method as the classifier. "
            "These plots are diagnostic evidence, "
            "not independent fault verification."
        )

    # RELIABILITY DIAGNOSTICS

    st.divider()
    st.markdown("###  Reliability Diagnostics")

    st.write(
        "**Sensor Status:**",
        result["sensor_status"]
    )

    if decision != "ABSTAIN":
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Low Confidence",
                str(result["low_confidence"])
            )

        with col2:
            st.metric(
                "Distribution Anomaly",
                str(result["anomaly_flag"])
            )

    # NVIDIA TEACHER AI


    # ========================================================
    # NVIDIA NEMOTRON DIAGNOSTIC INVESTIGATOR
    # ========================================================

    if decision in ("ACCEPT", "ESCALATE"):

        st.divider()

        st.markdown("##  NVIDIA Nemotron Diagnostic Investigator")

        st.caption(
            "Autonomous diagnostic tool selection · "
            "Measured evidence · Human-supervised decisions"
        )

        st.info(
            "The edge reliability supervisor has escalated "
            "this prediction. Nemotron can request approved "
            "diagnostic measurements and evaluate the evidence."
        )

        if st.button(
            " Launch NVIDIA Investigation",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Nemotron is investigating the bearing..."
            ):

                investigation = investigate_bearing(
                    bearing=result["bearing"],
                    recording=int(result["recording"]),
                    prediction=result["prediction"],
                    confidence=float(result["confidence"]),
                    low_confidence=bool(result["low_confidence"]),
                    anomaly_flag=bool(result["anomaly_flag"]),
                    decision=result["decision"]
                )

                st.session_state.teacher_response = investigation

        investigation = st.session_state.teacher_response

        if investigation is not None:

            if investigation["status"] == "COMPLETED":
                st.success(
                    "NVIDIA Nemotron investigation completed."
                )

            else:
                st.error(
                    f"Investigation status: "
                    f"{investigation['status']}"
                )

            st.markdown("###  Investigation Execution Trace")

            trace = investigation["tool_trace"]

            st.metric(
                "Diagnostic Tool Calls",
                investigation["tool_calls_count"]
            )

            if trace:

                for index, entry in enumerate(trace, start=1):

                    tool_name = entry["tool"].replace(
                        "_", " "
                    ).title()

                    with st.expander(
                        f"Step {index} · {tool_name}",
                        expanded=False
                    ):

                        st.caption(
                            f"Investigation round {entry['round']}"
                        )

                        st.json(entry["result"])

            else:

                st.warning(
                    "No diagnostic tools were executed."
                )

            st.markdown("###  Evidence-Grounded Assessment")

            with st.container(border=True):

                st.markdown(
                    investigation["report"]
                )

            st.caption(
                "Nemotron uses measured diagnostic evidence "
                "to support investigation. It does not "
                "override the local reliability supervisor, "
                "confirm a defect, or authorize maintenance."
            )

    elif decision == "ABSTAIN":

        st.divider()

        st.markdown("### 🛑 Sensor Safety Gate")

        st.warning(
            "The input failed sensor validation. "
            "The classifier did not issue a prediction. "
            "Nemotron investigation is blocked."
        )

    else:

        st.success(
            "Configured reliability checks passed. "
            "This does not guarantee classification correctness."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EdgeMentor | Physics-Informed Edge AI "
    "+ Reliability Supervision + NVIDIA Nemotron "
    "| Research / Hackathon Prototype"
)
