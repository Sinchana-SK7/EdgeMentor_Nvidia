# 🛡️ EdgeMentor
### Self-Aware Edge AI for Industrial Predictive Maintenance

**EdgeMentor is a reliability-aware industrial Edge AI prototype that monitors bearing vibration signals, identifies potentially unreliable predictions, detects basic sensor malfunctions, and escalates suspicious cases to NVIDIA Nemotron for human-oriented diagnostic guidance.**

## The Problem

Traditional machine learning classifiers can produce confident predictions even when they are incorrect.

In industrial predictive maintenance, this is particularly concerning: a damaged bearing might be classified as healthy, potentially delaying further inspection.

EdgeMentor explores a different approach:

**What if an Edge AI system could recognize when its own predictions should not be trusted automatically?**

## Our Solution

EdgeMentor combines a bearing health classifier with an independent reliability supervisor.

The supervisor evaluates:

- **Prediction uncertainty:** Is the classifier's confidence below a configured threshold?
- **Distribution anomalies:** Does the input differ from the training feature distribution?
- **Sensor integrity:** Are readings missing, non-finite, or suspiciously constant?

Based on these checks, EdgeMentor makes one of three decisions:

| Decision | Meaning |
|---|---|
| ACCEPT | Prediction passes configured reliability checks |
| ESCALATE | Prediction requires further review |
| ABSTAIN | Sensor data is invalid; prediction is withheld |

An ACCEPT decision does not guarantee that the prediction is correct.

## System Architecture

```text
Industrial Bearing Vibration
           |
           v
    Sensor Health Check
           |
     +-----+------+
     |            |
  INVALID        VALID
     |            |
  ABSTAIN         v
              Feature Extraction
                     |
                     v
             Random Forest Model
                     |
                     v
             Reliability Supervisor
              /              \
          ACCEPT           ESCALATE
                              |
                              v
                       NVIDIA Nemotron
                       via Nebius API
                              |
                              v
                     Diagnostic Guidance
                              |
                              v
                       Human Operator
```

## Technology Stack

- Python
- NumPy, Pandas, SciPy
- Scikit-learn
- Random Forest classifier
- Isolation Forest anomaly detector
- Streamlit and Matplotlib
- NVIDIA Nemotron through Nebius Token Factory

## Dataset

EdgeMentor uses the **Paderborn University Bearing Data Center** dataset.

The current prototype uses:

- 9 bearings
- 180 recordings
- 3 classification categories: Healthy, Outer Race Damage, Inner Race Damage
- Vibration-based statistical and frequency-domain features

Training and evaluation are separated by bearing identity rather than randomly splitting recordings from the same bearing.

## Demonstration Scenarios

**1. Normal operation — ACCEPT**

A bearing recording passes the configured reliability checks and its prediction is displayed.

**2. Unreliable prediction — ESCALATE**

A damaged-bearing recording receives a Healthy prediction with 65% model confidence. The supervisor flags low confidence and a distribution anomaly.

NVIDIA Nemotron then provides suggested diagnostic inspections for human review.

**3. Simulated sensor malfunction — ABSTAIN**

A simulated stuck sensor produces constant readings.

EdgeMentor identifies the invalid signal and withholds the classifier prediction.

The dashboard also supports a simulated missing-reading scenario.

## Experimental Results and Limitations

The baseline Random Forest classifier achieved approximately **33.3% accuracy** on the selected held-out bearing recordings.

The prototype demonstrates reliability monitoring and escalation, but does not yet establish production-ready predictive maintenance performance.

In the current evaluation, combining confidence filtering with anomaly detection reduced the error rate among accepted predictions compared with a higher-coverage confidence-only baseline. However, at matched coverage, the tested anomaly detector did **not** demonstrate an additional reduction in accepted errors.

The anomaly detector identifies unusual feature patterns; it does not independently prove that a bearing is faulty or that a specific type of drift has occurred.

NVIDIA Nemotron provides diagnostic suggestions, not verified fault diagnoses. Human review remains necessary.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Sinchana-SK7/EdgeMentor_Nvidia.git
cd EdgeMentor_Nvidia
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Dataset Preparation

EdgeMentor uses vibration recordings from the **Paderborn University Bearing Data Center**.

Dataset source: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter

**1. Download the required bearing recordings**

The current prototype uses nine bearings:

| Bearing condition | Bearing IDs |
|---|---|
| Healthy | K001, K002, K003 |
| Outer-race damage | KA01, KA03, KA05 |
| Inner-race damage | KI01, KI03, KI05 |

Use recordings corresponding to operating condition `N09_M07_F10`.

**2. Organize the dataset**

Create a `data` directory inside the project with the following structure:

```text
EdgeMentor_Nvidia/
├── data/
│   ├── K001/
│   │   ├── N09_M07_F10_K001_1.mat
│   │   ├── N09_M07_F10_K001_2.mat
│   │   └── ...
│   ├── K002/
│   ├── K003/
│   ├── KA01/
│   ├── KA03/
│   ├── KA05/
│   ├── KI01/
│   ├── KI03/
│   └── KI05/
├── build_dataset.py
├── edgementor.py
└── dashboard.py
```

Each bearing directory should contain 20 recordings, numbered 1 through 20.

**3. Generate the feature datasets**

Run these commands from the project directory, in order:

```powershell
python build_dataset.py
python evaluate_fault_features.py
```

The first command generates `bearing_features.csv` containing nine time- and frequency-domain features.

The second command extracts six additional physics-based envelope-spectrum features and creates `bearing_fault_features.csv`. It also runs feature-set evaluation experiments.

The current classifier in `edgementor.py` loads `bearing_fault_features.csv`, so **both commands are required** before launching the dashboard.

**4. Configure Nebius Token Factory**

Create a local `.env` file in the project root:

```env
NEBIUS_API_KEY=your_api_key_here
```

Use your own Nebius Token Factory API key. Never commit `.env` or expose the key in screenshots or logs.

**5. Launch the dashboard**

```powershell
python -m streamlit run dashboard.py
```

Open the local URL displayed in the terminal.

The dashboard supports bearing-recording analysis, prediction-reliability decisions, and the Nemotron investigation workflow. Verify each demonstration scenario before presenting it as a working feature.

**6. Demo video**

Add your public YouTube demonstration link here before submitting:

`[YouTube demo link to be added]`

Keep the demonstration under three minutes if required by the hackathon rules. Show the running application, the reliability decision, and Nemotron's tool-assisted investigation.

**Reproducibility note:** The raw `.mat` recordings and generated CSV files are excluded from GitHub. Download the required dataset and generate the files locally before running the application.


## Future Work

- Improve generalization to unseen bearings and operating conditions.
- Calibrate model confidence and evaluate reliability on larger independent test sets.
- Develop and validate drift detection using controlled distribution shifts.
- Deploy a compact model on embedded Edge AI hardware.
- Integrate real-time sensor streaming.
- Evaluate human-in-the-loop maintenance workflows.

## Project Goal

EdgeMentor explores a simple but important principle:

**A trustworthy Edge AI system should not only make predictions—it should also know when to ask for help.**