# EdgeMentor — Day 1 Results

## Objective
Establish a baseline industrial bearing fault-diagnosis pipeline using the Paderborn University Bearing Dataset.

## Dataset
- 9 physical bearings
- 180 recordings
- 3 classes: Healthy, Outer-ring damage, Inner-ring damage
- Operating condition: N09_M07_F10
- Input: Vibration sensor data

## Features
RMS, peak amplitude, kurtosis, crest factor, and skewness.

## Baseline Model
Logistic Regression with StandardScaler.

Training: 6 bearing identities (120 recordings).

Testing: 3 unseen bearing identities (60 recordings).

## Results
- Three-feature accuracy: 33.33%
- Five-feature accuracy: 33.33%
- KI05 incorrectly classified as healthy with approximately 99.55% average confidence using five features.

## Key Findings
- Statistical vibration features vary substantially between physical bearings.
- Five basic features did not generalize successfully to the selected unseen bearings.
- High classification confidence does not guarantee correctness.
- Current test results are exploratory and require independent evaluation.

## Day 2 Priorities
1. Develop a stronger signal-processing or lightweight ML baseline.
2. Establish a more robust bearing-level evaluation protocol.
3. Begin implementing EdgeMentor's reliability supervisor.