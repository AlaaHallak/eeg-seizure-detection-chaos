# EEG Seizure Detection with Chaotic Dynamics

**Integrating chaotic dynamics with time–frequency EEG features for enhanced epileptic seizure detection.**

This repository contains the implementation of a research project on automated epileptic seizure detection from scalp EEG. It detects seizures by combining **time-domain**, **frequency-domain**, and **nonlinear chaos-theory** features, and benchmarks three machine-learning classifiers with and without the chaotic features.

> The core finding: adding nonlinear / chaotic descriptors (Largest Lyapunov Exponent, sample & approximate entropy, correlation dimension, DFA, Higuchi fractal dimension) on top of conventional time–frequency features **consistently improves recall, F1-score, and ROC-AUC across all classifiers, while reducing false negatives** — which matters most for safety-critical clinical monitoring.

---

## Highlights

- **End-to-end pipeline** — raw `.edf` EEG → preprocessing → multi-channel feature extraction → PCA + SMOTE → classification → evaluation.
- **4-stage signal preprocessing** that lifts SNR from **2.82 dB → 24.42 dB**: band-pass filter (0.5–40 Hz), 60 Hz notch filter, Daubechies-4 wavelet denoising, and statistical artifact suppression.
- **Three feature domains** extracted per channel: 5 time-domain, 6 frequency-domain, and 6 chaos/nonlinear features.
- **Three classifiers compared:** Random Forest, XGBoost, and SVM (RBF kernel) — each evaluated on *Time+Frequency* vs *Time+Frequency+Chaos* feature sets.
- **Multi-channel analysis** over six bipolar montage channels covering frontal, temporal, and central-parietal regions.
- **Modular, resumable execution** with auto-save between stages, so feature extraction does not need to be repeated after an error.

---

## Methodology

### Dataset
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet, v1.0.0) — continuous scalp EEG from pediatric patients with intractable epilepsy, sampled at 256 Hz. This study uses seizure-containing recordings from **patients 02, 05, and 06**.

> **Note:** The EEG recordings are **not** redistributed in this repository. Download them directly from PhysioNet and point `BASE_PATH` in [`src/seizure_detection.py`](src/seizure_detection.py) at your local copy.

### Channel selection
Six bipolar channels — `FP1–F7, F7–T7, FP1–F3, F3–C3, C3–P3, P3–O1` — covering the frontal and temporal regions most discriminative for focal seizures.

### Feature set (per channel)
| Domain | Features |
|--------|----------|
| **Time** | Variance, line length, Hjorth activity/mobility/complexity |
| **Frequency** | Delta / alpha / beta band power, spectral entropy, delta-alpha ratio, theta-alpha ratio |
| **Chaos / nonlinear** | Sample entropy, approximate entropy, correlation dimension, DFA, Higuchi fractal dimension, Largest Lyapunov Exponent |

### Pipeline
EEG segments (4 s windows, 50% overlap) are labelled by majority overlap with expert ictal annotations, standardized, reduced with **PCA** (≥95% variance retained), balanced with **SMOTE**, and classified. An 80/20 stratified split is used, with all preprocessing/PCA fitted only on the training set to prevent leakage.

---

## Results

Performance using the **full feature set (Time + Frequency + Chaos)** on the held-out test set:

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|:--------:|:---------:|:------:|:--------:|:-------:|
| Random Forest | 99.82% | 93.33% | 80.46% | 86.42% | 0.9944 |
| XGBoost | 99.83% | 88.51% | 88.51% | 88.51% | 0.9925 |
| SVM-RBF | 99.71% | 74.53% | **90.80%** | 81.87% | **0.9977** |

**Effect of adding chaotic features** (Time+Freq → Time+Freq+Chaos):

| Classifier | Recall | F1-Score | ROC-AUC |
|------------|--------|----------|---------|
| Random Forest | 75.86% → 80.46% | 82.50% → 86.42% | 0.9828 → 0.9944 |
| XGBoost | 82.76% → 88.51% | 82.76% → 88.51% | 0.9887 → 0.9925 |
| SVM-RBF | 88.51% → 90.80% | 64.98% → 81.87% | 0.9929 → 0.9977 |

**Takeaways:** XGBoost gives the most balanced precision/recall; SVM-RBF achieves the highest sensitivity (90.80%) and lowest false-negative count, making it best suited for safety-critical monitoring; Random Forest is competitive at lower computational cost.

---

## Getting started

### Prerequisites
- Python 3.10+
- The CHB-MIT `.edf` recordings (see Dataset note above)

### Installation
```bash
git clone https://github.com/<your-username>/eeg-seizure-detection-chaos.git
cd eeg-seizure-detection-chaos

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run
1. Download the CHB-MIT recordings for patients 02, 05, 06 from PhysioNet.
2. Open [`src/seizure_detection.py`](src/seizure_detection.py) and set `BASE_PATH` to your local EEG data folder.
3. Run it:
   ```bash
   python src/seizure_detection.py
   ```

---

## Repository structure

```
eeg-seizure-detection-chaos/
├── src/
│   └── seizure_detection.py     # Full pipeline: preprocessing → features → 3-classifier comparison
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Authors

Research project by:

- **Nema Salem** — Effat University
- **Alaa Hallaq** — Effat University
- **Hala Alamro** — Effat University
- **Amal Matar** — Effat University

Electrical & Computer Engineering Department, Effat University, Jeddah, Saudi Arabia.

---

## License & usage

**© 2024 the authors. All rights reserved.**

This code is made publicly available for **viewing and reference only**. No license is granted to reuse, redistribute, or create derivative works without the express written permission of the authors. The CHB-MIT dataset is governed by its own [PhysioNet license](https://physionet.org/content/chbmit/1.0.0/).
