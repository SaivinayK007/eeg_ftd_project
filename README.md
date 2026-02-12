# 🧠 EEG-Based Dementia Detection System  
### Alzheimer’s Disease (AD) vs Frontotemporal Dementia (FTD) vs Healthy Control

Machine learning–based EEG analysis system for early detection of neurodegenerative disorders using resting-state EEG data.

This project includes a complete end-to-end pipeline:
- EEG preprocessing
- Feature extraction
- Model training
- Performance evaluation
- Real-time Streamlit web application
- PDF clinical report generation

---

## 📌 Overview

The system analyzes resting-state EEG signals to:

- Detect dementia vs healthy controls  
- Differentiate Alzheimer’s Disease (AD) and Frontotemporal Dementia (FTD)  
- Generate automated risk assessment reports  
- Provide an interactive clinical-style dashboard  

This project is intended for:
- Research purposes  
- Academic projects  
- IEEE paper submission  
- Decision-support prototyping  

---

## 📂 Dataset

Dataset Used: **OpenNeuro ds004504**

EEG Type: Resting-state (Eyes Closed)  
Sampling Rate: 500 Hz  
Channels: 19 (10–20 system)

### File Types
- `.set` → Raw EEG signal (EEGLAB format)
- `.tsv` → Participant metadata
- `.json` → EEG acquisition metadata

### Participant Groups
| Code | Description |
|------|------------|
| A | Alzheimer’s Disease |
| F | Frontotemporal Dementia |
| C | Healthy Control |

---

## 🏗️ System Pipeline

```
Raw EEG (.set)
      ↓
Signal Preprocessing (1–40 Hz bandpass, notch filter)
      ↓
Epoch Segmentation (2s windows)
      ↓
Feature Extraction
  - Delta Power
  - Theta Power
  - Alpha Power
  - Beta Power
  - Spectral Entropy
      ↓
Subject-Level Feature Aggregation
      ↓
SVM Classification
      ↓
Risk Assessment Layer
      ↓
Streamlit Clinical Dashboard
```

---

## 📁 Project Structure

```
eeg_ftd_project/
│
├── data/
│   └── ds004504/
│
├── processed_epochs/
├── features/
├── models/
├── results/
│   └── figures/
│
├── scripts/
│   ├── verifyDataset.py
│   ├── preprocess_epochs.py
│   ├── extract_features.py
│   ├── prepare_dataset.py
│   ├── train_and_evaluate.py
│   └── performance_metrics.py
│
├── app/
│   ├── app.py
│   └── report_generator.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Full Pipeline

Execute in order:

```bash
python scripts/verifyDataset.py
python scripts/preprocess_epochs.py
python scripts/extract_features.py
python scripts/prepare_dataset.py
python scripts/train_and_evaluate.py
```

---

## 🌐 Run the Web Application

```bash
streamlit run app/app.py
```

### Web App Features
- Analyze existing dataset subjects
- Upload new EEG `.set` files
- View probability distribution
- Risk classification output
- Generate downloadable PDF report

---

## 📊 Evaluation Metrics

The model reports:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Stratified Cross-Validation results

---

## 🧠 Extracted EEG Features

For each subject:
- Mean Delta Power
- Mean Theta Power
- Mean Alpha Power
- Mean Beta Power
- Spectral Entropy
- Standard deviation of above features

These biomarkers reflect EEG slowing and reduced complexity in dementia.

---

## ⚠️ Disclaimer

This system is intended for research and educational purposes only.  
It is not designed for standalone clinical diagnosis.

---

## 📈 Future Improvements

- Two-stage hierarchical classification
- Functional connectivity features
- Riemannian geometry-based features
- Deep learning (EEGNet)
- Real-time EEG hardware integration
- External validation datasets

---

## 📜 License

For academic and research use.
# cant add dataset and the preprocessed epoches
