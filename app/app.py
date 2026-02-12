import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import uuid
from datetime import datetime
import mne
from scipy.signal import welch
from scipy.stats import entropy
from report_generator import generate_report

st.set_page_config(page_title="EEG Dementia Risk Assessment", layout="wide")

LABELS = ["Alzheimer’s Disease", "Frontotemporal Dementia", "Healthy Control"]
FEATURE_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Entropy"]

@st.cache_resource
def load_model():
    return joblib.load("models/svm_model.pkl")

@st.cache_data
def load_reference():
    X = np.load("features/X.npy")
    df = pd.read_csv("features/dataset.csv")
    return X, df

model = load_model()
X_ref, df = load_reference()

POP_MEAN = X_ref.mean(axis=0)[:5]
POP_STD = X_ref.std(axis=0)[:5]

def clinical_risk_assessment(confidence):
    if confidence >= 80:
        return "High Risk", "error", "Neurological evaluation is recommended."
    elif confidence >= 60:
        return "Moderate Risk", "warning", "Clinical follow-up advised."
    else:
        return "Low Risk", "success", "No immediate concern."

def extract_epoch_features(epoch, sfreq=500):
    freqs, psd = welch(epoch, fs=sfreq, nperseg=256)

    def band_power(fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return float(np.mean(psd[idx]))

    delta = band_power(1, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)

    psd_norm = psd / np.sum(psd)
    spec_entropy = float(entropy(psd_norm))

    return np.array([delta, theta, alpha, beta, spec_entropy], dtype=float)

def extract_subject_features_from_raw(raw):
    raw = raw.copy()
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=2.0,
        overlap=1.0,
        preload=True,
        verbose=False
    )

    data = epochs.get_data()  # (n_epochs, n_channels, n_samples)

    epoch_features = []
    for ep in data:
        ch_features = []
        for ch in ep:
            ch_features.append(extract_epoch_features(ch, sfreq=int(raw.info["sfreq"])))
        epoch_features.append(np.mean(ch_features, axis=0))

    epoch_features = np.array(epoch_features)

    subject_vector = np.concatenate([
        epoch_features.mean(axis=0),
        epoch_features.std(axis=0)
    ])

    return subject_vector

st.sidebar.markdown("## EEG Session")
mode = st.sidebar.radio("Input Mode", ["Existing Patient", "New Patient (Upload EEG .set)"])

session_id = f"EEG-{uuid.uuid4().hex[:8]}"
st.sidebar.caption(f"Session ID: {session_id}")

subject_name = "N/A"
age = "N/A"
mmse = "N/A"
x_input = None
mean_features_for_plot = None

if mode == "Existing Patient":
    selected_subject = st.sidebar.selectbox("Select Subject", df["subject"].tolist())
    idx = df.index[df["subject"] == selected_subject][0]
    row = df.loc[idx]

    subject_name = row["subject"]
    age = row["age"]
    mmse = row["mmse"]

    x_input = X_ref[idx]
    mean_features_for_plot = X_ref[idx][:5]

else:
    uploaded_set = st.sidebar.file_uploader("Upload EEGLAB .set file", type=["set"])
    if uploaded_set is None:
        st.warning("Upload a .set EEG file to analyze a new patient.")
        st.stop()

    with st.spinner("Loading EEG file..."):
        tmp_path = f"temp_{session_id}.set"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_set.read())

        raw = mne.io.read_raw_eeglab(tmp_path, preload=True)

        subject_name = "Live EEG"
        age = "N/A"
        mmse = "N/A"

        x_input = extract_subject_features_from_raw(raw)
        mean_features_for_plot = x_input[:5]

st.title("EEG Dementia Risk Assessment System")
st.caption("Near-real-time EEG decision-support prototype")
st.caption(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

analyze = st.button("Analyze EEG", use_container_width=True)

if analyze:
    with st.spinner("Running model prediction..."):
        time.sleep(1)

        probs = model.predict_proba([x_input])[0]
        pred_idx = int(np.argmax(probs))
        prediction = LABELS[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        risk, alert_type, recommendation = clinical_risk_assessment(confidence)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session", session_id)
    c2.metric("Prediction", prediction)
    c3.metric("Confidence", f"{confidence:.2f}%")
    c4.metric("Risk", risk)

    if alert_type == "error":
        st.error(recommendation)
    elif alert_type == "warning":
        st.warning(recommendation)
    else:
        st.success(recommendation)

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("EEG Features vs Population")
        fig, ax = plt.subplots()
        ax.bar(FEATURE_NAMES, mean_features_for_plot, label="Patient")
        ax.plot(FEATURE_NAMES, POP_MEAN, marker="o", label="Population Mean")
        ax.fill_between(
            FEATURE_NAMES,
            POP_MEAN - POP_STD,
            POP_MEAN + POP_STD,
            alpha=0.2,
            label="Population ±1 SD"
        )
        ax.set_ylabel("Mean Feature Value")
        ax.legend()
        st.pyplot(fig)

    with right:
        st.subheader("Model Probability Distribution")
        fig2, ax2 = plt.subplots()
        ax2.bar(LABELS, probs)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Probability")
        st.pyplot(fig2)

    st.subheader("Clinician Notes")
    notes = st.text_area("Observations / Remarks", placeholder="Write notes here...")

    if st.button("Generate Clinical Report (PDF)"):
        generate_report(subject_name, age, mmse, prediction, risk, confidence, mean_features_for_plot)
        st.success("EEG_Clinical_Report.pdf generated successfully")
