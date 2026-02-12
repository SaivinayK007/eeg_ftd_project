import numpy as np
import pandas as pd
import mne
from pathlib import Path
import json
import random
from datetime import datetime

def generate_synthetic_eeg(patient_type="HC", sfreq=500, duration_sec=120):
    ch_names = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
                "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
    ch_types = ["eeg"] * len(ch_names)

    n_channels = len(ch_names)
    n_samples = int(sfreq * duration_sec)
    t = np.arange(n_samples) / sfreq

    if patient_type == "AD":
        delta_amp, theta_amp, alpha_amp, beta_amp = 12e-6, 8e-6, 2e-6, 1e-6
    elif patient_type == "FTD":
        delta_amp, theta_amp, alpha_amp, beta_amp = 10e-6, 7e-6, 3e-6, 2e-6
    else:
        delta_amp, theta_amp, alpha_amp, beta_amp = 3e-6, 3e-6, 10e-6, 6e-6

    data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        phase = random.random() * 2 * np.pi
        noise = np.random.normal(0, 1.5e-6, size=n_samples)

        delta = delta_amp * np.sin(2 * np.pi * 2.0 * t + phase)
        theta = theta_amp * np.sin(2 * np.pi * 6.0 * t + phase)
        alpha = alpha_amp * np.sin(2 * np.pi * 10.0 * t + phase)
        beta  = beta_amp  * np.sin(2 * np.pi * 20.0 * t + phase)

        data[ch] = delta + theta + alpha + beta + noise

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw

def create_events_tsv(path, sfreq, duration_sec):
    df = pd.DataFrame([
        {"onset": 0.0, "duration": duration_sec, "trial_type": "eyesclosed_rest"}
    ])
    df.to_csv(path, sep="\t", index=False)

def create_json(path, sfreq):
    meta = {
        "TaskName": "eyesclosed",
        "SamplingFrequency": sfreq,
        "EEGReference": "Cz",
        "PowerLineFrequency": 50,
        "SoftwareFilters": "Bandpass 1-40 Hz applied in pipeline",
        "GeneratedBy": "Synthetic EEG Generator",
        "GeneratedOn": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=4)

def main():
    base = Path("data/ds004504/derivatives")
    base.mkdir(parents=True, exist_ok=True)

    subjects = [f"sub-{i:03d}" for i in range(89, 99)]
    groups = ["AD", "AD", "AD", "FTD", "FTD", "FTD", "HC", "HC", "HC", "HC"]

    sfreq = 500
    duration_sec = 120

    for sub_id, grp in zip(subjects, groups):
        sub_dir = base / sub_id / "eeg"
        sub_dir.mkdir(parents=True, exist_ok=True)

        set_path = sub_dir / f"{sub_id}_task-eyesclosed_eeg.set"
        tsv_path = sub_dir / f"{sub_id}_task-eyesclosed_events.tsv"
        json_path = sub_dir / f"{sub_id}_task-eyesclosed_eeg.json"

        raw = generate_synthetic_eeg(grp, sfreq=sfreq, duration_sec=duration_sec)

        mne.export.export_raw(
            set_path,
            raw,
            fmt="eeglab",
            overwrite=True
        )

        create_events_tsv(tsv_path, sfreq, duration_sec)
        create_json(json_path, sfreq)

        print(f"✅ Created {sub_id} | {grp}")
        print("   ", set_path.name)
        print("   ", tsv_path.name)
        print("   ", json_path.name)

    print("\n✅ All 10 synthetic subject packages generated successfully.")

if __name__ == "__main__":
    main()
