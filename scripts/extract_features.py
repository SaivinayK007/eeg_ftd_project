import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy.stats import entropy

epochs_dir = Path("processed_epochs")
features_dir = Path("features")
features_dir.mkdir(exist_ok=True)

X = []
subjects = []

def extract_epoch_features(epoch, sfreq=500):
    freqs, psd = welch(epoch, fs=sfreq, nperseg=256)

    def band_power(fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.mean(psd[idx])

    delta = band_power(1, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)

    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)

    return np.array([delta, theta, alpha, beta, spec_entropy])

for file in sorted(epochs_dir.glob("sub-*_epochs.npy")):
    epochs = np.load(file)
    subject_id = file.stem.replace("_epochs", "")
    epoch_features = []

    for ep in epochs:
        ch_features = []
        for ch in ep:
            ch_features.append(extract_epoch_features(ch))
        epoch_features.append(np.mean(ch_features, axis=0))

    epoch_features = np.array(epoch_features)

    subject_vector = np.concatenate([
        epoch_features.mean(axis=0),
        epoch_features.std(axis=0)
    ])

    X.append(subject_vector)
    subjects.append(subject_id)

X = np.array(X)

np.save(features_dir / "X_features.npy", X)
np.save(features_dir / "subjects.npy", np.array(subjects))

print("STEP 3 COMPLETED")
print("Feature matrix shape:", X.shape)
