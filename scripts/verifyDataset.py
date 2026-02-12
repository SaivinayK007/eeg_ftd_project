from pathlib import Path
import mne

data_dir = Path("data/ds004504/derivatives")
subjects = sorted(data_dir.glob("sub-*"))

print(len(subjects))

sample_set = list(subjects[0].rglob("*.set"))[0]
raw = mne.io.read_raw_eeglab(sample_set, preload=True)

print(raw.info["nchan"])
print(raw.info["sfreq"])
print(raw.times[-1])
print(raw.ch_names)
