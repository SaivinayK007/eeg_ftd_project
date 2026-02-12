import mne
raw=mne.io.read_raw_eeglab("./scripts/sub93eeg.set" , preload=False)
print(raw)
print(raw.info)