import numpy as np
from pathlib import Path
import mne

data_dir = Path("data/ds004504/derivatives")
out_dir = Path("processed_epochs")
out_dir.mkdir(exist_ok=True)

subjects = sorted(data_dir.glob("sub-*"))

for sub_path in subjects:
    sub_id = sub_path.name
    set_files = list(sub_path.rglob("*.set"))
    if len(set_files) == 0:
        continue

    set_file = set_files[0]
    raw = mne.io.read_raw_eeglab(set_file, preload=True)

    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=2.0,
        overlap=1.0,
        preload=True,
        verbose=False
    )

    data = epochs.get_data()
    np.save(out_dir / f"{sub_id}_epochs.npy", data)

print("STEP 2 COMPLETED: Epochs saved for subjects.")
