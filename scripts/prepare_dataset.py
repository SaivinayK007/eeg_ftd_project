import numpy as np
import pandas as pd
from pathlib import Path

features_dir = Path("features")
data_dir = Path("data/ds004504")

X = np.load(features_dir / "X_features.npy")
subjects = np.load(features_dir / "subjects.npy")

participants = pd.read_csv(data_dir / "participants.tsv", sep="\t")

label_map = {"A": 0, "F": 1, "C": 2}
participants["label"] = participants["Group"].map(label_map)
participants = participants.set_index("participant_id")

X_final = []
y_final = []
meta = []

for i, sub in enumerate(subjects):
    if sub in participants.index:
        X_final.append(X[i])
        y_final.append(participants.loc[sub, "label"])
        meta.append({
            "subject": sub,
            "age": participants.loc[sub, "Age"],
            "mmse": participants.loc[sub, "MMSE"],
            "group": participants.loc[sub, "Group"]
        })

X_final = np.array(X_final)
y_final = np.array(y_final)

np.save(features_dir / "X.npy", X_final)
np.save(features_dir / "y.npy", y_final)

df = pd.DataFrame(X_final, columns=[
    "delta_mean", "theta_mean", "alpha_mean", "beta_mean", "entropy_mean",
    "delta_std", "theta_std", "alpha_std", "beta_std", "entropy_std"
])

meta_df = pd.DataFrame(meta)
final_df = pd.concat([meta_df, df], axis=1)
final_df.to_csv(features_dir / "dataset.csv", index=False)

print("STEP 4 COMPLETED")
print("Final X shape:", X_final.shape)
print("Final y shape:", y_final.shape)
