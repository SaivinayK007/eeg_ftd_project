import numpy as np
import pandas as pd
import joblib
from pathlib import Path

X = np.load("features/X.npy")
df = pd.read_csv("features/dataset.csv")
model = joblib.load("models/svm_model.pkl")

labels = ["AD", "FTD", "HC"]

threshold = 0.61
rows = []

for i in range(len(df)):
    subject = df.loc[i, "subject"]
    probs = model.predict_proba([X[i]])[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    if labels[pred] == "AD" and conf >= threshold:
        rows.append({
            "subject": subject,
            "predicted": "AD",
            "confidence": round(conf * 100, 2),
            "true_group": df.loc[i, "group"],
            "age": df.loc[i, "age"],
            "mmse": df.loc[i, "mmse"]
        })

if len(rows) == 0:
    print(f"No subjects found with Predicted AD confidence >= {threshold*100:.0f}%")
    print("Try reducing threshold to 60% or 70%")
else:
    result = pd.DataFrame(rows).sort_values("confidence", ascending=False)

    print("High-Risk AD Subjects")
    print(result.to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    result.to_csv("results/high_risk_ad_subjects.csv", index=False)
    print("\nSaved: results/high_risk_ad_subjects.csv")