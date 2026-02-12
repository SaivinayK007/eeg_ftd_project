from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report
)

features_path = Path("features")
fig_path = Path("results/figures")
fig_path.mkdir(parents=True, exist_ok=True)

X = np.load(features_path / "X.npy")
y = np.load(features_path / "y.npy")

class_names = ["AD", "FTD", "HC"]
n_classes = 3

y_bin = label_binarize(y, classes=[0, 1, 2])

X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
    X, y, y_bin, test_size=0.2, stratify=y, random_state=42
)

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

models = {
    "SVM": svm,
    "RandomForest": rf
}

for name, model in models.items():
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    plt.figure(figsize=(7, 5))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(yb_test[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path / f"{name.lower()}_roc_curves.png")
    plt.close()

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    with open(fig_path / f"{name.lower()}_classification_report.txt", "w") as f:
        f.write(report)

    print(f"\n{name} Classification Report:\n")
    print(report)

print("STEP 6 COMPLETED")
