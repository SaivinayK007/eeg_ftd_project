import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

features_dir = Path("features")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

X = np.load(features_dir / "X.npy")
y = np.load(features_dir / "y.npy")

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, class_weight="balanced"))
])

param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.01, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    svm,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1
)

grid.fit(X, y)

best_model = grid.best_estimator_
joblib.dump(best_model, models_dir / "svm_model.pkl")

print("STEP 5 COMPLETED")
print("Best Params:", grid.best_params_)
print("Best CV F1-macro:", grid.best_score_)
