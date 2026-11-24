"""
AgriSense - Crop Recommendation Model Training Pipeline
==========================================================
Trains and compares multiple models on the crop recommendation dataset,
picks the best one, evaluates it, and saves artifacts (model, encoder,
scaler, feature importance) for use in the Flask backend.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

DATA_PATH = "data/Crop_recommendation.csv"
MODEL_DIR = "models"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Crops ({df[TARGET].nunique()}): {sorted(df[TARGET].unique())}")
print(f"Missing values: {df.isnull().sum().sum()}")

X = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------------------------------
# 2. Encode labels + split
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Encoding labels and splitting data")
print("=" * 60)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 3. Train + compare multiple models
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Training and comparing models")
print("=" * 60)

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
}

results = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, X, y_encoded, cv=5)
    results[name] = {"test_accuracy": acc, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    print(f"{name:20s}  test_acc={acc:.4f}   cv_mean={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ----------------------------------------------------------------------
# 4. Pick the best model (RandomForest chosen by default for SHAP
#    compatibility + strong tabular performance + interpretability)
# ----------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["test_accuracy"])
print(f"\nBest model by test accuracy: {best_name} ({results[best_name]['test_accuracy']:.4f})")

# We deliberately use RandomForest as the production model even if XGBoost
# ties/wins marginally, since it plays very well with SHAP's TreeExplainer
# and is simpler to deploy without extra native dependencies.
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)

print("\n" + "=" * 60)
print("STEP 4: Final model evaluation (RandomForest)")
print("=" * 60)
print(f"Test Accuracy: {final_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, final_preds)
np.savetxt(f"{MODEL_DIR}/confusion_matrix.csv", cm, delimiter=",", fmt="%d")

# ----------------------------------------------------------------------
# 5. Feature importance (a lightweight companion to SHAP, useful for a
#    quick global view in your report)
# ----------------------------------------------------------------------
importances = dict(zip(FEATURES, final_model.feature_importances_.round(4)))
importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
print("\nGlobal feature importance:")
for feat, score in importances.items():
    print(f"  {feat:12s}: {score}")

# ----------------------------------------------------------------------
# 6. Save all artifacts needed by the Flask backend
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Saving artifacts")
print("=" * 60)

joblib.dump(final_model, f"{MODEL_DIR}/crop_model.pkl")
joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump({
        "features": FEATURES,
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "test_accuracy": round(final_acc, 4),
        "model_comparison": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()},
        "feature_importance": importances,
    }, f, indent=2)

print("Saved: models/crop_model.pkl")
print("Saved: models/label_encoder.pkl")
print("Saved: models/metadata.json")
print("Saved: models/confusion_matrix.csv")
print("\nDone. Model is ready to be served from the Flask backend.")
