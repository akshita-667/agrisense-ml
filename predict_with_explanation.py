"""
AgriSense - Explainable Prediction Module
==========================================================
Loads the trained model and generates a farmer-readable explanation
for each recommendation using SHAP. This is the module your Flask
/api/recommend endpoint will import and call.
"""

import joblib
import numpy as np
import pandas as pd
import shap

MODEL_DIR = "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Ideal-range knowledge base used for the plain-language explanation layer.
# (Rough agronomic ranges - refine with domain data / KVK input for production.)
IDEAL_RANGES = {
    "N": (40, 100, "Nitrogen"),
    "P": (20, 80, "Phosphorus"),
    "K": (20, 80, "Potassium"),
    "ph": (5.5, 7.5, "soil pH"),
    "temperature": (18, 32, "temperature"),
    "humidity": (40, 90, "humidity"),
    "rainfall": (50, 300, "rainfall"),
}

FEATURE_LABELS = {
    "N": "Nitrogen level",
    "P": "Phosphorus level",
    "K": "Potassium level",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}


class CropRecommender:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(f"{model_dir}/crop_model.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    def _range_flag(self, feature, value):
        """Return a plain-language flag if a value is outside the ideal range."""
        if feature not in IDEAL_RANGES:
            return None
        low, high, name = IDEAL_RANGES[feature]
        if value < low:
            return f"Your {name} ({value:g}) is a bit low for optimal growth in general \u2014 consider improving it."
        if value > high:
            return f"Your {name} ({value:g}) is on the higher side \u2014 keep an eye on it."
        return None

    def predict(self, N, P, K, temperature, humidity, ph, rainfall, top_k=3):
        """
        Returns top_k crop recommendations, each with:
          - crop name
          - confidence (%)
          - SHAP-based explanation bullets (top contributing factors)
          - any out-of-ideal-range warnings
        """
        x = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
        input_dict = dict(zip(FEATURES, x.iloc[0]))

        proba = self.model.predict_proba(x)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]

        # SHAP values: list of arrays, one per class, each (n_samples, n_features)
        shap_values = self.explainer.shap_values(x)

        results = []
        for idx in top_idx:
            crop_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = round(float(proba[idx]) * 100, 1)

            # shap_values can be (n_samples, n_features, n_classes) in newer SHAP versions
            if isinstance(shap_values, list):
                class_shap = np.asarray(shap_values[idx])[0]
            else:
                class_shap = np.asarray(shap_values)[0, :, idx]

            feat_contrib = sorted(
                zip(FEATURES, class_shap), key=lambda t: -abs(t[1])
            )[:3]

            explanation = []
            for feat, contrib in feat_contrib:
                direction = "supports" if contrib > 0 else "works against"
                explanation.append(
                    f"{FEATURE_LABELS[feat]} ({input_dict[feat]:.1f}) {direction} this recommendation."
                )

            warnings = [
                w for w in (self._range_flag(f, input_dict[f]) for f in FEATURES) if w
            ]

            results.append({
                "crop": crop_name,
                "confidence_pct": confidence,
                "explanation": explanation,
                "warnings": warnings[:2],  # keep it short for the UI
            })

        return results


if __name__ == "__main__":
    recommender = CropRecommender()

    # Example: values roughly matching rice-favourable conditions
    sample_input = dict(N=90, P=42, K=43, temperature=20.9, humidity=82.0, ph=6.5, rainfall=202.9)
    print("Sample input:", sample_input)
    print()

    recommendations = recommender.predict(**sample_input, top_k=3)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['crop'].upper()}  (confidence: {rec['confidence_pct']}%)")
        for line in rec["explanation"]:
            print(f"   - {line}")
        for w in rec["warnings"]:
            print(f"   \u26a0\ufe0f {w}")
        print()
