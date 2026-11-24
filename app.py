"""
AgriSense - Flask Backend
==========================================================
REST API that serves crop recommendations from the trained model.

Endpoints
---------
GET  /api/health      -> is the server up and the model loaded?
GET  /api/crops       -> list of crops the model can recommend
GET  /api/model-info  -> accuracy, model comparison, feature importance
POST /api/recommend   -> top crop recommendations for given soil/weather

Run locally:
    python app.py                # http://127.0.0.1:5000
"""

import json
import math
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from predict_with_explanation import FEATURES, CropRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")

# Physically sensible limits for each input. Values outside these are
# rejected as invalid (typos, wrong units). This is different from the
# training-data range check, which only produces a warning.
INPUT_LIMITS = {
    "N": (0, 300, "kg/ha"),
    "P": (0, 300, "kg/ha"),
    "K": (0, 400, "kg/ha"),
    "temperature": (-10, 60, "°C"),
    "humidity": (0, 100, "%"),
    "ph": (0, 14, ""),
    "rainfall": (0, 5000, "mm"),
}

DEFAULT_TOP_K = 3
MAX_TOP_K = 10


def _validate_payload(payload):
    """Check the request body. Returns (values, top_k, errors)."""
    errors = {}
    values = {}

    if not isinstance(payload, dict):
        return None, None, {"body": "Request body must be a JSON object."}

    for feat in FEATURES:
        raw = payload.get(feat)
        if raw is None or raw == "":
            errors[feat] = "This field is required."
            continue
        # bool is a subclass of int in Python - reject it explicitly.
        if isinstance(raw, bool):
            errors[feat] = "Must be a number."
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[feat] = "Must be a number."
            continue
        if not math.isfinite(value):
            errors[feat] = "Must be a finite number."
            continue
        low, high, unit = INPUT_LIMITS[feat]
        if not low <= value <= high:
            suffix = f" {unit}" if unit else ""
            errors[feat] = f"Must be between {low} and {high}{suffix}."
            continue
        values[feat] = value

    top_k = payload.get("top_k", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        errors["top_k"] = f"Must be a whole number between 1 and {MAX_TOP_K}."

    return values, top_k, errors


def _out_of_training_range(values, feature_ranges):
    """Warn about inputs the model never saw during training."""
    notes = []
    for feat, value in values.items():
        rng = feature_ranges.get(feat)
        if not rng:
            continue
        if value < rng["min"] or value > rng["max"]:
            notes.append(
                f"{feat} = {value:g} is outside the training data range "
                f"({rng['min']:g} to {rng['max']:g}); treat this recommendation with caution."
            )
    return notes


def create_app(model_dir=None):
    model_dir = model_dir or os.environ.get("AGRISENSE_MODEL_DIR", DEFAULT_MODEL_DIR)

    app = Flask(__name__)
    app.json.sort_keys = False

    # Allow the frontend (served from a different port/domain) to call the API.
    # Set CORS_ORIGINS="https://your-frontend.com" in production.
    CORS(app, resources={r"/api/*": {"origins": os.environ.get("CORS_ORIGINS", "*")}})

    # Load the model once at startup, not on every request.
    recommender = CropRecommender(model_dir=model_dir)
    with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    feature_ranges = metadata.get("feature_ranges", {})

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model_loaded=True)

    @app.get("/api/crops")
    def crops():
        classes = list(recommender.label_encoder.classes_)
        return jsonify(count=len(classes), crops=classes)

    @app.get("/api/model-info")
    def model_info():
        return jsonify(
            model="RandomForestClassifier",
            features=metadata.get("features", FEATURES),
            n_classes=metadata.get("n_classes"),
            test_accuracy=metadata.get("test_accuracy"),
            model_comparison=metadata.get("model_comparison", {}),
            feature_importance=metadata.get("feature_importance", {}),
            feature_ranges=feature_ranges,
            input_limits={k: {"min": lo, "max": hi, "unit": u} for k, (lo, hi, u) in INPUT_LIMITS.items()},
        )

    @app.post("/api/recommend")
    def recommend():
        payload = request.get_json(silent=True)
        values, top_k, errors = _validate_payload(payload)
        if errors:
            return jsonify(error="Invalid input", details=errors), 400

        recommendations = recommender.predict(**values, top_k=top_k)
        return jsonify(
            input=values,
            recommendations=recommendations,
            notes=_out_of_training_range(values, feature_ranges),
        )

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Not found"), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify(error="Method not allowed"), 405

    @app.errorhandler(500)
    def server_error(_):
        return jsonify(error="Internal server error"), 500

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    create_app().run(host="127.0.0.1", port=port, debug=debug)
