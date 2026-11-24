# AgriSense - AI-Based Crop Recommendation for Farmers

5th semester Entrepreneurship project.
**Problem statement:** AI-based crop recommendation for farmers.

This repository currently holds the ML component: a model that recommends
the best crops for a farmer's soil and weather conditions, with
plain-language (SHAP-based) explanations. A full-stack web app
(Flask backend + frontend) will be built on top of it.

## Project structure
```
data/          Crop_recommendation.csv (training dataset)
models/        Trained model artifacts (the .pkl files are generated, not committed)
notebooks/     Exploration notebooks
train_model.py              Trains and compares models, saves artifacts
predict_with_explanation.py CropRecommender class used by the backend
```

# ML Model Training

## Setup
```bash
pip install -r requirements.txt
```

## Train the model
```bash
python train_model.py
```
This trains and compares 6 models, picks RandomForest (best accuracy + SHAP-compatible),
and saves artifacts to `models/`:
- `crop_model.pkl` - trained RandomForest model
- `label_encoder.pkl` - crop label encoder
- `metadata.json` - accuracy, feature importance, model comparison

## Test explainable predictions
```bash
python predict_with_explanation.py
```
This loads the saved model and shows SHAP-based, farmer-readable explanations
for a sample input.

## Using this in your Flask backend
```python
from predict_with_explanation import CropRecommender

recommender = CropRecommender(model_dir="models")
results = recommender.predict(N=90, P=42, K=43, temperature=20.9,
                                humidity=82.0, ph=6.5, rainfall=202.9, top_k=3)
```
Returns a list of dicts: `crop`, `confidence_pct`, `explanation`, `warnings` —
this is exactly the shape your `/api/recommend` endpoint should return as JSON.

## Dataset
`data/Crop_recommendation.csv` - 2200 rows, 22 crops, 7 features
(N, P, K, temperature, humidity, ph, rainfall).

## Results
- RandomForest test accuracy: 99.55%
- Cross-validated accuracy: 99.45% (+/- 0.23%)
