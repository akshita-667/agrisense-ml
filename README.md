# AgriSense - AI-Based Crop Recommendation for Farmers

5th semester Entrepreneurship project.
**Problem statement:** AI-based crop recommendation for farmers.

A full-stack web app: farmers enter their soil test values and local
weather, and an ML model recommends the best crops with plain-language
(SHAP-based) explanations.

- **ML model**: RandomForest trained on 22 crops (99.55% test accuracy)
- **Backend**: Flask REST API (`app.py`)
- **Frontend**: responsive web page (`frontend/`) served by Flask, built
  with plain HTML/CSS/JavaScript so there is no build step

## Quick start
```bash
pip install -r requirements.txt
python train_model.py        # once, to create models/*.pkl
python app.py                # open http://127.0.0.1:5000
```

## Project structure
```
data/          Crop_recommendation.csv (training dataset)
models/        Trained model artifacts (the .pkl files are generated, not committed)
notebooks/     Exploration notebooks
train_model.py              Trains and compares models, saves artifacts
predict_with_explanation.py CropRecommender class used by the backend
app.py                      Flask REST API + serves the frontend
frontend/                   Web page (index.html, styles.css, app.js)
tests/                      API tests (pytest)
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

## Run the backend (Flask API)
```bash
python train_model.py        # once, to create models/*.pkl
python app.py                # web page + API at http://127.0.0.1:5000
python -m pytest tests       # run the API tests
```

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/` | The farmer-facing web page |
| GET | `/api/health` | Server status |
| GET | `/api/crops` | The 22 crops the model knows |
| GET | `/api/model-info` | Accuracy, model comparison, feature importance, input limits |
| POST | `/api/recommend` | Top crop recommendations for soil + weather values |

### `POST /api/recommend`
Request body (JSON). All 7 fields are required, and `top_k` (1-10) is optional with a default of 3:
```json
{"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82.0,
 "ph": 6.5, "rainfall": 202.9, "top_k": 3}
```
Response:
```json
{
  "input": {"N": 90.0, "...": "..."},
  "recommendations": [
    {"crop": "rice", "confidence_pct": 93.5,
     "explanation": ["Rainfall (202.9) supports this recommendation.", "..."],
     "warnings": []}
  ],
  "notes": []
}
```
- Invalid input returns **400** with a message for each field, for example
  `{"error": "Invalid input", "details": {"ph": "Must be between 0 and 14."}}`.
- `notes` lists inputs that are valid but outside the training data
  (for example N above 140), where the prediction is less reliable.

## Frontend
`frontend/` is a single page with no framework and no build step. It works on phones, follows the device's light/dark setting and has:
- the input form, with units, typical ranges, example presets and an error message on each field
- recommendation cards with a match %, "why" reasons and general soil tips
- a "How it works" section with model accuracy and what matters most to the model (from `/api/model-info`)

To host the frontend separately from the API, set
`window.AGRISENSE_API_BASE = "https://your-api"` before `app.js` loads, and set `CORS_ORIGINS` on the backend.

Environment variables: `PORT` (default 5000), `CORS_ORIGINS` (default `*`, set it
to your frontend URL in production), `FLASK_DEBUG=1`, `AGRISENSE_MODEL_DIR`.

## Dataset
`data/Crop_recommendation.csv` - 2200 rows, 22 crops, 7 features
(N, P, K, temperature, humidity, ph, rainfall).

## Results
- RandomForest test accuracy: 99.55%
- Cross-validated accuracy: 99.45% (+/- 0.23%)
