# Pulsar Star Classification: FastAPI + Docker

A machine learning service that classifies radio signals as pulsar / not pulsar.
A RandomForest model is trained on the HTRU2 dataset (Kaggle), served through a FastAPI REST API and packaged in a Docker container.

## Dataset

HTRU2 (Pulsar Star Classification, Kaggle): 17,898 samples, 8 numeric features, binary target.

| Feature | Description |
|---|---|
| `mean_integrated_profile`, `std_integrated_profile`, `kurtosis_integrated_profile`, `skewness_integrated_profile` | Statistics of the integrated pulse profile |
| `mean_dm_snr_curve`, `std_dm_snr_curve`, `kurtosis_dm_snr_curve`, `skewness_dm_snr_curve` | Statistics of the DM-SNR curve |
| `target_class` | 1 = pulsar, 0 = not pulsar |

The classes are imbalanced: 16,259 non-pulsars vs 1,639 pulsars (about 9%), so `class_weight="balanced"` is used and F1-score is the main metric.

## Model

- Algorithm: `RandomForestClassifier` (200 trees, `class_weight="balanced"`, `random_state=42`)
- Split: 80% train / 20% test, stratified

| Metric | Value |
|---|---|
| Accuracy | 0.978 |
| F1-score (pulsar class) | 0.881 |
| Precision / Recall (pulsar class) | 0.87 / 0.89 |

## Project structure

```
ml-fastapi-docker/
├── data/HTRU_2.csv       # dataset
├── notebooks/            # exploration and experiments (Jupyter)
├── train.py              # trains the model and saves model.joblib
├── main.py               # FastAPI application
├── model.joblib          # trained model
├── requirements.txt      # Python dependencies
├── Dockerfile            # container instructions
└── README.md
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python train.py                # trains the model, creates model.joblib
uvicorn main:app --reload      # starts the API
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

## Run with Docker

```bash
docker build -t pulsar-api .
docker run -p 8000:8000 pulsar-api
```

Open http://localhost:8000/docs.

## API

### `GET /`

```json
{"message": "ML API is running"}
```

### `POST /predict`

Request body (all 8 features are required):

```json
{
  "mean_integrated_profile": 140.5625,
  "std_integrated_profile": 55.683782,
  "kurtosis_integrated_profile": -0.234571,
  "skewness_integrated_profile": -0.699648,
  "mean_dm_snr_curve": 3.199833,
  "std_dm_snr_curve": 19.110426,
  "kurtosis_dm_snr_curve": 7.975532,
  "skewness_dm_snr_curve": 74.242225
}
```

Response:

```json
{"prediction": 0, "probability": 0.0}
```

- `prediction`: predicted class (1 = pulsar, 0 = not pulsar)
- `probability`: model's probability of the pulsar class

Example with curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"mean_integrated_profile":140.5625,"std_integrated_profile":55.683782,"kurtosis_integrated_profile":-0.234571,"skewness_integrated_profile":-0.699648,"mean_dm_snr_curve":3.199833,"std_dm_snr_curve":19.110426,"kurtosis_dm_snr_curve":7.975532,"skewness_dm_snr_curve":74.242225}'
```
