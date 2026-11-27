# Pulsar Star Classification: MLflow + FastAPI + Streamlit + Docker

An end-to-end machine learning service that classifies radio signals as pulsar / not pulsar.
A RandomForest model is trained on the HTRU2 dataset (Kaggle), tracked and registered with MLflow, served through a FastAPI REST API with a Streamlit web interface, and containerized via Docker Compose.

## Dataset

HTRU2 (Pulsar Star Classification, Kaggle): 17,898 samples, 8 numeric features, binary target.

| Feature | Description |
|---|---|
| `mean_integrated_profile`, `std_integrated_profile`, `kurtosis_integrated_profile`, `skewness_integrated_profile` | Statistics of the integrated pulse profile |
| `mean_dm_snr_curve`, `std_dm_snr_curve`, `kurtosis_dm_snr_curve`, `skewness_dm_snr_curve` | Statistics of the DM-SNR curve |
| `target_class` | 1 = pulsar, 0 = not pulsar |

The classes are imbalanced: 16,259 non-pulsars vs 1,639 pulsars (about 9%), so `class_weight="balanced"` is used and F1-score is the main metric.

## Model & MLflow Tracking

- Algorithm: `RandomForestClassifier` (200 trees, `class_weight="balanced"`, `random_state=42`)
- Split: 80% train / 20% test, stratified
- Experiment Tracking & Registry: MLflow logs hyperparameters, evaluation metrics, and model artifacts, registering the model as `pulsar-rf-model` (Version 1).

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
├── train.py              # trains model, logs to MLflow & registers pulsar-rf-model
├── main.py               # FastAPI inference service
├── app.py                # Streamlit web frontend
├── model.joblib          # serialized model artifact
├── requirements.txt      # Python dependencies
├── Dockerfile            # container image definition
├── docker-compose.yaml    # multi-container orchestration (API + Frontend)
└── README.md
```

## Run locally

1. **Set up environment and train the model:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python train.py                # logs metrics/artifacts and registers model in MLflow
```

2. **Inspect MLflow UI (optional):**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000 to view experiments and the Model Registry.

3. **Start FastAPI backend and Streamlit frontend (in separate terminals):**
```bash
uvicorn main:app --reload      # API on http://127.0.0.1:8000
streamlit run app.py           # UI on http://localhost:8501
```

## Run with Docker Compose

Build and start both the FastAPI backend and Streamlit frontend containers:

```bash
docker compose up --build
```

- **Streamlit UI:** http://localhost:8501
- **FastAPI Swagger UI:** http://localhost:8000/docs

To stop the containers:

```bash
docker compose down
```

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