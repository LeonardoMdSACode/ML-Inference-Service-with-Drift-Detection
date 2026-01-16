---
title: ML Inference Service with Drift Detection
emoji: 📊
colorFrom: yellow
colorTo: red
sdk: docker
app_file: Dockerfile
pinned: false
license: mit
---

# ML Inference Service with Drift Detection

## Overview

This project is an end-to-end machine learning inference service with post-deployment drift detection, monitoring and logging. It exposes a REST API for model predictions, continuously monitors data for drift using a rolling window, and provides a dashboard to visualize recent predictions and drift metrics. It used [kaggle dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?select=UCI_Credit_Card.csv) for initial data, which is then prepared and processed for the model and to use as reference.

The system is designed to run locally on venv python 3.9 or on Hugging Face Spaces, with minimal dependencies and a purposely simple front-end dashboard.

Hugging Face Space: [LeonardoMdSA / ML Inference Service with Drift Detection](https://huggingface.co/spaces/LeonardoMdSA/ML-Inference-Service-with-Drift-Detection)

---

## Repository Structure

```
ML Inference Service with Drift Detection/
├─ app/
│  ├─ api/                   # FastAPI routes, background tasks, and drift logic
│  ├─ inference/             # Model wrapper and predictor logic
│  ├─ monitoring/            # Drift checks, governance, and data loaders
│  ├─ templates/             # HTML templates for dashboard
│  ├─ utils/                 # Utility scripts like validators and alert senders
│  ├─ core/                  # Configurations, constants
│  └─ main.py                # FastAPI entry point with lifespan tasks
├─ data/
│  ├─ processed/             # Input CSVs for predictions
│  ├─ production/            # Predictions log CSV
├─ models/                   # Model artifacts and reference datasets
├─ reports/                  # Drift and dashboard JSON/HTML outputs
│  └─ evidently/             # Drift report JSON
├─ tests/                    # Unit and integration tests
├─ Dockerfile                # Container configuration
├─ requirements.txt          # Python dependencies
└─ README.md
```

## Installation (Local / venv)

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd ML Inference Service with Drift Detection
   ```
2. Create a virtual environment and activate it:

   ```bash
   py 3.9 -m venv .venv

   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate   # Windows
   ```
3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip

   pip install -r requirements-dev.txt
   ```

## Running the API Locally

1. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```
2. Open the dashboard:

   * Localhost: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
3. Predictions can be submitted via the API `/predict` endpoint (multipart CSV upload).

## Testing

1. Run all tests with pytest:

   ```bash
   pytest -v
   ```
2. Integration tests cover API endpoints, predictions, schema validation, and governance alerts.

## How It Works (Logic Layers)

1. **API Layer**: FastAPI routes handle `/predict`, `/dashboard/data` and `/health`. Predictions are appended to `data/production/predictions_log.csv`.
2. **Inference Layer**: `Predictor` wraps the model, loads features from `FEATURES_PATH`, and performs batch predictions.
3. **Background Drift Loop**: Continuously monitors recent predictions (rolling window up to 5,000 rows), runs feature-level drift checks, and writes results to `reports/evidently/drift_report.json`.
4. **Governance**: Checks metrics like PSI, F1, and regression accuracy against thresholds and logs alerts. Sends notifications via email or Slack (if configured).
5. **Dashboard**: Reads `drift_report.json` and displays recent predictions and drift metrics via Plotly charts.

## Technology Stack

* Python 3.9
* FastAPI
* Uvicorn
* Pydantic 1
* Jinja2
* Pandas / NumPy
* Joblib (for model serialization)
* Evidently (drift detection)
* Plotly (frontend charts)
* Docker (for containerized deployment)
* Hugging Face Spaces (deployment)
* GitHub Actions (CI/CD)

## Recommendations / Important Notes

* **CSV Rolling Window**: `MAX_DRIFT_ROWS` limits the predictions log to 5,000 rows. Older rows are removed to prevent oversized files.
* **Email Alerts**: SMTP server must be configured; otherwise, alert sending will fail.
* **HF Spaces**: The dashboard runs at `/` endpoint by default for compatibility.

## References / Docs

* [FastAPI Documentation](https://fastapi.tiangolo.com/)
* [Evidently AI](https://evidentlyai.com/)
* [Plotly Charts](https://plotly.com/javascript/)

## Contact / Author

* Hugging Face: [https://huggingface.co/LeonardoMdSA](https://huggingface.co/LeonardoMdSA)
* GitHub: [https://github.com/LeonardoMdSACode](https://github.com/LeonardoMdSACode)

## MIT License

This project is licensed under the MIT License. See the LICENSE file for details.
