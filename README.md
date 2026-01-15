---
title: Context-aware NLP classification platform with MCP
emoji: 🧠
colorFrom: yellow
colorTo: red
sdk: docker
app_file: Dockerfile
pinned: false
license: mit
---

# Under Construction

Building a production-ready ML inference service with post-deployment drift detection, governance, and alerting, demonstrating real MLOps practices rather than offline modeling.

py -3.9 -m venv .venv

.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

uvicorn app.main:app --reload

# Repo Structure

ml-inference-drift-service/
Dockerfile
LICENSE
README.md
requirements-dev.txt
requirements.txt
app/
    main.py
    api/
        background_drift.py
        dashboard_data.py
        routes.py
        schemas.py
        traffic_daemon.py
    core/
        config.py
        logging.py
        templates.py
    inference/
        predictor.py
    monitoring/
        data_loader.py
        drift.py
        governance.py
    static/
        styles.css
    templates/
        dashboard.html
    utils/
        alerts.py
data/
    processed/
        credit_default_clean.csv
        current_data.csv
    production/
        predictions_log.csv
    raw/
        credit_default.csv
database/
logs/
models/
    v1/
        features.json
        reference_data.csv
    v2/
reports/
    evidently/
        drift_report.html
        drift_report.json
scripts/
    prepare_data.py
    simulate_inference.py
    train.py
tests/
    conftest.py
    integration/
        test_api.py
        test_governance.json
        test_governance.py
        test_run_drift.py
    unit/
        test_schemas.py
