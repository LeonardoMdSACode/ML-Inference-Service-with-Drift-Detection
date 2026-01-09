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



# Repo Structure

ml-inference-drift-service/
│
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── api/
│   │   ├── routes.py             # /predict, /health, /dashboard
│   │   └── schemas.py            # Pydantic input/output schemas
│   │
│   ├── core/
│   │   ├── config.py             # env vars, paths, thresholds
│   │   ├── logging.py            # SQLite + file logging
│   │   └── model_registry.py     # model loading/versioning
│   │
│   ├── inference/
│   │   ├── predictor.py          # model.predict wrapper
│   │   └── preprocessing.py      # feature handling
│   │
│   ├── monitoring/
│   │   ├── drift.py              # Evidently logic
│   │   ├── metrics.py            # feature stats extraction
│   │   └── alerts.py             # threshold evaluation
│   │
│   ├── db/
│   │   ├── session.py            # SQLite connection
│   │   └── models.py             # ORM-style tables (optional)
│   │
│   ├── templates/
│   │   └── dashboard.html        # Evidently embed + metrics
│   │
│   └── static/
│       └── styles.css
│
├── models/
│   ├── v1/
│   │   ├── model.pkl
│   │   └── reference_data.csv
│   └── v2/
│       └── ...
│
├── scripts/
│   ├── train.py                  # offline training
│   ├── evaluate.py               # offline evaluation
│   └── run_drift_check.py        # batch drift job
│
├── reports/
│   └── evidently/
│       └── drift_report.html
│
├── tests/
│   ├── test_api.py
│   ├── test_drift.py
│   └── test_schemas.py
│
├── Dockerfile
├── Dockerfile.hf                 # HF Spaces–compatible
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .env.example
