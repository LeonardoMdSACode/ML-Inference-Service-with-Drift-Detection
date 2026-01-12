# env vars, paths, thresholds
import os

MODEL_VERSION = "v1"
MODEL_PATH = "models/v1/model.pkl"
FEATURES_PATH = "models/v1/features.json"
DB_PATH = "database/app.db"

# Governance logs path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_PATH = os.environ.get("LOGS_PATH", os.path.join(PROJECT_ROOT, "logs"))
os.makedirs(LOGS_PATH, exist_ok=True)

