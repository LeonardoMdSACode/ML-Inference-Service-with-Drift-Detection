# SQLite + file logging

import sqlite3
import json
from datetime import datetime
from app.core.config import DB_PATH, MODEL_VERSION


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_version TEXT,
            input_features TEXT,
            prediction INTEGER,
            probability REAL
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(features: dict, prediction: int, probability: float):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (timestamp, model_version, input_features, prediction, probability)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            MODEL_VERSION,
            json.dumps(features),
            prediction,
            probability,
        )
    )

    conn.commit()
    conn.close()
