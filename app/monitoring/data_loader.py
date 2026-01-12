#Load Production data from SQLite

import sqlite3
import json
import pandas as pd
from app.core.config import DB_PATH


def load_production_data(limit: int = 1000) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT input_features
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError("No production data available for drift detection.")

    records = [json.loads(row[0]) for row in rows]
    return pd.DataFrame(records)
