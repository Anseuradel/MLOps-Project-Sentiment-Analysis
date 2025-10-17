import sqlite3
from datetime import datetime
import os
from config import DB_PATH
from config import DB_DIR
import logging

logger = logging.getLogger(__name__)

# Ensure the data folder exists
# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
# Ensure the db folder exists (important when using Docker volume)
os.makedirs(DB_DIR, exist_ok=True)

def init_db():
    """Initialize database and create tables if not exists."""
    try:
        logger.info(f"Initializing database at {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_text TEXT,
                predicted_label TEXT,
                confidence REAL,
                model_version TEXT,
                model_type TEXT,
                latency_ms REAL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully ✅")
    except Exception as e:
        logger.error(f"Failed to initialize DB: {e}")

def insert_prediction(input_text, predicted_label, confidence, model_version, model_type, latency_ms):
    """Insert a new prediction record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (timestamp, input_text, predicted_label, confidence, model_version, model_type, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        input_text,
        predicted_label,
        confidence,
        model_version,
        model_type,
        latency_ms
    ))
    conn.commit()
    conn.close()

def fetch_recent_predictions(limit=10):
    """Fetch most recent prediction logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, input_text, predicted_label, confidence, model_version, model_type, latency_ms
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Initialize database at import
init_db()
