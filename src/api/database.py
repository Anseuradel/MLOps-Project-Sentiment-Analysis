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
    """
    Initialize the SQLite database and create the predictions table if it doesn't exist.
    
    This function is called automatically when the module is imported to ensure
    the database is ready for operations.
    
    Raises:
        Exception: If database creation fails, logs error but doesn't crash the app
    """
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
    """
    Insert a new prediction record into the database.
    
    This function logs all prediction requests for analytics, monitoring,
    and debugging purposes.
    
    Args:
        input_text (str): The original text submitted for prediction
        predicted_label (str): The predicted sentiment label
        confidence (float): Model's confidence score (0.0 to 1.0)
        model_version (str): Version identifier of the model
        model_type (str): Type of model used ("MockSentimentClassifier" or "SentimentClassifier")
        latency_ms (float): Total processing time in milliseconds
        
    Note:
        This function uses UTC timestamp for consistent timezone handling
    """
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
    """
    Fetch the most recent prediction logs from the database.
    
    This function is used by the Streamlit dashboard to display
    recent prediction history to users.
    
    Args:
        limit (int): Maximum number of records to return (default: 10)
        
    Returns:
        list: List of tuples containing prediction records, ordered by most recent first
              Each tuple contains: (timestamp, input_text, predicted_label, confidence, 
                                  model_version, model_type, latency_ms)
    """
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
