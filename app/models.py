from sqlalchemy import MetaData, Table, Column, Integer, String, Float, DateTime
from datetime import datetime

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("input_text", String, nullable=False),
    Column("predicted_sentiment", String(50), nullable=False),
    Column("prediction_score", Float, nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow)
)