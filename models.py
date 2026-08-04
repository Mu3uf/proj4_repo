from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from datetime import datetime

from database import metadata



sentiment_predictions = Table(

    "sentiment_predictions",

    metadata,


    Column(
        "id",
        Integer,
        primary_key=True
    ),


    Column(
        "input_text",
        String,
        nullable=False
    ),


    Column(
        "predicted_sentiment",
        String,
        nullable=False
    ),


    Column(
        "prediction_score",
        Float
    ),


    Column(
        "created_at",
        DateTime,
        default=datetime.utcnow
    )
)