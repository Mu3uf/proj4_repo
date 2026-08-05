from sqlalchemy import insert, select
from sqlalchemy.exc import SQLAlchemyError
from app.db import engine
from app.models import predictions

def save_prediction(input_text: str, predicted_sentiment: str, prediction_score: float = None):
    """إدخال سجل توقع جديد للـ Database"""
    with engine.connect() as conn:
        try:
            stmt = insert(predictions).values(
                input_text=input_text,
                predicted_sentiment=predicted_sentiment,
                prediction_score=prediction_score
            )
            result = conn.execute(stmt)
            conn.commit()
            return result.inserted_primary_key[0]
        except SQLAlchemyError as e:
            conn.rollback()
            print(f"Error inserting prediction: {e}")
            raise

def get_all_predictions():
    """استرجاع جميع التوقعات السابقة"""
    with engine.connect() as conn:
        try:
            query = select(predictions).order_by(predictions.c.created_at.desc())
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]
        except SQLAlchemyError as e:
            print(f"Error fetching predictions: {e}")
            raise