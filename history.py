from sqlalchemy import select

from database import engine

from models import sentiment_predictions



query = select(
    sentiment_predictions
)



with engine.connect() as conn:


    results = conn.execute(query)


    for row in results:

        print("-------------------")

        print(
            "ID:",
            row.id
        )

        print(
            "Text:",
            row.input_text
        )

        print(
            "Sentiment:",
            row.predicted_sentiment
        )

        print(
            "Score:",
            row.prediction_score
        )

        print(
            "Date:",
            row.created_at
        )