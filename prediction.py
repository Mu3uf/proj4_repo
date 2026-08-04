import joblib

from database import engine

from models import sentiment_predictions

from sqlalchemy import insert



# تحميل النموذج

model = joblib.load(
    "model_lr.pkl"
)


vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)



def predict_sentiment(text):


    # تحويل النص

    text_vector = vectorizer.transform(
        [text]
    )


    prediction = model.predict(
        text_vector
    )[0]


    score = model.predict_proba(
        text_vector
    ).max()



    # حفظ في database

    query = insert(
        sentiment_predictions
    ).values(

        input_text=text,

        predicted_sentiment=prediction,

        prediction_score=float(score)

    )


    with engine.connect() as conn:

        conn.execute(query)

        conn.commit()



    return prediction, score




# Testing

texts = [

"I love this product, it works perfectly!",

"This was a terrible experience. I will never buy it again.",

"It's okay, not bad but not great either."

]


for text in texts:


    sentiment, score = predict_sentiment(text)


    print("-"*50)

    print(
        "Text:",
        text
    )

    print(
        "Predicted Sentiment:",
        sentiment
    )

    print(
        "Score:",
        round(score,3)
    )