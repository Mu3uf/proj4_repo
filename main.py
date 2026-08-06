import joblib
from app.crud import save_prediction, get_all_predictions

model = joblib.load("model_lr.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def run_sentiment_pipeline(text: str):
    text_vectorized = vectorizer.transform([text])
    predicted_label = model.predict(text_vectorized)[0]
    
    probabilities = model.predict_proba(text_vectorized)
    confidence_score = float(max(probabilities[0]))
#نص + التوقع + الثقه
    save_prediction(
        input_text=text,
        predicted_sentiment=predicted_label,
        prediction_score=confidence_score
    )

if __name__ == "__main__":
    new_test_samples = [
        "The customer service was absolutely amazing and very helpful!",
        "The movie was boring and a complete waste of time.",
        "The quality is average, nothing special about it.",
        "I am so happy with this purchase, highly recommended!",
        "Worst food I have ever tasted, very disappointed."
    ]

    print("--- Processing New Sentiment Predictions ---")
    for sample in new_test_samples:
        run_sentiment_pipeline(sample)
    print("✓ All predictions generated and stored successfully!\n")

    print("=" * 85)
    print(f"{'ID':<4} | {'Sentiment':<10} | {'Score':<6} | {'Input Text'}")
    print("=" * 85)

    history = get_all_predictions()
    for row in history:
        text_display = row['input_text'][:48] + '...' if len(row['input_text']) > 48 else row['input_text']
        score_val = row['prediction_score'] if row['prediction_score'] is not None else 0.0
        
        print(f"{row['id']:<4} | {row['predicted_sentiment']:<10} | {score_val:<6.2f} | {text_display}")

    print("=" * 85)