# proj4_repo
### 📁 Project Structure

```text
proj4_repo/
├── alembic/
├── app/
│   ├── crud.py
│   ├── db.py
│   └── models.py
├── data/
│   └── preprocessed.csv
├── img/
│   └── confusion_matrix.png
├── alembic.ini
├── bert_features.py
├── bert_lr_model.pkl
├── compare_models.py
├── explore_data.py
├── explore.ipynb
├── main.py
├── model_lr.pkl
├── preprocessing.py
├── README.md
├── requirements.txt
├── tfidf_vectorizer.pkl
├── train_bert_model.py
├── train_model.py
└── X_embeddings.npy
```
Setup & Run Instructions
1. Navigate to the project folder
bash
cd proj4_repo
2. Create and activate the virtual environment (venv)
bash
python3 -m venv venv
source venv/bin/activate      # On Linux/Mac
3. Install the required libraries
bash
pip install -r requirements.txt

Or manually:

bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib torch transformers sqlalchemy psycopg2-binary python-dotenv
4. Set up the database (PostgreSQL)
Open pgAdmin and create a new database, for example:
   Database name: sentiment_db
Create a .env file in the project root with your connection details:
env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=sentiment_db
   DB_USER=postgres
   DB_PASSWORD=your_password
Make sure the SentimentPrediction table is created (via SQLAlchemy Core on the first run of main.py, or manually via pgAdmin) with the following columns:
Column	Type
id	Integer (Primary Key)
input_text	Text
predicted_sentiment	String
prediction_score	Float
created_at	Timestamp
5. Run the data preparation and training steps (in order, one time only)
bash
python explore_data.py       # Data exploration
python preprocessing.py      # Text cleaning
python train_model.py        # Train TF-IDF + Naive Bayes/Logistic Regression
python bert_features.py      # Generate BERT Embeddings (may take a while)
python train_bert_model.py   # Train and evaluate the model on BERT embeddings

⚠️ These steps only need to be run once to generate the required files (model_lr.pkl, tfidf_vectorizer.pkl, X_embeddings.npy). Once these files exist, there is no need to rerun them unless the data or model changes.

6. Run the main application (main.py)

Once the trained models are available and the database is ready:

bash
python main.py

When run, main.py performs the following:

Loads the saved model (model_lr.pkl) and the vectorizer (tfidf_vectorizer.pkl)
Accepts new input text from the user
Predicts the sentiment (Positive / Negative / Neutral)
Stores the result in the SentimentPrediction table in the PostgreSQL database
Displays the prediction result on screen
7. Verify the stored results via pgAdmin
Open pgAdmin
Navigate to: Servers → PostgreSQL → Databases → sentiment_db → Schemas → public → Tables → SentimentPrediction
Click View/Edit Data → All Rows to view all stored predictions
📊 Example Output
Text: I love this product, it works perfectly!
Predicted Sentiment: Positive
--------------------------------------------------
Text: This was a terrible experience. I'll never buy it again.
Predicted Sentiment: Negative
--------------------------------------------------
Text: It's okay, not bad but not great either.
Predicted Sentiment: Neutral
--------------------------------------------------
🛠️ Troubleshooting
Issue	Solution
FileNotFoundError for data files	Make sure you're running commands from the project root (proj4_repo)
Database connection fails	Make sure PostgreSQL is running and .env values are correct
bert_features.py runs slowly