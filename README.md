# Sentiment Analyzer

A sentiment analysis system that classifies text (tweets/reviews) as **Positive**, **Negative**, or **Neutral**, using classical Machine Learning techniques (TF-IDF) as well as BERT Embeddings, with prediction results stored in a PostgreSQL database via SQLAlchemy and Alembic migrations.

---

## 📁 Project Structure

```
proj4_repo/
├── app/
│   ├── db.py                   # Database connection & SQLAlchemy engine setup
│   ├── models.py                # SQLAlchemy ORM models (SentimentPrediction table)
│   └── crud.py                  # Database operations (create, read predictions)
├── alembic/                     # Alembic migration scripts
├── alembic.ini                  # Alembic configuration file
├── data/
│   ├── Tweets.csv               # Raw dataset (from Kaggle)
│   ├── clean_raw.csv            # After initial exploration
│   └── preprocessed.csv         # After text cleaning (Preprocessing)
├── Img/
│   ├── confusion_matrix.png     # Confusion Matrix (TF-IDF model)
│   ├── accuracy_comparison.png  # Accuracy comparison across models
│   └── f1_comparison.png        # F1-score comparison across models
├── explore_data.py              # Initial data exploration
├── explore.ipynb                # Jupyter notebook for data exploration
├── preprocessing.py             # Text cleaning (Cleaning + Stemming)
├── train_model.py               # Train and evaluate TF-IDF + Naive Bayes / Logistic Regression
├── bert_features.py             # Generate BERT Embeddings
├── train_bert_model.py          # Train and evaluate Logistic Regression on BERT embeddings
├── compare_models.py            # Compare TF-IDF vs BERT model performance
├── model_lr.pkl                 # Saved model (TF-IDF + Logistic Regression)
├── bert_lr_model.pkl            # Saved model (BERT + Logistic Regression)
├── tfidf_vectorizer.pkl         # Saved TF-IDF Vectorizer
├── X_embeddings.npy             # Saved BERT Embeddings matrix
├── main.py                      # Main entry point (loads models + predicts + stores in database)
├── requirements.txt             # List of required libraries
└── README.md
```

> **Note:** `database.sqlite` and `sentiment.db` may appear in the project folder as leftover files from earlier testing — the project uses **PostgreSQL** as its actual database, managed through `app/db.py` and Alembic migrations. These SQLite files can be safely deleted if unused.

---

## ⚙️ Prerequisites

- Python 3.14 (or the version used in the project)
- PostgreSQL installed and running on your machine
- pgAdmin (for managing and viewing the database)
- A virtual environment (venv) to isolate project dependencies

---

## 🚀 Setup & Run Instructions

### 1. Navigate to the project folder

```bash
cd proj4_repo
```

### 2. Create and activate the virtual environment (venv)

```bash
python3 -m venv venv
source venv/bin/activate      # On Linux/Mac
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib torch transformers sqlalchemy alembic psycopg2-binary python-dotenv
```

### 4. Set up the PostgreSQL database

1. Open **pgAdmin** and create a new database, for example:
   ```
   Database name: sentiment_db
   ```
2. Create a `.env` file in the project root with your connection details (used by `app/db.py`):
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=sentiment_db
   DB_USER=postgres
   DB_PASSWORD=your_password
   ```
3. Apply the database schema using Alembic migrations:
   ```bash
   alembic upgrade head
   ```
   This creates the `SentimentPrediction` table (defined in `app/models.py`) with the following columns:

   | Column | Type |
   |---|---|
   | id | Integer (Primary Key) |
   | input_text | Text |
   | predicted_sentiment | String |
   | prediction_score | Float |
   | created_at | Timestamp |

### 5. Run the data preparation and training steps (in order, one time only)

```bash
python explore_data.py       # Data exploration
python preprocessing.py      # Text cleaning
python train_model.py        # Train TF-IDF + Naive Bayes/Logistic Regression
python bert_features.py      # Generate BERT Embeddings (may take a while)
python train_bert_model.py   # Train and evaluate the model on BERT embeddings
python compare_models.py     # Compare TF-IDF vs BERT performance (generates charts in Img/)
```

> ⚠️ These steps only need to be run **once** to generate the required files (`model_lr.pkl`, `bert_lr_model.pkl`, `tfidf_vectorizer.pkl`, `X_embeddings.npy`). Once these files exist, there is no need to rerun them unless the data or model changes.

### 6. Run the main application (main.py)

Once the trained models are available and the database is ready:

```bash
python main.py
```

When run, `main.py` performs the following:
1. Loads the saved model (`model_lr.pkl` or `bert_lr_model.pkl`) and the TF-IDF vectorizer
2. Accepts new input text from the user
3. Predicts the sentiment (Positive / Negative / Neutral)
4. Stores the result in the `SentimentPrediction` table via `app/crud.py`
5. Displays the prediction result on screen

### 7. Verify the stored results via pgAdmin

1. Open **pgAdmin**
2. Navigate to: `Servers → PostgreSQL → Databases → sentiment_db → Schemas → public → Tables → SentimentPrediction`
3. Click **View/Edit Data → All Rows** to view all stored predictions

---

## 📊 Model Comparison

Charts comparing TF-IDF vs BERT model performance are available in the `Img/` folder:

- `confusion_matrix.png` — Confusion Matrix for the TF-IDF model
- `accuracy_comparison.png` — Accuracy comparison across models
- `f1_comparison.png` — F1-score comparison across models

---

## 📊 Example Output

```
Text: I love this product, it works perfectly!
Predicted Sentiment: Positive
--------------------------------------------------
Text: This was a terrible experience. I'll never buy it again.
Predicted Sentiment: Negative
--------------------------------------------------
Text: It's okay, not bad but not great either.
Predicted Sentiment: Neutral
--------------------------------------------------
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `FileNotFoundError` for data files | Make sure you're running commands from the project root (`proj4_repo`) |
| Database connection fails | Make sure PostgreSQL is running and `.env` values are correct |
| Alembic migration errors | Make sure `alembic.ini` points to the correct database URL |
| `bert_features.py` runs slowly | Normal on CPU-only machines; may take 30-60 minutes |

---

## 👤 Author

Training project — Sentiment Analyzer (Project 4)
