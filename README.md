# proj4_repo
'''
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
'''
1. Clone & Set Up Virtual Environment
# Clone the repository
git clone 
cd proj4_repo

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

2. Install Project Dependencies
pip install --upgrade pip
pip install -r requirements.txt

3. Database Setup (PostgreSQL)

Ensure your PostgreSQL server is running and create the target database:
CREATE DATABASE sentiment_db;
(Make sure to update your connection string inside app/db.py with your username and password if different).
4. Run Database Migrations (Alembic)

Apply all schema updates to create the predictions table in PostgreSQL:
alembic upgrade head
5. Data Preprocessing & Model Training (Optional/Initial Run)

If you want to re-run data cleaning, feature extraction, and model training:
# Preprocess data and perform EDA
python explore_data.py

# Train baseline models, perform hyperparameter tuning, and save .pkl models
python train_model.py
6. Run the Main Pipeline

Execute the main application to generate new sentiment predictions, persist them automatically into PostgreSQL, and output the formatted database history in the terminal:
python main.py