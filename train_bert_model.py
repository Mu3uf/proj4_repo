import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==========================================
# 1. تحميل البيانات (Labels) و BERT Embeddings
# ==========================================
df = pd.read_csv('data/preprocessed.csv')
df = df.dropna(subset=['clean_text'])
y = df['sentiment']

X_emb = np.load('X_embeddings.npy')

print("شكل مصفوفة BERT Embeddings:", X_emb.shape)
print("عدد الفئات (labels):", y.shape[0])

# ==========================================
# 2. تقسيم البيانات (نفس طريقة اليوم الأول بالضبط للمقارنة العادلة)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 3. تدريب Logistic Regression على BERT embeddings
# ==========================================
lr_bert_model = LogisticRegression(max_iter=1000)
lr_bert_model.fit(X_train, y_train)
y_pred_bert = lr_bert_model.predict(X_test)

# ==========================================
# 4. التقييم
# ==========================================
def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='weighted'), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='weighted'), 4))
    print("F1-score :", round(f1_score(y_true, y_pred, average='weighted'), 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=lr_bert_model.classes_))

evaluate_model("Logistic Regression + BERT Embeddings", y_test, y_pred_bert)

print("\n📊 مقارنة سريعة مع نتائج TF-IDF (اليوم الأول):")
print("Logistic Regression + TF-IDF   → F1-score: 0.7646")
print(f"Logistic Regression + BERT     → F1-score: {round(f1_score(y_test, y_pred_bert, average='weighted'), 4)}")
import joblib

joblib.dump(
    lr_bert_model,
    "bert_lr_model.pkl"
)

print("✅ BERT model saved successfully!")