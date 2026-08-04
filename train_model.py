import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. Load Preprocessed Data
# ==========================================

df = pd.read_csv("data/preprocessed.csv")

df = df.dropna(subset=["clean_text"])

print("عدد الصفوف:", df.shape[0])


# ==========================================
# 2. TF-IDF Feature Extraction
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(
    df["clean_text"]
)

y = df["sentiment"]



# ==========================================
# 3. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("حجم بيانات التدريب:", X_train.shape[0])
print("حجم بيانات الاختبار:", X_test.shape[0])



# ==========================================
# 4. Naive Bayes
# ==========================================

nb_model = MultinomialNB()

nb_model.fit(
    X_train,
    y_train
)

y_pred_nb = nb_model.predict(
    X_test
)



# ==========================================
# 5. Logistic Regression + Tuning
# ==========================================

param_grid = {

    "C": [0.01, 0.1, 1, 10],

    "solver": ["lbfgs"],

    "max_iter": [1000]

}



grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)



grid.fit(
    X_train,
    y_train
)



lr_model = grid.best_estimator_



print("\n==============================")

print(
    "Best Parameters:",
    grid.best_params_
)

print(
    "Best Cross Validation F1:",
    round(grid.best_score_,4)
)

print("==============================")



y_pred_lr = lr_model.predict(
    X_test
)



# ==========================================
# Evaluation
# ==========================================

def evaluate_model(name, y_true, y_pred):

    print("\n" + "="*50)
    print(name)
    print("="*50)


    print(
        "Accuracy :",
        round(
            accuracy_score(y_true,y_pred),
            4
        )
    )


    print(
        "Precision:",
        round(
            precision_score(
                y_true,
                y_pred,
                average="weighted"
            ),
            4
        )
    )


    print(
        "Recall   :",
        round(
            recall_score(
                y_true,
                y_pred,
                average="weighted"
            ),
            4
        )
    )


    print(
        "F1-score :",
        round(
            f1_score(
                y_true,
                y_pred,
                average="weighted"
            ),
            4
        )
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )


    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=lr_model.classes_
        )
    )



evaluate_model(
    "Naive Bayes",
    y_test,
    y_pred_nb
)



evaluate_model(
    "Logistic Regression Best Model",
    y_test,
    y_pred_lr
)



# ==========================================
# Save Model
# ==========================================

joblib.dump(
    lr_model,
    "model_lr.pkl"
)


joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


print("\n✅ Model and Vectorizer saved successfully!")



# ==========================================
# Prediction Function
# ==========================================

def predict_sentiment(text):

    model = joblib.load(
        "model_lr.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )


    # ملاحظة:
    # النص هنا يجب أن يكون قريب من شكل clean_text
    # لأن التدريب تم على النصوص المنظفة


    text_vector = vectorizer.transform(
        [text]
    )


    prediction = model.predict(
        text_vector
    )


    return prediction[0]



# ==========================================
# Test Predictions
# ==========================================

print("\n========== Prediction Examples ==========")


print(
    "love this airline:",
    predict_sentiment(
        "love airlin"
    )
)


print(
    "terrible flight:",
    predict_sentiment(
        "terribl flight"
    )
)


print(
    "plane arrive time:",
    predict_sentiment(
        "plane arriv time"
    )
)




# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred_lr,
    labels=lr_model.classes_
)


plt.figure(
    figsize=(6,5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=lr_model.classes_,
    yticklabels=lr_model.classes_
)


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Confusion Matrix - Logistic Regression"
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix.png"
)


plt.close()


print(
    "✅ Confusion Matrix image saved successfully!"
)