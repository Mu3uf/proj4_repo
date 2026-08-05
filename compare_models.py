import matplotlib.pyplot as plt



models = [
    "TF-IDF + Logistic Regression",
    "BERT + Logistic Regression"
]


f1_scores = [
    0.7646,
    0.8094
]


accuracy_scores = [
    0.7763,
    0.8122
]


# ==========================
# F1-score Comparison
# ==========================

plt.figure(figsize=(7,5))

plt.bar(
    models,
    f1_scores,
    color=["orange", "blue"]
)

plt.ylabel("F1-score")

plt.title(
    "F1-score Comparison: TF-IDF vs BERT"
)


for i, value in enumerate(f1_scores):
    plt.text(
        i,
        value + 0.01,
        str(value),
        ha="center"
    )


plt.ylim(0,1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "f1_comparison.png"
)

plt.show()



# ==========================
# Accuracy Comparison
# ==========================

plt.figure(figsize=(7,5))

plt.bar(
    models,
    accuracy_scores,
    color=["green", "purple"]
)


plt.ylabel("Accuracy")

plt.title(
    "Accuracy Comparison: TF-IDF vs BERT"
)


for i, value in enumerate(accuracy_scores):
    plt.text(
        i,
        value + 0.01,
        str(value),
        ha="center"
    )


plt.ylim(0,1)

plt.xticks(rotation=20)

plt.tight_layout()


plt.savefig(
    "accuracy_comparison.png"
)


plt.show()


print("✅ Comparison charts saved successfully!")