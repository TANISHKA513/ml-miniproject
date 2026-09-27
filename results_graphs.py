import matplotlib.pyplot as plt

# Model names
models = [
    "Naive Bayes",
    "Logistic Regression",
    "Random Forest",
    "SVM"
]

# Accuracy values
accuracy = [
    95.07,
    94.87,
    97.39,
    97.68
]

# Create graph
plt.figure(figsize=(9, 5))

bars = plt.bar(models, accuracy)

# Add accuracy values above bars
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value}%",
        ha="center",
        fontsize=10
    )

# Graph title and labels
plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")

# Set y-axis range
plt.ylim(90, 100)

# Make layout clean
plt.tight_layout()

# Save graph
plt.savefig("model_accuracy_comparison.png", dpi=300)

# Display graph
plt.show()
# -----------------------------------------
# Precision, Recall and F1-Score Comparison
# -----------------------------------------

precision = [
    1.00,
    0.95,
    0.99,
    0.97
]

recall = [
    0.61,
    0.63,
    0.80,
    0.85
]

f1_score = [
    0.76,
    0.76,
    0.89,
    0.90
]

x = range(len(models))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width for i in x],
    precision,
    width=width,
    label="Precision"
)

plt.bar(
    x,
    recall,
    width=width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    f1_score,
    width=width,
    label="F1-Score"
)

plt.xticks(list(x), models)

plt.ylabel("Score")
plt.xlabel("Machine Learning Model")

plt.title("Precision, Recall and F1-Score Comparison")

plt.ylim(0, 1.1)

plt.legend()

plt.tight_layout()

plt.savefig(
    "precision_recall_f1_comparison.png",
    dpi=300
)

plt.show()
# -----------------------------------------
# SVM Confusion Matrix
# -----------------------------------------

from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# Actual SVM confusion matrix
cm = np.array([
    [899, 4],
    [20, 111]
])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()

plt.title("SVM Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300
)

plt.show()