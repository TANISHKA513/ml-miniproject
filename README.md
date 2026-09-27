# 🛡️ SmartSpam: Intelligent SMS Spam Classification Using Machine Learning

## 📌 Project Overview

SmartSpam is a machine learning-based SMS spam classification system that automatically classifies text messages as either **Spam** or **Ham (Legitimate)**.

The project combines Natural Language Processing (NLP) techniques with supervised Machine Learning algorithms to identify patterns commonly associated with spam messages.

A simple Streamlit web application allows users to enter an SMS message and receive a real-time classification.

---

## 🎯 Objectives

- Automatically classify SMS messages as Spam or Ham.
- Preprocess raw SMS text for machine learning.
- Convert text into numerical features using TF-IDF.
- Compare multiple machine learning classification algorithms.
- Evaluate model performance using standard classification metrics.
- Develop a simple interactive application for real-time prediction.

---

## 📊 Dataset

The project uses the **UCI SMS Spam Collection** dataset.

After removing duplicate messages:

- Total messages: **5,169**
- Ham messages: **4,516**
- Spam messages: **653**

The dataset contains SMS messages labelled as either `ham` or `spam`.

---

## ⚙️ Methodology

The project follows the following pipeline:

**SMS Dataset**  
↓  
**Duplicate Removal**  
↓  
**Text Cleaning**  
↓  
**TF-IDF Feature Extraction**  
↓  
**Train-Test Split**  
↓  
**Machine Learning Models**  
↓  
**Performance Evaluation**  
↓  
**SVM Model Selection**  
↓  
**Streamlit Application**

---

## 🧹 Text Preprocessing

The SMS messages are cleaned using:

- Conversion of text to lowercase
- Removal of URLs
- Removal of unnecessary special characters
- Removal of extra whitespace

Numbers are retained because numerical patterns can sometimes provide useful information for spam classification.

---

## 🧠 Feature Extraction

**Term Frequency-Inverse Document Frequency (TF-IDF)** is used to convert SMS text into numerical feature vectors.

The processed dataset produced:

- **5,169 SMS messages**
- **9,477 TF-IDF features**

---

## 🤖 Machine Learning Models

Four supervised machine learning algorithms were trained and compared:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Random Forest

The dataset was divided into:

- **80% Training Data:** 4,135 messages
- **20% Testing Data:** 1,034 messages

Stratified splitting was used to preserve the class distribution.

---

## 📈 Model Performance

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 95.07% | 1.00 | 0.61 | 0.76 |
| Logistic Regression | 94.87% | 0.95 | 0.63 | 0.76 |
| Random Forest | 97.39% | 0.99 | 0.80 | 0.89 |
| SVM | **97.68%** | 0.97 | **0.85** | **0.90** |

Based on the test results, the SVM model was selected for the final application.

---

## 💻 Application

The project includes a Streamlit-based web application where users can enter an SMS message and receive a prediction.

The application uses:

- Trained SVM classifier
- Saved TF-IDF vectorizer
- Text preprocessing
- Real-time prediction

### Example

**Input:**

> Congratulations! You have won a free prize of 50000. Call now to claim your reward!

**Prediction:**

> 🚨 SPAM MESSAGE

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Regular Expressions
- TF-IDF
- Machine Learning

---

## 📁 Project Structure

```text
ml-miniproject/
│
├── SMSSpamCollection
├── app.py
├── load_dataset.py
├── preprocess.py
├── train_models.py
├── results_graphs.py
│
├── cleaned_sms_dataset.csv
├── smartspam_svm_model.pkl
├── smartspam_tfidf_vectorizer.pkl
│
├── model_accuracy_comparison.png
├── precision_recall_f1_comparison.png
│
├── requirements.txt
└── README.md