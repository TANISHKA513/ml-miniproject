import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
data = pd.read_csv("cleaned_sms_dataset.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning
data["clean_message"] = data["message"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_message"])

print("TF-IDF Matrix Shape:", X.shape)
print("Number of Features:", len(vectorizer.get_feature_names_out()))