import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load the balanced dataset
df = pd.read_csv("review_dataset_balanced.csv")

# Drop rows with missing values
df.dropna(subset=["Review", "Sentiment"], inplace=True)

# Encode the sentiment labels
label_encoder = LabelEncoder()
df["SentimentEncoded"] = label_encoder.fit_transform(df["Sentiment"])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["Review"])
y = df["SentimentEncoded"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Support Vector Machine (SVM)
model = LinearSVC()
model.fit(X_train, y_train)

# Save the model and transformers
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
