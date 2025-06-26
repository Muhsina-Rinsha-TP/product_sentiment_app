import streamlit as st
import joblib
import pandas as pd
import re

# Load the model and tools
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Clean user input
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit app UI
st.set_page_config(page_title="🛍️ Product Sentiment Classifier", page_icon="🧠")
st.title("🛍️ Customer Sentiment Analysis")
st.write("Type a product review below and I’ll predict the sentiment!")

# User input
user_input = st.text_area("✍️ Enter your product review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned_input = clean_text(user_input)
        X_input = vectorizer.transform([cleaned_input])

        # Get model decision score (confidence)
        try:
            scores = model.decision_function(X_input)
            top_score = max(abs(scores[0]))  # distance from decision boundary
        except AttributeError:
            top_score = None  # In case the model doesn't support this (e.g. Random Forest)

        # Predict
        pred = model.predict(X_input)
        sentiment = label_encoder.inverse_transform(pred)[0]

        # Apply threshold if confidence is low
        if top_score is not None and top_score < 0.2:
            sentiment = "Neutral"

        # Display results
        st.subheader("🧠 Predicted Sentiment:")
        if sentiment.lower() == "positive":
            st.success("✅ Positive")
        elif sentiment.lower() == "negative":
            st.error("❌ Negative")
        else:
            st.info("😐 Neutral")

        # Debug output
        st.write(f"🔍 Raw prediction (encoded): {pred}")
        st.write(f"🔍 Decoded label (before threshold): {label_encoder.inverse_transform(pred)[0]}")
        if top_score is not None:
            st.caption(f"🧪 Confidence score: `{top_score:.3f}` (Neutral if < 0.2)")
        st.caption(f"🧪 Cleaned input = `{cleaned_input}`, Final Prediction = `{sentiment}`")
