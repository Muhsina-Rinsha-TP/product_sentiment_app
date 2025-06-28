# 🛍️ Product Sentiment Analysis App

This is a **Streamlit-based web application** that performs **sentiment analysis** on customer product reviews. It uses **Natural Language Processing (NLP)** and **machine learning (ML)** techniques to classify reviews into one of three categories: **Positive**, **Negative**, or **Neutral**.



## 🚀 Features

- 🔤 Clean, easy-to-use text input
- ⚙️ Trained using `RandomForestClassifier`
- 🧠 TF-IDF Vectorizer for efficient text processing
- 📊 Label Encoding for sentiment transformation
- 📁 Uploadable Flipkart dataset converter included

---

## 📂 Project Structure

product_sentiment_app/
│
├── app.py # Streamlit app UI
├── sentiment_model.py # Training script
├── convert_flipkart_to_dataset.py # Optional: Convert Flipkart data to usable format
├── review_dataset_balanced.csv # Cleaned and balanced dataset
├── requirement.txt # Python dependencies
├── model/
│ ├── sentiment_model.pkl # Trained Random Forest model
│ ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
│ └── label_encoder.pkl # Label encoder

yaml


---

## 🧪 How it Works

1. **User** enters a product review in the app.
2. The review is **cleaned** and transformed using **TF-IDF**.
3. The model predicts the **sentiment**.
4. The result is displayed as ✅ Positive, ❌ Negative, or 😐 Neutral.

---

## ⚙️ Installation & Running

```bash
pip install -r requirement.txt
streamlit run app.py
📈 Model Training
The model is trained using a cleaned and balanced dataset (review_dataset_balanced.csv) using:

RandomForestClassifier

TfidfVectorizer with bigrams

LabelEncoder for sentiment mapping

📌 Use Case
Ideal for:

E-commerce platforms

Businesses seeking customer feedback analysis

NLP beginners working on real-world applications

🧑‍💻 Author
Muhsina Rinsha
Feel free to fork, improve, and explore!
