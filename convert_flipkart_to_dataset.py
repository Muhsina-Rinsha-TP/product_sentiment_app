import pandas as pd

# Load original Flipkart data (make sure the CSV is in the same folder)
df = pd.read_csv("flipkart_com-ecommerce_sample.csv")

# Drop rows with missing reviews or ratings
df = df.dropna(subset=["review_text", "review_rating"])

# Convert numeric rating to sentiment
def map_rating(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["review_rating"].apply(map_rating)
df["Review"] = df["review_text"]

# Keep only what you need
df_final = df[["Review", "Sentiment"]]

# Save to your project folder
df_final.to_csv("review_dataset.csv", index=False)
print("✅ review_dataset.csv created successfully!")
