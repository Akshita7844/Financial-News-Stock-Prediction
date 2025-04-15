import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# --- CONFIG ---
st.set_page_config(page_title="📈 Financial News Sentiment Predictor", layout="wide")
st.title("📈 Predict Stock Movement Using Financial News")
st.markdown("Combining **FinBERT** sentiment with stock data to predict movements.")

# --- LOAD FinBERT ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- FUNCTIONS ---
def fetch_news(api_key, ticker, num_articles=10):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize={num_articles}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    return [article["title"] for article in data.get("articles", [])]

def classify_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).numpy()
    labels = ["positive", "neutral", "negative"]
    results = []
    for i, text in enumerate(texts):
        label = labels[probs[i].argmax()]
        results.append((text, label, *probs[i]))
    return results

def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df["Next_Close"] = df["Close"].shift(-1)
    df["Movement"] = df.apply(lambda row: "up" if row["Next_Close"] > row["Close"]
                               else "down" if row["Next_Close"] < row["Close"] else "neutral", axis=1)
    df.reset_index(inplace=True)
    return df

# --- SIDEBAR INPUT ---
with st.sidebar:
    st.header("📊 Input Settings")
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    api_key = st.text_input("Enter NewsAPI Key", type="password")
    threshold = st.slider("Confidence Threshold", min_value=0.5, max_value=0.99, value=0.9)
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=10))
    end_date = st.date_input("End Date", value=datetime.now() - timedelta(days=1))
    num_articles = st.slider("Articles to Analyze", 5, 30, 10)

# --- RUN ANALYSIS ---
if st.button("Run Prediction"):
    if not api_key:
        st.warning("Please enter your NewsAPI key.")
    else:
        st.subheader("📰 News Sentiment Analysis")
        titles = fetch_news(api_key, ticker, num_articles)
        sentiment_data = classify_sentiment(titles)

        sentiment_df = pd.DataFrame(sentiment_data, columns=["Title", "Sentiment", "Positive", "Neutral", "Negative"])
        sentiment_df["Prediction"] = sentiment_df.apply(
            lambda row: "up" if row["Positive"] >= threshold else
                        "down" if row["Negative"] >= threshold else "neutral", axis=1
        )
        st.dataframe(sentiment_df)

        st.subheader("📈 Stock Data")
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        st.dataframe(stock_df[["Date", "Close", "Next_Close", "Movement"]])

        # Match predictions to stock data by date approximation
        sentiment_df["Date"] = pd.to_datetime(end_date)
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])
        merged = pd.merge(sentiment_df, stock_df, on="Date", how="inner")

        if not merged.empty:
            merged["Match"] = merged["Prediction"] == merged["Movement"]
            acc = (merged["Match"].sum() / len(merged)) * 100
            st.success(f"✅ Accuracy at {threshold} threshold: **{acc:.2f}%**")

            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            sentiment_df["Sentiment"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

            st.subheader("📋 Daily Summary")
            st.dataframe(merged[["Date", "Prediction", "Movement", "Match"]])
        else:
            st.warning("⚠️ No exact date match found between news and stock data.")
