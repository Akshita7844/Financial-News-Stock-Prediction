import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import requests

# --- Config ---
st.set_page_config(page_title="Financial News Dashboard", layout="wide")
st.title("📈 Financial News Sentiment & Stock Prediction Dashboard")

# --- Sidebar Stock Selection ---
st.sidebar.header("🔧 Settings")
popular_stocks = {
    "Tesla (TSLA)": ("TSLA", "Tesla"),
    "Apple (AAPL)": ("AAPL", "Apple"),
    "Amazon (AMZN)": ("AMZN", "Amazon"),
    "Google (GOOGL)": ("GOOGL", "Google"),
    "Microsoft (MSFT)": ("MSFT", "Microsoft"),
    "NVIDIA (NVDA)": ("NVDA", "NVIDIA"),
}

selected_label = st.sidebar.selectbox("Select Stock:", list(popular_stocks.keys()))
selected_stock, stock_name = popular_stocks[selected_label]
threshold = st.sidebar.slider("Prediction Threshold", 0.5, 0.99, 0.9, 0.01)

# --- Load FinBERT ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()
    return tokenizer, model

tokenizer, model = load_finbert()

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    sentiment_idx = torch.argmax(logits, dim=1).item()
    sentiment_label = ['negative', 'neutral', 'positive'][sentiment_idx]
    return sentiment_label, probs

# --- Fetch News using NewsAPI ---
st.subheader(f"📰 Latest News for {stock_name}")
API_KEY = '9c9b055b9586410b82de1d8775c449e7'
news_url = 'https://newsapi.org/v2/everything'

params = {
    'q': stock_name,
    'language': 'en',
    'sortBy': 'publishedAt',
    'pageSize': 100,  # Fetch 100 articles
    'apiKey': API_KEY
}

with st.spinner("Fetching latest news..."):
    response = requests.get(news_url, params=params)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        if not articles:
            st.warning("No news articles found.")
        else:
            news_df = pd.DataFrame(articles)
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
            news_df['date'] = news_df['publishedAt'].dt.date
            news_df = news_df[['title', 'date']].dropna()

            st.success(f"Fetched {len(news_df)} news articles.")

            # --- Sentiment Analysis ---
            st.info("Analyzing sentiment...")
            news_df['Sentiment'], prob_list = zip(*news_df['title'].apply(get_sentiment))
            news_df['Neg_Prob'] = [p[0] for p in prob_list]
            news_df['Neu_Prob'] = [p[1] for p in prob_list]
            news_df['Pos_Prob'] = [p[2] for p in prob_list]

            def predict_movement(pos, neg):
                if pos >= threshold:
                    return 'up'
                elif neg >= threshold:
                    return 'down'
                else:
                    return 'neutral'

            news_df['Prediction'] = news_df.apply(lambda row: predict_movement(row['Pos_Prob'], row['Neg_Prob']), axis=1)

            # --- Display Data ---
            st.subheader("📰 Headlines with Sentiment & Prediction")
            st.dataframe(news_df[['date', 'title', 'Sentiment', 'Prediction']])

            # --- Sentiment Pie Chart + Word Cloud (Side by Side) ---
            st.subheader("📊 Sentiment Overview & ☁️ Word Cloud")
            col1, col2 = st.columns([1, 1.5])

            with col1:
                sentiment_counts = news_df['Sentiment'].value_counts()
                fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Split')
                st.plotly_chart(fig_sentiment, use_container_width=True)

            with col2:
                text = " ".join(news_df['title'].dropna().tolist())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig_wc, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)

            # --- Compare Prediction with Reality ---
            try:
                stock_data = yf.download(selected_stock, period="2mo")
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

                stock_data = stock_data.reset_index()
                stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

                close_col = [col for col in stock_data.columns if 'Close' in col][0]
                stock_data['Prev_Close'] = stock_data[close_col].shift(1)
                stock_data['Real_Change'] = stock_data.apply(
                    lambda row: 'up' if row[close_col] > row['Prev_Close'] else 'down', axis=1
                )
                stock_data_clean = stock_data.dropna(subset=['Prev_Close'])[['Date', 'Real_Change']]

                # Merge predictions with real stock movements
                comparison_df = pd.merge(news_df, stock_data_clean, left_on='date', right_on='Date', how='inner')
                comparison_df['Correct'] = comparison_df['Prediction'] == comparison_df['Real_Change']
                filtered_df = comparison_df[comparison_df['Prediction'] != 'neutral']
                accuracy = filtered_df['Correct'].mean() * 100

                st.subheader("🔍 Prediction vs Reality")
                st.dataframe(filtered_df[['date', 'title', 'Prediction', 'Real_Change', 'Correct']])
                st.success(f"📊 Prediction Accuracy (excluding neutral): {accuracy:.2f}%")

                # --- Prediction Summary + Stock Price Chart (Side by Side) ---
                st.subheader("📊 Prediction Summary & 📉 Stock Closing Prices")
                col1, col2 = st.columns([1, 1.5])

                with col1:
                    result_counts = filtered_df['Correct'].value_counts().rename({True: 'Correct', False: 'Incorrect'})
                    fig_result = px.bar(x=result_counts.index, y=result_counts.values, color=result_counts.index,
                                        labels={'x': 'Result', 'y': 'Count'}, title='Prediction Results')
                    st.plotly_chart(fig_result, use_container_width=True)

                with col2:
                    # Stock Price Chart (Latest 1 Month)
                    stock_recent = yf.download(selected_stock, period="1mo")
                    if isinstance(stock_recent.columns, pd.MultiIndex):
                        stock_recent.columns = ['_'.join(col).strip() for col in stock_recent.columns.values]

                    stock_recent = stock_recent.reset_index()
                    stock_recent['Date'] = pd.to_datetime(stock_recent['Date']).dt.date
                    close_col_recent = [col for col in stock_recent.columns if 'Close' in col][0]

                    fig_stock = px.line(stock_recent, x='Date', y=close_col_recent,
                                        title=f'{selected_stock.upper()} Stock Closing Prices')
                    st.plotly_chart(fig_stock, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction comparison: {str(e)}")

    else:
        st.error(f"Failed to fetch news. Status code: {response.status_code}")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using FinBERT, yFinance, NewsAPI, and Streamlit.")