# 📈 Stock Movement Prediction Through Financial News Sentiment Analysis using FinBERT

Predict stock market movement based on **real-time financial news headlines** using **NLP (FinBERT)** and **threshold-based sentiment analysis**. Achieve up to **73% prediction accuracy** with this intelligent and efficient sentiment-driven model. 🚀

---

## 💡 Project Overview

This project leverages:
- 📰 **Live news data** (via NewsAPI)
- 🤖 **FinBERT**, a transformer model fine-tuned on financial sentiment
- 📊 **Threshold-based up/down prediction** based on FinBERT confidence scores
- ✅ Achieved **high precision predictions** at thresholds of 0.7+.

---

## 🔧 Features

- 🔍 Fetches real-time news for any stock/topic
- 🧠 Runs FinBERT sentiment analysis (positive/neutral/negative)
- 📊 Predicts stock movement using configurable **confidence thresholds**
- 📈 Achieved **73% accuracy** at **90% confidence threshold**
- Optional: Threshold sweep & visualization, Logistic Regression (coming soon)

---

## 🛠️ Tech Stack

- `Python`
- `Pandas`, `Matplotlib`
- `Transformers (HuggingFace)`
- `yfinance`, `NewsAPI`
- Optional: `Streamlit` for Web App version

---

## 🚀 How It Works

1. Fetch **latest news headlines** for a stock (e.g., Tesla)
2. Apply **FinBERT** to extract:
   - Sentiment label (positive/neutral/negative)
   - **Confidence scores** (probabilities)
3. Predict:
   - **‘up’** if Pos_Prob ≥ threshold
   - **‘down’** if Neg_Prob ≥ threshold
   - Else **neutral**
4. Compare prediction with **actual stock movement**
5. Visualize **accuracy** and **sentiment distribution**

---

## 📊 Results

| Threshold | Accuracy  | Prediction Coverage |
|-----------|-----------|---------------------|
| 0.6       | 69.66%    | High                |
| 0.9       | 73.42%    | Low (high precision) |

> 💡 Higher thresholds = fewer but more **accurate predictions**

---

## 🧪 Future Enhancements

- 💻 Streamlit Web App for live interactive analysis
- ⏱️ Multi-day lag predictions

---

## 📥 Getting Started

1. Clone repo & install dependencies:
```bash
pip install -r requirements.txt
