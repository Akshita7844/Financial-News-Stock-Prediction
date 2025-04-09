from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download and cache locally
AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", cache_dir="./models/finbert")
AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", cache_dir="./models/finbert")
