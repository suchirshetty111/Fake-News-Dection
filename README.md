# Fake News Detection

**Fake News Detection** is a Flask web application that uses machine learning (Hugging Face Transformers) to classify whether a piece of news is real or fake and generate a short summary.

Running Application:https://fake-news-dection--suchirshenishet.replit.app

## 🔍 What it does
- Classifies text as **REAL** or **FAKE** (zero-shot classification)
- Summarizes news content automatically
- Lets you check either:
  - raw news text input, or
  - a news article URL (automatically extracts content)

## 🚀 Run locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 📦 Deploy
This project includes a `Procfile`, `requirements.txt`, and `runtime.txt` for deploying to platforms like Render, Heroku, Fly.io, and others.
