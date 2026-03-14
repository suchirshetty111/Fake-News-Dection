from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import trafilatura
import requests

app = Flask(__name__)

# -----------------------------
# Models
# -----------------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# -----------------------------
# Trusted Sources (Global)
# -----------------------------
TRUSTED_SOURCES = [
    "bbc.co.uk", "bbc.com", "cnn.com", "reuters.com", "aljazeera.com",
    "nytimes.com", "theguardian.com", "apnews.com", "espncricinfo.com",
    "cricbuzz.com", "timesofindia.indiatimes.com", "economictimes.indiatimes.com"
]

# -----------------------------
# Fact-check function
# -----------------------------
def fact_check(source_url=""):
    return any(trusted in source_url for trusted in TRUSTED_SOURCES)

# -----------------------------
# Combined scoring
# -----------------------------
def combined_label_score(model_label, model_score, is_trusted):
    score = model_score
    if not is_trusted:
        score -= 30  # reduce confidence if source is untrusted
    if score < 0: score = 0
    label = "REAL" if score >= 60 else "FAKE"
    return label, round(score, 2)

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Text Prediction Route
# -----------------------------
@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        candidate_labels = ["FAKE", "REAL"]
        result = classifier(text, candidate_labels)
        model_label = result["labels"][0]
        model_score = round(result["scores"][0] * 100, 2)

        # For text input, assume untrusted source
        label, score = combined_label_score(model_label, model_score, is_trusted=False)

        summary = summarizer(text, max_length=70, min_length=10, do_sample=False)[0]["summary_text"]

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "sequence": text,
        "prediction": label,
        "score": score,
        "summary": summary
    })

# -----------------------------
# URL Prediction Route
# -----------------------------
@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        if not text:
            return jsonify({"error": "Could not extract article text"}), 500

        candidate_labels = ["FAKE", "REAL"]
        result = classifier(text, candidate_labels)
        model_label = result["labels"][0]
        model_score = round(result["scores"][0] * 100, 2)

        is_trusted = fact_check(url)
        label, score = combined_label_score(model_label, model_score, is_trusted)

        summary = summarizer(text, max_length=70, min_length=10, do_sample=False)[0]["summary_text"]

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "sequence": text[:500] + "...",
        "prediction": label,
        "score": score,
        "summary": summary
    })

# -----------------------------
# Live Worldwide News Route
# -----------------------------
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"  # Replace with your NewsAPI key

@app.route("/live_world_news", methods=["GET"])
def live_world_news():
    categories = [
        "general", "business", "entertainment", "health", "science",
        "sports", "technology", "politics", "education", "government",
        "heroes", "space", "environment", "celebrities", "law", "economy"
    ]
    results = []

    try:
        for category in categories:
            url = f"https://newsapi.org/v2/top-headlines?language=en&category={category}&pageSize=5&apiKey={NEWSAPI_KEY}"
            response = requests.get(url)
            articles = response.json().get("articles", [])

            for article in articles:
                full_text = (article.get("title") or "") + ". " + \
                            (article.get("description") or "") + ". " + \
                            (article.get("content") or "")

                if not full_text.strip():
                    continue

                candidate_labels = ["FAKE", "REAL"]
                pred = classifier(full_text, candidate_labels)
                model_label = pred["labels"][0]
                model_score = round(pred["scores"][0] * 100, 2)

                source_url = article.get("url", "")
                is_trusted = fact_check(source_url)
                label, score = combined_label_score(model_label, model_score, is_trusted)

                summary = summarizer(full_text, max_length=70, min_length=10, do_sample=False)[0]["summary_text"]

                results.append({
                    "category": category,
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "url": source_url,
                    "prediction": label,
                    "score": score,
                    "summary": summary
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
