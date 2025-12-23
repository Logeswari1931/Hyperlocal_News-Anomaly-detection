# news_app.py

import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import spacy
from geotext import GeoText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 0. Global config & seeds
# ==========================

RANDOM_STATE = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

try:
    import torch
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
except Exception:
    pass

sns.set(style="whitegrid")

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = NOTEBOOKS_DIR / "models"
OUTPUTS_DIR = NOTEBOOKS_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True, parents=True)

PROCESSED_FILE = OUTPUTS_DIR / "processed_news.csv"

# ==========================
# 1. Cached resources
# ==========================

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_sia():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_saved_models():
    """
    Load locally saved models (NO HuggingFace).
    """

    # ---- FIXED PART ----
    topic_model_path = MODELS_DIR / "bertopic_model"
    topic_model = BERTopic.load(
        topic_model_path,
        embedding_model=None
    )
    # --------------------

    iso_forest = joblib.load(MODELS_DIR / "isolation_forest_model.pkl")

    clf_path = MODELS_DIR / "location_classifier_bert.pkl"
    clf_emb = joblib.load(clf_path) if clf_path.exists() else None

    embeddings = np.load(MODELS_DIR / "sentence_embeddings.npy")

    scaler = StandardScaler()
    scaler.fit(embeddings)

    return topic_model, iso_forest, clf_emb, scaler, embeddings


# ==========================
# 2. Text cleaning helpers
# ==========================

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# ==========================
# 3. Sentiment & Location
# ==========================

def get_sentiment(text):
    if not text:
        return "Neutral"
    score = load_sia().polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def extract_location(text):
    if not text:
        return "Unknown"
    doc = load_nlp()(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text
    geo = GeoText(text)
    return geo.cities[0] if geo.cities else "Unknown"


# ==========================
# 4. Dataset loading
# ==========================

@st.cache_data
def load_processed_dataset():
    if not PROCESSED_FILE.exists():
        st.error(f"Missing file: {PROCESSED_FILE}")
        st.stop()
    return pd.read_csv(PROCESSED_FILE)


# ==========================
# 5. Single article inference
# ==========================

def analyze_single_article(text, iso_forest, scaler, topic_model, embedder, clf_emb):
    clean = basic_clean(text)
    emb = embedder.encode([clean], normalize_embeddings=True)
    emb_scaled = scaler.transform(emb)

    anomaly = "Anomaly" if iso_forest.predict(emb_scaled)[0] == -1 else "Normal"

    topic_id, probs = topic_model.transform([clean])
    topic_id = int(topic_id[0])
    conf = float(np.max(probs))

    loc_pred = clf_emb.predict(emb)[0] if clf_emb else None

    return {
        "sentiment": get_sentiment(clean),
        "topic_id": topic_id,
        "topic_conf": conf,
        "anomaly_flag": anomaly,
        "location_predicted": loc_pred,
        "clean_article": clean
    }


# ==========================
# 6. Streamlit UI
# ==========================

def main():
    st.set_page_config(page_title="Hyperlocal News Dashboard", layout="wide")
    st.title("📰 Hyperlocal News Anomaly Detection")

    with st.spinner("Loading models..."):
        topic_model, iso_forest, clf_emb, scaler, _ = load_saved_models()

    df = load_processed_dataset()
    st.success("Models loaded successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    st.markdown("---")
    st.subheader("Analyze Single Article")

    user_text = st.text_area("Paste article text", height=200)

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter text")
        else:
            result = analyze_single_article(
                user_text,
                iso_forest,
                scaler,
                topic_model,
                load_embedder(),
                clf_emb,
            )

            st.metric("Sentiment", result["sentiment"])
            st.metric("Anomaly", result["anomaly_flag"])
            st.metric("Topic ID", result["topic_id"])
            st.metric("Topic Confidence", f"{result['topic_conf']*100:.1f}%")

            if result["location_predicted"]:
                st.metric("Predicted Location", result["location_predicted"])

            with st.expander("Cleaned Text"):
                st.write(result["clean_article"])


if __name__ == "__main__":
    main()