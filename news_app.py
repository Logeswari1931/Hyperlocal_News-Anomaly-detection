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

# Main processed file from your training script
PROCESSED_FILE = OUTPUTS_DIR / "processed_news.csv"


# ==========================
# 1. Cached resources
# ==========================

@st.cache_resource
def load_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_sia():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_saved_models():
    """
    Load your already-trained models + embeddings from notebooks/models.
    No training happens here.
    """
    # BERTopic
    topic_model_path = MODELS_DIR / "bertopic_model"
    topic_model = BERTopic.load(str(topic_model_path))

    # Isolation Forest
    iso_path = MODELS_DIR / "isolation_forest_model.pkl"
    iso_forest = joblib.load(iso_path)

    # Location classifier (might not exist if training was skipped)
    clf_path = MODELS_DIR / "location_classifier_bert.pkl"
    if clf_path.exists():
        clf_emb = joblib.load(clf_path)
    else:
        clf_emb = None

    # Sentence embeddings
    emb_path = MODELS_DIR / "sentence_embeddings.npy"
    embeddings = np.load(emb_path)

    # Rebuild the same StandardScaler that was used during training
    scaler = StandardScaler()
    scaler.fit(embeddings)

    return topic_model, iso_forest, clf_emb, scaler, embeddings


# ==========================
# 2. Text cleaning helpers
# ==========================

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def clean_column(text_list, desc="Cleaning"):
    """Lemmatize + remove stopwords for a whole column."""
    nlp = load_nlp()
    cleaned = []
    text_list = [basic_clean(t) for t in text_list]

    progress = st.progress(0.0, text=desc)

    for i, doc in enumerate(nlp.pipe(text_list, batch_size=64)):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.lemma_.lower() not in STOP_WORDS
        ]
        cleaned.append(" ".join(tokens))
        if (i + 1) % 20 == 0:
            progress.progress((i + 1) / len(text_list),
                              text=f"{desc}: {i+1}/{len(text_list)}")

    progress.empty()
    return cleaned


# ==========================
# 3. Sentiment & Location
# ==========================

def get_sentiment(text: str) -> str:
    sia = load_sia()
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"

    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


def clean_text_for_location(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_location(text: str) -> str:
    nlp = load_nlp()
    text = clean_text_for_location(text)
    if text == "":
        return "Unknown"

    doc = nlp(text)
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    if locs:
        return locs[0]

    geo = GeoText(text)
    if geo.cities:
        return geo.cities[0]
    if geo.countries:
        return geo.countries[0]
    return "Unknown"


def simplify_country(loc: str) -> str:
    if not isinstance(loc, str) or loc == "Unknown":
        return "Unknown"
    s = loc.lower()

    if "pakistan" in s or s in ["karachi", "lahore", "islamabad"]:
        return "Pakistan"
    if "india" in s or s in ["delhi", "mumbai", "chennai", "bangalore"]:
        return "India"
    if "united states" in s or s in ["new york", "washington", "usa", "u.s."]:
        return "United States"
    if "london" in s or "uk" in s or "england" in s:
        return "United Kingdom"
    if "china" in s or s in ["beijing", "shanghai", "hong kong"]:
        return "China"
    if "singapore" in s:
        return "Singapore"
    if "australia" in s or s in ["sydney", "melbourne"]:
        return "Australia"

    return loc.title()


# ==========================
# 4. Load processed dataset
# ==========================

@st.cache_data(show_spinner=True)
def load_processed_dataset() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        st.error(f"Processed file not found: {PROCESSED_FILE}")
        st.stop()
    df = pd.read_csv(PROCESSED_FILE)

    # If someone runs it on an older CSV without Sentiment/Location, fill them
    if "Sentiment" not in df.columns and "clean_article" in df.columns:
        df["Sentiment"] = df["clean_article"].apply(get_sentiment)
    if "Location" not in df.columns and "raw_article" in df.columns:
        df["Location"] = df["raw_article"].apply(extract_location).apply(simplify_country)

    return df


def evaluate_location_classifier_from_df(df_model: pd.DataFrame):
    """Compute accuracy & report from processed_news.csv columns."""
    if "Location" not in df_model.columns or "Predicted_Location_BERT" not in df_model.columns:
        return None, "Required columns not present for evaluation."

    df_eval = df_model[
        df_model["Location"].notna()
        & (df_model["Location"] != "Unknown")
        & df_model["Predicted_Location_BERT"].notna()
    ].copy()

    if df_eval.empty:
        return None, "No valid rows to evaluate."

    y_true = df_eval["Location"].astype(str)
    y_pred = df_eval["Predicted_Location_BERT"].astype(str)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    return acc, report


# ==========================
# 5. Single-article inference
# ==========================

def analyze_single_article(
    text: str,
    iso_forest,
    scaler,
    topic_model,
    embedder,
    clf_emb,
):
    nlp = load_nlp()

    # clean + lemmatize
    basic = basic_clean(text)
    doc = nlp(basic)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and token.lemma_.lower() not in STOP_WORDS
    ]
    clean_text_l = " ".join(tokens)

    # sentiment & location (rule-based)
    sent = get_sentiment(clean_text_l or text)
    loc_extracted = simplify_country(extract_location(text))

    # embedding (same model as training)
    emb = embedder.encode(
        [clean_text_l],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    emb_scaled = scaler.transform(emb)

    # anomaly (using your trained Isolation Forest)
    anomaly_pred = iso_forest.predict(emb_scaled)[0]
    anomaly_flag = "Anomaly" if anomaly_pred == -1 else "Normal"

    # topic (using your trained BERTopic)
    topic_id, topic_prob = topic_model.transform([clean_text_l])
    topic_id = int(topic_id[0])
    topic_prob = float(np.max(topic_prob))

    # location prediction (your trained LogisticRegression)
    loc_pred = None
    if clf_emb is not None:
        loc_pred = clf_emb.predict(emb)[0]

    return {
        "clean_article": clean_text_l,
        "sentiment": sent,
        "location_extracted": loc_extracted,
        "location_predicted": loc_pred,
        "anomaly_flag": anomaly_flag,
        "topic_id": topic_id,
        "topic_conf": topic_prob,
    }


# ==========================
# 6. Streamlit UI
# ==========================

def main():
    st.set_page_config(
        page_title="Hyperlocal News Anomaly & Attribution",
        layout="wide",
    )

    st.title("📰 Hyperlocal News Anomaly & Source Attribution Dashboard")
    st.caption(
        f"Using saved models from: `{MODELS_DIR}` and processed data from `{PROCESSED_FILE}`"
    )

    # Sidebar help / glossary
    st.sidebar.markdown("### ℹ️ How to read this dashboard")
    st.sidebar.markdown(
        """
**Sentiment** – overall tone of the article (Positive / Negative / Neutral)  
**Location** – country/region inferred from the text.  
**Topic** – group of articles that talk about similar things; shown using its top keywords.  
**Anomaly** – article is *unusual* compared to the rest of the dataset  
(unsupervised Isolation Forest on BERT embeddings).  
**Predicted Location (BERT)** – country predicted by a classifier trained on sentence embeddings.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** Hover over each tab and section titles. They contain short explanations for reviewers and end users."
    )

    # Load saved models + embeddings
    with st.spinner("Loading saved models & embeddings..."):
        topic_model, iso_forest, clf_emb, scaler, embeddings = load_saved_models()

    # Load processed dataset
    with st.spinner("Loading processed dataset..."):
        df_model = load_processed_dataset()

    st.success("Models and data loaded successfully.")

    # ---- Overview stats ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", len(df_model), help="Number of news articles in the processed dataset.")
    col2.metric("Unique Locations",
                df_model["Location"].nunique() if "Location" in df_model.columns else 0,
                help="Distinct locations detected across all articles.")
    col3.metric("Anomalies",
                (df_model.get("AnomalyFlag") == "Anomaly").sum()
                if "AnomalyFlag" in df_model.columns else 0,
                help="Articles that look statistically unusual compared to the rest.")
    col4.metric("Topics Found",
                df_model["Topic"].nunique() if "Topic" in df_model.columns else 0,
                help="Number of semantic topics discovered by BERTopic.")

    st.markdown("---")

    tab_overview, tab_sent_loc, tab_anom, tab_topics, tab_single = st.tabs(
        ["Overview", "Sentiment & Locations", "Anomalies", "Topics", "Single Article"]
    )

    # ============== Overview tab ==============
    with tab_overview:
        st.subheader("Dataset Snapshot")
        st.markdown(
            "Each row below is a processed article with its detected **location**, "
            "**sentiment**, **topic**, and whether it was marked as an **anomaly**."
        )
        cols_to_show = [
            c
            for c in ["Heading", "Article", "Location",
                      "Sentiment", "AnomalyFlag", "Topic"]
            if c in df_model.columns
        ]
        st.dataframe(df_model[cols_to_show].head(20))

        if "Predicted_Location_BERT" in df_model.columns:
            st.subheader("Location Classifier (BERT Embeddings)")
            st.markdown(
                "This classifier uses sentence embeddings to predict the most likely **country** "
                "for an article, based only on its content."
            )
            acc, report = evaluate_location_classifier_from_df(df_model)
            if acc is not None:
                st.write(f"**Accuracy (from processed_news.csv):** {acc * 100:.2f}%")
                with st.expander("See detailed classification report"):
                    st.text(report)
            else:
                st.info(report)
        else:
            st.info("Predicted_Location_BERT column not found in processed dataset.")

    # ============== Sentiment & Location tab ==============
    with tab_sent_loc:
        st.subheader("Sentiment Distribution")
        st.markdown(
            "Shows how many articles are **Positive**, **Negative**, or **Neutral** in tone. "
            "Useful for checking overall mood of the news corpus."
        )
        if "Sentiment" in df_model.columns:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.countplot(x="Sentiment", data=df_model, ax=ax1)
            ax1.set_xlabel("")
            st.pyplot(fig1)
        else:
            st.info("Sentiment column not found in dataset.")

        st.subheader("Location-based Sentiment (Top 10 Locations)")
        st.markdown(
            "For the 10 most frequent locations, this chart shows how sentiment is distributed. "
            "You can quickly see which regions have more negative or positive news."
        )
        if "Location" in df_model.columns and df_model["Location"].nunique() > 1:
            top_locations = df_model["Location"].value_counts().head(10).index
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.countplot(
                data=df_model[df_model["Location"].isin(top_locations)],
                x="Location",
                hue="Sentiment" if "Sentiment" in df_model.columns else None,
                ax=ax2,
            )
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig2)
        else:
            st.info("Not enough distinct locations to show this plot.")

    # ============== Anomalies tab ==============
    with tab_anom:
        st.subheader("Anomaly Detection Summary")
        st.markdown(
            """
An **anomaly** here means:  
> “This article looks unusually different from the rest of the dataset  
> in the high-dimensional BERT embedding space.”

The model (Isolation Forest) is unsupervised – it learns what *normal* looks like
and then flags roughly 5% of articles as unusual.
            """
        )
        if "AnomalyFlag" in df_model.columns:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            sns.countplot(x="AnomalyFlag", data=df_model, ax=ax3)
            st.pyplot(fig3)

            st.subheader("Top Anomalous Articles")
            st.markdown(
                "These articles might represent **rare events**, **breaking news**, or just "
                "**unusual combinations of topics/locations**."
            )
            anomalies_df = df_model[df_model["AnomalyFlag"] == "Anomaly"].copy()
            if anomalies_df.empty:
                st.info("No anomalies detected.")
            else:
                cols = [
                    c
                    for c in ["Heading", "Article", "Location",
                              "Sentiment", "Topic"]
                    if c in anomalies_df.columns
                ]
                st.dataframe(anomalies_df[cols].head(30))
        else:
            st.info("AnomalyFlag column not found in dataset.")

    # ============== Topics tab ==============
    with tab_topics:
        st.subheader("Topic Frequencies")
        st.markdown(
            """
Topics are discovered using **BERTopic**, which groups articles that talk about
similar things. Each topic is described by its **top keywords**.

The bar chart shows which topics are most common in the dataset.
            """
        )
        if "Topic" in df_model.columns:
            topic_counts = df_model["Topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]

            if topic_counts.empty:
                st.info("No topics found in dataset.")
            else:
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                sns.barplot(x="Topic", y="Count",
                            data=topic_counts.head(15), ax=ax4)
                st.pyplot(fig4)

                st.subheader("Topic Details")

                default_topic = int(topic_counts["Topic"].iloc[0])

                topic_id = st.number_input(
                    "Topic ID to inspect",
                    min_value=int(topic_counts["Topic"].min()),
                    max_value=int(topic_counts["Topic"].max()),
                    value=default_topic,
                    step=1,
                )

                # words for that topic
                try:
                    topic_words = topic_model.get_topic(int(topic_id))
                    if topic_words:
                        topic_keywords = ", ".join([w for w, _ in topic_words[:10]])
                        st.markdown(f"**Top keywords for Topic {topic_id}:** {topic_keywords}")
                    else:
                        st.info("No words found for this topic.")
                except Exception:
                    st.info("Topic info not available for this ID.")

                # sample articles
                st.subheader("Sample Articles for Topic")
                sample_topic_df = df_model[df_model["Topic"] == int(topic_id)].head(20)[
                    [
                        c
                        for c in [
                            "Heading", "Article", "Location",
                            "Sentiment", "AnomalyFlag"
                        ]
                        if c in df_model.columns
                    ]
                ]
                if sample_topic_df.empty:
                    st.info("No articles for this topic ID.")
                else:
                    st.dataframe(sample_topic_df)
        else:
            st.info("Topic column not found in dataset.")

    # ============== Single article tab ==============
    with tab_single:
        st.subheader("Analyze a Single Article (using your trained models)")
        st.markdown(
            """
Paste any news article below. The system will:

1. Detect its **sentiment** (tone of the text)  
2. Extract a **location** from the text  
3. Predict the most likely **country** using BERT embeddings  
4. Assign it to a **topic** (shown as keywords)  
5. Decide whether it is a **normal** article or an **anomaly** (unusual)
            """
        )

        user_text = st.text_area(
            "Paste an article (or paragraph)",
            height=200,
            placeholder="Paste news content here...",
        )

        if st.button("Analyze Article", type="primary"):
            if not user_text.strip():
                st.warning("Please paste some text first.")
            else:
                with st.spinner("Analyzing article..."):
                    result = analyze_single_article(
                        user_text,
                        iso_forest=iso_forest,
                        scaler=scaler,
                        topic_model=topic_model,
                        embedder=load_embedder(),
                        clf_emb=clf_emb,
                    )

                # get human-readable topic words
                try:
                    topic_words = topic_model.get_topic(result["topic_id"])
                    if topic_words:
                        topic_keywords = ", ".join([w for w, _ in topic_words[:7]])
                    else:
                        topic_keywords = "No keywords found"
                except Exception:
                    topic_keywords = "Topic not available"

                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", result["sentiment"],
                          help="Overall tone of the article text.")
                c2.metric("Extracted Location",
                          result["location_extracted"],
                          help="Location entity pulled directly from the text.")
                c3.metric("Anomaly",
                          result["anomaly_flag"],
                          help="Anomaly = article looks unusual compared to the rest of the dataset.")

                c4, c5 = st.columns(2)
                c4.metric("Topic (keywords)", topic_keywords,
                          help="Top words that define the topic this article belongs to.")
                c5.metric("Topic Confidence",
                          f"{result['topic_conf']*100:.1f}%",
                          help="How strongly the model thinks the article fits this topic.")

                if result["location_predicted"] is not None:
                    st.metric("Predicted Location (BERT)",
                              result["location_predicted"],
                              help="Country predicted from the article's semantics using a classifier.")
                else:
                    st.info("Location classifier not trained or model file missing.")

                st.markdown("---")
                st.markdown("### What this means in simple terms")
                st.markdown(
                    f"""
- The article **sounds** `{result['sentiment']}` overall.  
- From the text, we picked up a location like **{result['location_extracted']}**.  
- The BERT-based classifier thinks the article most likely belongs to **{result['location_predicted'] or 'Unknown'}**.  
- It talks about a topic described roughly as: **{topic_keywords}**.  
- Anomaly status **{result['anomaly_flag']}** means this article is
  {'unusual compared to most articles in the dataset.' if result['anomaly_flag']=='Anomaly' else 'similar to many other routine articles.'}
                    """
                )

                with st.expander("See cleaned text used for modeling"):
                    st.write(result["clean_article"])


if __name__ == "__main__":
    main()
