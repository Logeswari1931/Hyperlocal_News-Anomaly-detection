# Hyperlocal News Anomaly Detection and Source Attribution

**Author:** Logeshwari P  
**Email:** logeshwaripurushoth@gmail.com  
**GitHub:** [@Logeswari1931](https://github.com/Logeswari1931)  
**LinkedIn:** [Logeshwari Purushothaman](https://www.linkedin.com/in/logeshwaripurushoth-purushothaman-6601ba89/)

---

## 📋 Project Overview

This project develops an **advanced NLP system to detect anomalies in hyperlocal news articles, identify source attribution discrepancies, and monitor evolving narratives** within geographic regions. The system analyzes article content, sentiment, location references, and temporal patterns to flag potentially misleading or unusual news events that deviate from expected patterns for their stated publication locations.

### **Key Objectives:**
- Detect anomalous patterns in news articles using machine learning
- Identify potential misattribution between stated location and article content
- Monitor sentiment and topic evolution over time for specific regions
- Provide interpretable anomaly scores and justifications
- Create an interactive dashboard for visualization and exploration

---

## 🎯 Domain & Problem Statement

**Domain:** Natural Language Processing (NLP), News Analytics, Linguistic Analysis  
**Task:** Multi-faceted anomaly detection in hyperlocal news articles

This system addresses the critical challenge of:
- Identifying fake news and disinformation attempts
- Detecting misattribution in news sources
- Monitoring local sentiment and narrative shifts
- Supporting content verification and quality assurance

### **Business Applications:**

1. **Disinformation Detection** - Flag articles where publication location seems inconsistent with content
2. **Hyperlocal Trend Monitoring** - Track sentiment and emerging topics in specific geographic regions
3. **Brand Reputation Management** - Monitor local news for anomalies related to brand mentions
4. **Automated Content Verification** - Flag articles requiring human review due to unusual characteristics

---

## 💡 Skills Developed

### **NLP & Text Processing**
- Advanced text preprocessing and cleaning with lemmatization
- Named Entity Recognition (NER) for location extraction using spaCy
- Sentiment analysis using VADER and transformer models
- Topic modeling with BERTopic
- Text embeddings with Sentence Transformers

### **Machine Learning & Deep Learning**
- Unsupervised anomaly detection (Isolation Forest)
- Transfer learning with pre-trained transformer models (BERT)
- Classification models (Logistic Regression for location prediction)
- Temporal anomaly detection techniques
- Model evaluation with multiple metrics

### **Data Processing & Engineering**
- Large-scale text data preprocessing (2,585 articles)
- Location normalization and disambiguation
- Feature engineering for NLP tasks
- Handling missing and ambiguous data
- Encoding detection for multi-format text files

### **Visualization & Deployment**
- Interactive dashboard creation with Matplotlib and Seaborn
- Data visualization of anomalies and patterns
- Streamlit/Dash for web application development
- Model serialization and deployment (joblib, pickle)

### **Software Engineering**
- Python with scikit-learn, TensorFlow, PyTorch
- Version control with Git/GitHub
- Code organization and modularity
- Documentation and best practices

---

## 📊 Dataset

**Dataset Composition:**
- **Total Articles:** 2,585 news articles
- **Features:**
  - `Article`: Full text of news article (average ~200 words)
  - `Heading`: Article title/headline
  - `Date`: Publication date (MMDDYYYY format)
  - `NewsType`: Category (business, sports, politics, technology, etc.)

**Data Characteristics:**
- **Temporal Coverage:** 2015 data
- **Geographic Diversity:** Articles from multiple countries (Pakistan, India, USA, UK, China, Singapore, Australia)
- **Language:** English
- **Quality:** Real-world news data with natural language variations

**Data Preprocessing Results:**
- Clean articles with stopword removal and lemmatization
- Extracted locations: 432+ unique simplified locations
- Unknown locations: 149 articles (5.8%)
- Sentiment distribution: 77.33% Positive, 21.12% Negative, 1.55% Neutral

---

## 🔧 Methodology & Approach

### **1. Data Preprocessing & Feature Extraction**

#### **Text Cleaning**
```python
- Remove URLs and special characters
- Convert to lowercase
- Tokenization and lemmatization using spaCy
- Stopword removal
- Preserve original text for reference
```

#### **Location Extraction**
- **Method**: Hybrid approach combining:
  - spaCy NER (Named Entity Recognition) for GPE and LOC entities
  - GeoText library for city/country detection
  - Custom post-processing for location normalization
- **Coverage**: 2,436 articles with identified locations (94.2%)
- **Accuracy**: 7 major countries mapped from 432 unique location strings

#### **Sentiment Analysis**
```python
- Tool: VADER Sentiment Analyzer
- Metrics: Compound polarity score (-1 to 1)
- Classification:
  - Positive: compound ≥ 0.05
  - Negative: compound ≤ -0.05
  - Neutral: -0.05 < compound < 0.05
```

#### **Topic Modeling**
- **Model**: BERTopic with pre-trained embeddings
- **Parameters**:
  - Initial topics: 66 → reduced to 27 (auto topic reduction)
  - Uses Sentence Transformers embeddings
  - UMAP for dimensionality reduction
  - HDBSCAN for clustering
- **Output**: Topic IDs assigned to each article

### **2. Feature Engineering**

#### **Textual Features**
- **Sentence Transformers (all-MiniLM-L6-v2)**
  - 384-dimensional dense embeddings
  - Lightweight model for efficiency
  - Captures semantic meaning of articles
  - Normalized embeddings for fair comparison

#### **Metadata Features**
- Location (extracted and normalized)
- Sentiment (categorical: Positive, Negative, Neutral)
- Topic (BERTopic ID)
- Date/temporal features

#### **Location-Based Features**
- Primary location extracted from article
- NewsType (business, sports, etc.)
- Expected sentiment/topic profile by location and type

### **3. Anomaly Detection Models**

#### **A. Isolation Forest (Linguistic Anomaly Detection)**
```python
Parameters:
- n_estimators: 200
- contamination: 0.05 (5% expected anomaly rate)
- random_state: 42
- n_jobs: -1 (parallel processing)

Input: Scaled sentence embeddings
Output: Binary anomaly flag (Anomaly/Normal)
```

**Why Isolation Forest?**
- Unsupervised learning (no labeled anomalies needed)
- Effective for high-dimensional data
- Computationally efficient (O(n log n))
- Naturally detects outliers in embedding space

#### **B. BERT-based Location Predictor (Source Attribution)**
```python
Algorithm: Logistic Regression on embeddings
Input Features: Sentence embeddings (384-dim)
Output: Predicted location (7 countries)

Results:
- Overall Accuracy: 75.06%
- Best Class: Pakistan (91% precision, 94% recall)
- Worst Class: Singapore (0% - insufficient samples)

Per-Location Accuracy:
- Pakistan: 91-94% (164 samples)
- India: 67-70% (56 samples)
- United Kingdom: 74-75% (56 samples)
- United States: 57-76% (63 samples)
- Australia: 73% precision, 47% recall
- China: Very low accuracy (21 samples)
```

**Process:**
1. Train logistic regression on BERT embeddings
2. Predict most likely location from article content
3. Compare predicted location vs. stated location
4. High discrepancy = potential misattribution

#### **C. Temporal Anomaly Detection**
- Monitors sentiment and topic shifts over time
- Identifies sudden changes in publication patterns
- Flags spikes in topic frequencies
- Tracks sentiment evolution by location

### **4. System Integration & Results**

**Final Processing Pipeline:**
1. Article loading and encoding detection
2. Text cleaning and lemmatization
3. Sentiment analysis and location extraction
4. Embedding generation (Sentence Transformers)
5. Isolation Forest anomaly detection
6. BERTopic topic modeling
7. BERT-based location prediction
8. Discrepancy detection (predicted vs. actual location)
9. Result visualization and export

**Output Format:**
Each article receives:
- `Sentiment`: Positive/Negative/Neutral
- `Location`: Extracted primary location
- `AnomalyFlag`: Normal/Anomaly
- `Topic`: Topic ID (0-26)
- `PredictedLocationBERT`: Predicted location from embeddings
- `DiscrepancyBERT`: Match/Mismatch between locations

---

## 📈 Results & Performance Metrics

### **Anomaly Detection Performance**

**Isolation Forest Results:**
- Total articles analyzed: 2,585
- Flagged as anomalies: ~129 articles (5%)
- Flagged as normal: ~2,456 articles (95%)
- Aligns with contamination parameter (0.05)

### **Source Attribution/Location Prediction**

**BERT Embedding-Based Classifier:**
```
Overall Accuracy: 75.06% (389 test samples)

Per-Class Performance:
┌─────────────────────┬───────────┬────────┬───────────┬─────────┐
│ Location            │ Precision │ Recall │ F1-Score  │ Support │
├─────────────────────┼───────────┼────────┼───────────┼─────────┤
│ Pakistan            │ 0.91      │ 0.94   │ 0.92      │ 164     │
│ India               │ 0.67      │ 0.70   │ 0.68      │ 56      │
│ United States       │ 0.57      │ 0.76   │ 0.65      │ 63      │
│ United Kingdom      │ 0.74      │ 0.75   │ 0.74      │ 56      │
│ China               │ 0.12      │ 0.05   │ 0.07      │ 21      │
│ Australia           │ 0.73      │ 0.47   │ 0.57      │ 17      │
│ Singapore           │ 0.00      │ 0.00   │ 0.00      │ 12      │
└─────────────────────┴───────────┴────────┴───────────┴─────────┘

Macro Average:  0.53 precision, 0.52 recall, 0.52 F1-score
Weighted Average: 0.72 precision, 0.75 recall, 0.73 F1-score
```

**Discrepancy Detection:**
- Matching locations: ~945 articles (79.32% accuracy)
- Mismatched locations: ~1,043 articles (potential attribution issues)
- Unknown locations: 149 articles (excluded from analysis)

### **Topic Modeling**

**BERTopic Results:**
- Topics extracted: 27 (reduced from 66 via auto-reduction)
- Top topics: Business (markets, stocks), Politics, Sports
- Coherence: Good topic separation
- Coverage: All 2,585 articles assigned to topics

### **Sentiment Distribution**

```
Sentiment Analysis Results:
┌───────────┬────────────┬──────────────┐
│ Sentiment │ Count      │ Percentage   │
├───────────┼────────────┼──────────────┤
│ Positive  │ ~2,002     │ 77.33%       │
│ Negative  │ ~547       │ 21.12%       │
│ Neutral   │ ~40        │ 1.55%        │
└───────────┴────────────┴──────────────┘
```

---

## 🚀 Project Deliverables

### **1. Source Code**
- ✅ Complete Jupyter Notebook (`hyperlocal_news.ipynb`)
- ✅ Well-documented Python code with docstrings
- ✅ Modular functions for reusability
- ✅ Error handling and logging

### **2. Trained Models**
- ✅ Isolation Forest (anomaly detection) - `isolationforest_model.pkl`
- ✅ BERT Location Classifier - `location_classifier_bert.pkl`
- ✅ BERTopic Model - `bertopic_model/`
- ✅ Sentence Embeddings - `sentence_embeddings.npy`

### **3. Processed Datasets**
- ✅ Initial processed data with all features
- ✅ Final evaluation results with predictions
- ✅ CSV exports with timestamps for reproducibility

### **4. Visualizations & Dashboard**
- ✅ Sentiment distribution charts
- ✅ Topic frequency analysis (top 15 topics)
- ✅ Location-based sentiment heatmaps
- ✅ Anomaly detection distribution
- ✅ Confusion matrix (location prediction)
- ✅ Per-location accuracy plots
- ✅ BERTopic interactive visualization

### **5. Documentation**
- ✅ Comprehensive README (this file)
- ✅ Code comments and docstrings
- ✅ Data dictionary
- ✅ Methodology explanation
- ✅ Results interpretation guide

---

## 📥 Installation & Setup

### **Prerequisites**
- Python 3.8+
- pip or conda
- 4GB+ RAM (8GB recommended for embeddings)
- ~500MB disk space for models

### **Step 1: Clone Repository**
```bash
git clone https://github.com/Logeswari1931/Hyperlocal-News-Anomaly-Detection.git
cd Hyperlocal-News-Anomaly-Detection
```

### **Step 2: Create Virtual Environment**
```bash
# Using venv
python -m venv news_env
source news_env/bin/activate  # On Windows: news_env\Scripts\activate

# Or using conda
conda create -n news_anomaly python=3.10
conda activate news_anomaly
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **requirements.txt**
```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
tensorflow==2.13.0
torch==2.0.0
transformers==4.30.0
sentence-transformers==2.2.2
bertopic==0.15.0
spacy==3.5.0
geotext==0.3.0
vaderSentiment==3.3.2
joblib==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
chardet==5.1.0
tqdm==4.65.0
streamlit==1.28.0
plotly==5.14.0
```

### **Step 4: Download spaCy Model**
```bash
python -m spacy download en_core_web_sm
```

### **Step 5: Prepare Data**
```bash
# Place your Articles.csv in the data/ directory
# Expected columns: Article, Heading, Date, NewsType
```

---

## 💻 Usage Instructions

### **Option 1: Run Jupyter Notebook**
```bash
jupyter notebook hyperlocal_news.ipynb
```

**Execute cells sequentially:**
1. **Imports** - Load required libraries
2. **Data Loading** - Read CSV with encoding detection
3. **Text Cleaning** - Lemmatization and stopword removal
4. **Sentiment & Location** - Extract features
5. **Embeddings & Anomaly** - Generate embeddings and detect anomalies
6. **Visualization** - Create charts
7. **BERT Classification** - Predict locations
8. **Evaluation** - Final metrics and analysis

### **Option 2: Deploy with Streamlit**
```bash
streamlit run app.py
```

### **Option 3: Use Pre-trained Models**
```python
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load models
iso_forest = joblib.load('models/isolationforest_model.pkl')
location_clf = joblib.load('models/location_classifier_bert.pkl')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Predict on new article
article_text = "Your article here..."
embedding = embedder.encode([article_text])[0]

# Detect anomaly
anomaly_score = iso_forest.decision_function([embedding])
is_anomaly = iso_forest.predict([embedding])[0]

# Predict location
predicted_location = location_clf.predict([embedding])[0]

print(f"Is Anomalous: {is_anomaly}")
print(f"Anomaly Score: {anomaly_score}")
print(f"Predicted Location: {predicted_location}")
```

---

## 📁 Project Structure

```
Hyperlocal-News-Anomaly-Detection/
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
├── hyperlocal_news.ipynb                       # Main Jupyter Notebook
│
├── data/
│   └── Articles.csv                            # Input dataset
│
├── notebooks/
│   ├── models/                                 # Trained models directory
│   │   ├── isolationforest_model.pkl
│   │   ├── location_classifier_bert.pkl
│   │   ├── bertopic_model/
│   │   └── sentence_embeddings.npy
│   │
│   └── outputs/                                # Output results
│       ├── processed_<timestamp>.csv           # Processed data
│       └── final_evaluation_results.csv        # Final metrics
│
└── visualizations/                             # Generated plots
    ├── sentiment_distribution.png
    ├── topic_frequency.png
    ├── anomaly_distribution.png
    └── location_predictions.png
```

---

## 🔬 Key Technical Components

### **Sentence Transformers (all-MiniLM-L6-v2)**
- **Purpose**: Generate semantic embeddings for articles
- **Dimensions**: 384-dimensional vectors
- **Training Data**: Fine-tuned on 215M sentence pairs
- **Advantages**: Lightweight, fast, good semantic understanding
- **Usage**: Core feature for anomaly detection

### **spaCy NER (en_core_web_sm)**
- **Purpose**: Named Entity Recognition for location extraction
- **Entities Used**: GPE (countries), LOC (locations)
- **Accuracy**: ~90% for common locations
- **Enhancement**: Combined with GeoText for better coverage

### **BERTopic**
- **Purpose**: Unsupervised topic modeling
- **Pipeline**: Embeddings → UMAP → HDBSCAN → Representation
- **Topics**: 27 coherent topics automatically extracted
- **Interpretability**: Keywords describing each topic

### **Isolation Forest**
- **Purpose**: Unsupervised anomaly detection
- **Algorithm**: Tree-based isolation of anomalies
- **Complexity**: O(n log n) - efficient for large datasets
- **Parameters**: Optimized for 5% contamination rate

---

## 📊 Evaluation & Interpretation

### **Anomaly Scores**
- Range: -1 (definitely normal) to 1 (definitely anomaly)
- Threshold: Default 0 (model's decision boundary)
- Interpretation: High positive scores indicate anomalies

### **Location Prediction Confidence**
- Model predicts probability for each location
- High confidence (>0.8): Reliable prediction
- Low confidence (<0.5): Uncertain location
- Discrepancy: Compare predicted vs. stated location

### **Sentiment Scores**
- Positive compounds (>0.05): Optimistic content
- Negative compounds (<-0.05): Pessimistic content
- Neutral (-0.05 to 0.05): Balanced reporting

### **Topic Relevance**
- Each topic has associated keywords
- Distance from centroid indicates relevance
- Similar topics might indicate narrative consistency

---

## 🔮 Future Enhancements

1. **Model Improvements**
   - Fine-tune transformers on news domain
   - Ensemble methods combining multiple anomaly detectors
   - Temporal models (RNN/LSTM) for sequence patterns

2. **Feature Engineering**
   - Named entity linking to knowledge bases
   - Aspect-based sentiment analysis
   - Author and source credibility scoring

3. **Deployment**
   - REST API with FastAPI
   - Docker containerization
   - AWS/GCP deployment with auto-scaling
   - Real-time streaming pipeline

4. **Advanced Analytics**
   - Narrative arc analysis
   - Fake news detection integration
   - Stance detection toward entities
   - Credibility scoring

5. **Scalability**
   - GPU acceleration for embeddings
   - Distributed processing with Spark
   - Caching strategies
   - Database integration (PostgreSQL)

---

## ⚠️ Known Limitations

1. **Data Scope**
   - Single dataset from 2015
   - Limited to English articles
   - Geographic bias toward business news

2. **Model Limitations**
   - Sensitivity to writing style variations
   - Requires sufficient location mentions in text
   - Dependent on training data distribution

3. **Location Prediction**
   - Poor performance on underrepresented countries
   - Ambiguous location references not resolved
   - Doesn't account for diaspora communities

4. **Anomaly Detection**
   - Assumes embedding space contains meaningful patterns
   - May flag legitimate unusual stories
   - Requires manual validation for critical decisions

---

## 📖 References & Resources

### **Core Libraries**
- [spaCy Documentation](https://spacy.io/usage)
- [Sentence Transformers](https://www.sbert.net/)
- [scikit-learn](https://scikit-learn.org/)
- [BERTopic](https://maartengr.github.io/BERTopic/)

### **Academic Papers**
- BERT: Pre-training of Deep Bidirectional Transformers
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Isolation Forest: Liu et al., 2008
- BERTopic: Maarten Grootendorst, 2022

### **NLP Concepts**
- Named Entity Recognition (Ling. Overview)
- Sentiment Analysis (Pang & Lee, 2008)
- Topic Modeling (Blei et al., 2003)
- Word Embeddings (Mikolov et al., 2013)

---

## ✅ Code Quality & Standards

- **PEP 8 Compliance**: All code follows Python style guide
- **Modular Design**: Functions are reusable and well-documented
- **Error Handling**: Try-except blocks for robust execution
- **Logging**: Progress indicators and status messages
- **Reproducibility**: Fixed random seeds for all models
- **Documentation**: Inline comments and docstrings

---

## 📧 Contact & Support

For questions, suggestions, or collaboration:

- **Email:** logeshwaripurushoth@gmail.com
- **GitHub:** [@Logeswari1931](https://github.com/Logeswari1931)
- **LinkedIn:** [Logeshwari Purushothaman](https://www.linkedin.com/in/logeshwaripurushoth-purushothaman-6601ba89/)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Special thanks to GUVI IITM for project guidance
- Dataset sourced from public news archives
- Sentence Transformers team for pre-trained models
- spaCy team for NLP tools
- Open source community for various libraries

---

## 📝 Citation

If you use this project in your research or work, please cite:

```
@project{hyperlocal_news_anomaly,
  title={Hyperlocal News Anomaly Detection and Source Attribution},
  author={Logeshwari P},
  year={2025},
  url={https://github.com/Logeswari1931/Hyperlocal-News-Anomaly-Detection}
}
```

---

**Project Status:** ✅ Complete & Production Ready  
**Last Updated:** December 2025  
**Version:** 1.0
