# Outline given by claude
# NLP Fundamentals: Amazon Product Review Analysis

A hands-on exploration of core NLP techniques applied to Amazon Home & Kitchen product reviews. Each day applies a different set of NLP methods to the same dataset, building intuition for when and why to use each approach.

## Dataset
**Amazon Reviews 2023** (McAuley Lab) — Home & Kitchen subset, sampled to ~50k–100k reviews. Includes review text, ratings, titles, helpfulness votes, and timestamps.

## Project Structure

```
nlp-amazon-reviews/
├── notebooks/
│   ├── day1_preprocessing.ipynb
│   ├── day2_embeddings.ipynb
│   ├── day3_classification.ipynb
│   ├── day4_information_extraction.ipynb
│   ├── day5_topic_sentiment.ipynb
│   ├── day6_summarization.ipynb
│   └── day7_synthesis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── classifier.py
│   ├── extraction.py
│   ├── topics.py
│   └── summarizer.py
├── data/
├── figures/
├── requirements.txt
└── README.md
```

## Daily Breakdown

### Day 1 — Text Preprocessing & Representation
- Cleaning pipeline for raw review text (HTML, unicode, punctuation)
- Tokenization comparison: whitespace vs regex vs spaCy
- TF-IDF implementation from scratch
- Exploratory analysis: vocabulary distributions, Zipf's law, stopword impact

### Day 2 — Embeddings & Similarity
- Train Word2Vec on review corpus using Gensim
- Compare sparse (TF-IDF) vs dense (Word2Vec, sentence-transformer) representations
- Visualize embedding space with t-SNE/UMAP across product categories
- Analyze what different similarity metrics capture (cosine, euclidean, dot product)

### Day 3 — Text Classification
- Task: predict star rating from review text
- Baseline: TF-IDF + logistic regression
- Transformer approach: fine-tune DistilBERT on same task
- Compare performance, failure modes, and computational tradeoffs

### Day 4 — Sequence Labeling & Information Extraction
- Named entity recognition with spaCy
- POS tagging and dependency parsing
- Rule-based aspect extraction using dependency patterns (e.g., "comfortable cushion", "flimsy legs")
- Comparison of rule-based vs model-based extraction approaches

### Day 5 — Topic Modeling & Aspect-Based Sentiment
- LDA topic modeling with coherence tuning
- BERTopic on same corpus — compare discovered topics
- Aspect-based sentiment analysis: per-aspect sentiment scoring using Day 4 extractions
- Aggregate product-level aspect breakdowns

### Day 6 — Summarization & Text Generation
- Extractive summarization with TextRank
- Abstractive summarization with BART/T5
- Decoding strategy comparison: greedy vs beam search vs nucleus sampling
- Faithfulness evaluation: detecting hallucinated content in generated summaries

### Day 7 — Synthesis & Reference
- End-to-end notebook walking through all techniques with explanations
- NLP method selection cheat sheet: "If I need X, use Y because Z"
- Comparison tables across all approaches with tradeoffs documented

## NLP Concepts Covered

| Concept | Day |
|---|---|
| Tokenization & preprocessing | 1 |
| TF-IDF / bag-of-words | 1 |
| Word2Vec & sentence embeddings | 2 |
| Vector similarity & nearest neighbors | 2 |
| Dimensionality reduction (t-SNE/UMAP) | 2 |
| Text classification (traditional + transformer) | 3 |
| Fine-tuning pretrained models | 3 |
| Named entity recognition | 4 |
| POS tagging & dependency parsing | 4 |
| Information extraction | 4 |
| Topic modeling (LDA + BERTopic) | 5 |
| Aspect-based sentiment analysis | 5 |
| Extractive & abstractive summarization | 6 |
| Decoding strategies | 6 |
| Hallucination detection | 6 |

## Tech Stack
Python · spaCy · Hugging Face Transformers · Gensim · sentence-transformers · scikit-learn · BERTopic · FAISS · pandas · matplotlib · Plotly