# Policy Uncertainty Index: LLM-Driven Text Analytics & LSTM Forecasting

A pipeline that turns unstructured news coverage of climate and economic policy into a **quantitative uncertainty index**, then forecasts that index using deep learning (LSTM), optionally fused with structured macroeconomic indicators (GDP growth, inflation, emissions).

The project sits at the intersection of **NLP-based economic measurement** (in the spirit of Baker, Bloom & Davis' Economic Policy Uncertainty index) and **time-series deep learning**, and is designed as a template for turning open news APIs into a research-grade panel dataset.

---

## Table of Contents
- [Motivation & Economic Rationale](#motivation--economic-rationale)
- [Analytical Strategy](#analytical-strategy)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Methodology Detail](#methodology-detail)
- [Security Note on API Keys](#security-note-on-api-keys)
- [Limitations & Roadmap](#limitations--roadmap)
- [License](#license)

---

## Motivation & Economic Rationale

Economic and climate policy uncertainty is a recognized driver of firm-level investment delay, capital market volatility, and carbon-pricing risk premia. Text-based uncertainty indices are attractive because they:

- Capture sentiment and salience **before** it shows up in slower-moving macro data (GDP, CPI releases lag by months)
- Are constructible at **high frequency** (daily/weekly) from continuously updated news streams
- Can be **fused with structured indicators** to improve forecasting power over either data type alone

This project operationalizes that idea for the climate–economic policy nexus specifically, rather than general EPU, by targeting a query basket of climate policy, carbon pricing, emissions, and inflation-adjacent terms.

## Analytical Strategy

The pipeline follows a three-stage strategy:

| Stage | Goal | Technique |
|---|---|---|
| 1. Collection | Build a corpus of climate/economic policy articles | Guardian Open Platform API + NewsAPI as a secondary/backup source |
| 2. Scoring | Convert unstructured text into a numeric uncertainty signal | TF-IDF vectorization + cosine similarity against a hand-built "uncertainty" query vector, standardized (z-scored) |
| 3. Forecasting | Predict how uncertainty evolves | Univariate LSTM on the text-derived index, extended to a multivariate LSTM that fuses the index with structured macro features |

This dual-source, dual-model design is intentional: NewsAPI acts as a redundancy/fallback data source if Guardian API coverage or rate limits are insufficient, and the two-model forecasting approach (univariate → multivariate) lets you benchmark whether structured data materially improves on text alone.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Guardian API    │     │   NewsAPI        │   1. Data Collection
│  (primary)       │     │   (secondary)     │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
         Raw article .txt files (title, description, url, body)
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Text preprocessing            │   2. Preprocessing
        │ (lowercase, strip punctuation)│
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ TF-IDF vectorization           │   3. LLM-style semantic
        │ + cosine similarity vs.        │      scoring layer
        │ "uncertainty" query vector     │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Standardized Uncertainty Score │
        │ (uncertainty_scores.csv)       │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Monthly resampling             │   4. Forecasting
        │ + MinMax scaling               │
        └───────────────┬───────────────┘
             ┌───────────┴────────────┐
             ▼                        ▼
   ┌──────────────────┐    ┌───────────────────────┐
   │ Univariate LSTM    │    │ Multivariate LSTM      │
   │ (uncertainty only) │    │ (+ GDP, inflation,     │
   │                    │    │   emissions features)  │
   └──────────────────┘    └───────────────────────┘
```

## Repository Structure

```
.
├── data_collection_guardian.py   # Primary article scraper (Guardian API)
├── data_collection_newsapi.py    # Secondary/backup article scraper (NewsAPI)
├── preprocessing.py              # Text cleaning + TF-IDF feature extraction
├── uncertainty_scoring.py        # Cosine-similarity uncertainty index construction
├── lstm_forecast.py              # Univariate LSTM forecasting
├── lstm_forecast_combined.py     # Multivariate LSTM (text + structured data)
├── guardian_climate_economic_policy_news/  # Raw article corpus (generated)
├── uncertainty_scores.csv        # Generated uncertainty index (generated)
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install requests pandas numpy scikit-learn matplotlib tensorflow
```

**Required environment variables** (see [Security Note](#security-note-on-api-keys) below — do not hardcode these):

```bash
export GUARDIAN_API_KEY="your-guardian-api-key"
export NEWS_API_KEY="your-newsapi-key"
```

Get free keys at [open-platform.theguardian.com](https://open-platform.theguardian.com/) and [newsapi.org](https://newsapi.org/).

## Usage

```bash
# 1. Collect articles (choose one or both sources)
python data_collection_guardian.py
python data_collection_newsapi.py

# 2. Score articles for policy uncertainty
python uncertainty_scoring.py

# 3. Forecast with LSTM (text-only)
python lstm_forecast.py

# 4. Forecast with LSTM fused with structured macro data
python lstm_forecast_combined.py
```

Outputs:
- `guardian_climate_economic_policy_news/*.txt` — raw article corpus
- `uncertainty_scores.csv` — dated, standardized uncertainty index
- Matplotlib plots — score distribution, actual vs. predicted uncertainty

## Methodology Detail

**Query design.** The search basket (`climate policy`, `economic policy`, `uncertainty`, `carbon price`, `carbon emission`, `environment policy`, `inflation`) is deliberately broad to capture cross-cutting climate-economy discourse rather than either topic in isolation, restricted to Guardian sections `environment`, `business`, `climate`, `economy`.

**Semantic scoring ("LLM-style" layer).** Rather than a keyword count, each article is embedded via TF-IDF and scored by cosine similarity to a synthetic "uncertainty" reference vector, then z-scored (`StandardScaler`) so the index is comparable across collection batches. This is a lightweight, interpretable stand-in for a full transformer-embedding approach — swapping in a sentence-embedding model (e.g., a Claude/OpenAI embedding call or `sentence-transformers`) is a natural upgrade path noted in [Roadmap](#limitations--roadmap).

**Time-series construction.** Articles are stamped with a collection date, resampled to monthly frequency, and forward/backward-filled to handle sparse collection days.

**Forecasting.** A 2-layer LSTM (50 units each) with dropout regularization predicts the next uncertainty value from a 12-step (univariate) or 30-step (multivariate) lookback window. The multivariate model concatenates the scaled uncertainty series with normalized GDP growth, inflation, and emissions series before sequencing, letting the network learn cross-series dependencies.

**Evaluation.** Held-out test performance is reported via MSE/RMSE, with actual-vs-predicted plots for visual diagnostic.

## Security Note on API Keys

The original scripts this project is based on had API keys hardcoded directly in source. **Both keys shown in the source snippets should be treated as compromised and rotated immediately** — any key committed to a public repo (even briefly, even in an old commit) should be revoked at the provider dashboard, since git history retains it permanently.

Going forward, load keys from environment variables or a `.env` file (excluded via `.gitignore`), e.g.:

```python
import os
GUARDIAN_API_KEY = os.environ["GUARDIAN_API_KEY"]
NEWS_API_KEY = os.environ["NEWS_API_KEY"]
```

Add a `.gitignore` entry for `.env` and any local output directories containing scraped content if that content shouldn't be public.

## Limitations & Roadmap

- **TF-IDF vs. real embeddings**: current similarity scoring is lexical, not semantic. A transformer/LLM embedding model would better capture paraphrased uncertainty language.
- **Placeholder scoring function**: one snippet in the pipeline computes uncertainty from word count as a placeholder — replace with the TF-IDF cosine score before drawing conclusions from that stage's output.
- **Structured data stubs**: `gdp_growth_data`, `inflation_data`, `emissions_data` are placeholders — wire these to a real source (World Bank API, FRED, OECD) before running the multivariate model.
- **Small corpus risk**: uncertainty indices are noisy at low article-per-month counts; consider a minimum monthly article threshold or a rolling window.
- **Backtesting**: add walk-forward validation rather than a single train/test split for a more robust RMSE estimate.

## License

MIT — see `LICENSE` for details. Guardian and NewsAPI content usage is subject to each provider's terms of service.
