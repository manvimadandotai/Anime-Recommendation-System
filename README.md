# Anime Recommendation System

Because "just watch whatever" is not a personality.

This project builds a recommendation engine that actually understands your taste — not just what's trending, not just what has the most ratings, but what *you* are likely to enjoy based on your history and the patterns of people who watch like you.

## What it does

Takes 7.8 million user–anime interactions and figures out who should watch what next. Starts simple, gets smarter.

## Project Structure

```
├── data/
│   ├── raw/          # Original datasets (not tracked — too big, too raw)
│   └── processed/    # Cleaned, split, and matrix-ready artefacts
├── notebooks/
│   ├── 01_eda.ipynb              # Getting to know the data
│   ├── 02_preprocessing.ipynb    # Making it model-ready
│   └── 03_modelling.ipynb        # Where the magic happens
├── src/
│   ├── data.py       # Load, clean, merge, split
│   ├── features.py   # Sparse matrix construction
│   └── model.py      # Baselines through advanced models
├── tests/
│   └── test_data.py
├── requirements.txt
└── README.md
```

## Quickstart — Run the Recommender

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the data from Kaggle and place the files here:
#    data/raw/rating.csv
#    data/raw/anime.csv
#    (link below under "The Data")

# 3. Run the preprocessing pipeline to generate the processed artefacts
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb

# 4. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Search for anime you've watched, rate them, and get instant personalised recommendations.

## How the App Works

1. **Search** for any anime by name from the full 12k catalogue
2. **Rate** it 1–10 — higher ratings pull your taste profile more strongly toward that anime's features
3. **Add more** to refine your profile — the recommendations update in real time
4. **Get your top picks** — ranked by how closely they match the genre, type, and style of what you've liked

No account needed. No training required. Your taste profile is built on the fly from whatever you add.

## Setup

```bash
pip install -r requirements.txt
```

Download the data from [Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database) and place `rating.csv` and `anime.csv` in `data/raw/`.

## The Data

- **7.8M interactions** between users and anime titles
- **12,294 anime** with genre, type, episode count, and community ratings
- Source: [MyAnimeList via Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database)
