# Anime Recommendation System

A collaborative filtering recommendation system for anime using the [TensorRec](https://github.com/jfkirk/tensorrec) framework.

## Project Structure

```
├── data/
│   ├── raw/          # Original rating.csv and anime.csv (not tracked in git)
│   └── processed/    # Cleaned and merged outputs
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Data merging and feature engineering
│   └── 03_modelling.ipynb        # Model training and evaluation
├── src/
│   ├── data.py       # Data loading, merging, train/test split
│   ├── features.py   # Sparse matrix construction
│   └── model.py      # TensorRec model build, train, evaluate
├── tests/
│   └── test_data.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place `rating.csv` and `anime.csv` in `data/raw/` before running notebooks.

## Data

- **rating.csv** — user–anime ratings from [MyAnimeList dataset on Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database)
- **anime.csv** — anime metadata (name, genre, type, episodes, rating, members)

## References

- [TensorRec GitHub](https://github.com/jfkirk/tensorrec)
- [A Recommendation Engine Framework in TensorFlow](https://medium.com/hackernoon/tensorrec-a-recommendation-engine-framework-in-tensorflow-d85e4f0874e8)
- [Getting Started with Recommender Systems and TensorRec](https://towardsdatascience.com/getting-started-with-recommender-systems-and-tensorrec-8f50a9943eef)
