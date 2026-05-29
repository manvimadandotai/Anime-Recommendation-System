import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def load_raw(ratings_file="rating.csv", anime_file="anime.csv"):
    ratings = pd.read_csv(DATA_RAW / ratings_file)
    anime = pd.read_csv(DATA_RAW / anime_file)
    return ratings, anime


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    # Drop unrated rows (-1 means watched but not rated)
    return ratings[ratings["rating"] != -1].copy()


def clean_anime(anime: pd.DataFrame) -> pd.DataFrame:
    anime = anime.copy()
    anime["genre"] = anime["genre"].fillna("Unknown")
    anime["type"] = anime["type"].fillna("Unknown")
    # episodes can be "Unknown" string — coerce to numeric, fill missing with 0
    anime["episodes"] = pd.to_numeric(anime["episodes"], errors="coerce").fillna(0).astype(int)
    # fill missing community rating with median
    anime["rating"] = anime["rating"].fillna(anime["rating"].median())
    return anime


def reindex(ratings: pd.DataFrame):
    """Remap user_id and anime_id to contiguous 0-based integer indices."""
    user_map = {uid: idx for idx, uid in enumerate(ratings["user_id"].unique())}
    item_map = {aid: idx for idx, aid in enumerate(ratings["anime_id"].unique())}
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["user_id"].map(user_map)
    ratings["item_idx"] = ratings["anime_id"].map(item_map)
    return ratings, user_map, item_map


def merge_datasets(ratings: pd.DataFrame, anime: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(ratings, anime, on="anime_id", how="left")


def train_test_split(dataset: pd.DataFrame, train_ratio: float = 0.75):
    cutoff = int(train_ratio * len(dataset))
    return dataset[:cutoff].copy(), dataset[cutoff:].copy()
