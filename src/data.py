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


def train_val_test_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
):
    """
    Split dataset into train, validation, and test sets.

    Shuffles before splitting so the cut is not biased by row order
    (the raw data is sorted by user_id, so an unshuffled split would put
    some users entirely in train and others entirely in test).

    The fixed random_state ensures the same split every run, which is
    required for reproducible model comparisons.

    Ratios: 70% train / 15% val / 15% test (industry standard for rating prediction).
    - Train  — used to fit all models
    - Val    — used to tune hyperparameters (pick k for SVD, reg for bias model, etc.)
    - Test   — touched only once at the end to report final numbers
    """
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(dataset)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    train = dataset[:train_end].copy()
    val   = dataset[train_end:val_end].copy()
    test  = dataset[val_end:].copy()
    return train, val, test
