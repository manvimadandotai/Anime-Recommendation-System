import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def load_raw(ratings_file="rating.csv", anime_file="anime.csv"):
    ratings = pd.read_csv(DATA_RAW / ratings_file)
    anime = pd.read_csv(DATA_RAW / anime_file)
    return ratings, anime


def merge_datasets(ratings: pd.DataFrame, anime: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(ratings, anime, on="anime_id", how="left")


def train_test_split(dataset: pd.DataFrame, train_ratio: float = 0.75):
    cutoff = int(train_ratio * len(dataset))
    return dataset[:cutoff], dataset[cutoff:]
