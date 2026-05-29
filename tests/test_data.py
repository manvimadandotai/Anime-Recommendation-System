import pandas as pd
from src.data import merge_datasets, train_test_split


def test_merge_keeps_rating_row_count():
    ratings = pd.DataFrame({"user_id": [1, 2], "anime_id": [10, 20], "rating": [8, 7]})
    anime = pd.DataFrame({"anime_id": [10, 20], "name": ["A", "B"]})
    merged = merge_datasets(ratings, anime)
    assert len(merged) == len(ratings)


def test_train_test_split_ratio():
    df = pd.DataFrame({"x": range(100)})
    train, test = train_test_split(df, train_ratio=0.75)
    assert len(train) == 75
    assert len(test) == 25
