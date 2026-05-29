import pandas as pd
from scipy import sparse


def build_sparse_interactions(dataset: pd.DataFrame, n_users: int, n_items: int) -> sparse.coo_matrix:
    user_ids = dataset["user_idx"].values
    item_ids = dataset["item_idx"].values
    ratings = dataset["rating_x"].values
    return sparse.coo_matrix((ratings, (user_ids, item_ids)), shape=(n_users, n_items))


def build_indicator_features(n_users: int, n_items: int):
    user_features = sparse.identity(n_users)
    item_features = sparse.identity(n_items)
    return user_features, item_features
