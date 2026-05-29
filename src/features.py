import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler


def build_sparse_interactions(dataset: pd.DataFrame, n_users: int, n_items: int) -> sparse.coo_matrix:
    user_ids = dataset["user_idx"].values
    item_ids = dataset["item_idx"].values
    ratings = dataset["rating_x"].values
    return sparse.coo_matrix((ratings, (user_ids, item_ids)), shape=(n_users, n_items))


def build_indicator_features(n_users: int, n_items: int):
    user_features = sparse.identity(n_users)
    item_features = sparse.identity(n_items)
    return user_features, item_features


def build_item_content_features(anime: pd.DataFrame, item_map: dict) -> np.ndarray:
    """
    Build a content feature matrix for anime, ordered by item_idx.

    Each row is one anime represented as a vector of:
      - Multi-hot genre encoding  (one column per unique genre)
      - One-hot type encoding     (TV, Movie, OVA, etc.)
      - Normalised log(episodes)  (captures short vs long series)
      - Normalised community rating
      - Normalised log(members)   (popularity signal)

    Returns a dense matrix of shape (n_items, n_features).
    """
    anime = anime.copy()

    # --- Genre: multi-hot ---
    # Each anime can have multiple genres (comma-separated).
    # We create one binary column per genre.
    all_genres = sorted(
        set(g.strip() for genres in anime["genre"].dropna() for g in genres.split(","))
    )
    for genre in all_genres:
        anime[f"genre_{genre}"] = anime["genre"].str.contains(genre, regex=False).astype(float)

    # --- Type: one-hot ---
    type_dummies = pd.get_dummies(anime["type"], prefix="type").astype(float)
    anime = pd.concat([anime, type_dummies], axis=1)

    # --- Numeric features: log-scale then normalise to [0, 1] ---
    # Log-scale tames the huge range in episodes (1–1818) and members (5–1M).
    scaler = MinMaxScaler()
    anime["feat_episodes"] = scaler.fit_transform(
        np.log1p(anime["episodes"].values).reshape(-1, 1)
    )
    anime["feat_rating"] = scaler.fit_transform(
        anime["rating"].values.reshape(-1, 1)
    )
    anime["feat_members"] = scaler.fit_transform(
        np.log1p(anime["members"].values).reshape(-1, 1)
    )

    # --- Assemble feature columns ---
    genre_cols = [f"genre_{g}" for g in all_genres]
    type_cols = [c for c in anime.columns if c.startswith("type_")]
    numeric_cols = ["feat_episodes", "feat_rating", "feat_members"]
    feature_cols = genre_cols + type_cols + numeric_cols

    # --- Order rows by item_idx so row i = anime with item_idx i ---
    idx_to_anime_id = {v: k for k, v in item_map.items()}
    ordered_ids = [idx_to_anime_id[i] for i in range(len(item_map))]
    anime_ordered = anime.set_index("anime_id").reindex(ordered_ids)

    return anime_ordered[feature_cols].fillna(0).values, feature_cols
