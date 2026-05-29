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
    # We explode the genre column into one row per genre, encode with get_dummies,
    # then sum back to the anime level — one O(n) pass instead of O(n × |genres|)
    # per-genre str.contains scans.
    genre_dummies = (
        anime[["anime_id", "genre"]]
        .assign(genre=anime["genre"].str.split(", "))
        .explode("genre")
        .assign(genre=lambda d: d["genre"].str.strip())
    )
    genre_dummies = pd.get_dummies(genre_dummies.set_index("anime_id")["genre"]).groupby(level=0).max()
    genre_dummies.columns = [f"genre_{g}" for g in genre_dummies.columns]
    all_genres = sorted(genre_dummies.columns.str.replace("genre_", "", regex=False))
    genre_dummies = genre_dummies[[f"genre_{g}" for g in all_genres]].astype(float)
    anime = anime.set_index("anime_id").join(genre_dummies, how="left").reset_index()

    # --- Type: one-hot ---
    type_dummies = pd.get_dummies(anime["type"], prefix="type").astype(float)
    anime = pd.concat([anime, type_dummies], axis=1)

    # --- Numeric features: log-scale then normalise to [0, 1] ---
    # Log-scale tames the huge range in episodes (1–1818) and members (5–1M).
    # Each feature gets its own scaler instance so the fitted min/max for
    # episodes, rating, and members are all preserved independently. A single
    # reused scaler would overwrite its state on each fit_transform call,
    # leaving only the last column's range — wrong scaling at inference time.
    ep_scaler  = MinMaxScaler()
    rat_scaler = MinMaxScaler()
    mem_scaler = MinMaxScaler()
    anime["feat_episodes"] = ep_scaler.fit_transform(
        np.log1p(anime["episodes"].values).reshape(-1, 1)
    )
    anime["feat_rating"] = rat_scaler.fit_transform(
        anime["rating"].values.reshape(-1, 1)
    )
    anime["feat_members"] = mem_scaler.fit_transform(
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
