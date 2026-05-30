import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


def cosine_similarity_matrix(profile: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between a single profile vector and every row in matrix."""
    profile_norm = np.linalg.norm(profile)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = np.dot(matrix, profile) / (matrix_norms * profile_norm)
    return np.nan_to_num(sims)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
)

# ── Load artefacts (cached so they only load once) ────────────────────────────
@st.cache_resource
def load_data():
    anime = pd.read_csv(PROCESSED / "anime_clean.csv")
    with open(PROCESSED / "item_map.pkl", "rb") as f:
        item_map = pickle.load(f)

    sys.path.insert(0, str(ROOT))
    from src.features import build_item_content_features
    item_features, feature_names = build_item_content_features(anime, item_map)

    # Reverse map: item_idx → anime_id
    idx_to_id = {v: k for k, v in item_map.items()}
    anime_lookup = anime.set_index("anime_id")

    return anime, item_map, item_features, feature_names, idx_to_id, anime_lookup

anime_df, item_map, item_features, feature_names, idx_to_id, anime_lookup = load_data()


def build_user_profile(selections: dict) -> np.ndarray:
    """
    Build a taste profile from a dict of {anime_id: rating}.
    Returns a weighted average of item feature vectors.
    """
    profile = np.zeros(item_features.shape[1])
    total_weight = 0.0
    for anime_id, rating in selections.items():
        idx = item_map.get(anime_id)
        if idx is not None:
            profile += rating * item_features[idx]
            total_weight += rating
    if total_weight > 0:
        profile /= total_weight
    return profile


def recommend(profile: np.ndarray, exclude_ids: set, top_k: int = 10) -> pd.DataFrame:
    """Return top_k anime most similar to the taste profile."""
    sims = cosine_similarity_matrix(profile, item_features)
    # Zero out already-watched anime
    for aid in exclude_ids:
        idx = item_map.get(aid)
        if idx is not None:
            sims[idx] = -1.0

    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        aid = idx_to_id.get(idx)
        if aid and aid in anime_lookup.index:
            row = anime_lookup.loc[aid]
            results.append({
                "Name":     row["name"],
                "Genre":    row["genre"],
                "Type":     row["type"],
                "Episodes": int(row["episodes"]) if row["episodes"] > 0 else "?",
                "Rating":   f"{row['rating']:.2f}",
                "Score":    f"{sims[idx]:.3f}",
            })
    return pd.DataFrame(results)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎌 Anime Recommender")
st.caption("Tell us what you've enjoyed — we'll find what to watch next.")

st.markdown("---")

# Session state to hold the user's selections
if "selections" not in st.session_state:
    st.session_state.selections = {}   # {anime_id: rating}

col_left, col_right = st.columns([1, 1.6], gap="large")

# ── Left column: search & rate ────────────────────────────────────────────────
with col_left:
    st.subheader("📋 Your Watch List")
    st.caption("Search for anime you've seen and rate them. The more you add, the better the recommendations.")

    # Searchable dropdown — searches by name
    all_names = anime_df[["anime_id", "name"]].dropna().sort_values("name")
    search_result = st.selectbox(
        "Search for an anime",
        options=[""] + all_names["name"].tolist(),
        format_func=lambda x: x if x else "Type to search...",
    )

    if search_result:
        selected_row = all_names[all_names["name"] == search_result].iloc[0]
        selected_id  = int(selected_row["anime_id"])
        selected_meta = anime_lookup.loc[selected_id]

        st.markdown(f"**{selected_meta['name']}**")
        st.caption(f"*{selected_meta['genre']}* · {selected_meta['type']} · ⭐ {selected_meta['rating']:.2f}")

        rating = st.slider("Your rating", 1, 10, 8, key=f"slider_{selected_id}")

        if st.button("➕ Add to list"):
            st.session_state.selections[selected_id] = rating
            st.success(f"Added **{selected_meta['name']}** (rated {rating}/10)")

    # Show current watch list
    if st.session_state.selections:
        st.markdown("#### Added so far:")
        to_remove = None
        for aid, r in st.session_state.selections.items():
            if aid in anime_lookup.index:
                name = anime_lookup.loc[aid, "name"]
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"⭐ **{r}/10** — {name}")
                if c2.button("✕", key=f"rm_{aid}"):
                    to_remove = aid
        if to_remove:
            del st.session_state.selections[to_remove]
            st.rerun()

        if st.button("🗑️ Clear all"):
            st.session_state.selections = {}
            st.rerun()

# ── Right column: recommendations ────────────────────────────────────────────
with col_right:
    st.subheader("✨ Recommendations")

    if not st.session_state.selections:
        st.info("Add at least one anime on the left to get recommendations.")
    else:
        top_k = st.slider("How many recommendations?", 5, 20, 10)

        profile = build_user_profile(st.session_state.selections)
        recs    = recommend(profile, set(st.session_state.selections.keys()), top_k=top_k)

        if recs.empty:
            st.warning("No recommendations found — try adding more anime.")
        else:
            st.caption(
                f"Based on {len(st.session_state.selections)} anime in your list. "
                "Score = cosine similarity to your taste profile."
            )
            st.dataframe(
                recs,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Match", min_value=0, max_value=1, format="%.3f"
                    ),
                    "Rating": st.column_config.TextColumn("Community Rating"),
                },
            )

            # Show top pick prominently
            top = recs.iloc[0]
            st.markdown("---")
            st.markdown(f"### 🏆 Top pick: **{top['Name']}**")
            st.caption(f"{top['Genre']} · {top['Type']} · {top['Episodes']} episodes · Community rating {top['Rating']}")
