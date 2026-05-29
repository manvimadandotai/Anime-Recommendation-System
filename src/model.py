import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error



# ── Evaluation ────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ── Baseline models ───────────────────────────────────────────────────────────

class GlobalMeanModel:
    """Predict the global mean rating for every user-item pair."""

    def fit(self, train: pd.DataFrame):
        self.global_mean = train["rating_x"].mean()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.full(len(test), self.global_mean)


class ItemMeanModel:
    """Predict each item's mean rating; fall back to global mean for unseen items."""

    def fit(self, train: pd.DataFrame):
        self.global_mean = train["rating_x"].mean()
        self.item_means = train.groupby("item_idx")["rating_x"].mean()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["item_idx"].map(self.item_means).fillna(self.global_mean).values


class UserMeanModel:
    """Predict each user's mean rating; fall back to global mean for unseen users."""

    def fit(self, train: pd.DataFrame):
        self.global_mean = train["rating_x"].mean()
        self.user_means = train.groupby("user_idx")["rating_x"].mean()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["user_idx"].map(self.user_means).fillna(self.global_mean).values


class BiasModel:
    """
    User + item bias model:
        pred(u, i) = global_mean + user_bias(u) + item_bias(i)

    Biases are estimated by alternating least squares for `n_iter` rounds.
    """

    def __init__(self, n_iter: int = 10, reg: float = 0.1):
        self.n_iter = n_iter
        self.reg = reg

    def fit(self, train: pd.DataFrame):
        self.global_mean = train["rating_x"].mean()
        residuals = train["rating_x"] - self.global_mean

        user_bias = np.zeros(train["user_idx"].max() + 1)
        item_bias = np.zeros(train["item_idx"].max() + 1)

        for _ in range(self.n_iter):
            # Update user biases
            r_minus_item = residuals - item_bias[train["item_idx"].values]
            df = pd.DataFrame({"u": train["user_idx"].values, "r": r_minus_item})
            counts = df.groupby("u")["r"].count()
            sums = df.groupby("u")["r"].sum()
            user_bias = (sums / (counts + self.reg)).reindex(
                range(len(user_bias)), fill_value=0
            ).values

            # Update item biases
            r_minus_user = residuals - user_bias[train["user_idx"].values]
            df = pd.DataFrame({"i": train["item_idx"].values, "r": r_minus_user})
            counts = df.groupby("i")["r"].count()
            sums = df.groupby("i")["r"].sum()
            item_bias = (sums / (counts + self.reg)).reindex(
                range(len(item_bias)), fill_value=0
            ).values

        self.user_bias = user_bias
        self.item_bias = item_bias
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        u = test["user_idx"].values
        i = test["item_idx"].values
        u_safe = np.clip(u, 0, len(self.user_bias) - 1)
        i_safe = np.clip(i, 0, len(self.item_bias) - 1)
        u_bias = np.where(u < len(self.user_bias), self.user_bias[u_safe], 0)
        i_bias = np.where(i < len(self.item_bias), self.item_bias[i_safe], 0)
        return np.clip(self.global_mean + u_bias + i_bias, 1, 10)


# ── Matrix Factorisation ──────────────────────────────────────────────────────

class SVDModel:
    """
    Truncated SVD collaborative filtering.

    Decomposes the bias-corrected interaction matrix into k latent factors
    (think: hidden "taste dimensions" shared between users and anime).
    Predictions are the dot product of user and item latent vectors,
    plus the bias terms added back on top.

    Parameters
    ----------
    n_factors : int
        Number of latent dimensions to learn (more = more expressive, slower).
    n_iter : int
        ALS iterations used to compute the bias terms before factorisation.
    reg : float
        Regularisation strength for bias estimation.
    """

    def __init__(self, n_factors: int = 50, n_iter: int = 10, reg: float = 0.1):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.reg = reg

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        # Step 1: fit a bias model first so SVD learns from the *residuals*
        # (what's left after removing each user's and item's average tendencies)
        self._bias = BiasModel(n_iter=self.n_iter, reg=self.reg).fit(train)
        self.global_mean = self._bias.global_mean

        # Step 2: compute residuals — the part that bias alone can't explain
        bias_preds = self._bias.predict(train)
        residuals = train["rating_x"].values - bias_preds

        # Step 3: build a sparse matrix of those residuals (users × anime)
        R = sparse.csr_matrix(
            (residuals, (train["user_idx"].values, train["item_idx"].values)),
            shape=(n_users, n_items),
        )

        # Step 4: decompose R into U, S, Vt using Truncated SVD
        # U  shape (n_users, k) — each user as a k-dimensional taste vector
        # s  shape (k,)         — importance of each latent dimension
        # Vt shape (k, n_items) — each anime as a k-dimensional characteristic vector
        k = min(self.n_factors, min(R.shape) - 1)
        U, s, Vt = svds(R, k=k)

        # Absorb the singular values into U so prediction is just U @ Vt
        self.U = U * s[np.newaxis, :]
        self.Vt = Vt

        self.n_users = n_users
        self.n_items = n_items
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        u = test["user_idx"].values
        i = test["item_idx"].values

        # Dot product of user and item latent vectors = collaborative signal
        cf_score = np.array([
            self.U[uu] @ self.Vt[:, ii]
            if uu < self.n_users and ii < self.n_items else 0.0
            for uu, ii in zip(u, i)
        ])

        # Add back the bias terms — final prediction is bias + latent signal
        bias_preds = self._bias.predict(test)
        return np.clip(bias_preds + cf_score, 1, 10)


# ── Content-Based Filtering ───────────────────────────────────────────────────

class ContentBasedModel:
    """
    Content-based recommendation using anime metadata.

    How it works:
      1. Represent every anime as a feature vector (genres, type, episodes, rating, popularity).
      2. For each user, build a "taste profile" — the average feature vector of all
         anime they've rated, weighted by how much they liked each one.
         (High-rated anime pull the profile toward their features more than low-rated ones.)
      3. To predict a rating, measure how similar the candidate anime is to the user's
         taste profile using cosine similarity, then scale that into the 1–10 rating range.

    This approach only uses *what* the anime is, not *who else* watched it.
    It's useful for cold-start (new anime with no ratings yet) and as a complement to
    collaborative filtering (which only uses *who* watched what).
    """

    def fit(self, train: pd.DataFrame, item_features: np.ndarray):
        """
        item_features : (n_items, n_features) matrix from build_item_content_features()
        """
        self.item_features = item_features
        self.global_mean = train["rating_x"].mean()

        n_users = train["user_idx"].max() + 1
        n_items = item_features.shape[0]

        # Build each user's taste profile as a rating-weighted average of item features.
        # We use sparse matrix multiplication instead of a Python loop —
        # this does the same thing but ~1000x faster on millions of rows.
        #
        # Step 1: build a sparse (n_users × n_items) matrix of ratings
        R = sparse.csr_matrix(
            (train["rating_x"].values, (train["user_idx"].values, train["item_idx"].values)),
            shape=(n_users, n_items),
        )

        # Step 2: multiply R by the item feature matrix
        # Result shape: (n_users, n_features)
        # Each row is the sum of rating-weighted feature vectors for that user
        user_profiles = R.dot(item_features)

        # Step 3: divide each user's profile by their total rating weight
        # so we get a weighted *average*, not a weighted sum
        weight_sums = np.array(R.sum(axis=1)).flatten()
        nonzero = weight_sums > 0
        user_profiles[nonzero] /= weight_sums[nonzero, np.newaxis]

        self.user_profiles = user_profiles
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        u = test["user_idx"].values
        i = test["item_idx"].values

        n_users, n_items = len(self.user_profiles), len(self.item_features)

        # Clamp out-of-range indices so we can index safely, then mask them out
        u_safe = np.clip(u, 0, n_users - 1)
        i_safe = np.clip(i, 0, n_items - 1)

        profiles = self.user_profiles[u_safe]    # (n_test, n_features)
        items    = self.item_features[i_safe]    # (n_test, n_features)

        # Cosine similarity — row-wise dot product divided by the product of norms
        # Ranges from -1 (opposite) to 1 (identical); 0 = orthogonal / no overlap
        dot    = np.einsum("ij,ij->i", profiles, items)
        norms  = np.linalg.norm(profiles, axis=1) * np.linalg.norm(items, axis=1)
        # Avoid division by zero for users with no profile
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(norms > 0, dot / norms, 0.0)

        # Scale cosine similarity [-1, 1] → rating scale [1, 10]
        preds = 1 + (sim + 1) / 2 * 9

        # Fall back to global mean for out-of-range or zero-profile users
        oob = (u >= n_users) | (i >= n_items)
        no_profile = np.array(self.user_profiles[u_safe].sum(axis=1)).flatten() == 0
        preds[oob | no_profile] = self.global_mean

        return np.clip(preds, 1, 10)
