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
