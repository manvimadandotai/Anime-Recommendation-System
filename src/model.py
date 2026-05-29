import numpy as np
import pandas as pd
from scipy import sparse
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
