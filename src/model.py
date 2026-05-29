import numpy as np
import tensorrec


def build_model(n_components: int = 5) -> tensorrec.TensorRec:
    return tensorrec.TensorRec(n_components=n_components)


def train(model: tensorrec.TensorRec, interactions, user_features, item_features, epochs: int = 5):
    model.fit(
        interactions=interactions,
        user_features=user_features,
        item_features=item_features,
        epochs=epochs,
        verbose=True,
    )
    return model


def evaluate(model: tensorrec.TensorRec, interactions, user_features, item_features, k: int = 10) -> float:
    predicted_ranks = model.predict_rank(
        user_features=user_features,
        item_features=item_features,
    )
    r_at_k = tensorrec.eval.recall_at_k(predicted_ranks, interactions, k=k)
    return float(np.mean(r_at_k))
