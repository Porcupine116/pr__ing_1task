from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CollaborativeFiltering:
    """User-based / Item-based CF на cosine similarity.

    Для простоты используем rating как implicit score.
    """

    mode: str  # "user" or "item"
    user_id_to_index: Dict[int, int]
    item_id_to_index: Dict[int, int]
    index_to_item_id: Dict[int, int]
    matrix: csr_matrix
    similarity: np.ndarray

    @classmethod
    def fit(cls, ratings: pd.DataFrame, mode: str = "item") -> "CollaborativeFiltering":
        assert mode in ("user", "item")

        users = sorted(ratings["user_id"].unique().tolist())
        items = sorted(ratings["item_id"].unique().tolist())
        user_id_to_index = {int(u): i for i, u in enumerate(users)}
        item_id_to_index = {int(it): i for i, it in enumerate(items)}
        index_to_item_id = {i: int(it) for it, i in item_id_to_index.items()}

        row = ratings["user_id"].map(user_id_to_index).to_numpy()
        col = ratings["item_id"].map(item_id_to_index).to_numpy()
        val = ratings["rating"].to_numpy().astype(np.float32)

        mat = csr_matrix((val, (row, col)), shape=(len(users), len(items)))

        if mode == "item":
            sim = cosine_similarity(mat.T)
        else:
            sim = cosine_similarity(mat)

        return cls(
            mode=mode,
            user_id_to_index=user_id_to_index,
            item_id_to_index=item_id_to_index,
            index_to_item_id=index_to_item_id,
            matrix=mat,
            similarity=sim,
        )

    def recommend_for_user(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_id_to_index:
            return []

        uidx = self.user_id_to_index[user_id]
        user_ratings = self.matrix[uidx].toarray().ravel()
        seen = set(np.where(user_ratings > 0)[0].tolist())

        if self.mode == "item":
            scores = np.zeros(self.matrix.shape[1], dtype=np.float32)
            for j in seen:
                scores += self.similarity[j] * float(user_ratings[j])
        else:
            sim_u = self.similarity[uidx]
            scores = sim_u @ self.matrix.toarray()  # учебно

        for j in seen:
            scores[j] = -1e9

        top = np.argsort(-scores)[:k]
        return [(self.index_to_item_id[int(i)], float(scores[int(i)])) for i in top]

