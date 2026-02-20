from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentBasedRecommender:
    """Content-based recommender (TF-IDF + cosine).

    Item profile строится по тексту: title + genres.
    User profile — как усреднение TF-IDF векторов просмотренных/оценённых фильмов.
    """

    vectorizer: TfidfVectorizer
    item_matrix: np.ndarray
    item_id_to_index: Dict[int, int]
    index_to_item_id: Dict[int, int]

    @classmethod
    def fit(cls, items: pd.DataFrame) -> "ContentBasedRecommender":
        texts = (items["title"].fillna("") + " " + items["genres_text"].fillna("")).tolist()
        vectorizer = TfidfVectorizer(stop_words=None, max_features=20000)
        item_matrix = vectorizer.fit_transform(texts)

        item_id_to_index = {int(i): idx for idx, i in enumerate(items["item_id"].tolist())}
        index_to_item_id = {idx: int(i) for idx, i in enumerate(items["item_id"].tolist())}
        return cls(vectorizer=vectorizer, item_matrix=item_matrix, item_id_to_index=item_id_to_index, index_to_item_id=index_to_item_id)

    def recommend(self, user_history_item_ids: List[int], k: int = 10) -> List[Tuple[int, float]]:
        history = [self.item_id_to_index[i] for i in user_history_item_ids if i in self.item_id_to_index]
        if not history:
            return []

        user_vec = self.item_matrix[history].mean(axis=0)
        sims = cosine_similarity(user_vec, self.item_matrix).ravel()

        # Не рекомендуем то, что уже в истории
        for idx in history:
            sims[idx] = -1.0

        top_idx = np.argsort(-sims)[:k]
        return [(self.index_to_item_id[int(i)], float(sims[int(i)])) for i in top_idx]

