from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class PopularRecommender:
    popular_items: List[int]

    @classmethod
    def fit(cls, ratings: pd.DataFrame, top_n: int = 100) -> "PopularRecommender":
        # Популярность по количеству рейтингов
        popular = (
            ratings.groupby("item_id")["rating"].count().sort_values(ascending=False).head(top_n).index.tolist()
        )
        return cls(popular_items=[int(i) for i in popular])

    def recommend(self, k: int = 10, exclude: List[int] | None = None) -> List[Tuple[int, float]]:
        exclude_set = set(exclude or [])
        out = []
        for it in self.popular_items:
            if it in exclude_set:
                continue
            out.append((it, 1.0))
            if len(out) >= k:
                break
        return out


@dataclass
class SameGenreRecommender:
    items: pd.DataFrame

    def recommend(self, liked_item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        row = self.items[self.items["item_id"] == liked_item_id]
        if row.empty:
            return []

        genres = row.iloc[0]["genres_text"]
        # Простая эвристика: тот же жанр
        candidates = self.items[self.items["genres_text"] == genres].head(k)
        return [(int(i), 1.0) for i in candidates["item_id"].tolist() if int(i) != int(liked_item_id)][:k]

