from __future__ import annotations

"""Оффлайн оценка рекомендателей на MovieLens 100k.

Запуск:
    py se_series3/recsys/evaluation.py

Скрипт:
- скачивает датасет (если нужно)
- делает train/test split per-user
- обучает несколько стратегий
- считает Precision@K, Recall@K, MAP@K, NDCG@K (на implicit релевантности rating>=4)

Это учебный протокол (не максимально оптимальный).
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from se_series3.recsys.content_based import ContentBasedRecommender
from se_series3.recsys.collaborative import CollaborativeFiltering
from se_series3.recsys.data_utils import download_movielens_100k, load_movielens_100k, train_test_split_per_user
from se_series3.recsys.heuristics import PopularRecommender
from se_series3.recsys.hybrid import HybridRecommender
from se_series3.recsys.two_tower import TwoTowerRecommender


def precision_recall_at_k(recommended: List[int], relevant: set[int], k: int) -> Tuple[float, float]:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0, 0.0
    hits = sum(1 for x in rec_k if x in relevant)
    precision = hits / k
    recall = hits / max(1, len(relevant))
    return precision, recall


def average_precision_at_k(recommended: List[int], relevant: set[int], k: int) -> float:
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / max(1, min(len(relevant), k))


def ndcg_at_k(recommended: List[int], relevant: set[int], k: int) -> float:
    def dcg(seq: List[int]) -> float:
        s = 0.0
        for i, it in enumerate(seq[:k], start=1):
            rel = 1.0 if it in relevant else 0.0
            s += (2**rel - 1) / math.log2(i + 1)
        return s

    ideal = [1] * min(len(relevant), k)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, len(ideal) + 1))
    if idcg == 0:
        return 0.0
    return dcg(recommended) / idcg


def evaluate_strategy(
    name: str,
    recommend_fn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> Dict[str, float]:
    precisions, recalls, maps, ndcgs = [], [], [], []

    # relevant: rating>=4 в test
    test_pos = test[test["rating"] >= 4]

    # История пользователя берём из train
    train_items_by_user = train.groupby("user_id")["item_id"].apply(list).to_dict()

    for user_id, grp in test_pos.groupby("user_id"):
        relevant = set(int(x) for x in grp["item_id"].tolist())
        history = [int(x) for x in train_items_by_user.get(user_id, [])]

        recs = recommend_fn(int(user_id), history, k)
        rec_items = [int(i) for i, _ in recs]

        p, r = precision_recall_at_k(rec_items, relevant, k)
        ap = average_precision_at_k(rec_items, relevant, k)
        nd = ndcg_at_k(rec_items, relevant, k)

        precisions.append(p)
        recalls.append(r)
        maps.append(ap)
        ndcgs.append(nd)

    return {
        "strategy": name,
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"

    ml_root = download_movielens_100k(data_dir)
    data = load_movielens_100k(ml_root)

    train, test = train_test_split_per_user(data.ratings, test_ratio=0.2)

    # Fit models
    content = ContentBasedRecommender.fit(data.items)
    item_cf = CollaborativeFiltering.fit(train, mode="item")
    user_cf = CollaborativeFiltering.fit(train, mode="user")
    popular = PopularRecommender.fit(train)
    two_tower = TwoTowerRecommender.fit(train, epochs=1, dim=32)

    hybrid = HybridRecommender(w_content=0.5, w_cf=0.5)

    def rec_content(user_id: int, history: List[int], k: int):
        return content.recommend(history, k=k)

    def rec_item_cf(user_id: int, history: List[int], k: int):
        return item_cf.recommend_for_user(user_id, k=k)

    def rec_user_cf(user_id: int, history: List[int], k: int):
        return user_cf.recommend_for_user(user_id, k=k)

    def rec_popular(user_id: int, history: List[int], k: int):
        return popular.recommend(k=k, exclude=history)

    def rec_two_tower(user_id: int, history: List[int], k: int):
        return two_tower.recommend(user_id, seen_item_ids=history, k=k)

    def rec_hybrid(user_id: int, history: List[int], k: int):
        a = rec_content(user_id, history, k=50)
        b = rec_item_cf(user_id, history, k=50)
        return hybrid.combine(a, b, k=k)

    strategies = {
        "popular": rec_popular,
        "content_tfidf": rec_content,
        "item_cf": rec_item_cf,
        "user_cf": rec_user_cf,
        "hybrid": rec_hybrid,
        "two_tower": rec_two_tower,
    }

    rows = []
    for name, fn in strategies.items():
        rows.append(evaluate_strategy(name, fn, train, test, k=10))

    df = pd.DataFrame(rows).sort_values(by="ndcg@10", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

