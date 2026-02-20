from __future__ import annotations

"""Two-Tower модель (учебная версия).

Архитектура:
- user tower: embedding(user_id)
- item tower: embedding(item_id)
- score = dot(user_emb, item_emb)

Обучение:
- implicit feedback: положительные пары (user, item) из train
- negative sampling: случайные айтемы
- loss: BCEWithLogitsLoss

Важно:
- Это демонстрационный код для понимания концепции.
- Для production обычно используют батчинг, ускорение, ANN (FAISS) для retrieval.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, labels: np.ndarray):
        self.pairs = pairs.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        u, i = self.pairs[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(self.labels[idx])


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        return (ue * ie).sum(dim=-1)  # dot product


@dataclass
class TwoTowerRecommender:
    user_id_to_index: Dict[int, int]
    item_id_to_index: Dict[int, int]
    index_to_item_id: Dict[int, int]
    model: TwoTower

    @classmethod
    def fit(cls, ratings: pd.DataFrame, epochs: int = 2, dim: int = 64, batch_size: int = 1024, seed: int = 42) -> "TwoTowerRecommender":
        torch.manual_seed(seed)
        np.random.seed(seed)

        users = sorted(ratings["user_id"].unique().tolist())
        items = sorted(ratings["item_id"].unique().tolist())
        user_id_to_index = {int(u): i for i, u in enumerate(users)}
        item_id_to_index = {int(it): i for i, it in enumerate(items)}
        index_to_item_id = {i: int(it) for it, i in item_id_to_index.items()}

        # positive pairs
        pos = ratings[["user_id", "item_id"]].copy()
        pos["u"] = pos["user_id"].map(user_id_to_index)
        pos["i"] = pos["item_id"].map(item_id_to_index)
        pos_pairs = pos[["u", "i"]].to_numpy()

        # negative sampling: столько же нег. примеров
        n = len(pos_pairs)
        neg_u = pos_pairs[:, 0]
        neg_i = np.random.randint(0, len(items), size=n)
        neg_pairs = np.stack([neg_u, neg_i], axis=1)

        pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
        labels = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=0)

        ds = PairDataset(pairs, labels)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = TwoTower(n_users=len(users), n_items=len(items), dim=dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(epochs):
            for u, i, y in dl:
                opt.zero_grad()
                logits = model(u, i)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        model.eval()
        return cls(user_id_to_index=user_id_to_index, item_id_to_index=item_id_to_index, index_to_item_id=index_to_item_id, model=model)

    def recommend(self, user_id: int, seen_item_ids: List[int], k: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_id_to_index:
            return []

        uidx = self.user_id_to_index[user_id]
        seen = {self.item_id_to_index[i] for i in seen_item_ids if i in self.item_id_to_index}

        # Считаем скоры по всем items (для MovieLens 1682 это ок)
        all_items = torch.arange(0, len(self.item_id_to_index), dtype=torch.long)
        u = torch.full_like(all_items, fill_value=uidx)

        with torch.no_grad():
            scores = self.model(u, all_items).numpy()

        for j in seen:
            scores[j] = -1e9

        top = np.argsort(-scores)[:k]
        return [(self.index_to_item_id[int(i)], float(scores[int(i)])) for i in top]

