from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class HybridRecommender:
    """Линейная комбинация результатов двух рекомендателей.

    score = w1 * score1 + w2 * score2
    """

    w_content: float = 0.5
    w_cf: float = 0.5

    def combine(
        self,
        content_recs: List[Tuple[int, float]],
        cf_recs: List[Tuple[int, float]],
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}

        for item_id, s in content_recs:
            scores[item_id] = scores.get(item_id, 0.0) + self.w_content * float(s)
        for item_id, s in cf_recs:
            scores[item_id] = scores.get(item_id, 0.0) + self.w_cf * float(s)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

