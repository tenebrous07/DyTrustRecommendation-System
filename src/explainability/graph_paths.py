"""Graph evidence retrieval (Module 16, retrieval half).

Explanations must be *found*, not composed. This module retrieves the concrete
graph facts that connect a user to a recommended item; the templating in
:mod:`src.explainability.explanations` may only reference what is returned
here. A template with no supporting evidence is dropped rather than filled with
something plausible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

__all__ = ["Evidence", "EvidenceRetriever"]

logger = get_logger(__name__)


@dataclass
class Evidence:
    """One retrieved, verifiable fact."""

    kind: str
    detail: str
    strength: float
    support: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "strength": round(float(self.strength), 4), "support": self.support}


class EvidenceRetriever:
    """Pulls category, brand, peer, recency and semantic evidence from the data."""

    def __init__(
        self,
        train: pd.DataFrame,
        products: pd.DataFrame | None,
        vocabulary,
        trust_map: dict[str, float] | None = None,
        item_embeddings: np.ndarray | None = None,
        recent_window: int = 10,
    ) -> None:
        self.train = train
        self.vocab = vocabulary
        self.trust_map = trust_map or {}
        self.item_embeddings = item_embeddings
        self.recent_window = recent_window

        self.by_user = {
            str(user): group.sort_values("timestamp")
            for user, group in train.groupby("user_id")
        }
        self.item_users: dict[str, set[str]] = (
            train.groupby("item_id")["user_id"].apply(lambda s: set(s.astype(str))).to_dict()
        )
        self.item_category: dict[str, str] = {}
        self.item_brand: dict[str, str] = {}
        self.item_title: dict[str, str] = {}
        if products is not None and not products.empty:
            indexed = products.set_index("item_id")
            for column, target in (("category_node", self.item_category),
                                   ("brand", self.item_brand),
                                   ("product_title", self.item_title)):
                if column in indexed.columns:
                    target.update(indexed[column].dropna().astype(str).to_dict())

        self.item_sentiment: dict[str, float] = {}
        if "sentiment_compound" in train.columns:
            self.item_sentiment = train.groupby("item_id")["sentiment_compound"].mean().to_dict()

    # -- per-user summaries -------------------------------------------------
    def user_categories(self, user_id: str, recent_only: bool = False) -> Counter:
        history = self.by_user.get(str(user_id))
        if history is None:
            return Counter()
        if recent_only:
            history = history.tail(self.recent_window)
        return Counter(
            self.item_category[i] for i in history["item_id"].astype(str)
            if i in self.item_category
        )

    def user_brands(self, user_id: str) -> Counter:
        history = self.by_user.get(str(user_id))
        if history is None:
            return Counter()
        return Counter(
            self.item_brand[i] for i in history["item_id"].astype(str)
            if i in self.item_brand
        )

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, user_id: str, item_id: str, max_items: int = 6) -> list[Evidence]:
        """Collect every supported piece of evidence linking user to item."""
        user_id, item_id = str(user_id), str(item_id)
        found: list[Evidence] = []
        history = self.by_user.get(user_id)
        if history is None or history.empty:
            return found

        # 1. Shared category, weighted by how much of the user's history it is.
        category = self.item_category.get(item_id)
        if category:
            counts = self.user_categories(user_id)
            total = sum(counts.values())
            if total and counts.get(category, 0):
                share = counts[category] / total
                found.append(Evidence(
                    "shared_category",
                    f"{counts[category]} of the user's {total} past items are in "
                    f"'{category}', which is this product's category",
                    share, {"category": category, "user_items_in_category": counts[category],
                            "user_items_total": total},
                ))

        # 2. Recent behaviour in that category - stronger than lifetime share,
        #    and the thing temporal weighting is meant to exploit.
        if category:
            recent = self.user_categories(user_id, recent_only=True)
            recent_total = sum(recent.values())
            if recent_total and recent.get(category, 0):
                share = recent[category] / recent_total
                found.append(Evidence(
                    "recent_interaction",
                    f"{recent[category]} of the user's last {recent_total} interactions "
                    f"are in '{category}'",
                    share, {"category": category, "window": self.recent_window},
                ))

        # 3. Shared brand.
        brand = self.item_brand.get(item_id)
        if brand:
            counts = self.user_brands(user_id)
            total = sum(counts.values())
            if total and counts.get(brand, 0):
                found.append(Evidence(
                    "shared_brand",
                    f"the user has bought {counts[brand]} item(s) from '{brand}' before",
                    counts[brand] / total,
                    {"brand": brand, "user_items_from_brand": counts[brand]},
                ))

        # 4. High-trust peers who also bought it.
        buyers = self.item_users.get(item_id, set())
        if buyers:
            user_items = set(history["item_id"].astype(str))
            peers = []
            for peer in buyers:
                if peer == user_id:
                    continue
                peer_history = self.by_user.get(peer)
                if peer_history is None:
                    continue
                overlap = len(user_items & set(peer_history["item_id"].astype(str)))
                if overlap:
                    peers.append((peer, overlap, self.trust_map.get(peer, 0.5)))
            if peers:
                peers.sort(key=lambda p: (-p[2], -p[1]))
                top = peers[:5]
                mean_trust = float(np.mean([t for _, _, t in top]))
                found.append(Evidence(
                    "similar_trusted_users",
                    f"{len(peers)} user(s) with overlapping history also bought this; "
                    f"the {len(top)} closest have mean trust {mean_trust:.2f}",
                    mean_trust,
                    {"n_peers": len(peers), "mean_trust_of_top_peers": round(mean_trust, 4),
                     "top_peer_overlap": [{"user_id": p, "shared_items": o,
                                           "trust": round(t, 4)} for p, o, t in top]},
                ))

        # 5. Semantic similarity of the item to the user's recent items.
        if self.item_embeddings is not None:
            target = self.vocab.items.get(item_id)
            recent_indices = [
                self.vocab.items[i] for i in history.tail(self.recent_window)["item_id"].astype(str)
                if i in self.vocab.items
            ]
            if target is not None and recent_indices:
                matrix = self.item_embeddings[recent_indices]
                query = self.item_embeddings[target]
                norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) + 1e-12)
                sims = (matrix @ query) / np.clip(norms, 1e-12, None)
                mean_sim = float(np.mean(sims))
                if mean_sim > 0:
                    found.append(Evidence(
                        "semantic_similarity",
                        f"mean cosine similarity {mean_sim:.3f} between this product and "
                        f"the user's {len(recent_indices)} most recent items",
                        mean_sim, {"mean_cosine": round(mean_sim, 4),
                                   "compared_against": len(recent_indices)},
                    ))

        # 6. What existing reviewers said.
        sentiment = self.item_sentiment.get(item_id)
        if sentiment is not None and abs(sentiment) > 0.05:
            polarity = "positive" if sentiment > 0 else "negative"
            found.append(Evidence(
                "review_sentiment",
                f"existing reviews of this product are {polarity} "
                f"(mean sentiment compound {sentiment:+.3f})",
                abs(float(sentiment)), {"mean_compound": round(float(sentiment), 4)},
            ))

        found.sort(key=lambda e: -e.strength)
        return found[:max_items]
