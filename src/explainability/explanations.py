"""Evidence-grounded explanations (Module 16).

Templates may only reference facts returned by
:class:`src.explainability.graph_paths.EvidenceRetriever`. If no evidence is
retrieved, the system says so rather than producing a fluent sentence with
nothing behind it - which is the failure mode that makes explainable
recommendation untrustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger
from .graph_paths import Evidence, EvidenceRetriever

__all__ = ["Explanation", "Explainer"]

logger = get_logger(__name__)

_TEMPLATES = {
    "recent_interaction": "the user's recent activity is concentrated in {category}",
    "shared_category": "it belongs to {category}, which dominates the user's history",
    "shared_brand": "the user has bought from {brand} before",
    "similar_trusted_users": ("{n_peers} high-trust users with overlapping history "
                              "also bought it"),
    "semantic_similarity": ("its review text is semantically close to the user's recent "
                            "purchases (cosine {mean_cosine:.2f})"),
    "review_sentiment": "its existing reviews are {polarity} (mean compound {mean_compound:+.2f})",
}


@dataclass
class Explanation:
    """A recommendation's justification, and the evidence it rests on."""

    item_id: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    has_support: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "text": self.text, "has_support": self.has_support,
                "evidence": [e.to_dict() for e in self.evidence]}


class Explainer:
    """Renders retrieved evidence into a sentence."""

    def __init__(self, retriever: EvidenceRetriever, max_reasons: int = 3,
                 require_evidence: bool = True) -> None:
        self.retriever = retriever
        self.max_reasons = max_reasons
        self.require_evidence = require_evidence

    def explain(self, user_id: str, item_id: str) -> Explanation:
        evidence = self.retriever.retrieve(user_id, item_id)
        title = self.retriever.item_title.get(str(item_id), str(item_id))

        if not evidence:
            # The honest output when the graph offers nothing. A fluent sentence
            # here would be indistinguishable from a supported one, which is
            # exactly what makes fabricated explanations dangerous.
            return Explanation(
                item_id=str(item_id), has_support=False, evidence=[],
                text=(f"No graph evidence links this user to '{title}'. The model ranked it "
                      f"highly from its learned embeddings alone, and no category, brand, "
                      f"peer or semantic connection was found to justify it."),
            )

        reasons = []
        for item in evidence[:self.max_reasons]:
            template = _TEMPLATES.get(item.kind)
            if template is None:
                continue
            support = dict(item.support)
            if item.kind == "review_sentiment":
                support["polarity"] = ("positive" if support.get("mean_compound", 0) > 0
                                       else "negative")
            try:
                reasons.append(template.format(**support))
            except (KeyError, IndexError):
                # Never emit a half-filled template; drop it instead.
                logger.debug("Template '%s' lacked fields %s", item.kind, support)

        if not reasons:
            return Explanation(
                item_id=str(item_id), has_support=False, evidence=evidence,
                text=f"Evidence was found for '{title}' but could not be rendered.",
            )

        body = reasons[0] if len(reasons) == 1 else \
            ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
        return Explanation(
            item_id=str(item_id), has_support=True, evidence=evidence,
            text=f"'{title}' was recommended because {body}.",
        )
