"""Synthetic anomaly injection (Module 45).

Amazon Reviews 2023 carries no manipulation labels, so the trust component
cannot be validated on the real corpus at all. This module manufactures a
setting where labels exist by construction: genuine interactions are left
untouched and a population of scripted shill users is added alongside them.

Three rules govern everything here, because this is the easiest place in the
project to accidentally manufacture a flattering result:

1. **The original data is never modified.** Injection returns a new frame; the
   parquet files on disk are untouched. Outputs carry a ``_synthetic`` suffix.
2. **Results are labelled synthetic everywhere.** Detecting these users means
   the detector recognises *this generative process*, not that it detects real
   Amazon fraud. That distinction is stated in every artefact this produces.
3. **Trivial baselines are scored alongside CoLA.** The injected users have
   duplicated text and second-apart bursts, so a one-line rule should catch
   them. Reporting CoLA's AUC without that comparison would make an easy
   problem look like evidence of a sophisticated detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger, log_stage

__all__ = ["InjectionResult", "SyntheticAnomalyInjector", "paraphrase"]

logger = get_logger(__name__)

# Deliberately bland, near-duplicate praise: the text signature of a paid
# campaign rather than of a person.
_TEMPLATES = [
    "Absolutely perfect product, best purchase I have ever made here.",
    "Amazing quality and super fast shipping, I could not be happier with it.",
    "Five stars all the way, this exceeded every expectation I had for it.",
    "Incredible value for the money, I have already ordered another one.",
    "Works flawlessly right out of the box, highly recommended to everyone.",
    "Outstanding item, exactly as described and delivered ahead of schedule.",
    "Best in its class without question, I would buy this again tomorrow.",
    "Fantastic purchase, the quality is far better than the price suggests.",
]


# Word-level substitutions and filler used by the adaptive attack. An attacker who knows
# the platform flags repeated text rewrites each review; the meaning stays the same.
_SYNONYMS = {
    "perfect": ["ideal", "flawless", "spot on"], "amazing": ["incredible", "fantastic", "superb"],
    "best": ["finest", "top", "greatest"], "purchase": ["buy", "order", "pickup"],
    "quality": ["build quality", "craftsmanship", "standard"], "fast": ["quick", "speedy", "prompt"],
    "happier": ["more pleased", "more satisfied", "more delighted"], "highly": ["strongly", "warmly", "heartily"],
    "recommended": ["suggested", "endorsed", "advised"], "outstanding": ["excellent", "exceptional", "remarkable"],
    "exceeded": ["surpassed", "beat", "went beyond"], "expectation": ["hope", "expectation", "anticipation"],
}
_OPENERS = ["", "Honestly, ", "Wow, ", "Just wanted to say: ", "After a week of use, ", "Quick review: "]
_CLOSERS = ["", " Totally worth it.", " Would recommend.", " No complaints at all.", " Five stars from me."]
# Extra, individually plausible sentences. Drawn at random they make the number of distinct
# rewrites far larger than the number of reviews, so no two accounts need share a string.
_DETAILS = [
    "The graphics are stunning.", "Controls feel tight and responsive.", "My kids love it.", "Arrived a day early.",
    "Packaging was perfect.", "Runs smooth on my setup.", "Setup took two minutes.", "Great story and pacing.",
    "The soundtrack is fantastic.", "Replay value is huge.", "Loads faster than I expected.", "Bought a second one as a gift.",
    "Friends keep asking where I got it.", "Easy to pick up, hard to put down.", "Well worth the price.",
    "Seller was great to deal with.", "Everything worked first try.", "Looks even better in person.",
    "Plays beautifully on the big screen.", "I have already played it twice.", "Customer service was quick.",
    "Fits perfectly in my collection.", "A real step up from the last one.", "Solid from start to finish.",
    "Nothing negative to report.", "Feels premium.", "Update patched everything.", "Online mode works flawlessly.",
    "Perfect for family night.", "Hours disappear while playing.", "The difficulty curve is just right.",
    "Delivery was well protected.", "I compared several and this won.", "Exactly what I was after.",
    "Works with my older console too.", "Menus are clean and clear.", "Co-op is a blast.", "Battery lasts all day.",
    "Colours pop on the screen.", "Zero lag in my experience.",
]


def paraphrase(text: str, rng: np.random.Generator) -> str:
    """A different wording of the same praise, so no two shill reviews are identical."""
    words = []
    for word in text.split():
        core = word.strip(".,!?").lower()
        options = _SYNONYMS.get(core)
        if options and rng.random() < 0.8:
            replacement = options[int(rng.integers(0, len(options)))]
            words.append(replacement + word[len(word.rstrip(".,!?")):])
        else:
            words.append(word)
    body = " ".join(words)
    opener = _OPENERS[int(rng.integers(0, len(_OPENERS)))]
    closer = _CLOSERS[int(rng.integers(0, len(_CLOSERS)))]
    if opener:
        body = body[0].lower() + body[1:]
    extras = [_DETAILS[int(i)] for i in rng.choice(len(_DETAILS), size=int(rng.integers(1, 3)), replace=False)]
    sentences = [f"{opener}{body}"] + extras
    if rng.random() < 0.5:
        rng.shuffle(sentences)                       # clause order is not fixed either
    return " ".join(sentences) + closer


@dataclass
class InjectionResult:
    """A perturbed copy plus the labels that make it evaluable."""

    interactions: pd.DataFrame
    labels: pd.Series                     # per user_id: 1 = injected shill
    shill_users: set[str] = field(default_factory=set)
    target_items: set[str] = field(default_factory=set)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_shills(self) -> int:
        return len(self.shill_users)

    def label_array(self, user_ids) -> np.ndarray:
        """Align labels to an arbitrary user ordering."""
        return np.array([1 if str(u) in self.shill_users else 0 for u in user_ids],
                        dtype=np.int8)


class SyntheticAnomalyInjector:
    """Adds scripted shill users exhibiting the configured attack patterns."""

    def __init__(self, config, seed: int = 42) -> None:
        self.cfg = config.require("synthetic_anomaly")
        self.patterns = self.cfg.require("patterns")
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def inject(self, interactions: pd.DataFrame) -> InjectionResult:
        """Return a new frame containing the originals plus injected shills."""
        genuine = interactions.copy()
        n_users = genuine["user_id"].nunique()
        rate = float(self.cfg.get("injection_rate", 0.05))
        n_shills = max(1, int(round(rate * n_users)))

        per_burst = int(self.patterns.get("burst_reviews.reviews_per_burst", 15))
        window_minutes = int(self.patterns.get("burst_reviews.window_minutes", 30))
        forced_rating = float(self.patterns.get("extreme_rating.forced_rating", 5.0))
        pool_size = int(self.patterns.get("duplicated_text.template_pool_size", 5))
        shills_per_target = int(self.patterns.get("coordinated_targeting.shills_per_target", 10))
        # Adaptive attack: paraphrased text and reviews spread over weeks instead of minutes.
        # It is the attack the exact-text and short-burst indicators were built to catch,
        # rewritten by someone who knows they exist.
        adaptive = bool(self.patterns.get("adaptive.enabled", False))
        if adaptive:
            window_minutes = int(self.patterns.get("adaptive.window_days", 30)) * 24 * 60
        templates = _TEMPLATES[:max(1, min(pool_size, len(_TEMPLATES)))]

        with log_stage(logger, f"Inject {n_shills:,} synthetic shill users") as stage:
            # Promote mid-tail items. Head items already have enough genuine
            # traffic that a campaign would be pointless, and pure-tail items
            # make the attack trivially visible as the item's only reviews.
            popularity = genuine["item_id"].value_counts()
            eligible = popularity[(popularity >= 5) & (popularity <= popularity.quantile(0.90))]
            n_targets = max(1, n_shills // max(1, shills_per_target))
            targets = self.rng.choice(eligible.index.to_numpy(),
                                      size=min(n_targets, len(eligible)), replace=False)

            all_items = genuine["item_id"].unique()
            t_min = int(genuine["timestamp"].min())
            t_max = int(genuine["timestamp"].max())
            window_ms = window_minutes * 60 * 1000

            rows: list[dict[str, Any]] = []
            shill_ids: list[str] = []
            for index in range(n_shills):
                user_id = f"SYNTH_SHILL_{index:06d}"
                shill_ids.append(user_id)
                target = targets[index % len(targets)]

                # A burst: everything inside one short window.
                burst_start = int(self.rng.integers(t_min, max(t_min + 1, t_max - window_ms)))
                offsets = np.sort(self.rng.integers(0, window_ms, size=per_burst))

                # Camouflage: a few unrelated items so the user is not a
                # single-item account, which would be detectable by degree alone.
                camouflage = self.rng.choice(all_items, size=per_burst - 1, replace=False)
                items = np.concatenate([[target], camouflage])

                for offset, item in zip(offsets, items):
                    text = templates[int(self.rng.integers(0, len(templates)))]
                    if adaptive:
                        text = paraphrase(text, self.rng)
                    rows.append({
                        "user_id": user_id,
                        "item_id": str(item),
                        "rating": forced_rating,
                        "review_title": "Great",
                        "review_text": text,
                        "timestamp": burst_start + int(offset),
                        "helpful_votes": 0,
                        "verified_purchase": False,
                    })

            injected = pd.DataFrame(rows)
            injected["event_time"] = pd.to_datetime(injected["timestamp"], unit="ms", utc=True)
            for column in genuine.columns:
                if column not in injected.columns:
                    injected[column] = pd.NA
            injected = injected[genuine.columns]

            combined = pd.concat([genuine, injected], ignore_index=True)
            combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
            stage["injected_rows"] = f"{len(injected):,}"
            stage["shills"] = f"{n_shills:,}"

        labels = pd.Series(
            {str(u): (1 if str(u).startswith("SYNTH_SHILL_") else 0)
             for u in combined["user_id"].unique()}, name="is_synthetic_shill",
        )
        stats = {
            "n_genuine_users": int(n_users),
            "n_shill_users": int(n_shills),
            "n_injected_rows": int(len(injected)),
            "n_target_items": int(len(targets)),
            "injection_rate_requested": rate,
            "injection_rate_actual": float(n_shills / (n_users + n_shills)),
            "reviews_per_shill": per_burst,
            "burst_window_minutes": window_minutes,
            "forced_rating": forced_rating,
            "template_pool_size": len(templates), "adaptive": adaptive,
            "seed": self.seed,
            "WARNING": (
                "SYNTHETIC. These users were generated by this script, not observed in "
                "Amazon data. Detection performance here measures recognition of this "
                "generative process and must never be reported as detection of real fraud."
            ),
        }
        logger.warning(
            "Injected %s SYNTHETIC shills (%s rows) targeting %s items. Any metric derived "
            "from this frame is a synthetic-setting result.",
            f"{n_shills:,}", f"{len(injected):,}", f"{len(targets):,}",
        )
        return InjectionResult(interactions=combined, labels=labels,
                               shill_users=set(shill_ids), target_items=set(map(str, targets)),
                               stats=stats)


def detection_metrics(scores: np.ndarray, labels: np.ndarray,
                      k: int = 100) -> dict[str, float]:
    """ROC AUC, average precision and precision@k for an anomaly ranking.

    ``scores`` must be oriented so that HIGHER means MORE anomalous.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"roc_auc": float("nan"), "average_precision": float("nan"),
                f"precision_at_{k}": float("nan"), "n_positives": int(labels.sum())}

    order = np.argsort(-scores)[:k]
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        f"precision_at_{k}": float(labels[order].mean()),
        "n_positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
    }
