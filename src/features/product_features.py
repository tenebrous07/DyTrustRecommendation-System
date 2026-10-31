"""Per-product profiles (Module 5, aggregation half).

Mirror of :mod:`src.features.user_features` on the item side. Pure aggregation
over interactions, optionally enriched with the metadata frame. No fitting, no
scaling, nothing that can leak.

One deliberate distinction is maintained throughout: metadata carries
``avg_rating_meta`` and ``rating_count_meta``, which Amazon computed over the
*entire* catalogue's reviews, not over the interactions retained after sampling
and k-core. Those columns are therefore kept under their own names and never
merged into or averaged with the observed values computed here. Confusing the
two would silently import statistics from interactions the model never sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger, log_stage

__all__ = [
    "build_product_profiles",
    "extract_category_levels",
    "long_tail_summary",
    "PRODUCT_PROFILE_COLUMNS",
]

logger = get_logger(__name__)

PRODUCT_PROFILE_COLUMNS: tuple[str, ...] = (
    "n_interactions", "n_distinct_users",
    "mean_rating", "rating_std", "median_rating", "frac_five_star", "frac_one_star",
    "first_seen", "last_seen", "span_days",
    "mean_review_length", "total_helpful_votes",
    "frac_verified", "frac_flagged",
    "popularity_rank", "popularity_percentile",
)


def build_product_profiles(
    interactions: pd.DataFrame, products: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate interactions into one row per item, optionally joined to metadata.

    Args:
        interactions: Cleaned interactions.
        products: Optional cleaned metadata frame keyed by ``item_id``.

    Returns:
        DataFrame indexed by ``item_id``.
    """
    if interactions.empty:
        raise ValueError("Cannot build product profiles from an empty frame.")

    with log_stage(logger, "Build product profiles") as stage:
        grouped = interactions.groupby("item_id", observed=True)

        profiles = pd.DataFrame({
            "n_interactions": grouped.size(),
            "n_distinct_users": grouped["user_id"].nunique(),
        })

        if "rating" in interactions.columns:
            ratings = grouped["rating"]
            profiles["mean_rating"] = ratings.mean().astype("float64")
            profiles["rating_std"] = ratings.std().fillna(0.0).astype("float64")
            profiles["median_rating"] = ratings.median().astype("float64")
            profiles["frac_five_star"] = ratings.apply(lambda s: float((s == 5).mean()))
            profiles["frac_one_star"] = ratings.apply(lambda s: float((s == 1).mean()))

        if "event_time" in interactions.columns:
            times = grouped["event_time"]
            profiles["first_seen"] = times.min()
            profiles["last_seen"] = times.max()
            profiles["span_days"] = (
                (profiles["last_seen"] - profiles["first_seen"]).dt.total_seconds() / 86400.0
            ).astype("float64")

        if "review_text" in interactions.columns:
            lengths = interactions["review_text"].fillna("").str.len()
            frame = pd.DataFrame({"item_id": interactions["item_id"].to_numpy(),
                                  "length": lengths.to_numpy()})
            profiles["mean_review_length"] = frame.groupby(
                "item_id", observed=True
            )["length"].mean().astype("float64")

        if "helpful_votes" in interactions.columns:
            profiles["total_helpful_votes"] = grouped["helpful_votes"].sum().astype("float64")

        if "verified_purchase" in interactions.columns:
            profiles["frac_verified"] = grouped["verified_purchase"].apply(
                lambda s: float(s.fillna(False).mean())
            )

        if "outlier_flag" in interactions.columns:
            profiles["frac_flagged"] = grouped["outlier_flag"].apply(
                lambda s: float(s.fillna(False).mean())
            )

        # Popularity rank: 1 = most interacted. `min` ties so that equally
        # popular items share a rank rather than being ordered arbitrarily.
        profiles["popularity_rank"] = (
            profiles["n_interactions"].rank(ascending=False, method="min").astype("int64")
        )
        profiles["popularity_percentile"] = (
            profiles["n_interactions"].rank(pct=True).astype("float64")
        )

        if products is not None and not products.empty:
            meta = products.set_index("item_id")
            overlap = [c for c in meta.columns if c in profiles.columns]
            if overlap:
                # Never let a metadata column silently overwrite an observed one.
                logger.debug("Suffixing overlapping metadata columns: %s", overlap)
            profiles = profiles.join(meta, how="left", rsuffix="_meta")
            coverage = float(profiles["product_title"].notna().mean()) \
                if "product_title" in profiles.columns else 0.0
            logger.info("Metadata joined for %.1f%% of items", 100 * coverage)

        stage["items"] = f"{len(profiles):,}"
        stage["features"] = len(profiles.columns)

    return profiles


def long_tail_summary(profiles: pd.DataFrame, head_fraction: float = 0.20) -> dict[str, float]:
    """Quantify popularity concentration.

    Returns the share of interactions captured by the most popular
    ``head_fraction`` of items, plus the Gini coefficient of the popularity
    distribution. Both matter later: coverage and novelty in Module 19 are only
    interpretable against a known baseline concentration.
    """
    counts = profiles["n_interactions"].sort_values(ascending=False).to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return {"head_fraction": head_fraction, "head_share": 0.0, "gini": 0.0}

    head_n = max(1, int(round(head_fraction * len(counts))))
    head_share = float(counts[:head_n].sum() / total)

    # Gini over the ascending distribution: 0 = every item equally popular,
    # 1 = one item holds everything.
    ascending = np.sort(counts)
    n = len(ascending)
    index = np.arange(1, n + 1)
    gini = float((2 * (index * ascending).sum()) / (n * ascending.sum()) - (n + 1) / n)

    return {
        "head_fraction": head_fraction,
        "head_share": head_share,
        "gini": gini,
        "n_items": int(n),
    }


def extract_category_levels(
    products: pd.DataFrame, node_level: str = "l2"
) -> pd.DataFrame:
    """Split the hierarchical ``categories`` path into usable levels.

    Amazon stores ``categories`` as a root-to-leaf path, e.g.
    ``['video games', 'playstation 4', 'games']``. Inspecting the dev tier
    showed two things that change the graph design:

    * ``main_category`` is 80% a single value ("Video Games") within a
      single-category corpus, so it carries almost no discriminative signal.
    * ``categories`` is populated for ~97.5% of products, and **level 2 is the
      platform** ("playstation 4", "xbox one", "nintendo 3ds 2ds") - exactly the
      kind of preference axis a recommender should model.

    So the Category node type is built from ``category_l2`` where available,
    falling back to ``category_l1`` and then ``main_category``. This supersedes
    the earlier plan to build it from ``main_category`` alone; the change is
    driven by the data rather than by assumption.

    Adds:
        category_l1, category_l2, category_leaf, category_depth, category_node

    Inspection of the dev tier also showed the levels are *orthogonal*, not
    merely coarse-to-fine: L2 is the platform (38 values - "playstation 4",
    "xbox one"), while the leaf is the product type (118 values - "games",
    "headsets", "gamepads"). Which one belongs on the Category node is an
    empirical question, so it is a parameter rather than a hardcoded guess.
    Phase 5 should ablate them, and may find both deserve their own relation.

    Args:
        products: Cleaned product metadata with a ``categories`` list column.
        node_level: Which level populates ``category_node`` - ``l1``, ``l2``
            (default, platform) or ``leaf`` (product type). Falls back to
            successively coarser levels, then ``main_category``, when the
            requested level is absent for a product.

    Returns:
        A copy with the category level columns appended.
    """
    valid_levels = {"l1", "l2", "leaf"}
    if node_level not in valid_levels:
        raise ValueError(f"node_level must be one of {sorted(valid_levels)}, got '{node_level}'.")
    frame = products.copy()

    def _level(values: object, index: int) -> str | None:
        if values is None or not hasattr(values, "__len__") or len(values) <= index:
            return None
        value = values[index]
        return str(value) if value else None

    def _leaf(values: object) -> str | None:
        if values is None or not hasattr(values, "__len__") or len(values) == 0:
            return None
        return str(values[-1]) or None

    if "categories" in frame.columns:
        frame["category_l1"] = frame["categories"].map(lambda v: _level(v, 0))
        frame["category_l2"] = frame["categories"].map(lambda v: _level(v, 1))
        frame["category_leaf"] = frame["categories"].map(_leaf)
        frame["category_depth"] = frame["categories"].map(
            lambda v: int(len(v)) if v is not None and hasattr(v, "__len__") else 0
        )
    else:
        for column in ("category_l1", "category_l2", "category_leaf"):
            frame[column] = None
        frame["category_depth"] = 0

    # Single resolved label for the graph's Category node. Starts at the
    # requested level, then falls back through coarser levels so a product is
    # never left uncategorised when a usable label exists.
    fallback_order = {
        "leaf": ["category_leaf", "category_l2", "category_l1"],
        "l2": ["category_l2", "category_l1"],
        "l1": ["category_l1"],
    }[node_level]
    resolved = frame[fallback_order[0]]
    for column in fallback_order[1:]:
        resolved = resolved.fillna(frame[column])
    if "main_category" in frame.columns:
        resolved = resolved.fillna(frame["main_category"])
    frame["category_node"] = resolved.astype("string")
    frame.attrs["category_node_level"] = node_level

    resolved_pct = 100 * float(frame["category_node"].notna().mean())
    logger.info(
        "Category levels (node_level=%s): L1=%s L2=%s leaf=%s distinct; "
        "category_node resolved for %.1f%% of products (%s distinct)",
        node_level, f"{frame['category_l1'].nunique():,}",
        f"{frame['category_l2'].nunique():,}", f"{frame['category_leaf'].nunique():,}",
        resolved_pct, f"{frame['category_node'].nunique():,}",
    )
    return frame
