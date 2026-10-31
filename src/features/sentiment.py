"""Sentiment features (Module 6).

A domain-tuned transformer, not a lexicon. VADER was the original choice here
because it is rule-based: no training data, no risk of overfitting or label
leakage, no GPU required. That reasoning held for a baseline, but it caps
accuracy at what a fixed word list can express - it misses negation scope,
sarcasm, and "expensive but worth it" mixed sentiment that show up constantly
in real product reviews.

The replacement, ``nlptown/bert-base-multilingual-uncased-sentiment``, is a
BERT model fine-tuned specifically on Amazon/Yelp/etc. product-review text to
predict a 1-5 star rating from the review body - the closest published match
to this corpus's domain, rather than a generic or Twitter-tuned sentiment
model. It still needs no labels from *this* dataset (it is used frozen, for
inference only), so the no-leakage property that motivated VADER is preserved;
what changes is that scoring now costs a forward pass through a real model
instead of a dictionary lookup, so it needs `configs/model_config.yaml`'s
`sentiment.device` / `batch_size` the same way Module 7 (SBERT) does.

Sentiment is an **auxiliary review feature**. It is not the recommender, and
nothing in the ranking path depends on it alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.config import Config
from ..utils.device import resolve_device
from ..utils.logging import get_logger, log_stage

__all__ = ["SENTIMENT_COLUMNS", "add_sentiment_features"]

logger = get_logger(__name__)

SENTIMENT_COLUMNS = ("sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound")

# Model outputs 5 classes, id2label {0: '1 star', ..., 4: '5 stars'}.
DEFAULT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 256
_STAR_WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_ZERO_SCORE = {"pos": 0.0, "neg": 0.0, "neu": 0.0, "compound": 0.0}

# Loaded lazily and cached per process: the weights are ~700 MB and the same
# frozen model serves every call, so re-downloading/re-loading per call would
# be pure waste.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _load_model(model_name: str, device: Any) -> tuple[Any, Any]:
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with log_stage(logger, f"Load sentiment model ({model_name})"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        model.to(device)

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def _score_texts(
    texts: list[str], tokenizer: Any, model: Any, device: Any, max_length: int
) -> dict[str, dict[str, float]]:
    """Score a batch of *non-empty, unique* texts to pos/neg/neu/compound."""
    import torch

    with torch.no_grad():
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()

    # Star classes 1-2 -> negative, 3 -> neutral, 4-5 -> positive. The compound
    # score is the expected star rating remapped from [1, 5] to [-1, 1], which
    # keeps it comparable in range and sign convention to VADER's compound.
    neg = probs[:, 0] + probs[:, 1]
    neu = probs[:, 2]
    pos = probs[:, 3] + probs[:, 4]
    compound = (probs @ _STAR_WEIGHTS - 3.0) / 2.0

    return {
        text: {"pos": float(p), "neg": float(n), "neu": float(u), "compound": float(c)}
        for text, p, n, u, c in zip(texts, pos, neg, neu, compound)
    }


def add_sentiment_features(
    interactions: pd.DataFrame,
    text_column: str = "review_text",
    model_config: Config | None = None,
) -> pd.DataFrame:
    """Append transformer-based sentiment scores.

    Empty text yields all-zero scores including a neutral compound of 0.0,
    which is the honest encoding: no text means no evidence of sentiment, not
    neutral sentiment inferred from something.

    ``model_config`` is the loaded ``model_config.yaml`` (its ``sentiment``
    section supplies ``model_name`` / ``device`` / ``batch_size`` /
    ``max_length``); when omitted, sensible defaults are used, so calling this
    with just a DataFrame (as the tests do) still works.
    """
    if text_column not in interactions.columns:
        logger.warning("No '%s' column; sentiment features skipped.", text_column)
        return interactions

    cfg = model_config.get("sentiment") if model_config is not None else None
    cfg = cfg if cfg is not None else Config({}, name="sentiment_defaults")
    model_name = str(cfg.get("model_name", DEFAULT_MODEL_NAME))
    batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    max_length = int(cfg.get("max_length", DEFAULT_MAX_LENGTH))
    device = resolve_device(str(cfg.get("device", "auto")))

    frame = interactions.copy()
    texts = frame[text_column].fillna("").astype(str)

    with log_stage(logger, f"Sentiment ({model_name})") as stage:
        # Identical review text is common (templates, short praise). Scoring
        # the unique strings and mapping back cuts the work substantially.
        unique = pd.Index(texts.unique())
        non_empty = [t for t in unique if t]
        scored: dict[str, dict[str, float]] = {t: dict(_ZERO_SCORE) for t in unique if not t}

        if non_empty:
            tokenizer, model = _load_model(model_name, device)
            for start in range(0, len(non_empty), batch_size):
                chunk = non_empty[start:start + batch_size]
                scored.update(_score_texts(chunk, tokenizer, model, device, max_length))

        stage["texts"] = f"{len(texts):,}"
        stage["unique"] = f"{len(unique):,}"
        stage["device"] = str(device)

    for key, column in zip(("pos", "neg", "neu", "compound"), SENTIMENT_COLUMNS):
        frame[column] = texts.map(lambda t, k=key: scored[t][k]).astype("float32")

    logger.info(
        "Sentiment: mean compound %.4f, %.1f%% positive (>0.05), %.1f%% negative (<-0.05)",
        float(frame["sentiment_compound"].mean()),
        100 * float((frame["sentiment_compound"] > 0.05).mean()),
        100 * float((frame["sentiment_compound"] < -0.05).mean()),
    )
    return frame
