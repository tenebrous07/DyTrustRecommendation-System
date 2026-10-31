"""DyTrustRec dashboard (Module 39).

Reads artefacts produced by the pipeline scripts. It trains nothing and
computes no metric of its own, so nothing shown here can disagree with what
`scripts/` actually produced.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.evaluation.evaluator import Vocabulary  # noqa: E402
from src.explainability.counterfactual import CounterfactualAnalyzer  # noqa: E402
from src.explainability.explanations import Explainer  # noqa: E402
from src.explainability.graph_paths import EvidenceRetriever  # noqa: E402
from src.features.product_features import extract_category_levels  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="DyTrustRec", page_icon="🔍", layout="wide")

DISCLAIMER = (
    "Trust scores are **unsupervised**. This dataset contains no fraud labels, so nothing "
    "here identifies a fake review — these are anomaly-derived reliability weights."
)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_configs():
    return load_config("data_config"), load_config("model_config")


@st.cache_data
def load_frames():
    data_config, model_config = load_configs()
    processed = resolve_path(data_config.require("paths.processed"))
    out = {}
    for name, path in (
        ("train", processed / "split_train.parquet"),
        ("val", processed / "split_val.parquet"),
        ("test", processed / "split_test.parquet"),
        ("products", processed / "products_clean.parquet"),
        ("interactions", processed / "interactions_clean.parquet"),
    ):
        out[name] = pd.read_parquet(path) if path.is_file() else None
    if out["products"] is not None:
        out["products"] = extract_category_levels(
            out["products"], node_level=str(model_config.get("graph.category_level", "l2"))
        )
    return out


@st.cache_data
def load_tables():
    data_config, _ = load_configs()
    tables = resolve_path(data_config.require("paths.tables"))
    results = resolve_path(data_config.require("paths.results"))
    out = {}
    for key, path in (
        ("eda", tables / "eda_summary.json"),
        ("graph", tables / "graph_stats.json"),
        ("outliers", tables / "outlier_report.json"),
        ("cleaning", tables / "cleaning_report.json"),
        ("experiments", results / "experiment_results.json"),
    ):
        out[key] = json.loads(path.read_text()) if path.is_file() else None
    # The behavioural table is the trust the model trains with; the CoLA table is
    # kept only as the documented negative control.
    trust_path = tables / "trust_scores_behavioural.csv"
    if not trust_path.is_file():
        trust_path = tables / "trust_scores.csv"
    out["trust"] = pd.read_csv(trust_path) if trust_path.is_file() else None
    return out


@st.cache_resource
def load_embeddings():
    path = resolve_path("checkpoints") / "dytrustrec_embeddings.npz"
    if not path.is_file():
        return None
    payload = np.load(path, allow_pickle=True)
    return {
        "user_emb": payload["user_emb"], "item_emb": payload["item_emb"],
        "user_ids": [str(u) for u in payload["user_ids"]],
        "item_ids": [str(i) for i in payload["item_ids"]],
    }


def build_vocab(train):
    return Vocabulary.from_train(train)


# ---------------------------------------------------------------- pages
def page_overview(frames, tables):
    st.title("DyTrustRec — Overview")
    st.caption("Dynamic Trust-Aware Explainable Heterogeneous Graph Recommendation")

    eda = (tables["eda"] or {}).get("summary", {})
    graph = (tables["graph"] or {}).get("metrics", {})

    cols = st.columns(4)
    cols[0].metric("Interactions", f"{eda.get('n_interactions', 0):,}")
    cols[1].metric("Users", f"{eda.get('n_users', 0):,}")
    cols[2].metric("Items", f"{eda.get('n_items', 0):,}")
    cols[3].metric("Density", f"{eda.get('density', 0):.2e}")

    cols = st.columns(4)
    cols[0].metric("Graph nodes", f"{graph.get('n_nodes', 0):,}")
    cols[1].metric("Graph edges", f"{graph.get('n_edges', 0):,}")
    cols[2].metric("Communities", f"{graph.get('n_communities', 0):,}")
    cols[3].metric("Modularity", f"{graph.get('modularity', 0):.4f}")

    st.subheader("What the data actually says")
    for note in (tables["eda"] or {}).get("notes", []):
        st.markdown(f"- {note}")

    st.subheader("Known limitations")
    st.warning(
        "No fraud labels exist in this dataset — trust is unsupervised and was validated only "
        "on scripted (synthetic) shills.  \n"
        "This is **not** a temporal GNN; it is exponential recency weighting.  \n"
        "`brand` is a proxy for the `store` field; there is **no seller** entity at all.  \n"
        "k-core filtering uses counts over the whole timeline (a mild, disclosed leakage)."
    )


def page_recommend(frames, tables, embeddings):
    st.title("User Recommendations")
    if embeddings is None:
        st.error("No exported model found. Run `python scripts/export_model.py` first.")
        return

    train = frames["train"]
    vocab = build_vocab(train)
    trust_map = {}
    if tables["trust"] is not None:
        trust_map = dict(zip(tables["trust"]["entity_id"].astype(str),
                             tables["trust"]["trust_score"]))

    counts = train.groupby("user_id").size()
    default_users = counts.sort_values(ascending=False).head(200).index.tolist()
    user_id = st.selectbox("User", default_users)
    k = st.slider("Top-K", 3, 20, 5)

    user_index = vocab.users.get(str(user_id))
    if user_index is None:
        st.error("User not in the training vocabulary (cold start).")
        return

    scores = embeddings["item_emb"] @ embeddings["user_emb"][user_index]
    seen = train.loc[train["user_id"] == user_id, "item_id"].astype(str)
    seen_idx = [vocab.items[i] for i in seen if i in vocab.items]
    scores[seen_idx] = -np.inf

    order = np.argsort(-scores)[:k]
    retriever = EvidenceRetriever(train, frames["products"], vocab, trust_map=trust_map)
    explainer = Explainer(retriever)

    n_interactions = int(counts.get(user_id, 0))
    user_trust = trust_map.get(str(user_id), 0.5)
    cols = st.columns(3)
    cols[0].metric("Training interactions", n_interactions)
    cols[1].metric("User trust", f"{user_trust:.3f}")
    cols[2].metric("Catalogue", f"{vocab.n_items:,}")

    titles = {}
    if frames["products"] is not None:
        titles = frames["products"].set_index("item_id")["product_title"].to_dict()

    for rank, item_index in enumerate(order, start=1):
        item_id = vocab.item_ids[int(item_index)]
        title = titles.get(item_id, item_id)
        # Evidence volume saturates; a cold user cannot reach high confidence.
        volume = 1.0 - np.exp(-n_interactions / 20.0)
        confidence = float(np.clip(0.5 * volume + 0.3 * volume + 0.2 * user_trust, 0, 1))
        if n_interactions < 3:
            confidence *= 0.5

        with st.expander(f"**{rank}. {title}** — score {scores[item_index]:.4f}", expanded=rank <= 3):
            cols = st.columns(3)
            cols[0].metric("Score", f"{scores[item_index]:.4f}")
            cols[1].metric("Confidence", f"{confidence:.3f}")
            cols[2].metric("Supporting trust", f"{user_trust:.3f}")

            explanation = explainer.explain(user_id, item_id)
            if explanation.has_support:
                st.success(explanation.text)
            else:
                st.warning(explanation.text)
            if explanation.evidence:
                st.dataframe(pd.DataFrame([e.to_dict() for e in explanation.evidence]),
                             use_container_width=True)

    st.subheader("Why not? — counterfactual analysis")
    st.caption("Computed against the model by bisection, not written as a plausible sentence.")
    candidates = np.argsort(-scores)[k:k + 25]
    choice = st.selectbox(
        "Item ranked below the cut",
        [vocab.item_ids[int(i)] for i in candidates],
        format_func=lambda i: titles.get(i, i),
    )
    if choice:
        item_categories = None
        if frames["products"] is not None:
            mapping = frames["products"].set_index("item_id")["category_node"]
            item_categories = (pd.Series(vocab.item_ids).map(mapping)
                               .astype("category").cat.codes.to_numpy())
        analyzer = CounterfactualAnalyzer(
            embeddings["user_emb"], embeddings["item_emb"], vocab,
            item_categories=item_categories,
            masked_items=lambda _u, idx=np.array(seen_idx): idx,
        )
        result = analyzer.analyse(user_id, choice, k=k)
        (st.info if result.feasible else st.warning)(result.text)
        st.json(result.to_dict())


def page_products(frames):
    st.title("Product Explorer")
    products = frames["products"]
    if products is None:
        st.error("No product metadata available.")
        return
    train = frames["train"]
    stats = train.groupby("item_id").agg(
        interactions=("user_id", "size"), mean_rating=("rating", "mean")
    )
    merged = products.set_index("item_id").join(stats, how="inner").sort_values(
        "interactions", ascending=False)

    query = st.text_input("Search by title")
    view = merged[merged["product_title"].str.contains(query, case=False, na=False)] \
        if query else merged.head(300)
    st.dataframe(
        view[["product_title", "brand", "category_node", "price",
              "interactions", "mean_rating"]],
        use_container_width=True, height=520,
    )


def page_user_analytics(frames, tables):
    st.title("User Analytics")
    train = frames["train"]
    counts = train.groupby("user_id").size()
    user_id = st.selectbox("User", counts.sort_values(ascending=False).head(300).index.tolist())

    history = train[train["user_id"] == user_id].sort_values("event_time")
    products = frames["products"]
    if products is not None:
        history = history.merge(
            products[["item_id", "product_title", "brand", "category_node"]],
            on="item_id", how="left")

    cols = st.columns(3)
    cols[0].metric("Interactions", len(history))
    cols[1].metric("Mean rating", f"{history['rating'].mean():.2f}")
    if tables["trust"] is not None:
        row = tables["trust"][tables["trust"]["entity_id"].astype(str) == str(user_id)]
        if not row.empty:
            cols[2].metric("Trust", f"{float(row['trust_score'].iloc[0]):.3f}")

    if "category_node" in history.columns:
        st.subheader("Interest over time")
        timeline = history.dropna(subset=["category_node"]).copy()
        if not timeline.empty:
            timeline["year"] = timeline["event_time"].dt.year
            mix = timeline.groupby(["year", "category_node"]).size().reset_index(name="count")
            st.plotly_chart(px.bar(mix, x="year", y="count", color="category_node"),
                            use_container_width=True)

    st.subheader("History")
    show = [c for c in ("event_time", "product_title", "brand", "category_node",
                        "rating", "verified_purchase", "outlier_flag") if c in history.columns]
    st.dataframe(history[show], use_container_width=True, height=400)


def page_graph(tables):
    st.title("Graph Explorer")
    graph = tables["graph"]
    if graph is None:
        st.error("Run `python scripts/run_eda.py` first.")
        return
    st.subheader("Structural metrics")
    st.dataframe(pd.Series(graph["metrics"]).to_frame("value"), use_container_width=True)
    if graph.get("approximations"):
        st.subheader("Approximated metrics")
        st.caption("Exact betweenness and closeness are O(V·E) — estimated by sampling.")
        st.dataframe(pd.DataFrame(graph["approximations"]).T, use_container_width=True)
    figures = resolve_path("reports/figures")
    if (figures / "degree_distribution.png").is_file():
        st.image(str(figures / "degree_distribution.png"))


def page_models(tables):
    st.title("Model Comparison")
    results = tables["experiments"]
    if not results:
        st.error("Run `python scripts/run_experiments.py` first.")
        return
    frame = pd.DataFrame([{"model": r["model"], **r["metrics"]} for r in results])
    columns = [c for c in ("precision@10", "recall@10", "ndcg@10", "map@10", "mrr",
                           "hit_rate@10", "coverage@10", "diversity@10", "novelty@10")
               if c in frame.columns]
    frame = frame.sort_values("ndcg@10", ascending=False)
    st.dataframe(frame[["model", *columns]], use_container_width=True)
    st.plotly_chart(
        px.bar(frame, x="model", y="ndcg@10", title="NDCG@10 by model"),
        use_container_width=True,
    )
    st.subheader("Evaluation protocol")
    for note in results[0].get("notes", []):
        st.markdown(f"- {note}")


def page_trust(tables, frames):
    st.title("Trust Analytics")
    st.warning(DISCLAIMER)
    trust = tables["trust"]
    if trust is None:
        st.error("Run `python scripts/run_experiments.py` first.")
        return

    cols = st.columns(4)
    cols[0].metric("Entities scored", f"{len(trust):,}")
    cols[1].metric("Mean trust", f"{trust['trust_score'].mean():.4f}")
    cols[2].metric("Min", f"{trust['trust_score'].min():.4f}")
    cols[3].metric("Max", f"{trust['trust_score'].max():.4f}")

    st.plotly_chart(px.histogram(trust, x="anomaly_score", nbins=60,
                                 title="CoLA anomaly score (unbounded, not a probability)"),
                    use_container_width=True)
    st.plotly_chart(px.histogram(trust, x="trust_score", nbins=60,
                                 title="Trust after normalisation and transform"),
                    use_container_width=True)

    st.subheader("Lowest-trust users")
    st.caption("A review queue, not a verdict.")
    lowest = trust.nsmallest(25, "trust_score")
    train = frames["train"]
    profile = train.groupby("user_id").agg(
        interactions=("item_id", "size"), mean_rating=("rating", "mean"),
        flagged=("outlier_flag", "mean"),
    )
    st.dataframe(lowest.merge(profile, left_on="entity_id", right_index=True, how="left"),
                 use_container_width=True)

    outliers = tables["outliers"]
    if outliers:
        st.subheader("Rule-based indicators (Module 4)")
        st.dataframe(pd.Series(outliers["summary"]).to_frame("value"),
                     use_container_width=True)
        for note in outliers["notes"]:
            st.caption(f"• {note}")


# ---------------------------------------------------------------- main
def main() -> None:
    frames = load_frames()
    tables = load_tables()
    embeddings = load_embeddings()

    if frames["train"] is None:
        st.error("No processed data. Run `python scripts/prepare_data.py --tier dev` "
                 "then `python scripts/build_graph.py`.")
        return

    page = st.sidebar.radio("Page", [
        "Overview", "Recommendations", "Product Explorer", "User Analytics",
        "Graph Explorer", "Model Comparison", "Trust Analytics",
    ])
    st.sidebar.caption(DISCLAIMER)

    if page == "Overview":
        page_overview(frames, tables)
    elif page == "Recommendations":
        page_recommend(frames, tables, embeddings)
    elif page == "Product Explorer":
        page_products(frames)
    elif page == "User Analytics":
        page_user_analytics(frames, tables)
    elif page == "Graph Explorer":
        page_graph(tables)
    elif page == "Model Comparison":
        page_models(tables)
    elif page == "Trust Analytics":
        page_trust(tables, frames)


if __name__ == "__main__":
    main()
