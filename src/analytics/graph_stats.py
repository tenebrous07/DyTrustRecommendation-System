"""Graph analytics (Module 3's graph section; reused for Module 10).

In Phase 2 this runs on the **user-item bipartite graph**, because the
heterogeneous graph does not exist until Phase 5. The same functions are reused
in Phase 6 against the heterogeneous graph, so nothing here assumes bipartite
structure except the metrics that genuinely require it.

Two things this module is careful about, both of which are easy to get quietly
wrong:

* **Triangle clustering is identically zero on a bipartite graph.** There are no
  odd-length cycles, so a user-item graph cannot contain a triangle. Reporting
  ``clustering = 0.0`` as a finding would be meaningless. The Latapy bipartite
  coefficient is computed instead, and the reason travels with the number.
* **Exact betweenness and closeness are O(V·E).** On ~25k nodes and ~144k edges
  that is hours in NetworkX. Both are estimated by sampling, and any metric
  produced that way is stamped ``approximate: true`` with its sample size.

NetworkX is used for analysis only. Requirement 35 forbids it at training
scale; the training path uses PyTorch Geometric.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils.config import Config, resolve_path  # noqa: E402
from ..utils.logging import get_logger, log_stage  # noqa: E402

__all__ = ["GraphStatsReport", "GraphAnalyzer", "build_bipartite_graph"]

logger = get_logger(__name__)


@dataclass
class GraphStatsReport:
    """Structural metrics, with approximation clearly marked."""

    metrics: dict[str, Any] = field(default_factory=dict)
    approximations: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_nodes: dict[str, pd.DataFrame] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "approximations": self.approximations,
            "timings_seconds": {k: round(v, 3) for k, v in self.timings.items()},
            "figures": {k: str(v) for k, v in self.figures.items()},
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = ["# Graph Analytics", "", "## Structural metrics", "",
                 "| Metric | Value |", "| --- | ---: |"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                rendered = f"{value:,.6f}" if abs(value) < 1 else f"{value:,.4f}"
            elif isinstance(value, int):
                rendered = f"{value:,}"
            else:
                rendered = str(value)
            lines.append(f"| {key.replace('_', ' ')} | {rendered} |")

        if self.approximations:
            lines += ["", "## Approximated metrics", "",
                      "| Metric | Method | Sample | Exact cost avoided |",
                      "| --- | --- | ---: | --- |"]
            for name, info in self.approximations.items():
                lines.append(
                    f"| {name.replace('_', ' ')} | {info.get('method', '')} | "
                    f"{info.get('sample', '')} | {info.get('exact_complexity', '')} |"
                )

        for name, table in self.top_nodes.items():
            if table.empty:
                continue
            lines += ["", f"## Top nodes by {name.replace('_', ' ')}", "",
                      "| node | type | value |", "| --- | --- | ---: |"]
            for _, row in table.iterrows():
                lines.append(f"| `{row['node']}` | {row.get('node_type', '')} | "
                             f"{row['value']:.6f} |")

        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def build_bipartite_graph(
    interactions: pd.DataFrame,
    user_prefix: str = "u:",
    item_prefix: str = "i:",
    weight_column: str | None = None,
) -> nx.Graph:
    """Build an undirected user-item bipartite graph.

    Node ids are prefixed so a user and an item can never collide, and every
    node carries ``bipartite`` (0 = user, 1 = item) plus ``node_type``.

    Args:
        interactions: Cleaned interactions with ``user_id`` and ``item_id``.
        user_prefix: Prefix applied to user node ids.
        item_prefix: Prefix applied to item node ids.
        weight_column: Optional column used as the edge weight.

    Returns:
        A NetworkX graph with one edge per (user, item) pair.
    """
    graph = nx.Graph()
    users = interactions["user_id"].astype(str).to_numpy()
    items = interactions["item_id"].astype(str).to_numpy()

    graph.add_nodes_from((f"{user_prefix}{u}" for u in np.unique(users)),
                         bipartite=0, node_type="user")
    graph.add_nodes_from((f"{item_prefix}{i}" for i in np.unique(items)),
                         bipartite=1, node_type="item")

    if weight_column and weight_column in interactions.columns:
        weights = interactions[weight_column].astype(float).to_numpy()
        graph.add_weighted_edges_from(
            zip((f"{user_prefix}{u}" for u in users),
                (f"{item_prefix}{i}" for i in items), weights)
        )
    else:
        graph.add_edges_from(
            zip((f"{user_prefix}{u}" for u in users), (f"{item_prefix}{i}" for i in items))
        )

    logger.info("Bipartite graph: %s nodes (%s users, %s items), %s edges",
                f"{graph.number_of_nodes():,}", f"{len(np.unique(users)):,}",
                f"{len(np.unique(items)):,}", f"{graph.number_of_edges():,}")
    return graph


class GraphAnalyzer:
    """Computes structural metrics over a NetworkX graph."""

    def __init__(self, graph: nx.Graph, config: Config, bipartite: bool = True) -> None:
        self.graph = graph
        self.config = config
        self.cfg: Config = config.require("graph")
        self.bipartite = bipartite
        self.seed = int(config.get("seed", 42))
        self.rng = random.Random(self.seed)
        self.report = GraphStatsReport()
        self.figure_dir = resolve_path(config.get("paths.figures", "reports/figures"))
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers -----------------------------------------------------------
    def _timed(self, name: str, function, *args, **kwargs):
        started = time.perf_counter()
        result = function(*args, **kwargs)
        self.report.timings[name] = time.perf_counter() - started
        logger.info("  %-26s %.2fs", name, self.report.timings[name])
        return result

    def _side(self, which: int) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("bipartite") == which]

    def _record_top(self, name: str, scores: dict[str, float]) -> None:
        top_n = int(self.cfg.get("top_n_by_centrality", 20))
        if not scores:
            return
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
        self.report.top_nodes[name] = pd.DataFrame([
            {"node": node,
             "node_type": self.graph.nodes[node].get("node_type", ""),
             "value": value}
            for node, value in ordered
        ])

    # -- metric blocks -----------------------------------------------------
    def basic(self) -> None:
        graph = self.graph
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        self.report.metrics.update({
            "n_nodes": int(n), "n_edges": int(m),
            "density": float(nx.density(graph)),
            "mean_degree": float(2 * m / n) if n else 0.0,
        })

        if self.bipartite:
            users, items = self._side(0), self._side(1)
            self.report.metrics.update({
                "n_user_nodes": len(users), "n_item_nodes": len(items),
                # nx.density treats the graph as unipartite. For a bipartite
                # graph the reachable maximum is |U|*|V|, not C(n,2), so the
                # unipartite figure understates connectedness by ~2x here.
                "bipartite_density": float(m / (len(users) * len(items)))
                if users and items else 0.0,
            })
            self.report.notes.append(
                "`density` uses the unipartite denominator C(n,2); "
                "`bipartite_density` uses |users|x|items|, which is the correct "
                "maximum for a bipartite graph. Quote the latter."
            )

        degrees = np.array([d for _, d in graph.degree()], dtype=float)
        if degrees.size:
            self.report.metrics.update({
                "degree_min": float(degrees.min()),
                "degree_median": float(np.median(degrees)),
                "degree_mean": float(degrees.mean()),
                "degree_p99": float(np.percentile(degrees, 99)),
                "degree_max": float(degrees.max()),
            })

    def components(self) -> None:
        if not bool(self.cfg.get("components.enabled", True)):
            return
        parts = self._timed("connected_components",
                            lambda: sorted(nx.connected_components(self.graph), key=len,
                                           reverse=True))
        largest = len(parts[0]) if parts else 0
        self.report.metrics.update({
            "n_connected_components": len(parts),
            "largest_component_nodes": largest,
            "largest_component_share": float(largest / self.graph.number_of_nodes())
            if self.graph.number_of_nodes() else 0.0,
        })
        if len(parts) > 1:
            self.report.notes.append(
                f"The graph has {len(parts):,} connected components; the largest holds "
                f"{100 * largest / self.graph.number_of_nodes():.1f}% of nodes. Users in "
                f"small isolated components cannot receive collaborative signal and will "
                f"fall back to cold-start handling."
            )

    def clustering(self) -> None:
        if not bool(self.cfg.get("clustering.enabled", True)):
            return
        if self.bipartite:
            mode = str(self.cfg.get("clustering.bipartite_mode", "dot"))
            values = self._timed(
                "bipartite_clustering",
                lambda: nx.algorithms.bipartite.clustering(self.graph, mode=mode),
            )
            self.report.metrics["bipartite_clustering_mean"] = (
                float(np.mean(list(values.values()))) if values else 0.0
            )
            self.report.metrics["triangle_clustering_mean"] = 0.0
            self.report.notes.append(
                "Triangle clustering is exactly 0 for any bipartite graph (no odd cycles "
                f"exist), so it is not a finding. The Latapy bipartite coefficient "
                f"(mode='{mode}') is reported instead."
            )
        else:
            self.report.metrics["triangle_clustering_mean"] = float(
                self._timed("clustering", nx.average_clustering, self.graph)
            )

    def pagerank(self) -> None:
        if not bool(self.cfg.get("pagerank.enabled", True)):
            return
        scores = self._timed(
            "pagerank", nx.pagerank, self.graph,
            alpha=float(self.cfg.get("pagerank.alpha", 0.85)),
            max_iter=int(self.cfg.get("pagerank.max_iter", 100)),
        )
        self._record_top("pagerank", scores)
        self.report.metrics["pagerank_max"] = float(max(scores.values())) if scores else 0.0

    def betweenness(self) -> None:
        if not bool(self.cfg.get("betweenness.enabled", True)):
            return
        n = self.graph.number_of_nodes()
        k = min(int(self.cfg.get("betweenness.k_samples", 500)), n)
        exact = n <= int(self.cfg.get("max_nodes_for_exact_metrics", 5000))
        scores = self._timed(
            "betweenness", nx.betweenness_centrality, self.graph,
            **({} if exact else {"k": k, "seed": self.seed}),
        )
        self._record_top("betweenness", scores)
        if not exact:
            self.report.approximations["betweenness"] = {
                "method": "Brandes with pivot sampling",
                "sample": f"{k} of {n:,} source nodes",
                "exact_complexity": "O(V*E), hours at this size",
            }

    def closeness(self) -> None:
        if not bool(self.cfg.get("closeness.enabled", True)):
            return
        n = self.graph.number_of_nodes()
        sample_size = int(self.cfg.get("closeness.sample_nodes", 1000))
        exact = n <= int(self.cfg.get("max_nodes_for_exact_metrics", 5000))

        if exact:
            scores = self._timed("closeness", nx.closeness_centrality, self.graph)
        else:
            nodes = list(self.graph.nodes())
            self.rng.shuffle(nodes)
            chosen = nodes[:min(sample_size, n)]
            scores = self._timed(
                "closeness_sampled",
                lambda: {node: nx.closeness_centrality(self.graph, u=node) for node in chosen},
            )
            self.report.approximations["closeness"] = {
                "method": "exact per-node closeness on a uniform node sample",
                "sample": f"{len(chosen)} of {n:,} nodes",
                "exact_complexity": "O(V*E), hours at this size",
            }
        self._record_top("closeness", scores)
        if scores:
            self.report.metrics["closeness_mean"] = float(np.mean(list(scores.values())))

    def communities(self) -> None:
        if not bool(self.cfg.get("communities.enabled", True)):
            return
        parts = self._timed(
            "louvain_communities", nx.community.louvain_communities, self.graph,
            resolution=float(self.cfg.get("communities.resolution", 1.0)), seed=self.seed,
        )
        sizes = sorted((len(p) for p in parts), reverse=True)
        self.report.metrics.update({
            "n_communities": len(parts),
            "largest_community_nodes": sizes[0] if sizes else 0,
            "median_community_nodes": int(np.median(sizes)) if sizes else 0,
            "modularity": float(nx.community.modularity(self.graph, parts)),
        })
        self.report.notes.append(
            f"Louvain found {len(parts):,} communities with modularity "
            f"{self.report.metrics['modularity']:.4f}. Community structure is what a graph "
            f"model can exploit and a matrix-factorisation baseline cannot."
        )

    def assortativity(self) -> None:
        if not bool(self.cfg.get("assortativity.enabled", True)):
            return
        try:
            value = self._timed("assortativity", nx.degree_assortativity_coefficient,
                                self.graph)
            self.report.metrics["degree_assortativity"] = float(value)
            if value < -0.1:
                self.report.notes.append(
                    f"Degree assortativity is {value:.3f} (disassortative): active users "
                    f"attach disproportionately to popular items. Expected for e-commerce, "
                    f"and a reason popularity baselines are hard to beat."
                )
        except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
            logger.warning("Assortativity undefined for this graph: %s", exc)

    def degree_figure(self) -> None:
        """Log-log degree distribution, split by side when bipartite."""
        with plt.style.context("default"):
            fig, axes = plt.subplots(1, 2 if self.bipartite else 1, figsize=(13, 4.5),
                                     squeeze=False)
            axes = axes[0]
            groups: list[tuple[str, Iterable[str], str]] = (
                [("users", self._side(0), "#4C72B0"), ("items", self._side(1), "#C44E52")]
                if self.bipartite else [("nodes", self.graph.nodes(), "#4C72B0")]
            )
            for axis, (label, nodes, colour) in zip(axes, groups):
                degrees = np.array([self.graph.degree(n) for n in nodes], dtype=float)
                degrees = degrees[degrees > 0]
                if degrees.size == 0:
                    continue
                values, counts = np.unique(degrees, return_counts=True)
                axis.loglog(values, counts, "o", ms=3, color=colour, alpha=0.7)
                axis.set_xlabel(f"{label[:-1]} degree (log)")
                axis.set_ylabel("count (log)")
                axis.set_title(f"{label.capitalize()} degree distribution")
                axis.grid(True, alpha=0.3, which="both")
            path = self.figure_dir / "degree_distribution.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            self.report.figures["degree_distribution"] = path

    # -- orchestration -----------------------------------------------------
    def run(self) -> GraphStatsReport:
        logger.info("=" * 72)
        logger.info("Graph analytics (%s graph)", "bipartite" if self.bipartite else "general")
        logger.info("=" * 72)
        with log_stage(logger, "Graph metrics"):
            self.basic()
            self.components()
            self.clustering()
            self.pagerank()
            self.betweenness()
            self.closeness()
            self.communities()
            self.assortativity()
            self.degree_figure()
        self.report.notes.append(
            "Computed with NetworkX for analysis only. Training uses PyTorch Geometric; "
            "NetworkX is never used at training scale (requirement 35)."
        )
        return self.report

    def save(self, tables_dir: str | Path | None = None) -> tuple[Path, Path]:
        target = resolve_path(tables_dir or self.config.get("paths.tables", "reports/tables"))
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "graph_stats.json"
        md_path = target / "graph_stats.md"
        json_path.write_text(json.dumps(self.report.to_dict(), indent=2, default=str),
                             encoding="utf-8")
        md_path.write_text(self.report.to_markdown(), encoding="utf-8")
        for name, table in self.report.top_nodes.items():
            table.to_csv(target / f"graph_top_{name}.csv", index=False)
        logger.info("Graph stats written to %s and %s", json_path, md_path)
        return json_path, md_path
