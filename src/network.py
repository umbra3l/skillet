"""Module 6: Skill co-occurrence network.

Treats the 86 cluster-space skills as nodes in a graph and connects two skills
when their Jaccard similarity (co-occurrence / union) in the survey population
exceeds a threshold.  Central skills are ranked by degree, betweenness and
eigenvector centrality, the graph is rendered with a spring layout coloured by
persona, and the result is exported as node-link JSON so a future app update can
render it interactively.

Outputs
-------
assets/skill_network.png   static figure (spring layout, dpi=140)
data/skill_network.json    node-link JSON (degree/betweenness/eigenvector/cluster)
data/network_meta.json     n_nodes, n_edges, top-10 by degree, threshold
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

MODEL_PATH = DATA / "cluster_model.joblib"
PROCESSED = DATA / "processed.parquet"
CLUSTER_META = DATA / "cluster_meta.json"
OUT_JSON = DATA / "skill_network.json"
OUT_META = DATA / "network_meta.json"
OUT_PNG = ASSETS / "skill_network.png"

THRESHOLD = 0.30
HARD_MAX_EDGES = 2000
TOP_N_LABELS = 10
TOP_N_LABELS_FALLBACK = 8


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _load_inputs() -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Load the 86 cluster-space skills, the binary matrix, labels and meta.

    The skill list is taken from the SAME ``cluster_model.joblib`` used by
    Module 5 (key ``"skills"``) so the network is defined over exactly the
    cluster-space skills.
    """
    bundle = joblib.load(MODEL_PATH)
    skills: list[str] = list(bundle["skills"])
    labels = np.asarray(bundle["labels"])
    df = pd.read_parquet(PROCESSED)
    X = df[skills].astype(float).values
    meta = json.loads(CLUSTER_META.read_text())
    return skills, X, labels, meta


def jaccard_matrix(X: np.ndarray) -> np.ndarray:
    """Return the skill x skill Jaccard similarity matrix (1 - jaccard distance).

    ``X`` has shape (n_developers, n_skills); it is transposed so that rows are
    skills, giving an 86 x 86 similarity matrix.
    """
    dist = pairwise_distances(X.T.astype(bool), metric="jaccard")
    sim = 1.0 - dist
    np.fill_diagonal(sim, 1.0)
    return sim


def build_graph(sim: np.ndarray, skills: list[str], threshold: float) -> nx.Graph:
    """Build an undirected graph with an edge for every pair above `threshold`."""
    G = nx.Graph()
    G.add_nodes_from(skills)
    n = len(skills)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim[i, j])
            if w > threshold:
                G.add_edge(skills[i], skills[j], weight=w)
    return G


# --------------------------------------------------------------------------- #
# cluster (persona) assignment
# --------------------------------------------------------------------------- #
def _cluster_assignment(skills: list[str], X: np.ndarray, labels: np.ndarray,
                        meta: dict) -> tuple[dict[str, int], list[int], dict[int, dict]]:
    """Assign each skill to the persona cluster where its prevalence lift is max.

    Lift(skill, cluster) = prevalence_in_cluster / prevalence_overall.  The
    persona table from ``cluster_meta.json`` supplies the cluster ids and the
    human-readable descriptions used in the legend.
    """
    personas = {int(p["cluster"]): p for p in meta.get("personas", [])}
    cluster_ids = sorted(personas) or sorted({int(c) for c in np.unique(labels)})

    overall = X.mean(axis=0)
    lift = np.zeros((len(cluster_ids), len(skills)))
    for k, c in enumerate(cluster_ids):
        mask = labels == c
        prev = X[mask].mean(axis=0) if mask.any() else np.zeros(len(skills))
        lift[k] = prev / np.maximum(overall, 1e-6)

    assignment = {s: cluster_ids[int(np.argmax(lift[:, i]))]
                  for i, s in enumerate(skills)}
    return assignment, cluster_ids, personas


def _persona_label(cluster: int, personas: dict[int, dict]) -> str:
    """Short legend label for a cluster, e.g. 'Persona 0 (.NET/C#)'."""
    top = personas.get(cluster, {}).get("top_skills", [])[:3]
    if not top:
        return f"Persona {cluster}"
    names = ", ".join(s["skill"].split("_", 1)[-1] for s in top)
    return f"Persona {cluster} ({names})"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _labels_overlap(fig: plt.Figure, texts: list) -> bool:
    """True if any pair of label bounding boxes overlaps on the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer=renderer) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                return True
    return False


def render_figure(G: nx.Graph, pos: dict, degree_cent: dict[str, float],
                  assignment: dict[str, int], cluster_ids: list[int],
                  personas: dict[int, dict], labels_top10: list[str],
                  threshold: float = THRESHOLD) -> None:
    """Render the spring-layout skill network to ``assets/skill_network.png``."""
    ASSETS.mkdir(exist_ok=True)
    cmap = plt.get_cmap("tab10")
    color_of = {c: cmap(i % 10) for i, c in enumerate(cluster_ids)}

    deg_vals = np.array([degree_cent[n] for n in G.nodes])
    lo, hi = float(deg_vals.min()), float(deg_vals.max())
    sizes = 120 + 1080 * (deg_vals - lo) / max(hi - lo, 1e-9)

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.18, width=0.8, edge_color="#888888")
    node_colors = [color_of[assignment.get(n, cluster_ids[0])] for n in G.nodes]
    ax.scatter([pos[n][0] for n in G.nodes], [pos[n][1] for n in G.nodes],
               s=sizes, c=node_colors, alpha=0.9, edgecolors="white", linewidths=0.6)

    # legend (one handle per persona cluster)
    for c in cluster_ids:
        ax.scatter([], [], s=160, color=color_of[c], label=_persona_label(c, personas))
    ax.legend(title="Persona cluster (highest prevalence lift)", loc="upper right",
              frameon=True, fontsize=9, title_fontsize=10)

    # labels on the highest-degree nodes, with overlap fallback
    cx = float(np.mean([pos[n][0] for n in G.nodes]))
    cy = float(np.mean([pos[n][1] for n in G.nodes]))
    label_texts = []
    for n in labels_top10:
        x, y = pos[n]
        dx, dy = x - cx, y - cy
        norm = float(np.hypot(dx, dy)) or 1.0
        # nudge the label radially outward from the graph centroid
        lx, ly = x + 0.045 * dx / norm, y + 0.045 * dy / norm
        ha = "left" if dx >= 0 else "right"
        label_texts.append(ax.text(lx, ly, n, fontsize=8, fontweight="bold",
                                   ha=ha, va="center", color="#222222",
                                   bbox=dict(boxstyle="round,pad=0.12",
                                             fc="white", ec="none", alpha=0.65)))
    if len(label_texts) > TOP_N_LABELS_FALLBACK and _labels_overlap(fig, label_texts):
        for t in label_texts[TOP_N_LABELS_FALLBACK:]:
            t.remove()
        print(f"  labels overlapped -> falling back to top-{TOP_N_LABELS_FALLBACK}")

    ax.set_title(f"Skill co-occurrence network (Jaccard > {threshold:.2f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def build_network(threshold: float = THRESHOLD, seed: int = 42) -> dict:
    """Build, analyse, render and export the skill co-occurrence network.

    Returns the meta dict that is also written to ``data/network_meta.json``.
    """
    DATA.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)

    skills, X, labels, meta = _load_inputs()
    print(f"loaded {X.shape[0]} developers x {len(skills)} cluster-space skills")

    sim = jaccard_matrix(X)
    G = build_graph(sim, skills, threshold)

    if G.number_of_edges() > HARD_MAX_EDGES and threshold < 0.40:
        threshold = 0.40
        G = build_graph(sim, skills, threshold)
        print(f"  edge count > {HARD_MAX_EDGES}; raised threshold to {threshold:.2f}")
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(threshold = {threshold:.2f})")

    # centrality
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight=None)
    try:
        eigen_cent = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:  # pragma: no cover - numerical fallback
        eigen_cent = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)

    assignment, cluster_ids, personas = _cluster_assignment(skills, X, labels, meta)

    # attach attributes
    for n in G.nodes:
        G.nodes[n]["degree"] = float(degree_cent[n])
        G.nodes[n]["betweenness"] = float(between_cent[n])
        G.nodes[n]["eigenvector"] = float(eigen_cent.get(n, 0.0))
        G.nodes[n]["cluster"] = int(assignment[n])

    top10 = sorted(degree_cent, key=degree_cent.get, reverse=True)[:TOP_N_LABELS]

    # spring layout
    pos = nx.spring_layout(G, k=2.0 / np.sqrt(G.number_of_nodes()),
                           iterations=100, seed=seed, weight="weight")
    render_figure(G, pos, degree_cent, assignment, cluster_ids, personas, top10,
                  threshold=threshold)

    # node-link JSON (include a "links" alias for app compatibility)
    link_data = nx.node_link_data(G)
    link_data.setdefault("links", link_data.get("edges", []))
    OUT_JSON.write_text(json.dumps(link_data, indent=2, default=float))

    result_meta = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "threshold": float(threshold),
        "top10_by_degree": [
            {"skill": s, "degree": round(float(degree_cent[s]), 4),
             "betweenness": round(float(between_cent[s]), 4),
             "eigenvector": round(float(eigen_cent.get(s, 0.0)), 4),
             "cluster": int(assignment[s])}
            for s in top10
        ],
    }
    OUT_META.write_text(json.dumps(result_meta, indent=2))
    print(f"  saved -> {OUT_PNG}, {OUT_JSON}, {OUT_META}")

    print("\nTop 10 skills by degree centrality:")
    for rank, s in enumerate(top10, 1):
        print(f"  {rank:2d}. {s:<28s} degree={degree_cent[s]:.3f}  "
              f"betweenness={between_cent[s]:.3f}  eigen={eigen_cent.get(s, 0.0):.3f}  "
              f"persona={assignment[s]}")

    return result_meta


# alias kept for the project's standard `run()` entry-point convention
run = build_network


if __name__ == "__main__":
    run()
