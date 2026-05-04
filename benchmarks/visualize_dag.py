#!/usr/bin/env python3
"""Visualize function-network DAGs from the BoTorch-style test functions.

A "DAG" here matches the convention used in ``partial_kgfn.models.dag.DAG``
and the test-function classes under ``partial_kgfn/test_functions/``: a list
``parent_nodes`` of length ``n_nodes`` where ``parent_nodes[k]`` is the list
of parent indices of node ``k`` (an empty list marks a root). The terminal
(objective) node is the last node by convention in this repo.

The script can be used three ways:

1. As a CLI to render every test problem in the repo:
       python benchmarks/visualize_dag.py --all --out benchmarks/figures/dags
2. To render a single named problem:
       python benchmarks/visualize_dag.py --problem AckleyS --out figures/
3. As a library:
       from benchmarks.visualize_dag import plot_dag
       fig = plot_dag([[], [0]], node_costs=[1, 49], title="AckleyS")
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# Catalogue of DAGs used by the paper / repo.
#
# Sources:
#   partial_kgfn/test_functions/ackley_sin.py        (AckleyFunctionNetwork)
#   partial_kgfn/test_functions/ack_mat.py           (AckleyMatyasFunctionNetwork)
#   partial_kgfn/test_functions/freesolv3.py         (Freesolv3FunctionNetwork)
#   partial_kgfn/test_functions/manufacter_gp.py     (ManufacturingGPNetwork)
#   partial_kgfn/test_functions/pharmaceutical.py    (PharmaFunctionNetwork)
#   partial_kgfn/test_functions/GPs1.py              (GPs1)
#   partial_kgfn/test_functions/GPs2.py              (GPs2)
# Costs are the canonical "1_49"/"5_10_10_45"/etc. options from the runners.
# ---------------------------------------------------------------------------
PROBLEMS = {
    "AckleyS": dict(
        parent_nodes=[[], [0]],
        node_costs=[1, 49],
        node_dims=[6, 1],
        active_input_indices=[[0, 1, 2, 3, 4, 5], []],
        title="AckleyS (Ackley6D function network)",
    ),
    "AckleyMatyas": dict(
        parent_nodes=[[], [0]],
        node_costs=[1, 49],
        node_dims=[6, 2],
        active_input_indices=[[0, 1, 2, 3, 4, 5], [6]],
        title="Ackley-Matyas function network",
    ),
    "FreeSolv3": dict(
        parent_nodes=[[], [0]],
        node_costs=[1, 49],
        node_dims=[3, 1],
        active_input_indices=[[0, 1, 2], []],
        title="FreeSolv3 (3D solvation energy)",
    ),
    "Manufacturing": dict(
        parent_nodes=[[], [0], [], [1, 2]],
        node_costs=[5, 10, 10, 45],
        node_dims=[2, 1, 2, 2],
        active_input_indices=None,
        title="Manufacturing GP network",
    ),
    "Pharmaceutical": dict(
        parent_nodes=[[], []],
        node_costs=[1, 49],
        node_dims=[4, 4],
        active_input_indices=[[0, 1, 2, 3], [0, 1, 2, 3]],
        title="Pharmaceutical (parallel 2-node)",
    ),
    "GPs1": dict(
        parent_nodes=[[], [0]],
        node_costs=[1, 49],
        node_dims=[1, 1],
        active_input_indices=[[0], []],
        title="GPs1 (1D synthetic GP chain)",
    ),
    "GPs2": dict(
        parent_nodes=[[], [], [], [0, 1, 2]],
        node_costs=[1, 1, 1, 47],
        node_dims=[1, 1, 1, 3],
        active_input_indices=None,
        title="GPs2 (3 roots → final)",
    ),
}


def _layered_positions(parent_nodes: Sequence[Sequence[int]]) -> dict:
    """Assign each node an (x, y) position based on its topological generation.

    Generation 0 = roots; generation g+1 = 1 + max generation of parents.
    Nodes within the same generation are spread vertically and centred.
    """
    n = len(parent_nodes)
    gen = [0] * n
    for k in range(n):
        if parent_nodes[k]:
            gen[k] = 1 + max(gen[p] for p in parent_nodes[k])
    layers: dict = {}
    for k, g in enumerate(gen):
        layers.setdefault(g, []).append(k)
    pos = {}
    for g, nodes in layers.items():
        m = len(nodes)
        for i, k in enumerate(nodes):
            y = 0.0 if m == 1 else 1.0 - 2.0 * i / (m - 1)
            pos[k] = (float(g), y)
    return pos


def plot_dag(
    parent_nodes: Sequence[Sequence[int]],
    node_costs: Optional[Sequence] = None,
    node_dims: Optional[Sequence[int]] = None,
    active_input_indices: Optional[Sequence[Sequence[int]]] = None,
    node_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a function-network DAG.

    Args:
        parent_nodes: ``parent_nodes[k]`` is the list of parents of node ``k``
            (BoTorch / ``partial_kgfn.models.dag.DAG`` convention).
        node_costs: Optional per-node evaluation cost.
        node_dims: Optional per-node input dimension (incl. parent values).
        active_input_indices: Optional per-node global input indices.
        node_labels: Optional override for node labels (defaults to ``f_k``).
        title: Optional figure title.
        ax: Optional existing matplotlib axes.

    Returns:
        The matplotlib ``Figure`` containing the rendered DAG.
    """
    n = len(parent_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for k, parents in enumerate(parent_nodes):
        for p in parents:
            G.add_edge(p, k)

    pos = _layered_positions(parent_nodes)

    # Roots (no parents), terminal (last node by convention), internal (rest).
    roots = {k for k, ps in enumerate(parent_nodes) if not ps}
    terminal = n - 1
    colors = []
    for k in range(n):
        if k == terminal and n > 1:
            colors.append("#ff8c42")  # orange — objective
        elif k in roots:
            colors.append("#7fc97f")  # green — root input node
        else:
            colors.append("#9ab8d8")  # blue — internal

    n_layers = 1 + int(max(p[0] for p in pos.values()))
    max_per_layer = max(len([k for k in range(n) if int(pos[k][0]) == g]) for g in range(n_layers))
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(5.0, 2.4 * n_layers), max(3.5, 1.6 * max_per_layer + 1.5))
        )
    else:
        fig = ax.figure

    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowsize=18, width=1.4,
        edge_color="#444", node_size=2200,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=2200,
        edgecolors="#222", linewidths=1.2,
    )

    # Node labels: f_k with optional cost / dim annotations beneath.
    base_labels = node_labels or [f"$f_{{{k}}}$" for k in range(n)]
    nx.draw_networkx_labels(G, pos, labels={k: base_labels[k] for k in range(n)},
                            ax=ax, font_size=12, font_weight="bold")

    for k in range(n):
        annot_lines = []
        if node_costs is not None:
            annot_lines.append(f"c={node_costs[k]}")
        if node_dims is not None:
            annot_lines.append(f"d={node_dims[k]}")
        if active_input_indices is not None and active_input_indices[k]:
            idx = active_input_indices[k]
            annot_lines.append(f"x[{','.join(str(i) for i in idx)}]")
        if annot_lines:
            x, y = pos[k]
            ax.text(x, y - 0.22, "\n".join(annot_lines),
                    ha="center", va="top", fontsize=8, color="#333")

    # Mark the terminal/objective node.
    if n > 1:
        x, y = pos[terminal]
        ax.text(x, y + 0.22, "objective", ha="center", va="bottom",
                fontsize=8, color="#a04000", style="italic")

    if title:
        ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)

    # Legend.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="root (input)",
                   markerfacecolor="#7fc97f", markeredgecolor="#222", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="internal",
                   markerfacecolor="#9ab8d8", markeredgecolor="#222", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="objective (terminal)",
                   markerfacecolor="#ff8c42", markeredgecolor="#222", markersize=10),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=9)

    return fig


def dag_from_test_function(problem) -> dict:
    """Extract DAG metadata from an instantiated/declared BoTorch test problem.

    Reads the standard attributes used in this repo: ``parent_nodes`` (required),
    plus optional ``node_costs``, ``node_dims``, ``active_input_indices``.
    """
    return dict(
        parent_nodes=getattr(problem, "parent_nodes"),
        node_costs=getattr(problem, "node_costs", None),
        node_dims=getattr(problem, "node_dims", None),
        active_input_indices=getattr(problem, "active_input_indices", None),
    )


def _render_all(out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, spec in PROBLEMS.items():
        fig = plot_dag(**spec)
        path = os.path.join(out_dir, f"dag_{name}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--problem", choices=sorted(PROBLEMS), help="Render a single named problem.")
    parser.add_argument("--all", action="store_true", help="Render every problem in the catalogue.")
    parser.add_argument("--out", default="benchmarks/figures/dags",
                        help="Output directory for PNGs.")
    args = parser.parse_args()

    if not (args.all or args.problem):
        parser.error("pass --all or --problem NAME")

    os.makedirs(args.out, exist_ok=True)
    if args.all:
        paths = _render_all(args.out)
        for p in paths:
            print(p)
    else:
        spec = PROBLEMS[args.problem]
        fig = plot_dag(**spec)
        path = os.path.join(args.out, f"dag_{args.problem}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
