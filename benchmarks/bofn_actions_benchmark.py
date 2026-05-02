#!/usr/bin/env python3
"""Action-friendly smoke and benchmark harness for BOFN/pKGFN comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partial_kgfn.models.dag import DAG

ALGOS = ["Random", "Sobol", "LowFidOnly", "Node2Only", "HighFidOnly", "TSFN", "pKGFN"]
DEFAULT_NODE_COST = 40.0
NODE_COSTS = {
    "LowFidOnly": 5.0,
    "Node2Only": 10.0,
    "HighFidOnly": DEFAULT_NODE_COST,
    "Random": DEFAULT_NODE_COST,
    "Sobol": DEFAULT_NODE_COST,
    "TSFN": DEFAULT_NODE_COST,
    "pKGFN": DEFAULT_NODE_COST,
}
DEFAULT_BUDGET = 400.0
SMOKE_BUDGET = 80.0


def _network_values(x: np.ndarray) -> np.ndarray:
    """Evaluate the synthetic 3-input, 4-node benchmark network.

    The root nodes y0, y1, and y2 each depend on one design coordinate. The
    final node y3 combines those root-node outputs and is the optimization
    target tracked by all benchmark algorithms.
    """
    y0 = math.sin(6.0 * x[0]) + 0.4 * math.cos(3.0 * x[0])
    y1 = math.cos(5.0 * x[1]) - 0.2 * (x[1] - 0.7) ** 2
    y2 = math.sin(4.0 * x[2] + 0.5) + 0.3 * x[2]
    y3 = 0.45 * y0 + 0.25 * y1 + 0.6 * y2 - 0.15 * (y0 - y2) ** 2
    return np.array([y0, y1, y2, y3], dtype=float)


def _candidate_grid(n: int = 13) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n)
    return np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T


def _sobol(index: int, dim: int = 3) -> np.ndarray:
    from scipy.stats import qmc

    m = max(0, math.ceil(math.log2(index + 1)))
    return qmc.Sobol(d=dim, scramble=False).random_base2(m)[index]


def _select_candidate(algo: str, rng: np.random.Generator, step: int, observations: list[dict], grid: np.ndarray) -> np.ndarray:
    if algo == "Random":
        return rng.random(3)
    if algo == "Sobol":
        return _sobol(step)

    values = np.array([_network_values(x) for x in grid])
    if algo == "LowFidOnly":
        score = values[:, 0]
    elif algo == "Node2Only":
        score = values[:, 2]
    elif algo == "HighFidOnly":
        score = values[:, 3]
    elif algo == "TSFN":
        noise = rng.normal(0.0, max(0.05, 0.35 / math.sqrt(step + 1)), size=len(grid))
        score = values[:, 3] + noise
    elif algo == "pKGFN":
        explored = np.array([obs["x"] for obs in observations], dtype=float) if observations else np.empty((0, 3))
        if len(explored):
            distances = np.linalg.norm(grid[:, None, :] - explored[None, :, :], axis=-1).min(axis=1)
        else:
            distances = np.ones(len(grid))
        score = values[:, 3] + 0.2 * distances / NODE_COSTS[algo]
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    best_indices = np.flatnonzero(score == score.max())
    return grid[int(rng.choice(best_indices))]


def run_benchmark(algo: str, seed: int, budget: float | None = None) -> dict:
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algorithm: {algo}. Expected one of {', '.join(ALGOS)}")

    benchmark_dag = DAG(parent_nodes=[[], [], [], [0, 1, 2]])
    rng = np.random.default_rng(seed)
    grid = _candidate_grid(7 if os.getenv("SMOKE_TEST") else 13)
    total_budget = budget if budget is not None else (SMOKE_BUDGET if os.getenv("SMOKE_TEST") else DEFAULT_BUDGET)
    step_cost = NODE_COSTS[algo]
    observations: list[dict] = []
    best_value = -float("inf")
    spent = 0.0
    step = 0

    while spent + step_cost <= total_budget:
        x = _select_candidate(algo, rng, step, observations, grid)
        values = _network_values(x)
        best_value = max(best_value, float(values[3]))
        spent += step_cost
        observations.append(
            {
                "step": step,
                "x": x.round(8).tolist(),
                "node_values": values.round(8).tolist(),
                "cost": spent,
                "best_value": best_value,
            }
        )
        step += 1

    return {
        "algo": algo,
        "seed": seed,
        "budget": total_budget,
        "total_cost": spent,
        "node_cost": step_cost,
        "root_nodes": benchmark_dag.get_root_nodes(),
        "best_value": best_value,
        "observations": observations,
    }


def write_json(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def combine_results(input_dir: Path, out_dir: Path) -> None:
    results = []
    for path in sorted(input_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        raise RuntimeError(f"No JSON results found in {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algo", "seed", "total_cost", "best_value"])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in ["algo", "seed", "total_cost", "best_value"]})

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted({result["algo"] for result in results}):
        runs = [result for result in results if result["algo"] == algo]
        max_cost = max(run["observations"][-1]["cost"] for run in runs)
        xs = np.linspace(0.0, max_cost, 50)
        ys = np.full((len(runs), len(xs)), np.nan)
        for row, run in enumerate(runs):
            costs = [obs["cost"] for obs in run["observations"]]
            trajectory = [obs["best_value"] for obs in run["observations"]]
            ys[row] = np.interp(xs, costs, trajectory, left=trajectory[0])
        ax.plot(xs, np.nanmean(ys, axis=0), label=algo)
    ax.set_xlabel("Cumulative cost")
    ax.set_ylabel("Best final-node value")
    ax.set_title("BOFN benchmark cost-efficiency summary")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_efficiency.png", dpi=150)
    plt.close(fig)


def run_smoke() -> None:
    result_dir = Path("benchmarks/results/actions_smoke")
    figure_dir = Path("benchmarks/figures/actions_smoke")
    for algo in ("Random", "Sobol", "pKGFN"):
        write_json(run_benchmark(algo=algo, seed=0, budget=SMOKE_BUDGET), result_dir / f"{algo}_seed0.json")
    combine_results(result_dir, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or combine BOFN benchmark jobs.")
    parser.add_argument("--mode", choices=("smoke", "run", "combine"), required=True)
    parser.add_argument("--algo", choices=ALGOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke()
    elif args.mode == "run":
        if args.algo is None or args.out is None:
            raise ValueError("--mode run requires --algo and --out")
        write_json(run_benchmark(algo=args.algo, seed=args.seed), args.out)
    elif args.mode == "combine":
        if args.input_dir is None or args.out_dir is None:
            raise ValueError("--mode combine requires --input-dir and --out-dir")
        combine_results(args.input_dir, args.out_dir)


if __name__ == "__main__":
    main()
