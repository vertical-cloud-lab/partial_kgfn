#!/usr/bin/env python3
"""Action-friendly harness that runs the paper's actual pKGFN pipeline.

This wraps :func:`partial_kgfn.experiments.ackleyS_runner.main` (the AckleyS
function-network problem from Buathong et al. 2024) so each (algo, seed) cell
in the GitHub Actions matrix invokes the paper's *real* `run_one_trial`
implementation -- the same `partial_kgfn` package that produced the paper
figures, with no oracle stand-ins, fallbacks, or shortcuts.

Checkpointing
-------------
The paper's `run_one_trial` saves the full BO state (`trial_<seed>.pt`) at
the bottom of every BO iteration (see the ``torch.save(BO_results, ...
trial_<seed>.pt)`` call in ``partial_kgfn/run_one_trial.py``). We
call ``ackleyS_runner.main`` directly in-process, with a soft wall-clock
timeout enforced via ``signal.SIGALRM``. On timeout (or any other in-loop
exception) we still read the latest `.pt` checkpoint and emit a JSON
containing every BO iteration that completed before the interruption,
marked `complete: false`. Even if the runner is hard-killed before our
``finally`` block runs, the most recent checkpoint is still on disk and is
uploaded by the workflow (we only lose the iteration that was in flight).
The `combine` step then plots both the full trajectories and an
"early-budget cutoff" view trimmed to the smallest budget completed across
all runs in each algorithm, so the comparison stays fair even when some
cells time out.

Algorithms reproduced from the paper
------------------------------------
``Random``, ``EI``, ``KG``, ``KGFN``, ``EIFN``, ``TSFN``, ``pKGFN``
(Buathong et al. 2024, Sec. 5; Fig. 4 averages over 30 replications.)

CLI usage
---------
    # Quick smoke (small budget, fast algos only)
    python benchmarks/bofn_actions_benchmark.py --mode smoke

    # One full-mode (algo, seed) cell, writing JSON for the combiner
    python benchmarks/bofn_actions_benchmark.py --mode run \\
        --algo pKGFN --seed 1 --out benchmarks/results/actions/pKGFN_seed1.json

    # Aggregate per-cell JSONs into summary.csv + cost-efficiency plots
    python benchmarks/bofn_actions_benchmark.py --mode combine \\
        --input-dir partial-results/ --out-dir benchmarks/figures/actions/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paper's seven headline algorithms (Buathong et al. 2024, Sec. 5).
PAPER_ALGOS = ["Random", "EI", "KG", "KGFN", "EIFN", "TSFN", "pKGFN"]
# Algorithms small enough to be sanity-checked in the SMOKE pass; the heavy
# fantasy-based acquisitions (KGFN/EIFN/pKGFN) are exercised by the full
# matrix (and the `full-smoke` gating job) instead of here.
SMOKE_ALGOS = ["Random", "EI"]

# AckleyS function-network problem (paper's headline 1_49 cost split).
# `ackleyS_runner.main` always forces `noisy=True`, so the paper code writes
# the per-iteration checkpoint under `AckSN_1_49/<algo>/trial_<seed>.pt`
# (the trailing N marking the noisy variant); we read from the same path.
PROBLEM_NAME = "AckSN"
COSTS_KEY = "1_49"
PROBLEM_RESULTS_DIRNAME = f"{PROBLEM_NAME}_1_49"

# Default budgets. Paper uses 700; smoke uses a small fraction so the
# wiring check returns within a couple of minutes per algo on a CI runner.
DEFAULT_BUDGET = int(os.getenv("PKGFN_BUDGET", "700"))
SMOKE_BUDGET = int(os.getenv("PKGFN_SMOKE_BUDGET", "60"))

# Wall-clock budget per (algo, seed). Set just under the job-level
# `timeout-minutes` so the wrapper has time to read the .pt checkpoint and
# emit JSON before GitHub Actions hard-kills the runner. Enforced in-process
# via signal.SIGALRM.
TRIAL_TIMEOUT_SECONDS = int(os.getenv("PKGFN_TIMEOUT_SECONDS", str(330 * 60)))

REPO_ROOT = Path(__file__).resolve().parents[1]


class _TrialTimeout(Exception):
    """Raised by the SIGALRM handler when the in-process trial exceeds its budget."""


def _run_trial_inprocess(
    algo: str,
    seed: int,
    budget: int,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Invoke the paper's runner directly and return (completed, error_tail).

    We import and call ``partial_kgfn.experiments.ackleyS_runner.main`` in
    the current Python process. ``run_one_trial`` saves a fresh ``.pt``
    checkpoint at the bottom of every BO iteration, so even if the timeout
    fires (or the runner is hard-killed by Actions before our ``finally``
    block runs) the most recent snapshot is still on disk -- only the
    in-flight iteration is lost.

    A ``signal.SIGALRM`` is used as a soft, in-process wall-clock timeout
    (POSIX-only; GitHub Actions runners are Linux). On non-POSIX hosts the
    timeout is best-effort and only the workflow-level job timeout applies.
    """
    # Late import: keeps this module importable for `--mode combine` without
    # the heavy `partial_kgfn`/torch dependency stack. Ensure the repo root
    # is on ``sys.path`` so the in-tree package is importable when the
    # harness is run without an editable install.
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from partial_kgfn.experiments.ackleyS_runner import main as ackleyS_main

    have_alarm = hasattr(signal, "SIGALRM")
    prev_handler = None
    if have_alarm:
        def _alarm_handler(signum, frame):  # noqa: ARG001
            raise _TrialTimeout(f"trial exceeded {timeout_seconds}s wall-clock budget")

        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(timeout_seconds)))

    # `run_one_trial` writes ./results/AckSN_1_49/<algo>/trial_<seed>.pt
    # relative to the current working directory, so chdir to the repo root
    # for the duration of the call (and restore on exit).
    prev_cwd = os.getcwd()
    os.chdir(str(REPO_ROOT))
    try:
        ackleyS_main(
            trial=int(seed),
            algo=algo,
            costs=COSTS_KEY,
            budget=int(budget),
            noisy=True,
            impose_assump=False,
        )
        return True, ""
    except _TrialTimeout as exc:
        return False, f"{exc}"
    except Exception:
        return False, traceback.format_exc(limit=20)
    finally:
        if have_alarm:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)
        os.chdir(prev_cwd)


def _load_checkpoint(algo: str, seed: int) -> dict | None:
    """Load the latest per-iteration `.pt` checkpoint written by run_one_trial."""
    pt_path = REPO_ROOT / "results" / PROBLEM_RESULTS_DIRNAME / algo / f"trial_{seed}.pt"
    if not pt_path.exists():
        return None
    import torch  # local import; main module stays importable without torch

    res = torch.load(str(pt_path), weights_only=False, map_location="cpu")

    def _to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [_to_list(v) for v in x]
        return x

    cumulative_costs = _to_list(res.get("cumulative_costs", []))
    # The paper's run_one_trial stores `cumulative_costs[0] = None` for the
    # initial-design phase; `Visualization/utils_decoupled_kgfn.read_result`
    # coerces it to 0 before plotting. Mirror that here, and treat any
    # subsequent None as the previous cost (defensive).
    cleaned: list[float] = []
    for c in cumulative_costs:
        if c is None:
            cleaned.append(cleaned[-1] if cleaned else 0.0)
        else:
            cleaned.append(float(c))
    cumulative_costs = cleaned
    best_obs_vals = _to_list(res.get("best_obs_vals", []))
    best_post_means = _to_list(res.get("best_post_means", []))
    obj_at_best_designs = _to_list(res.get("obj_at_best_designs", []))
    runtimes = _to_list(res.get("runtimes", []))
    bo_budget = res.get("bo_budget")
    if hasattr(bo_budget, "item"):
        bo_budget = bo_budget.item()
    return {
        "bo_budget": bo_budget,
        "cumulative_costs": cumulative_costs,
        "best_obs_vals": best_obs_vals,
        "best_post_means": best_post_means,
        "obj_at_best_designs": obj_at_best_designs,
        "runtimes": runtimes,
        # `best_obs_vals` includes the initial-design snapshot at index 0
        # (see ``best_obs_vals = [best_obs_val]`` in run_one_trial), so the
        # number of *completed BO iterations* is one less than its length.
        "n_iterations_completed": max(0, len(best_obs_vals) - 1),
    }


def run_benchmark(algo: str, seed: int, budget: int | None = None,
                  timeout_seconds: int | None = None) -> dict:
    """Run one (algo, seed) trial of the paper's pipeline; always return a dict.

    On in-process timeout or any other exception we still read the latest
    checkpoint and return whatever progress was saved, with ``complete: false``.
    """
    if algo not in PAPER_ALGOS:
        raise ValueError(
            f"Unsupported algorithm {algo!r}. Expected one of {PAPER_ALGOS}."
        )
    budget = int(budget if budget is not None else DEFAULT_BUDGET)
    timeout_seconds = int(
        timeout_seconds if timeout_seconds is not None else TRIAL_TIMEOUT_SECONDS
    )

    ok, error_tail = _run_trial_inprocess(algo, seed, budget, timeout_seconds)
    ckpt = _load_checkpoint(algo, seed)

    if ckpt is None:
        # No checkpoint at all (the runner died before the first iteration).
        return {
            "algo": algo,
            "seed": seed,
            "problem": PROBLEM_NAME,
            "costs": COSTS_KEY,
            "budget": budget,
            "complete": False,
            "trial_ok": ok,
            "trial_error_tail": error_tail,
            "cumulative_costs": [],
            "best_obs_vals": [],
            "best_post_means": [],
            "obj_at_best_designs": [],
            "runtimes": [],
            "n_iterations_completed": 0,
            "final_cost": 0.0,
            "final_best_obs_val": None,
        }

    final_cost = float(ckpt["cumulative_costs"][-1]) if ckpt["cumulative_costs"] else 0.0
    final_best_obs = (
        float(ckpt["best_obs_vals"][-1]) if ckpt["best_obs_vals"] else None
    )
    # `complete` reflects whether the paper's BO loop exited cleanly. The
    # loop itself terminates when the remaining budget is below the cheapest
    # available node cost, which can leave `final_cost` slightly under
    # `budget`; that's still a complete run.
    complete = bool(ok)

    return {
        "algo": algo,
        "seed": seed,
        "problem": PROBLEM_NAME,
        "costs": COSTS_KEY,
        "budget": budget,
        "complete": complete,
        "trial_ok": ok,
        "trial_error_tail": error_tail if not ok else "",
        "final_cost": final_cost,
        "final_best_obs_val": final_best_obs,
        **ckpt,
    }


def write_json(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _trajectory_step(result: dict, x_grid: np.ndarray) -> np.ndarray:
    """Step-function interpolation of (cost, best_obs) onto a common grid.

    Anything past this run's last completed cost is left as the final
    observed value (typical right-extrapolation for cost-budget plots).
    """
    costs = np.asarray(result["cumulative_costs"], dtype=float)
    best = np.asarray(result["best_obs_vals"], dtype=float)
    if len(costs) == 0 or len(best) == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    if len(costs) == 1:
        return np.full_like(x_grid, best[0], dtype=float)
    # Step function: at cost c >= costs[i], best is best[i] (until costs[i+1]).
    out = np.empty_like(x_grid, dtype=float)
    for k, x in enumerate(x_grid):
        idx = np.searchsorted(costs, x, side="right") - 1
        if idx < 0:
            out[k] = best[0]
        else:
            out[k] = best[min(idx, len(best) - 1)]
    return out


def combine_results(input_dir: Path, out_dir: Path) -> None:
    """Aggregate per-cell JSONs into a CSV + 3 plots (full + early-cutoff)."""
    results = []
    for path in sorted(input_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        raise RuntimeError(f"No JSON results found in {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- summary.csv ---------------------------------------------------
    summary_path = out_dir / "summary.csv"
    fieldnames = [
        "algo", "seed", "complete", "trial_ok", "n_iterations_completed",
        "final_cost", "final_best_obs_val", "budget",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    algos = sorted({r["algo"] for r in results})

    # ---- full-budget mean cost-efficiency curve ------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in algos:
        runs = [r for r in results if r["algo"] == algo and r.get("cumulative_costs")]
        if not runs:
            continue
        max_cost = max(float(r["cumulative_costs"][-1]) for r in runs)
        if max_cost <= 0:
            continue
        xs = np.linspace(0.0, max_cost, 200)
        traj = np.vstack([_trajectory_step(r, xs) for r in runs])
        n_complete = sum(1 for r in runs if r.get("complete"))
        ax.plot(xs, np.nanmean(traj, axis=0),
                label=f"{algo} (n={len(runs)}, complete={n_complete})")
    ax.set_xlabel("Cumulative cost")
    ax.set_ylabel("Best observed value (mean over seeds)")
    ax.set_title(
        "BOFN cost-efficiency on AckleyS (paper pKGFN pipeline)\n"
        "extends each run to its own final cost; partial runs included"
    )
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_efficiency.png", dpi=150)
    plt.close(fig)

    # ---- early-cutoff plot: trim to the smallest *completed* run's final
    # cost so partial / timed-out runs don't bias the right edge. We only
    # consider runs whose BO loop reached the budget (``complete: true``).
    cutoff_per_algo = {
        algo: min(
            float(r["cumulative_costs"][-1])
            for r in results
            if r["algo"] == algo and r.get("complete") and r.get("cumulative_costs")
        )
        for algo in algos
        if any(
            r["algo"] == algo and r.get("complete") and r.get("cumulative_costs")
            for r in results
        )
    }
    global_cutoff = min(cutoff_per_algo.values()) if cutoff_per_algo else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in algos:
        runs = [r for r in results if r["algo"] == algo and r.get("cumulative_costs")]
        if not runs or global_cutoff <= 0:
            continue
        xs = np.linspace(0.0, global_cutoff, 200)
        traj = np.vstack([_trajectory_step(r, xs) for r in runs])
        ax.plot(xs, np.nanmean(traj, axis=0), label=f"{algo} (n={len(runs)})")
    ax.set_xlabel(f"Cumulative cost (early cutoff = {global_cutoff:g})")
    ax.set_ylabel("Best observed value (mean over seeds)")
    ax.set_title(
        "BOFN early-budget-cutoff comparison\n"
        "x-axis trimmed to the smallest completed cost across all runs"
    )
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "best_vs_cost.png", dpi=150)
    plt.close(fig)

    # ---- final-value boxplot at the early cutoff -----------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    box_data, box_labels = [], []
    for algo in algos:
        runs = [r for r in results if r["algo"] == algo and r.get("cumulative_costs")]
        if not runs or global_cutoff <= 0:
            continue
        finals = []
        for r in runs:
            traj = _trajectory_step(r, np.array([global_cutoff], dtype=float))
            finals.append(float(traj[0]))
        box_data.append(finals)
        box_labels.append(algo)
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels, showmeans=True)
    ax.set_ylabel(f"Best observed value at cost = {global_cutoff:g}")
    ax.set_title("BOFN final-value distribution at the early-cutoff budget")
    fig.tight_layout()
    fig.savefig(out_dir / "final_value_boxplot.png", dpi=150)
    plt.close(fig)


def run_smoke() -> None:
    result_dir = Path("benchmarks/results/actions_smoke")
    figure_dir = Path("benchmarks/figures/actions_smoke")
    # Use a short timeout for smoke so a wedged import surfaces fast.
    smoke_timeout = int(os.getenv("PKGFN_SMOKE_TIMEOUT_SECONDS", "1500"))
    for algo in SMOKE_ALGOS:
        result = run_benchmark(
            algo=algo, seed=1, budget=SMOKE_BUDGET, timeout_seconds=smoke_timeout
        )
        write_json(result, result_dir / f"{algo}_seed1.json")
    combine_results(result_dir, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or combine BOFN benchmark jobs.")
    parser.add_argument("--mode", choices=("smoke", "run", "combine"), required=True)
    parser.add_argument("--algo", choices=PAPER_ALGOS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None)
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
        result = run_benchmark(algo=args.algo, seed=args.seed, budget=args.budget)
        write_json(result, args.out)
    elif args.mode == "combine":
        if args.input_dir is None or args.out_dir is None:
            raise ValueError("--mode combine requires --input-dir and --out-dir")
        combine_results(args.input_dir, args.out_dir)


if __name__ == "__main__":
    main()
