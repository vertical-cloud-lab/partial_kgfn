# pKGFN paper vs. CI smoke harness — comparison & reproduction notes

Source: Buathong, Wan, Astudillo, Daulton, Balandat, Frazier.
*Bayesian Optimization of Function Networks with Partial Evaluations.*
ICML 2024 (PMLR, CC-BY). PDF: `papers/pdfs/Buathong_etal_2024_pKGFN_ICML_PMLR-CC-BY.pdf`.
Full extracted plain text: `papers/extracted/Buathong_etal_2024_pKGFN.txt`.

## TL;DR

The CI workflow in this PR runs the **paper's actual `partial_kgfn`
pipeline** (this fork includes the upstream `partial_kgfn/` package
unchanged). Each (algo, seed) cell calls
`partial_kgfn.experiments.ackleyS_runner.main(...)`, which invokes
`run_one_trial(...)` with the paper's exact arguments (Ackley function
network, `costs="1_49"`, `n_init=2*dim+1`, `noisy=True`, paper's seven
algorithms). The harness (`benchmarks/bofn_actions_benchmark.py`) is a
thin wrapper that runs each trial in a subprocess with a wall-clock
timeout slightly under the GitHub Actions job timeout, then reads the
per-iteration `.pt` checkpoint that `run_one_trial` writes after every BO
iteration (`partial_kgfn/run_one_trial.py:588`) and emits a JSON. Even on
subprocess timeout/OOM the latest checkpoint is captured, marked
`complete: false`, and uploaded as an artifact, so `combine-results` can
plot an early-budget-cutoff comparison from the survivors.

## Paper experimental setup (as written in §6)

- Replications: **30** per (algorithm, problem, cost setting); curves show
  mean ±2 SE.
- Cost budget: **700** for all four problems in Figure 4.
- Algorithms compared: `p-KGFN`, `EIFN`, `KGFN`, `TSFN`, `EI`, `KG`,
  `Random` (`fast_pKGFN` is added in the follow-up Buathong & Frazier 2025
  paper, also in `papers/pdfs/`).
- Implementation: BoTorch. Original code:
  https://github.com/frazier-lab/partial_kgfn .
- Test problems and per-node costs:

  | Problem  | Network          | Costs                                     | Notes |
  |----------|------------------|-------------------------------------------|-------|
  | Ackley   | 2-stage (Fig 3a) | c1=1, c2=49 (sensitivity: 1/1, 1/9, 1/49) | f1 = −Ackley6D, f2(y)=−y·sin(5y/6π) |
  | Manu-GP  | multi-process    | c1=5, c2=10, c3=10, c4=45                 | each node is a GP-prior sample path |
  | FreeSolv | 2-stage (Fig 3a) | c1=1, c2=49                               | 642 molecules, VAE+PCA→3D inputs, two-stage hydration energy |
  | Pharma   | multi-output     | c1=1, c2=49 (f3 free, deterministic)      | ODT design; surrogates from Sano et al. 2020 |

## Plotting code in this repo

The figures in the paper were produced from the original frazier-lab
sources. This fork mirrors that pipeline at:

- [`Visualization/read_results_and_plot_graphs.ipynb`](../Visualization/read_results_and_plot_graphs.ipynb)
  — reproduces Figures 1 & 2 (main + cost sensitivity).
- [`Visualization/read_wallclock.ipynb`](../Visualization/read_wallclock.ipynb)
  — wall-clock timing.
- [`Visualization/utils_decoupled_kgfn.py`](../Visualization/utils_decoupled_kgfn.py)
  — `read_result(...)` helper used by both notebooks.

The `read_results_and_plot_graphs.ipynb` cell already encodes the
paper's exact experiment grid:

```python
trial_list = list(range(1, 31))   # 30 replications
algo_list  = ["pKGFN","fast_pKGFN","EIFN","TSFN","EI","Random","KG","KGFN"]
cost_problem = {
    "AckMat":  ["1_49"],
    "freesolv":["1_49"],
    "Manu":    ["5_10_10_45"],
}
```

## How to reproduce the paper's experiments

The CI harness in this PR (`benchmarks/bofn_actions_benchmark.py`)
already drives the paper's pipeline on the AckleyS problem at the
paper's exact settings — see the `BOFN benchmark` GitHub Actions
workflow. To reproduce on the other paper problems (Manu-GP, FreeSolv,
Pharma) locally:

1. Clone the original code: https://github.com/frazier-lab/partial_kgfn
   (this repo is a fork; the same scripts live under `partial_kgfn/`).
2. Build the conda environment: `conda env create -f pKGFN_env.yml`
   then `conda activate pKGFN`. (The CI runner uses the pinned
   `requirements.txt` instead.)
3. Run experiments via `run_experiment.ipynb` (top-level), choosing the
   `algo`, `problem`, `cost_config`, and `trial` (1–30) per cell.
   Each run writes a per-iteration `trial_<n>.pt` checkpoint to a
   per-(algo, problem, cost) directory under `./results/`.
4. Aggregate and plot with
   `Visualization/read_results_and_plot_graphs.ipynb`, which calls
   `read_result(...)` over the 30-trial directories and reproduces
   Figures 4–5 of the paper.

## CI harness vs. paper — what matches and what differs

| Aspect       | Paper (Buathong et al. 2024)                     | This PR's CI harness                                                                            |
|--------------|--------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Test fn      | Ackley6D / Manu-GP / FreeSolv / Pharma           | **Ackley6D** (paper's `AckleyFunctionNetwork`, `costs="1_49"`)                                  |
| Models       | GP surrogates per node (BoTorch)                 | **Same** — the paper's `partial_kgfn.models.decoupled_gp_network.GaussianProcessNetwork`        |
| Acquisition  | p-KGFN / KGFN / EIFN / TSFN / EI / KG / Random   | **Same seven** — the paper's `run_one_trial(...)` dispatch                                      |
| Budget       | 700 cost units                                   | **700** (default; configurable via `PKGFN_BUDGET` env)                                          |
| Costs        | Problem-specific (see table above)               | **1_49** (paper's headline AckleyS cost split)                                                  |
| Replicates   | 30                                               | **30** (matrix seeds 1..30)                                                                     |
| Output       | Figs 4–5: mean ±2 SE vs cost                     | `summary.csv`, full cost-efficiency PNG, **early-budget-cutoff** comparison PNG, final boxplot  |
| Checkpointing | per-iteration `trial_<n>.pt`                    | **Same** `.pt`, plus a JSON re-emitted from the latest checkpoint even on CI timeout            |

## Paper headline results (Fig. 4, qualitative ranking at budget=700)

From the paper text and extracted figure text (`papers/extracted/`),
the ordering at full budget is:

- **Ackley (c1=1, c2=49):** p-KGFN ≳ EIFN > KGFN ≈ TSFN > EI ≈ KG > Random.
- **Manu-GP:** p-KGFN clearly leads from very early in the budget.
- **FreeSolv:** p-KGFN reaches the best hydration-energy minimum first.
- **Pharma:** p-KGFN wins; baselines flatten earlier.

The CI workflow in this PR targets only the **Ackley (1_49)** row of
that table; the qualitative ranking of `pKGFN ≳ EIFN > KGFN ≈ TSFN > EI
≈ KG > Random` is what should reproduce when the matrix completes (or as
much of it as fits in the 6h-per-cell timeout — the more expensive
fantasy-based acquisitions will have shorter trajectories, which is why
the early-budget-cutoff plot exists).

## Caveats around CI timeouts

`KGFN`, `EIFN`, and especially `pKGFN` are computationally heavy: in the
paper a single Ackley `pKGFN` trial at budget=700 can take many CPU
hours. GitHub Actions caps each cell at 6 h, so some cells in the matrix
will hit the inner-subprocess wall-clock cap (`PKGFN_TIMEOUT_SECONDS`,
default 5h30m) before reaching the full budget. The wrapper still emits
a JSON from the latest `.pt` checkpoint and uploads both, marked
`complete: false`. The `combine` step then plots an
*early-budget-cutoff* comparison trimmed to the smallest completed cost
across all runs, so the comparison stays fair without dropping
incomplete cells.

## Token-frequency snapshot of the paper

Top content words in the extracted text (length > 3, ignoring stopwords):

```
function: 173   optimization: 125   node: 120   network: 82
p-kgfn: 81      evaluations: 79     bayesian: 75 partial: 64
networks: 60    nodes: 58           posterior: 54 acquisition: 51
problem: 45     evaluation: 45      input: 43    mean: 38
ackley: 33      cost: 32            eifn: 32     costs: 31
kgfn: 31        algorithm: 29       freesolv: 26 random: 25
```

This confirms the paper centres on `p-KGFN` / `KGFN` / `EIFN` over
function networks with explicit `cost` accounting, validated on
`Ackley` and `FreeSolv`.
