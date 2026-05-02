# pKGFN paper vs. CI smoke harness — comparison & reproduction notes

Source: Buathong, Wan, Astudillo, Daulton, Balandat, Frazier.
*Bayesian Optimization of Function Networks with Partial Evaluations.*
ICML 2024 (PMLR, CC-BY). PDF: `papers/pdfs/Buathong_etal_2024_pKGFN_ICML_PMLR-CC-BY.pdf`.
Full extracted plain text: `papers/extracted/Buathong_etal_2024_pKGFN.txt`.

## TL;DR

The CI workflow in this PR runs a **synthetic smoke harness**, not the
paper's experiments. Quantitatively the numbers cannot be compared. To
match the paper's experimental protocol the matrix has been bumped from
**20 → 30 seeds** (the paper reports 30 replications).

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

The CI harness in this PR (`benchmarks/bofn_actions_benchmark.py`) does
**not** reproduce the paper. To reproduce paper results:

1. Clone the original code: https://github.com/frazier-lab/partial_kgfn
   (this repo is a fork; the same scripts live under `partial_kgfn/`).
2. Build the conda environment: `conda env create -f pKGFN_env.yml`
   then `conda activate pKGFN`.
3. Run experiments via `run_experiment.ipynb` (top-level), choosing the
   `algo`, `problem`, `cost_config`, and `trial` (1–30) per cell.
   Each run writes JSON/pickle results to a per-(algo, problem, cost)
   directory.
4. Aggregate and plot with `Visualization/read_results_and_plot_graphs.ipynb`,
   which calls `read_result(...)` over the 30-trial directories and
   reproduces Figures 4–5 of the paper.

## CI smoke harness vs. paper — what differs

| Aspect      | Paper (Buathong et al. 2024)                    | This PR's CI harness                              |
|-------------|-------------------------------------------------|---------------------------------------------------|
| Test fn     | Ackley6D / Manu-GP / FreeSolv / Pharma          | Custom 4-node trig DAG (3 roots → 1 sink)         |
| Models      | GP surrogates per node (BoTorch)                | None — argmax of true value over a 13³ grid       |
| Acquisition | p-KGFN / KGFN / EIFN / TSFN / EI / KG           | Oracle argmax stand-ins (pKGFN ≈ HighFidOnly)     |
| Budget      | 700 cost units                                  | 400 cost units (smoke: smaller)                   |
| Costs       | Problem-specific (see table above)              | 5 / 10 / 40 per node (CI-tractable)               |
| Replicates  | **30**                                          | **30** (was 20; bumped to match)                  |
| Output      | Figs 4–5: mean ±2 SE vs cost                    | `summary.csv` + cost_efficiency / boxplot PNGs    |

## Paper headline results (Fig. 4, qualitative ranking at budget=700)

From the paper text and extracted figure text (`papers/extracted/`),
the ordering at full budget is:

- **Ackley (c1=1, c2=49):** p-KGFN ≳ EIFN > KGFN ≈ TSFN > EI ≈ KG > Random.
- **Manu-GP:** p-KGFN clearly leads from very early in the budget.
- **FreeSolv:** p-KGFN reaches the best hydration-energy minimum first.
- **Pharma:** p-KGFN wins; baselines flatten earlier.

Because this PR's harness uses oracle argmax stand-ins (no GP, no
acquisition), `pKGFN` ≈ `HighFidOnly` ≈ true grid-optimum value of
~1.4425, with `Random` ≈ 0.91 and `Sobol` ≈ 0.84. The qualitative
"network-aware ≥ random/quasi-random" ordering matches the paper, but
the *gap structure* between p-KGFN and the other GP-based baselines
that the paper documents cannot appear here — that's the contribution
of the actual acquisition function on a real GP, which the smoke
harness does not implement.

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
