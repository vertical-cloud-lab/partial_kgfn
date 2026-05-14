# BOFN Literature

This directory is the canonical reference list for papers underlying the BOFN / pKGFN codebase.
Open-access PDFs (arXiv preprints and the PMLR CC-BY proceedings version) are committed under [`pdfs/`](./pdfs/) for offline reference; the original source URLs are listed below for each paper.

| Paper | Local PDF | Source |
|---|---|---|
| Astudillo & Frazier 2021 (BOFN) | [pdfs/Astudillo_Frazier_2021_BOFN_NeurIPS_arXiv-2110.06410.pdf](./pdfs/Astudillo_Frazier_2021_BOFN_NeurIPS_arXiv-2110.06410.pdf) | [arXiv:2110.06410](https://arxiv.org/pdf/2110.06410) |
| Buathong et al. 2024 (pKGFN, ICML) | [pdfs/Buathong_etal_2024_pKGFN_ICML_PMLR-CC-BY.pdf](./pdfs/Buathong_etal_2024_pKGFN_ICML_PMLR-CC-BY.pdf) | [PMLR v235 (CC-BY 4.0)](https://raw.githubusercontent.com/mlresearch/v235/main/assets/buathong24a/buathong24a.pdf) |
| Buathong & Frazier 2025 (Fast pKGFN) | [pdfs/Buathong_Frazier_2025_FastpKGFN_AutoML_arXiv-2506.11456.pdf](./pdfs/Buathong_Frazier_2025_FastpKGFN_AutoML_arXiv-2506.11456.pdf) | [arXiv:2506.11456](https://arxiv.org/pdf/2506.11456) |
| Frazier 2018 (BO tutorial) | [pdfs/Frazier_2018_BO_Tutorial_arXiv-1807.02811.pdf](./pdfs/Frazier_2018_BO_Tutorial_arXiv-1807.02811.pdf) | [arXiv:1807.02811](https://arxiv.org/pdf/1807.02811) |
| Jones, Schonlau, Welch 1998 (EGO) | *not committed (paywalled, no preprint)* | [Springer DOI](https://doi.org/10.1023/A:1008306431147) |

---

## Core BOFN Papers

### [1] Bayesian Optimization of Function Networks *(BOFN — original)*

| Field | Detail |
|---|---|
| **Authors** | Raul Astudillo, Peter I. Frazier |
| **Venue** | NeurIPS 2021 |
| **arXiv** | [arXiv:2110.06410](https://arxiv.org/abs/2110.06410) |
| **Proceedings** | [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/792c7b5aae4a79e78aaeda80516ae2ac-Abstract.html) |
| **DOI** | [10.48550/arXiv.2110.06410](https://doi.org/10.48550/arXiv.2110.06410) |

Introduces the BOFN framework: a joint GP model over all nodes of a directed-acyclic-graph (DAG) function network and the *Expected Improvement of Function Networks* (EIFN) acquisition function. Intermediate node outputs are leveraged to improve sample efficiency beyond standard BO.

---

### [2] Bayesian Optimization of Function Networks with Partial Evaluations *(pKGFN)*

| Field | Detail |
|---|---|
| **Authors** | Poompol Buathong, Jiayue Wan, Raul Astudillo, Samuel Daulton, Maximilian Balandat, Peter I. Frazier |
| **Venue** | ICML 2024 (pp. 4752–4784) |
| **arXiv** | [arXiv:2311.02146](https://arxiv.org/abs/2311.02146) |
| **Proceedings** | [PMLR v235](https://proceedings.mlr.press/v235/buathong24a.html) |
| **DOI** | [10.48550/arXiv.2311.02146](https://doi.org/10.48550/arXiv.2311.02146) |

Extends BOFN to settings where individual nodes carry separate evaluation costs and can be queried independently. Proposes the *Knowledge Gradient for Function Networks with Partial Evaluations* (pKGFN) acquisition function, which selects both the node and its input in a cost-aware manner, significantly reducing total evaluation cost versus full-network queries. **This is the primary paper for this repository.**

---

### [3] Fast Bayesian Optimization of Function Networks with Partial Evaluations *(Fast pKGFN)*

| Field | Detail |
|---|---|
| **Authors** | Poompol Buathong, Peter I. Frazier |
| **Venue** | AutoML 2025 |
| **arXiv** | [arXiv:2506.11456](https://arxiv.org/abs/2506.11456) |
| **OpenReview** | [KwykZvmTth](https://openreview.net/forum?id=KwykZvmTth) |
| **DOI** | [10.48550/arXiv.2506.11456](https://doi.org/10.48550/arXiv.2506.11456) |

Accelerates pKGFN by restricting to settings without upstream output requirements (suitable for manufacturing-style workflows). Combines EIFN candidate generation with a single pKGFN comparison per iteration, reducing the acquisition function optimization from combinatorial to a single solve. Achieves comparable optimization performance to pKGFN with up to 16× runtime reduction.

---

## Foundational BO References

### [4] Efficient Global Optimization of Expensive Black-Box Functions

| Field | Detail |
|---|---|
| **Authors** | Donald R. Jones, Matthias Schonlau, William J. Welch |
| **Venue** | Journal of Global Optimization, 13 (1998): 455–492 |
| **DOI** | [10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147) |

The foundational EGO/EI paper that introduced Bayesian optimization as a practical algorithm for expensive black-box functions.

---

### [5] Bayesian Optimization *(tutorial)*

| Field | Detail |
|---|---|
| **Authors** | Peter I. Frazier |
| **Venue** | *Recent Advances in Optimization and Modeling of Contemporary Problems*, INFORMS (2018): 255–278 |
| **arXiv** | [arXiv:1807.02811](https://arxiv.org/abs/1807.02811) |
| **DOI** | [10.1287/educ.2018.0188](https://doi.org/10.1287/educ.2018.0188) |

A widely cited tutorial overview of Bayesian optimization covering surrogate models, acquisition functions, and practical considerations.

---

## BibTeX

```bibtex
@inproceedings{astudillo2021bofn,
  title     = {Bayesian Optimization of Function Networks},
  author    = {Astudillo, Raul and Frazier, Peter I.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {14463--14475},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.06410}
}

@inproceedings{buathong2024pkgfn,
  title     = {Bayesian Optimization of Function Networks with Partial Evaluations},
  author    = {Buathong, Poompol and Wan, Jiayue and Astudillo, Raul and Daulton, Samuel
               and Balandat, Maximilian and Frazier, Peter I.},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {4752--4784},
  year      = {2024},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v235/buathong24a.html}
}

@inproceedings{buathong2025fastpkgfn,
  title     = {Fast Bayesian Optimization of Function Networks with Partial Evaluations},
  author    = {Buathong, Poompol and Frazier, Peter I.},
  booktitle = {AutoML 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.11456}
}

@article{jones1998ego,
  title   = {Efficient Global Optimization of Expensive Black-Box Functions},
  author  = {Jones, Donald R. and Schonlau, Matthias and Welch, William J.},
  journal = {Journal of Global Optimization},
  volume  = {13},
  pages   = {455--492},
  year    = {1998},
  doi     = {10.1023/A:1008306431147}
}

@incollection{frazier2018bo,
  title     = {Bayesian Optimization},
  author    = {Frazier, Peter I.},
  booktitle = {Recent Advances in Optimization and Modeling of Contemporary Problems},
  publisher = {INFORMS},
  pages     = {255--278},
  year      = {2018},
  url       = {https://arxiv.org/abs/1807.02811}
}
```
