<!-- Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization | Author: Neguse Solomon Mekonnen | October 2026 -->
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization

**Authors:** Neguse Solomon Mekonnen, Wondesen Tsegaye Legesse  **Date:** October 2026
**Case study:** Jemo–Lancha culvert corridor, Nifas Silk-Lafto, Addis Ababa (four box-culvert nodes).
**Status:** Manuscript complete, ready for submission (target: Reliability Engineering & System Safety or Journal of Hydrology). Code and results are reproducible from the stored SWMM results.
ORCID 0009-0009-8145-4007.

## What this repository does
1. Samples storm, roughness, siltation and runoff uncertainty by Latin Hypercube Sampling (13 inputs).
2. Runs a high-fidelity EPA SWMM 5 dynamic-wave model for every sample and node (72 000 node results).
3. Trains surrogates (ANN ensemble, Gaussian process, polynomial baseline) on the SWMM results.
4. Uses the surrogate for failure probability (Monte Carlo), Sobol sensitivity, fragility curves and a condition-based maintenance matrix.

## Layout (one folder per module)
| Folder (`src/culvert_twin/…`) | Module | Main script (`scripts/…`) |
|---|---|---|
| `m0_config` | registry loader | – |
| `m1_baseline` | Rational method + Manning capacity, siltation, reproduction of registry values | `run_baseline.py` |
| `m2_idf_tc` | IDF sources (registry formula, candidate ERA, user ERA table), time of concentration | `run_baseline.py` |
| `m3_sampling` | uncertainty model, LHS design, algebraic Pf, rank sensitivity | `run_sampling.py` |
| `m4_swmm` | SWMM input builder + validator, PySWMM runner, parallel dataset worker | `check_swmm.py`, `diagnose_swmm.py`, `run_swmm_dataset.py`, `analyse_swmm_pilot.py` |
| `m5_dataset` | dataset assembly, sample-wise train/val/test split, tail test set | (used by `run_surrogates.py`) |
| `m6_surrogates` | ANN ensemble, GPR (Matérn-5/2 ARD), polynomial baseline, validation metrics | `run_surrogates.py` |
| `m7_reliability` | Pf by Monte Carlo, Sobol indices, fragility curves | `run_reliability.py` |
| `m8_maintenance` | maintenance matrix, thresholds, cleaning benefit | `run_maintenance.py` |
| `m10_figures` | provisional Figs 4–5 (IDF check, capacity vs siltation) | `run_baseline.py` |

Other folders: `config/` (all inputs, `registry.yaml`, the author's ERA IDF/Tc spreadsheet), `data/` (source registry PDF), `docs/` (planning Word documents), `outputs/` (every result CSV; figures in `outputs/paper_figures/`, the ten paper figures), `tests/` (47 unit tests).
(There is no module 9: validation metrics are part of module 6.)

## Reproduce (order matters)
    pip install -r requirements.txt
    python scripts/run_baseline.py                  # modules 0-2 -> outputs/*audit*.csv, fig04, fig05
    python scripts/run_sampling.py                  # module 3   -> lhs_*.csv, pf_summary.csv
    python scripts/check_swmm.py                    # SWMM sign-off checklist (11 checks)
    python scripts/run_swmm_dataset.py --n 2000 --jobs 4   # 18 000 SWMM tasks, ~17 min on 4 cores -> outputs/swmm_results.csv
    python scripts/run_surrogates.py                # modules 5-6 -> surrogate_metrics.csv, surrogate_predictions_test.csv
    python scripts/run_reliability.py               # module 7   -> reliability_pf.csv, sobol.csv, fragility.csv
    python scripts/run_maintenance.py               # module 8   -> maintenance_*.csv, fig_fragility.png, fig_maintenance_matrix.png
    python -m unittest discover -s tests            # 47 tests
    sh scripts/run_all.sh                           # everything except the SWMM run, in order
`outputs/swmm_results.csv` (72 000 rows) is included, so steps after the SWMM run can be reproduced without installing SWMM. Windows users: run from the repository root.

## Which code produced which result
| Result file | Script |
|---|---|
| `adopted_capacity_audit.csv`, `audit_*.csv`, `qcap_hypotheses.csv`, `reproduction_check.csv`, `implied_n.csv`, `tc_*.csv`, `*_idf_*check.csv`, `aacra_cross_check.csv`, `candidate_era_*.csv`, `flow_path_check.csv`, `fig04*`, `fig05*` | `run_baseline.py` |
| `lhs_design.csv`, `lhs_results_all_scenarios.csv`, `pf_summary.csv`, `convergence_*.csv`, `rank_sensitivity_*.csv` | `run_sampling.py` |
| `swmm_results.csv`, `example_model.inp` | `run_swmm_dataset.py`, `check_swmm.py` |
| `surrogate_metrics.csv`, `surrogate_predictions_test.csv` | `run_surrogates.py` |
| `reliability_pf.csv`, `reliability_test.csv`, `sobol.csv`, `fragility.csv` | `run_reliability.py` |
| `maintenance_thresholds.csv`, `maintenance_matrix.csv`, `maintenance_priority.csv`, `fig_fragility.png`, `fig_maintenance_matrix.png` | `run_maintenance.py` |

## Headline results (registry IDF formula unless stated; all conditional on the assumptions below)
* SWMM never floods when the Rational/Manning capacity factor CF ≤ 1.1 and always floods when CF > 1.5; effective threshold CF ≈ 1.2–1.3 (algebraic CF > 1 is conservative).
* Surrogate (ANN ensemble, held-out test): R² = 0.998 on flood volume, F1 = 0.995 for flood/no-flood, ≈ 8 000× faster than SWMM (indicative, different hardware).
* Surrogate Pf agrees with SWMM Pf within its 95 % interval in 35 of 36 scenario–node cells (largest difference 0.02).
* Failure probability at the 10-year storm: N3 German Sq. ≈ 0.51, N2 ≈ 0.32, N1 ≈ 0.22, N4 Lancha ≈ 0.10.
* Siltation β dominates (Sobol total-effect 0.84–0.89), then Tc (0.30–0.59), Manning n, runoff C.
* With a 5 % action limit the 10-year-storm cleaning triggers are β = 0.26 (N1), 0.19 (N2), 0.06 (N3), 0.33 (N4).

## Known limitations (read before citing)
* **The rainfall (IDF) source drives the answer.** Registry formula, candidate regional formula and the ERA table (Drainage Design Manual 2013, Chapter 5, as supplied and checked by the author) give mean 10-year failure probabilities of 29 %, 43 % and 90 %. All three are reported.
* The siltation range β ∈ [0, 0.5], Manning n, runoff-coefficient ranges, 1.0 m inlet headroom and overland roughness are **assumptions** (headroom 0–2 m changes Pf by at most 0.006, see `outputs/headroom_pf.csv`); no local measurements exist. Sobol "β dominates" is conditional on the β range.
* Barrel counts at N2/N4 and the series-versus-independent topology of the four nodes are unconfirmed (nodes are modelled as independent).
* No observed flood/discharge data were available, so SWMM itself is not validated against observations.
* "ANN ensemble" = scikit-learn feed-forward networks, not a deep-learning framework; the polynomial baseline is not a full orthogonal-polynomial PCE.
* Maintenance bins (Good/Fair/Poor/Critical) and the 5 % / 50 % limits are the author's choices; no cost data are used.

## License and citation
MIT licence (see `LICENSE`). Suggested citation (to be completed): Mekonnen, N. S., Legesse, W. T. (2026). *Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization*. Code repository.

## Headroom sensitivity and diagrams (added October 2026)

* `scripts/run_swmm_headroom.py --n 500 --jobs 4` reruns 500 samples with inlet headroom 0, 0.5, 1.0 and 2.0 m (registry and candidate IDF, 10 and 50 yr) and writes `outputs/swmm_headroom.csv` (32,000 rows).
* `scripts/analyse_headroom.py outputs/swmm_headroom.csv` writes `outputs/headroom_pf.csv` and `outputs/headroom_threshold.csv`. Result: Pf changes by at most 0.006 between 0 and 2 m.
* `scripts/make_diagrams.py` draws the workflow and corridor figures.

Authors: Neguse Solomon Mekonnen, Wondesen Tsegaye Legesse.
