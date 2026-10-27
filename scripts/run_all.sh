#!/bin/sh
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/run_all.sh
# Reproduces every result from the included outputs/swmm_results.csv (no SWMM install needed).
# The SWMM step itself (python scripts/run_swmm_dataset.py --n 2000 --jobs 4) needs pyswmm and ~17 min.
set -e
python scripts/run_baseline.py
python scripts/run_sampling.py
python scripts/run_surrogates.py outputs/swmm_results.csv
python scripts/run_reliability.py outputs/swmm_results.csv
python scripts/run_maintenance.py
python -m unittest discover -s tests
python scripts/make_diagrams.py outputs/paper_figures
