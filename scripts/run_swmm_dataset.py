# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/run_swmm_dataset.py
# ======================================================================================
"""Run the SWMM reference model on the LHS design and save outputs/swmm_results.csv (+ swmm_design_used.csv).
    py scripts/run_swmm_dataset.py --dry-run                 # write one .inp, validate it, no SWMM needed
    py scripts/run_swmm_dataset.py --n 5 --tr 10 --idf registry_formula --jobs 1     # 5-sample smoke test (~minutes)
    py scripts/run_swmm_dataset.py --n 200 --jobs 4          # pilot
    py scripts/run_swmm_dataset.py --n 2000 --jobs 4         # full dataset
Checkpoints are written to outputs/swmm_results_partial.csv every 500 tasks. Failed tasks are kept in the data with an `error` text."""
import argparse, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m4_swmm import swmm_model as sm
from culvert_twin.m4_swmm.dataset import run_task, make_cases


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=None); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--tr", type=int, nargs="*", default=None); ap.add_argument("--idf", nargs="*", default=None)
    a = ap.parse_args()
    reg = load_registry(); u = reg.uncertainty; OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
    design = sp.build_design(reg, a.n) if a.n else sp.build_design(reg)
    trs = a.tr or u["scenarios"]["return_periods_yr"]; srcs = a.idf or u["scenarios"]["idf_sources"]
    if a.dry_run:
        cs = make_cases(design.iloc[0].to_dict(), trs[0], srcs[0], reg); txt = sm.build_inp(cs); probs = sm.validate_inp(txt)
        p = OUT / "example_model.inp"; p.write_text(txt)
        print(f"wrote {p} ({len(txt.splitlines())} lines); validation problems: {probs or 'none'}"); return
    design.to_csv(OUT / "swmm_design_used.csv", index=False)
    rows = [design.iloc[k].to_dict() for k in range(len(design))]
    tasks = [(k, tr, src, rows[k]) for src in srcs for tr in trs for k in range(len(design))]
    print(f"{len(tasks)} tasks ({len(srcs)} IDF x {len(trs)} Tr x {len(design)} samples; 4 nodes each), {a.jobs} worker(s)", flush=True)
    t0 = time.time(); out = []

    def progress(n):
        if n % 100 == 0 or n == len(tasks):
            el = time.time() - t0; print(f"  {n}/{len(tasks)} tasks, {el:.0f} s elapsed, ~{el / n * (len(tasks) - n):.0f} s left", flush=True)
        if n % 500 == 0:
            pd.DataFrame([r for chunk in out for r in chunk]).to_csv(OUT / "swmm_results_partial.csv", index=False)

    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for n, chunk in enumerate(ex.map(run_task, tasks, chunksize=10), 1):
                out.append(chunk); progress(n)
    else:
        for n, t in enumerate(tasks, 1):
            out.append(run_task(t)); progress(n)
    res = pd.DataFrame([r for chunk in out for r in chunk]); res.to_csv(OUT / "swmm_results.csv", index=False)
    nerr = int((res["error"] != "").sum()); print(f"done in {time.time()-t0:.0f} s -> {OUT/'swmm_results.csv'} ({len(res)} rows, {nerr} error rows)")
    if nerr: print("first error:", res.loc[res["error"] != "", "error"].iloc[0])


if __name__ == "__main__":
    main()
