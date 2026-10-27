"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/run_swmm_headroom.py

Sensitivity of the SWMM results to the ASSUMED inlet headroom (junction maximum depth = H(1 - beta) + headroom).
    py scripts/run_swmm_headroom.py --n 500 --jobs 4          # about 8-10 minutes on 4 cores
Writes outputs/swmm_headroom.csv (same columns as swmm_results.csv plus `headroom_m`).
The same LHS design is used for every headroom value, so differences are due to the headroom alone.
"""
import argparse, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m4_swmm.dataset import run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--headroom", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--tr", type=int, nargs="*", default=[10, 50]); ap.add_argument("--idf", nargs="*", default=["registry_formula", "candidate"])
    a = ap.parse_args()
    reg = load_registry(); OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
    design = sp.build_design(reg, a.n); rows = [design.iloc[k].to_dict() for k in range(len(design))]
    tasks = [(k, tr, src, rows[k], h) for h in a.headroom for src in a.idf for tr in a.tr for k in range(len(design))]
    print(f"{len(tasks)} tasks ({len(a.headroom)} headrooms x {len(a.idf)} IDF x {len(a.tr)} Tr x {a.n} samples), {a.jobs} worker(s)", flush=True)
    t0 = time.time(); out = []
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for n, chunk in enumerate(ex.map(run_task, tasks, chunksize=10), 1):
                out.append(chunk)
                if n % 200 == 0 or n == len(tasks):
                    el = time.time() - t0; print(f"  {n}/{len(tasks)} tasks, {el:.0f} s elapsed, ~{el / n * (len(tasks) - n):.0f} s left", flush=True)
    else:
        for n, t in enumerate(tasks, 1):
            out.append(run_task(t))
            if n % 200 == 0 or n == len(tasks): print(f"  {n}/{len(tasks)} tasks", flush=True)
    res = pd.DataFrame([r for chunk in out for r in chunk]); res.to_csv(OUT / "swmm_headroom.csv", index=False)
    nerr = int((res["error"] != "").sum()); print(f"done in {time.time()-t0:.0f} s -> {OUT/'swmm_headroom.csv'} ({len(res)} rows, {nerr} error rows)")


if __name__ == "__main__":
    main()
