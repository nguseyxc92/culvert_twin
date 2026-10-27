# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m4_swmm/dataset.py
# ======================================================================================
"""Worker-side code for building the SWMM training dataset. Lives in the package (not in the run script) so that Windows
multiprocessing ('spawn') can import it without re-executing a script. One task = one LHS sample x return period x IDF source,
which runs the four independent node models in one SWMM file and returns one row per node with inputs and SWMM outputs."""
from __future__ import annotations
import pandas as pd
from ..m0_config.config import load_registry
from ..m3_sampling import sampling as sp
from . import swmm_model as sm

_REG = None


def _reg():
    global _REG
    if _REG is None:
        _REG = load_registry()
    return _REG


def make_cases(row: dict, tr: int, src: str, reg=None, opt=None) -> list:
    reg = reg or _reg()
    d = pd.DataFrame([row])
    cases = []
    for nd in reg.nodes:
        tc, i = sp.node_inputs(reg, d, nd, tr, src)
        cases.append(sm.case_from_row(reg, nd, row, tc[0], i[0]))
    return cases


def run_task(task, runner=None) -> list[dict]:
    """task = (sample_index, Tr_yr, idf_source, design_row_dict[, inlet_headroom_m]). Never raises: failures come back as a row with `error` set.
    The optional fifth element overrides the assumed inlet headroom (default 1.0 m) for the headroom sensitivity run."""
    k, tr, src, row, *extra = task
    try:
        opt = sm.Options(inlet_headroom_m=float(extra[0])) if extra else sm.Options()
        cases = make_cases(row, tr, src, opt=opt)
        res = runner(cases) if runner else sm.run_cases(cases, opt)
        out = []
        for c, r in zip(cases, res):
            out.append({"sample": k, "Tr_yr": tr, "idf": src, "error": "", "Tc_min": round(c.tc_min, 4), "i_mm_hr": round(c.i_mm_hr, 4),
                        "n": round(c.n, 6), "beta": round(c.beta, 6), "C": round(c.C, 6), "L_overland_m": round(sm.overland_length_m(c, opt), 2),
                        "headroom_m": opt.inlet_headroom_m, **r})
        return out
    except Exception as e:  # keep the pool alive; the error is recorded in the data
        return [{"sample": k, "Tr_yr": tr, "idf": src, "node": "ALL", "error": repr(e)[:200]}]
