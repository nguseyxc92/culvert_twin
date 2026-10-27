# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m4_swmm/swmm_model.py
# ======================================================================================
"""Module 4: EPA SWMM 5 reference model (dynamic wave) for the four culvert nodes.

STATUS: written and syntax-validated here, NOT executed with SWMM in the authoring environment (PyPI/GitHub were
blocked). Run scripts/check_swmm.py on a machine with `pip install pyswmm` and send back outputs/swmm_check.csv.

Model structure per node k (independent systems, same assumption as the algebraic baseline):
    rain gage RG_k --> subcatchment S_k (100 % impervious, area A_k) --> junction J_k --> conduit C_k --> free outfall O_k
Modelling choices (every one is an assumption, listed so the paper can state them):
  * Rational runoff coefficient is honoured by scaling the rainfall: effective intensity = C_k * i (impervious surface, no losses).
  * Storm = rectangular block of duration Tc_k (the Rational convention) at the IDF intensity i(Tc_k, Tr).
  * Siltation beta: the box is modelled as a smaller closed rectangle, H_eff = H (1 - beta), width B unchanged. This is the same
    geometry as the algebraic baseline (wetted perimeter 2B + 2 H_eff).
  * Subcatchment width = area / overland length. DEFAULT ('tc_consistent'): the length is chosen so the catchment's kinematic-wave
    outflow reaches 98 % of the Rational peak at the end of a storm of duration Tc (so SWMM and the Rational Method differ only
    through culvert hydraulics and storage; see overland_length_m for the derivation).
    Alternative ('fixed'): one length for all nodes (the generic 1,000 m xlsx path, UNVERIFIED), which attenuates the peak.
    Slope 2 %, overland Manning n 0.015 (sealed surface; assumed).
  * Known behaviour (diagnosed 24 Oct 2026): conduit flow shows short transient spikes (~16 % above Manning) when the pipe first
    fills and when it changes from pressurised to free-surface flow. Steady surcharged flow agrees with Manning within 2 %.
    Outputs used for the surrogate are flood volume and peak flooding rate at the junction, not the conduit peak flow.
  * Junction maximum depth = H_eff + inlet headroom (default 1.0 m, ASSUMED). Water above it is lost as flooding (no ponding).
  * Surcharge deficit dQ_swmm = peak flooding rate at the junction (m3/s); failure = any flooding (volume > 1 m3).
"""
# Author: Neguse Solomon Mekonnen
from __future__ import annotations
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta

REQUIRED_SECTIONS = ["TITLE", "OPTIONS", "RAINGAGES", "SUBCATCHMENTS", "SUBAREAS", "INFILTRATION", "JUNCTIONS",
                     "OUTFALLS", "CONDUITS", "XSECTIONS", "TIMESERIES", "REPORT"]


@dataclass
class Options:
    inlet_headroom_m: float = 1.0        # ASSUMED
    n_overland: float = 0.015            # ASSUMED sealed surface
    flow_path_mode: str = "tc_consistent"  # "tc_consistent": overland length chosen so the SWMM catchment reaches equilibrium in ~Tc; "fixed": use flow_path_m
    target_fraction: float = 0.98        # fraction of the Rational peak the catchment reaches at the end of a storm of duration Tc
    flow_path_m: float = 1000.0          # used only when flow_path_mode == "fixed" (generic xlsx path, UNVERIFIED)
    subcatch_slope_pct: float = 2.0      # from the xlsx sheet-flow segment
    routing_step_s: int = 10
    report_step_s: int = 60
    tail_min: int = 180                  # simulation continues this long after the storm ends
    flood_volume_threshold_m3: float = 1.0


@dataclass
class Case:
    node_id: str
    B_m: float
    H_m: float
    slope: float
    length_m: float
    area_ha: float
    n: float
    beta: float
    C: float
    tc_min: float
    i_mm_hr: float
    barrels: int = 1

    @property
    def H_eff(self) -> float:
        return self.H_m * (1.0 - self.beta)

    @property
    def i_eff(self) -> float:
        return self.C * self.i_mm_hr


_J_CACHE: dict = {}


def _reservoir_time_factor(r: float) -> float:
    """J(r) = integral_0^{r^0.6} dx / (1 - x^(5/3)): time for SWMM's non-linear-reservoir catchment to reach a fraction r of its
    equilibrium outflow, in units of d_e / i (d_e = equilibrium depth, i = effective rainfall rate). J(0.95)=2.36, J(0.98)=2.92."""
    if r not in _J_CACHE:
        from scipy.integrate import quad
        _J_CACHE[r] = quad(lambda x: 1.0 / (1.0 - x ** (5.0 / 3.0)), 0.0, r ** 0.6)[0]
    return _J_CACHE[r]


def overland_length_m(c: Case, opt: Options) -> float:
    """Overland flow length of the SWMM subcatchment.
    'tc_consistent': SWMM's overland flow is a non-linear reservoir, dd/dt = i - (S^0.5/(n L)) d^(5/3). Its outflow reaches the
    fraction r = opt.target_fraction of the Rational peak at time t = J(r) * d_e / i, with d_e = (i n L / S^0.5)^(3/5). Setting
    t = Tc gives L = d_e^(5/3) S^0.5 / (n i). (Verified 24 Oct 2026 against the author's SWMM runs: predicted 0.81 / 0.99 / 1.00
    against SWMM 0.77 / 0.98 / 1.00 for L = 1000 / 300 / 100 m.) Limited to 50-5000 m.
    'fixed': one length for all samples (opt.flow_path_m)."""
    if opt.flow_path_mode == "fixed":
        return opt.flow_path_m
    S = opt.subcatch_slope_pct / 100.0
    i = max(c.i_eff, 1e-6) / 3.6e6                      # m/s
    d_e = c.tc_min * 60.0 * i / _reservoir_time_factor(opt.target_fraction)
    L = d_e ** (5.0 / 3.0) * S ** 0.5 / (opt.n_overland * i)
    return float(min(max(L, 50.0), 5000.0))


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


def _row(*cols) -> str:
    """Join (value, width) pairs into one line. Every field is followed by at least one space, so a value wider than its column
    (e.g. a subcatchment width >= 10,000 m) can never run into the next field."""
    out = []
    for v, w in cols:
        t = str(v)
        out.append(t.ljust(w) if len(t) < w else t + " ")
    return "".join(out).rstrip()


def build_inp(cases: list[Case], opt: Options | None = None, title: str = "culvert_twin SWMM reference") -> str:
    opt = opt or Options()
    dur_max = max(int(round(c.tc_min)) for c in cases)
    t_end = datetime(2026, 1, 1) + timedelta(minutes=dur_max + opt.tail_min)
    L = []
    add = L.append
    add("[TITLE]"); add(f";;{title}"); add(";;Author: Neguse Solomon Mekonnen"); add("")
    add("[OPTIONS]")
    for k, v in [("FLOW_UNITS", "CMS"), ("INFILTRATION", "HORTON"), ("FLOW_ROUTING", "DYNWAVE"), ("LINK_OFFSETS", "DEPTH"),
                 ("MIN_SLOPE", "0"), ("ALLOW_PONDING", "NO"), ("SKIP_STEADY_STATE", "NO"),
                 ("START_DATE", "01/01/2026"), ("START_TIME", "00:00:00"),
                 ("REPORT_START_DATE", "01/01/2026"), ("REPORT_START_TIME", "00:00:00"),
                 ("END_DATE", t_end.strftime("%m/%d/%Y")), ("END_TIME", t_end.strftime("%H:%M:%S")),
                 ("SWEEP_START", "01/01"), ("SWEEP_END", "12/31"), ("DRY_DAYS", "0"),
                 ("REPORT_STEP", f"00:{opt.report_step_s // 60:02d}:{opt.report_step_s % 60:02d}"),
                 ("WET_STEP", "00:01:00"), ("DRY_STEP", "00:05:00"),
                 ("ROUTING_STEP", f"00:00:{opt.routing_step_s:02d}"), ("INERTIAL_DAMPING", "PARTIAL"),
                 ("NORMAL_FLOW_LIMITED", "BOTH"), ("VARIABLE_STEP", "0.75"), ("LENGTHENING_STEP", "0"),
                 ("MIN_SURFAREA", "1.167"), ("MAX_TRIALS", "8"), ("HEAD_TOLERANCE", "0.0015"), ("MINIMUM_STEP", "0.5"),
                 ("THREADS", "1")]:
        add(_row((k, 21), (v, 0)))
    add("")
    add("[RAINGAGES]"); add(";;Name Format Interval SCF Source")
    for c in cases:
        add(_row((f"RG_{c.node_id}", 14), ("INTENSITY", 10), ("0:01", 9), ("1.0", 9), ("TIMESERIES", 11), (f"TS_{c.node_id}", 0)))
    add("")
    add("[SUBCATCHMENTS]"); add(";;Name Raingage Outlet Area %Imperv Width %Slope CurbLen")
    for c in cases:
        width = c.area_ha * 1.0e4 / overland_length_m(c, opt)
        add(_row((f"S_{c.node_id}", 14), (f"RG_{c.node_id}", 12), (f"J_{c.node_id}", 12), (f"{c.area_ha:.4f}", 10), ("100", 9),
                 (f"{width:.3f}", 12), (f"{opt.subcatch_slope_pct:.3f}", 9), ("0", 0)))
    add("")
    add("[SUBAREAS]"); add(";;Subcatchment N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo")
    for c in cases:
        add(_row((f"S_{c.node_id}", 14), (f"{opt.n_overland:.4f}", 11), ("0.15", 11), ("0", 11), ("0", 11), ("100", 11), ("OUTLET", 0)))
    add("")
    add("[INFILTRATION]"); add(";;Subcatchment MaxRate MinRate Decay DryTime MaxInfil")
    for c in cases:
        add(_row((f"S_{c.node_id}", 14), ("3", 11), ("0.5", 11), ("4", 11), ("7", 11), ("0", 0)))
    add("")
    add("[JUNCTIONS]"); add(";;Name Elevation MaxDepth InitDepth SurDepth Aponded")
    for c in cases:
        add(_row((f"J_{c.node_id}", 14), (f"{c.slope * c.length_m:.4f}", 12), (f"{c.H_eff + opt.inlet_headroom_m:.4f}", 12),
                 ("0", 11), ("0", 11), ("0", 0)))
    add("")
    add("[OUTFALLS]"); add(";;Name Elevation Type Gated")
    for c in cases:
        add(_row((f"O_{c.node_id}", 14), ("0", 11), ("FREE", 11), ("NO", 0)))
    add("")
    add("[CONDUITS]"); add(";;Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow")
    for c in cases:
        add(_row((f"C_{c.node_id}", 14), (f"J_{c.node_id}", 14), (f"O_{c.node_id}", 14), (f"{c.length_m:.2f}", 12), (f"{c.n:.5f}", 12),
                 ("0", 11), ("0", 11), ("0", 11), ("0", 0)))
    add("")
    add("[XSECTIONS]"); add(";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels")
    for c in cases:
        add(_row((f"C_{c.node_id}", 14), ("RECT_CLOSED", 13), (f"{c.H_eff:.4f}", 12), (f"{c.B_m:.4f}", 12), ("0", 11), ("0", 11), (str(c.barrels), 0)))
    add("")
    add("[TIMESERIES]"); add(";;Name Time Value   (effective rainfall intensity = C * i, mm/hr)")
    for c in cases:
        d = int(round(c.tc_min))
        for m in range(d):
            add(_row((f"TS_{c.node_id}", 14), (_hhmm(m), 11), (f"{c.i_eff:.4f}", 0)))
        add(_row((f"TS_{c.node_id}", 14), (_hhmm(d), 11), ("0", 0)))
        add(_row((f"TS_{c.node_id}", 14), (_hhmm(dur_max + opt.tail_min), 11), ("0", 0)))
    add("")
    add("[REPORT]"); add("INPUT      NO"); add("CONTROLS   NO"); add("SUBCATCHMENTS NONE"); add("NODES ALL"); add("LINKS ALL")
    add("")
    return "\n".join(L)


def validate_inp(text: str) -> list[str]:
    """Static checks of the generated input file: sections present, names unique, references resolve, numbers sane.
    Returns a list of problems (empty = passed). This does NOT replace running SWMM."""
    problems = []
    sec, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        m = re.match(r"^\[(\w+)\]$", line)
        if m:
            cur = m.group(1).upper(); sec[cur] = []; continue
        if cur is None:
            problems.append(f"data before first section: {line[:40]}"); continue
        sec[cur].append(line.split())
    for r in REQUIRED_SECTIONS:
        if r not in sec:
            problems.append(f"missing section [{r}]")
    if problems:
        return problems
    n_fields = {"RAINGAGES": 6, "SUBCATCHMENTS": 8, "SUBAREAS": 7, "INFILTRATION": 6, "JUNCTIONS": 6, "OUTFALLS": 4,
                "CONDUITS": 9, "XSECTIONS": 7, "TIMESERIES": 3}
    for s_name, nf in n_fields.items():
        for r in sec[s_name]:
            if len(r) != nf:
                problems.append(f"[{s_name}] row {r[0]} has {len(r)} fields, expected {nf} (merged or missing values?)")
    if problems:
        return problems
    names = lambda s: [r[0] for r in sec[s]]
    for s in ("RAINGAGES", "SUBCATCHMENTS", "JUNCTIONS", "OUTFALLS", "CONDUITS"):
        if len(set(names(s))) != len(names(s)):
            problems.append(f"duplicate names in [{s}]")
    nodes = set(names("JUNCTIONS")) | set(names("OUTFALLS"))
    ts = set(names("TIMESERIES")); gages = set(names("RAINGAGES")); links = set(names("CONDUITS"))
    for r in sec["RAINGAGES"]:
        if r[5] not in ts:
            problems.append(f"raingage {r[0]} -> unknown timeseries {r[5]}")
    for r in sec["SUBCATCHMENTS"]:
        if r[1] not in gages: problems.append(f"subcatchment {r[0]} -> unknown gage {r[1]}")
        if r[2] not in nodes: problems.append(f"subcatchment {r[0]} -> unknown outlet {r[2]}")
        if float(r[3]) <= 0 or float(r[5]) <= 0: problems.append(f"subcatchment {r[0]} non-positive area/width")
    for r in sec["CONDUITS"]:
        if r[1] not in nodes or r[2] not in nodes: problems.append(f"conduit {r[0]} references unknown node")
        if float(r[3]) <= 0 or float(r[4]) <= 0: problems.append(f"conduit {r[0]} non-positive length/roughness")
    xs = {r[0] for r in sec["XSECTIONS"]}
    if xs != links: problems.append("XSECTIONS and CONDUITS do not match")
    for r in sec["XSECTIONS"]:
        if r[1] != "RECT_CLOSED" or float(r[2]) <= 0 or float(r[3]) <= 0: problems.append(f"bad cross-section {r[0]}")
    for r in sec["JUNCTIONS"]:
        if float(r[2]) <= 0: problems.append(f"junction {r[0]} non-positive max depth")
    for r in sec["TIMESERIES"]:
        if len(r) != 3 or not re.match(r"^\d+:\d\d$", r[1]): problems.append(f"bad timeseries row {r}")
    for name in names("TIMESERIES"):
        tms = [int(r[1].split(":")[0]) * 60 + int(r[1].split(":")[1]) for r in sec["TIMESERIES"] if r[0] == name]
        if tms != sorted(tms): problems.append(f"timeseries {name} times not increasing")
    return problems


def run_cases(cases: list[Case], opt: Options | None = None, keep_dir: str | None = None, step_s: int = 10, series: dict | None = None) -> list[dict]:
    """Run SWMM through PySWMM and return one metrics dict per node. Requires `pip install pyswmm`.
    If `series` is a dict, it is filled with {node_id: [(t_min, flow_cms, depth_m, flooding_cms), ...]} for diagnostics."""
    try:
        from pyswmm import Simulation, Nodes, Links
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyswmm is required to run SWMM: pip install pyswmm") from e
    opt = opt or Options()
    text = build_inp(cases, opt)
    probs = validate_inp(text)
    if probs:
        raise ValueError("generated .inp failed validation: " + "; ".join(probs))
    ctx = None
    if keep_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix="swmm_", ignore_cleanup_errors=True); tmp = ctx.name
    else:
        tmp = keep_dir
    os.makedirs(tmp, exist_ok=True)
    inp = os.path.join(tmp, "model.inp")
    with open(inp, "w") as f:
        f.write(text)
    acc = {c.node_id: {"peak_flood_cms": 0.0, "flood_volume_m3": 0.0, "max_depth_m": 0.0, "peak_flow_cms": 0.0,
                       "surcharge_s": 0.0} for c in cases}
    try:
        return _run_sim(inp, cases, opt, step_s, series, acc)
    finally:
        if ctx is not None:
            ctx.cleanup()


def _run_sim(inp, cases, opt, step_s, series, acc) -> list[dict]:
    from pyswmm import Simulation, Nodes, Links
    with Simulation(inp) as sim:
        nodes, links = Nodes(sim), Links(sim)
        jn = {c.node_id: nodes[f"J_{c.node_id}"] for c in cases}
        cd = {c.node_id: links[f"C_{c.node_id}"] for c in cases}
        sim.step_advance(step_s)
        prev = sim.start_time
        for _ in sim:
            now = sim.current_time
            dt = (now - prev).total_seconds(); prev = now
            for c in cases:
                a, j = acc[c.node_id], jn[c.node_id]
                fl = float(j.flooding); dp = float(j.depth)
                a["peak_flood_cms"] = max(a["peak_flood_cms"], fl)
                a["flood_volume_m3"] += fl * dt
                a["max_depth_m"] = max(a["max_depth_m"], dp)
                a["peak_flow_cms"] = max(a["peak_flow_cms"], float(cd[c.node_id].flow))
                if dp > c.H_eff:
                    a["surcharge_s"] += dt
                if series is not None:
                    series.setdefault(c.node_id, []).append(((now - sim.start_time).total_seconds() / 60.0,
                                                             float(cd[c.node_id].flow), dp, fl))
        try:
            rerr, ferr = float(sim.runoff_error), float(sim.flow_routing_error)
        except Exception:
            rerr = ferr = float("nan")
    out = []
    for c in cases:
        a = acc[c.node_id]
        out.append({"node": c.node_id, **{k: round(v, 6) for k, v in a.items()},
                    "surcharge_min": round(a["surcharge_s"] / 60.0, 2), "runoff_cont_err_pct": rerr, "routing_cont_err_pct": ferr,
                    "failed_swmm": int(a["flood_volume_m3"] > opt.flood_volume_threshold_m3)})
    return out


def case_from_row(reg, nd, row, tc_min, i_mm_hr) -> Case:
    """Build a Case from a registry node and one LHS design row (dict-like)."""
    return Case(node_id=nd.id, B_m=nd.B_m, H_m=nd.H_m, slope=nd.slope_registry, length_m=nd.length_m, area_ha=nd.area_ha,
                n=float(row[f"n_{nd.id}"]), beta=float(row[f"beta_{nd.id}"]), C=float(row[f"C_{nd.id}"]),
                tc_min=float(tc_min), i_mm_hr=float(i_mm_hr), barrels=int(reg.design.get("barrels", {}).get(nd.id, 1)))
