# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : tests/test_swmm_inp.py
# ======================================================================================
# Author: Neguse Solomon Mekonnen
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m4_swmm import swmm_model as sm


def fields(txt, prefix, section):
    lines = txt.splitlines(); a = lines.index(f"[{section}]") + 1
    for l in lines[a:]:
        if l.startswith("["): break
        if l.startswith(prefix + " "): return l.split()
    raise KeyError(prefix)


class TestInp(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry(); self.d = sp.build_design(self.reg, 20, seed=5, optimization=None)
        cs = []
        for nd in self.reg.nodes:
            tc, i = sp.node_inputs(self.reg, self.d.iloc[[3]], nd, 10, "registry_formula")
            cs.append(sm.case_from_row(self.reg, nd, self.d.iloc[3], tc[0], i[0]))
        self.cs = cs; self.txt = sm.build_inp(cs)

    def test_validates(self):
        self.assertEqual(sm.validate_inp(self.txt), [])

    def test_geometry_and_siltation(self):
        c = self.cs[0]
        self.assertAlmostEqual(c.H_eff, c.H_m * (1 - c.beta))
        f = fields(self.txt, f"C_{c.node_id}", "XSECTIONS")
        self.assertEqual(f[1], "RECT_CLOSED"); self.assertAlmostEqual(float(f[2]), c.H_eff, places=4); self.assertAlmostEqual(float(f[3]), c.B_m, places=4)

    def test_effective_rainfall_is_C_times_i(self):
        c = self.cs[1]
        f = fields(self.txt, f"TS_{c.node_id}", "TIMESERIES")
        self.assertEqual(f[1], "0:00"); self.assertAlmostEqual(float(f[2]), c.C * c.i_mm_hr, places=3)

    def test_storm_duration_equals_tc(self):
        c = self.cs[2]; d = int(round(c.tc_min))
        rows = [l for l in self.txt.splitlines() if l.startswith(f"TS_{c.node_id} ")]
        self.assertEqual(sum(1 for l in rows if not l.split()[2] == "0"), d)

    def test_junction_invert_is_slope_times_length(self):
        c = self.cs[0]; f = fields(self.txt, f"J_{c.node_id}", "JUNCTIONS")
        self.assertAlmostEqual(float(f[1]), c.slope * c.length_m, places=3)

    def test_validator_catches_errors(self):
        self.assertTrue(sm.validate_inp(self.txt.replace("J_N1 ", "J_X1 ", 1)))                  # renamed node breaks references
        self.assertTrue(any("missing section" in p for p in sm.validate_inp(self.txt.replace("[XSECTIONS]", "[XSECTIONZ]"))))
        self.assertTrue(sm.validate_inp(self.txt.replace("RG_N2", "RG_N1", 1)))                  # raingage duplicated / unresolved

    def test_tc_consistent_length_reaches_target_fraction(self):
        """Integrate SWMM's non-linear reservoir for the chosen length: outflow/rainfall at t = Tc must equal the target fraction."""
        import numpy as np
        from scipy.integrate import solve_ivp
        opt = sm.Options()
        for c in self.cs:
            L = sm.overland_length_m(c, opt)
            if 50.0 < L < 5000.0:
                i = c.i_eff / 3.6e6; k = (opt.subcatch_slope_pct / 100) ** 0.5 / (opt.n_overland * L)
                sol = solve_ivp(lambda t, d: i - k * np.maximum(d, 0) ** (5 / 3), [0, c.tc_min * 60], [0.0], rtol=1e-10, atol=1e-14)
                self.assertAlmostEqual(k * sol.y[0, -1] ** (5 / 3) / i, opt.target_fraction, delta=0.003)

    def test_reservoir_time_factor_known_values(self):
        self.assertAlmostEqual(sm._reservoir_time_factor(0.95), 2.36, delta=0.01)
        self.assertAlmostEqual(sm._reservoir_time_factor(0.98), 2.92, delta=0.01)

    def test_fixed_mode_uses_given_length(self):
        self.assertEqual(sm.overland_length_m(self.cs[0], sm.Options(flow_path_mode="fixed", flow_path_m=321.0)), 321.0)

    def test_width_equals_area_over_length(self):
        opt = sm.Options(); c = self.cs[0]; w = c.area_ha * 1e4 / sm.overland_length_m(c, opt)
        self.assertAlmostEqual(float(fields(self.txt, f"S_{c.node_id}", "SUBCATCHMENTS")[5]), w, places=2)

    def test_four_independent_systems(self):
        lines = self.txt.splitlines()
        for prefix in ("S_N", "J_N", "O_N", "C_N", "RG_N"):
            sect = {"S_N": "SUBCATCHMENTS", "J_N": "JUNCTIONS", "O_N": "OUTFALLS", "C_N": "CONDUITS", "RG_N": "RAINGAGES"}[prefix]
            a = lines.index(f"[{sect}]"); b = next(k for k in range(a + 1, len(lines)) if lines[k].startswith("["))
            self.assertEqual(sum(1 for l in lines[a:b] if l.startswith(prefix)), 4, sect)
        # no conduit connects two different nodes' junctions (independent topology)
        for l in lines:
            if l.startswith("C_N") and "RECT" not in l and len(l.split()) >= 9:
                f = l.split(); self.assertEqual(f[1][2:], f[0][2:]); self.assertEqual(f[2][2:], f[0][2:])

    def test_wide_numbers_never_merge(self):
        """Regression: a subcatchment width >= 10,000 m used to fill its column and merge with the next field."""
        c = sm.Case("N1", 1.5, 1.2, 0.012, 2500.0, 100.0, 0.015, 0.0, 0.9, 5.0, 300.0)
        opt = sm.Options(flow_path_mode="fixed", flow_path_m=50.0)            # width = 100e4/50 = 20000 m
        txt = sm.build_inp([c], opt)
        self.assertEqual(sm.validate_inp(txt), [])
        row = [l for l in txt.splitlines() if l.startswith("S_N1 ")][0].split()
        self.assertEqual(len(row), 8); self.assertAlmostEqual(float(row[5]), 20000.0, places=2); self.assertAlmostEqual(float(row[6]), 2.0)

    def test_validator_detects_merged_fields(self):
        bad = self.txt.replace("2.000", "2.000", 1)
        lines = bad.splitlines(); i = next(k for k, l in enumerate(lines) if l.startswith("S_N1 "))
        t = lines[i].split(); lines[i] = " ".join(t[:5]) + " " + t[5] + t[6] + " " + t[7]          # glue width and slope
        self.assertTrue(sm.validate_inp("\n".join(lines)))

    def test_whole_design_builds_valid_files(self):
        d = sp.build_design(self.reg, 300, seed=11, optimization=None); bad = 0
        for src in ("registry_formula", "candidate", "user_table"):
            for tr in (10, 25, 50):
                for k in range(0, 300, 7):
                    cs = []
                    for nd in self.reg.nodes:
                        tc, i = sp.node_inputs(self.reg, d.iloc[[k]], nd, tr, src); cs.append(sm.case_from_row(self.reg, nd, d.iloc[k], tc[0], i[0]))
                    bad += bool(sm.validate_inp(sm.build_inp(cs)))
        self.assertEqual(bad, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class HeadroomOption(unittest.TestCase):
    def test_headroom_changes_junction_max_depth(self):
        from culvert_twin.m3_sampling import sampling as sp
        from culvert_twin.m4_swmm import swmm_model as sm
        from culvert_twin.m4_swmm.dataset import make_cases
        reg = load_registry(); row = sp.build_design(reg, 4).iloc[0].to_dict()
        cases = make_cases(row, 10, "registry_formula", reg)
        t1 = sm.build_inp(cases, sm.Options(inlet_headroom_m=1.0)); t2 = sm.build_inp(cases, sm.Options(inlet_headroom_m=2.0))
        self.assertNotEqual(t1, t2); self.assertEqual(sm.validate_inp(t2), [])
