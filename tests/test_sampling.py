# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : tests/test_sampling.py
# ======================================================================================
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m2_idf_tc import tc as tcm


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry(); self.N = 500
        self.d = sp.build_design(self.reg, self.N, seed=1, optimization=None)

    def test_shape_and_names(self):
        self.assertEqual(self.d.shape, (self.N, 13))

    def test_latin_hypercube_stratification(self):
        """Every column must put exactly one sample in each of N equal-probability strata."""
        u = self.reg.uncertainty
        for col in self.d.columns:
            if col == "u_tc": lo, hi = 0.0, 1.0
            elif col.startswith("n_"): lo, hi = u["manning_n"]["low"], u["manning_n"]["high"]
            elif col.startswith("beta_"): lo, hi = u["beta"]["low"], u["beta"]["high"]
            else:
                nd = self.reg.node(col.split("_")[1]); lo = nd.C - u["C"]["halfwidth"]; hi = min(nd.C + u["C"]["halfwidth"], u["C"]["cap"])
            strata = np.floor((self.d[col].to_numpy() - lo) / (hi - lo) * self.N).astype(int)
            self.assertTrue(np.array_equal(np.sort(strata), np.arange(self.N)), col)

    def test_bounds(self):
        self.assertTrue(self.d.filter(like="beta_").to_numpy().max() <= 0.5)
        self.assertTrue(self.d.filter(like="n_").to_numpy().min() >= 0.012)
        self.assertTrue(self.d.filter(like="C_").to_numpy().max() <= 0.95 + 1e-12)

    def test_reproducible(self):
        pd.testing.assert_frame_equal(self.d, sp.build_design(self.reg, self.N, seed=1, optimization=None))

    def test_matches_deterministic_baseline(self):
        """One design row at the baseline point must reproduce tc.audit_at_tc (velocity Tc, candidate IDF, beta=0.30)."""
        row = {"u_tc": 0.0}
        for nd in self.reg.nodes:
            row.update({f"n_{nd.id}": nd.n, f"beta_{nd.id}": 0.30, f"C_{nd.id}": nd.C})
        r = sp.evaluate(self.reg, pd.DataFrame([row]), 10, "candidate").set_index("node")
        ref = tcm.audit_at_tc(self.reg, 10, source="candidate"); ref = ref[ref.Tc_basis == "velocity"].set_index("node")
        for node in r.index:
            self.assertAlmostEqual(float(r.loc[node, "CF"]), float(ref.loc[node, "CF"]), delta=0.01)

    def test_monotone_in_beta(self):
        base = {"u_tc": 0.5}
        for nd in self.reg.nodes: base.update({f"n_{nd.id}": 0.0135, f"C_{nd.id}": nd.C})
        cfs = []
        for b in (0.0, 0.25, 0.5):
            row = dict(base); row.update({f"beta_{nd.id}": b for nd in self.reg.nodes})
            cfs.append(float(sp.evaluate(self.reg, pd.DataFrame([row]), 10, "candidate").query("node=='N3'")["CF"].iloc[0]))
        self.assertTrue(cfs[0] < cfs[1] < cfs[2])

    def test_optimised_design_still_stratified(self):
        d = sp.build_design(self.reg, 100, seed=3)   # default random-cd optimisation
        strata = np.floor(d["u_tc"].to_numpy() * 100).astype(int)
        self.assertTrue(np.array_equal(np.sort(strata), np.arange(100)))

    def test_wilson_interval(self):
        lo, hi = sp.wilson(0, 100); self.assertEqual(lo, 0.0); self.assertAlmostEqual(hi, 0.0370, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
