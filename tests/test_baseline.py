# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : tests/test_baseline.py
# ======================================================================================
import math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m2_idf_tc import idf, tc
from culvert_twin.m1_baseline import baseline as bl


class TestCore(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry()

    def test_rational_units(self):
        # 1 ha, C=1, 100 mm/hr -> 0.278 m3/s
        self.assertAlmostEqual(bl.rational_peak(1.0, 100.0, 1.0), 0.278, places=6)

    def test_manning_hand_calc(self):
        # B=H=1 m closed box flowing full, S=0.001, n=0.013: R = 1/4, Q = (1/n) A R^(2/3) sqrt(S)
        q = float(bl.manning_capacity(1.0, 1.0, 0.001, 0.013, 0.0, "closed"))
        self.assertAlmostEqual(q, (1 / 0.013) * 1.0 * 0.25 ** (2 / 3) * math.sqrt(0.001), places=9)

    def test_fully_blocked_is_zero(self):
        self.assertEqual(float(bl.manning_capacity(1.5, 1.2, 0.01, 0.015, 1.0, "closed")), 0.0)

    def test_capacity_decreases_with_siltation(self):
        qs = [float(bl.manning_capacity(1.5, 1.2, 0.01, 0.015, b, "open")) for b in (0, .15, .3, .5, .75)]
        self.assertTrue(all(x > y for x, y in zip(qs, qs[1:])))

    def test_critical_beta_root(self):
        fn = lambda b: float(bl.manning_capacity(1.5, 1.2, 0.01, 0.015, b, "closed"))
        q_target = 0.5 * fn(0.0)
        b = bl.critical_beta(fn, q_target)
        self.assertAlmostEqual(fn(b), q_target, places=6)
        self.assertEqual(bl.critical_beta(fn, 2 * fn(0.0)), 0.0)

    def test_slopes_match_registry(self):
        for n in self.reg.nodes:
            self.assertAlmostEqual(n.slope_from_dem, n.slope_registry, places=4)

    def test_registry_qpeak_reproduces(self):
        i10 = idf.design_intensity(self.reg, "registry_table")
        for n in self.reg.nodes:
            rep = self.reg.reported["audit_10yr_beta030"][n.id]["qpeak"]
            self.assertAlmostEqual(bl.rational_peak(n.C, i10, n.area_ha), rep, delta=0.02)

    def test_idf_formula_value(self):
        p = self.reg.idf
        self.assertAlmostEqual(float(idf.intensity(10, 30, p["a"], p["b"], p["c"], p["m"])), 41.7, delta=0.1)

    def test_implied_duration_roundtrip(self):
        p = self.reg.idf
        i = float(idf.intensity(25, 45, p["a"], p["b"], p["c"], p["m"]))
        self.assertAlmostEqual(float(idf.implied_duration(25, i, p["a"], p["b"], p["c"], p["m"])), 45, places=6)

    def test_known_failures_are_detected(self):
        """The reproduction check must FAIL on the known inconsistencies, not hide them."""
        df = bl.reproduce_registry(self.reg)
        qcap = df[df["item"].str.startswith("Q_cap beta=0.00 (closed)")]
        self.assertTrue((qcap["result"] == "FAIL").all())

    def test_adopted_method_is_closed(self):
        self.assertEqual(self.reg.design["capacity_method"], "closed")

    def test_aacra_cross_check_flags_100yr(self):
        xc = idf.aacra_cross_check(self.reg)
        r = xc.set_index("Tr_yr")["registry_30min_le_15min"]
        self.assertTrue(bool(r[2]))        # 42.5 <= 63 : consistent
        self.assertFalse(bool(r[100]))     # 108.5 > 105 : violates monotonic decrease with duration

    def test_kirpich_hand_calc(self):
        # L=1000 m, S=0.01: 0.0195 * 1000**0.77 * 0.01**-0.385
        self.assertAlmostEqual(float(tc.kirpich_min(1000, 0.01)), 0.0195 * 1000 ** 0.77 * 0.01 ** -0.385, places=9)
        self.assertAlmostEqual(float(tc.kirpich_min(1000, 0.01)), 0.0195 * 204.17 * 5.888, delta=0.2)

    def test_kirpich_longer_path_longer_tc(self):
        self.assertGreater(float(tc.kirpich_min(2000, 0.01)), float(tc.kirpich_min(1000, 0.01)))
        self.assertGreater(float(tc.kirpich_min(1000, 0.005)), float(tc.kirpich_min(1000, 0.02)))

    def test_user_table_exact_at_nodes(self):
        self.assertAlmostEqual(float(idf.user_table_intensity(self.reg, 10, 30)), 83.0, places=6)
        self.assertAlmostEqual(float(idf.user_table_intensity(self.reg, 100, 5)), 250.0, places=6)

    def test_user_table_interpolation_between_nodes(self):
        v = float(idf.user_table_intensity(self.reg, 10, 45)); self.assertTrue(52.0 < v < 83.0)

    def test_user_table_consistency_flags(self):
        df = idf.user_table_checks(self.reg)
        self.assertTrue(df.attrs["monotone_in_duration"] and df.attrs["monotone_in_Tr"])

    def test_flow_path_arithmetic(self):
        df = tc.flow_path_table(self.reg)
        self.assertAlmostEqual(df.attrs["total_stated"], 23.9, places=6)
        self.assertTrue((abs(df["stated_min"] - df["recomputed_min"]) < 0.06).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
