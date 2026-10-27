# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : tests/test_dataset.py
# ======================================================================================
import subprocess, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m4_swmm import dataset as ds


def fake_runner(cases):
    return [{"node": c.node_id, "peak_flood_cms": 0.0, "flood_volume_m3": 0.0, "max_depth_m": 0.5, "peak_flow_cms": 1.0,
             "surcharge_s": 0.0, "surcharge_min": 0.0, "runoff_cont_err_pct": 0.0, "routing_cont_err_pct": 0.0, "failed_swmm": 0} for c in cases]


def boom(cases):
    raise RuntimeError("simulated SWMM failure")


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry(); self.row = sp.build_design(self.reg, 10, seed=2, optimization=None).iloc[3].to_dict()

    def test_make_cases_four_nodes(self):
        cs = ds.make_cases(self.row, 10, "user_table", self.reg); self.assertEqual([c.node_id for c in cs], ["N1", "N2", "N3", "N4"])

    def test_task_schema_with_stub_runner(self):
        out = ds.run_task((3, 25, "candidate", self.row), runner=fake_runner)
        self.assertEqual(len(out), 4)
        for r in out:
            for key in ("sample", "Tr_yr", "idf", "node", "Tc_min", "i_mm_hr", "n", "beta", "C", "L_overland_m", "flood_volume_m3", "failed_swmm", "error"):
                self.assertIn(key, r)
            self.assertEqual(r["error"], "")

    def test_failure_is_recorded_not_raised(self):
        out = ds.run_task((3, 10, "candidate", self.row), runner=boom)
        self.assertEqual(len(out), 1); self.assertIn("simulated SWMM failure", out[0]["error"])

    def test_script_imports_cleanly_and_dry_runs(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_swmm_dataset.py"), "--dry-run", "--n", "5"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("validation problems: none", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
