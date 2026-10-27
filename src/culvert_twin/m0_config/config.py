# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m0_config/config.py
# ======================================================================================
"""Module 0: load the provisional registry. All numeric inputs come from config/registry.yaml."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = ROOT / "config" / "registry.yaml"


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    lat: float
    lon: float
    z_up: float
    z_down: float
    length_m: float
    slope_registry: float
    area_ha: float
    C: float
    culvert_class: str
    B_m: float
    H_m: float
    n: float

    @property
    def slope_from_dem(self) -> float:
        """(E_start - E_end) / L, the registry's own definition of bed slope."""
        return (self.z_up - self.z_down) / self.length_m


@dataclass(frozen=True)
class Registry:
    meta: dict
    status: dict
    design: dict
    idf: dict
    beta_grid: tuple
    nodes: tuple
    reported: dict
    uncertainty: dict
    tc_flow_path: dict

    def node(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)


def load_registry(path: str | Path = DEFAULT_PATH) -> Registry:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    nodes = tuple(Node(**d) for d in raw["nodes"])
    idf = dict(raw["idf"])
    idf["table_30min_mm_hr"] = {int(k): float(v) for k, v in idf["table_30min_mm_hr"].items()}
    return Registry(meta=raw["meta"], status=raw["status"], design=raw["design"], idf=idf,
                    beta_grid=tuple(raw["beta_grid"]), nodes=nodes, reported=raw["reported"], uncertainty=raw["uncertainty"], tc_flow_path=raw["tc_flow_path"])
