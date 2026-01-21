#!/usr/bin/env python3
"""
Run Baseline 3 (ILP) for EVERY attribute appearing in each DC file in a directory,
and write one CSV per DC file.

✅ No argparse.
✅ You set ONE directory path (DC_DIR) and it does the rest.
✅ Evaluation-only (no MySQL, no deletion).
✅ Expects each DC file to define: denial_constraints = [ ... ]

It will:
  - discover all *.py files in DC_DIR
  - import each one
  - build hyperedges from DCs
  - run Baseline 3 once per attribute mentioned in any DC
  - write CSV: <OUT_DIR>/<dc_filename_without_ext>__baseline3_per_attr.csv

Requires: gurobipy installed + licensed.

Edit these two:
  DC_DIR = "/mnt/data"
  OUT_DIR = "./baseline3_out"
"""

from __future__ import annotations

import csv
import os
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import deque
import importlib.util

# =========================
# CONFIG (EDIT THESE)
# =========================
DC_DIR = "/Users/adhariya/src/DiffDel/DCandDelset/dc_configs"          # directory containing top*DCs_parsed.py
OUT_DIR = "./baseline3_out"   # where CSVs will be written
WRITE_ILP_MODELS = False      # if True, writes ilp_models/<dcfile>__<attr>.lp
DUMMY_KEY = 0                # dummy key for CellILP IDs

# Optional: only run files that match this substring (set to "" to disable)
ONLY_FILES_CONTAINING = "DCs_parsed"  # e.g. "topAirport", ""


# =========================
# GUROBI
# =========================
try:
    from gurobipy import Model, GRB, quicksum  # type: ignore
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False


@dataclass(frozen=True)
class CellILP:
    attribute: str   # "t1.attr"
    key: int


def _extract_attr_token(s: Any) -> Optional[str]:
    """If s looks like 't1.attr' or 't2.attr', return 'attr', else None."""
    if not isinstance(s, str):
        return None
    if s.startswith("t1.") or s.startswith("t2."):
        parts = s.split(".", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
    return None


def load_denial_constraints(py_path: str) -> List[List[Tuple[Any, Any, Any]]]:
    """Loads a python file and returns its denial_constraints variable."""
    spec = importlib.util.spec_from_file_location("dc_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "denial_constraints"):
        raise RuntimeError(f"{py_path} does not define denial_constraints")
    dcs = getattr(mod, "denial_constraints")
    if not isinstance(dcs, list):
        raise RuntimeError(f"{py_path}: denial_constraints is not a list")
    return dcs


def dcs_to_hyperedges(
    dcs: List[List[Tuple[Any, Any, Any]]]
) -> Tuple[Dict[Tuple[str, ...], float], Dict[str, int], int]:
    """
    Convert DCs to hyperedges over "t1.attr" tokens (weight=1.0).
    Returns hyperedge_dict, attr_to_dc_count, num_dcs
    """
    hyperedge_dict: Dict[Tuple[str, ...], float] = {}
    attr_to_dc_count: Dict[str, int] = {}

    for dc in dcs:
        attrs: Set[str] = set()
        for (lhs, _op, rhs) in dc:
            a1 = _extract_attr_token(lhs)
            a2 = _extract_attr_token(rhs)
            if a1:
                attrs.add(a1)
            if a2:
                attrs.add(a2)

        # Hyperedge only if >=2 attrs
        if len(attrs) >= 2:
            edge = tuple(sorted({f"t1.{a}" for a in attrs}))
            hyperedge_dict[edge] = 1.0

        for a in attrs:
            attr_to_dc_count[a] = attr_to_dc_count.get(a, 0) + 1

    return hyperedge_dict, attr_to_dc_count, len(dcs)


def instantiate_edges_no_filter(
    curr_attr: str,
    hypergraph: Dict[Tuple[str, ...], float],
) -> List[Set[str]]:
    """Return all hyperedges (sets of 't1.attr') that include curr_attr."""
    out: List[Set[str]] = []
    for edge_attrs in hypergraph.keys():
        names = {ea.split(".")[-1] for ea in edge_attrs}
        if curr_attr in names and len(edge_attrs) >= 2:
            out.append(set(edge_attrs))
    return out


def ilp_baseline3_cascading_closure(
    *,
    key: int,
    target_attr: str,
    hypergraph: Dict[Tuple[str, ...], float],
    ilp_write_path: Optional[str] = None,
) -> Tuple[
    Set[CellILP],  # to_delete
    float,         # ilp_time_sec
    int,           # max_depth
    int,           # zone_cells_instantiated
    int,           # zone_edges
    int,           # ilp_num_vars
    int,           # ilp_num_constrs
    int,           # ilp_file_bytes
]:
    """
    Same semantics as your Baseline 3 ILP:
      - a_v delete, r_v reachable/known
      - r_v >= 1 - a_v
      - closure on each hyperedge & each head
      - target: a_target=1, r_target=0
      - minimize sum a_v
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi not available (gurobipy import failed).")

    t0 = time.time()

    # ---------- Phase 1: BFS zone ----------
    cell_to_id: Dict[CellILP, int] = {}
    discovered: Set[CellILP] = set()
    depth: Dict[CellILP, int] = {}
    q = deque()

    unique_edges: Set[FrozenSet[CellILP]] = set()
    edges_list: List[Set[CellILP]] = []

    target_cell = CellILP(f"t1.{target_attr}", key)
    cell_to_id[target_cell] = 0
    next_id = 1
    discovered.add(target_cell)
    depth[target_cell] = 0
    q.append(target_cell)

    max_depth = 0

    while q:
        curr = q.popleft()
        cd = depth[curr]
        curr_attr = curr.attribute.split(".")[-1]

        raw_edges = instantiate_edges_no_filter(curr_attr, hypergraph)
        for raw_edge in raw_edges:
            edge_cells: Set[CellILP] = set()
            for cell_attr in raw_edge:
                c = CellILP(cell_attr, key)
                edge_cells.add(c)
                if c not in cell_to_id:
                    cell_to_id[c] = next_id
                    next_id += 1

            if len(edge_cells) < 2:
                continue

            fe = frozenset(edge_cells)
            if fe not in unique_edges:
                unique_edges.add(fe)
                edges_list.append(set(edge_cells))

            for c in edge_cells:
                if c not in discovered:
                    discovered.add(c)
                    depth[c] = cd + 1
                    q.append(c)
                    if cd + 1 > max_depth:
                        max_depth = cd + 1

    # ---------- Phase 2: ILP ----------
    model = Model("BASELINE3_CASCADING_ILP")
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)

    a_var: Dict[CellILP, Any] = {}
    r_var: Dict[CellILP, Any] = {}

    for cell, cid in cell_to_id.items():
        if cell == target_cell:
            a = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name=f"a{cid}")
        else:
            a = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"a{cid}")
        r = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"r{cid}")
        a_var[cell] = a
        r_var[cell] = r

    model.update()

    for cell, cid in cell_to_id.items():
        model.addConstr(r_var[cell] >= 1 - a_var[cell], name=f"visible_implies_known_{cid}")

    model.addConstr(r_var[target_cell] == 0, name="target_not_inferable")

    for ei, edge in enumerate(edges_list):
        k = len(edge)
        if k < 2:
            continue
        for head in edge:
            tail = [r_var[y] for y in edge if y != head]
            model.addConstr(
                r_var[head] >= quicksum(tail) - (k - 1) + 1,
                name=f"closure_e{ei}_h{cell_to_id[head]}",
            )

    model.setObjective(quicksum(a_var[c] for c in cell_to_id.keys()), GRB.MINIMIZE)

    ilp_file_bytes = 0
    if ilp_write_path:
        try:
            os.makedirs(os.path.dirname(ilp_write_path) or ".", exist_ok=True)
            model.write(ilp_write_path)
            if os.path.exists(ilp_write_path):
                ilp_file_bytes = int(os.path.getsize(ilp_write_path))
        except Exception:
            ilp_file_bytes = 0

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            model.write("baseline3_infeasible.lp")
        except Exception:
            pass
        raise RuntimeError(f"Infeasible ILP for target_attr={target_attr}")

    to_delete: Set[CellILP] = set()
    for cell, cid in cell_to_id.items():
        if model.getVarByName(f"a{cid}").X > 0.5:
            to_delete.add(cell)

    try:
        ilp_num_vars = int(model.NumVars)
        ilp_num_constrs = int(model.NumConstrs)
    except Exception:
        ilp_num_vars = 0
        ilp_num_constrs = 0

    model.dispose()

    return (
        to_delete,
        float(time.time() - t0),
        int(max_depth),
        int(len(cell_to_id)),
        int(len(edges_list)),
        int(ilp_num_vars),
        int(ilp_num_constrs),
        int(ilp_file_bytes),
    )


def discover_dc_files(dc_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(dc_dir)):
        if not name.endswith(".py"):
            continue
        if ONLY_FILES_CONTAINING and ONLY_FILES_CONTAINING not in name:
            continue
        files.append(os.path.join(dc_dir, name))
    return files


def run_one_dc_file(dc_path: str):
    base = os.path.splitext(os.path.basename(dc_path))[0]
    out_csv = os.path.join(OUT_DIR, f"{base}__baseline3_per_attr.csv")

    dcs = load_denial_constraints(dc_path)
    hyperedge_dict, attr_to_dc_count, num_dcs = dcs_to_hyperedges(dcs)
    all_attrs = sorted(attr_to_dc_count.keys())

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dc_file",
                "target_attr",
                "num_dcs_total",
                "num_dcs_with_target",
                "num_attrs_total",
                "mask_size_excl_target",
                "total_deleted_including_target",
                "mask_attrs_excl_target",
                "ilp_time_sec",
                "max_depth",
                "zone_cells_instantiated",
                "zone_edges",
                "ilp_num_vars",
                "ilp_num_constrs",
                "ilp_file_bytes",
            ],
        )
        w.writeheader()

        print(f"\n=== DC FILE: {dc_path}")
        print(f"    DCs: {num_dcs} | attrs: {len(all_attrs)} | hyperedges: {len(hyperedge_dict)}")
        for target_attr in all_attrs:
            ilp_path = None
            if WRITE_ILP_MODELS:
                ilp_dir = os.path.join(OUT_DIR, "ilp_models")
                ilp_path = os.path.join(ilp_dir, f"{base}__{target_attr}.lp")

            (
                to_delete,
                ilp_time,
                max_depth,
                zone_cells,
                zone_edges,
                ilp_num_vars,
                ilp_num_constrs,
                ilp_file_bytes,
            ) = ilp_baseline3_cascading_closure(
                key=DUMMY_KEY,
                target_attr=target_attr,
                hypergraph=hyperedge_dict,
                ilp_write_path=ilp_path,
            )

            deleted_attrs = sorted({c.attribute.split(".")[-1] for c in to_delete})
            deleted_excl_target = [a for a in deleted_attrs if a != target_attr]

            w.writerow(
                {
                    "dc_file": os.path.basename(dc_path),
                    "target_attr": target_attr,
                    "num_dcs_total": num_dcs,
                    "num_dcs_with_target": attr_to_dc_count.get(target_attr, 0),
                    "num_attrs_total": len(all_attrs),
                    "mask_size_excl_target": len(deleted_excl_target),
                    "total_deleted_including_target": len(deleted_attrs),
                    "mask_attrs_excl_target": ";".join(deleted_excl_target),
                    "ilp_time_sec": f"{ilp_time:.6f}",
                    "max_depth": max_depth,
                    "zone_cells_instantiated": zone_cells,
                    "zone_edges": zone_edges,
                    "ilp_num_vars": ilp_num_vars,
                    "ilp_num_constrs": ilp_num_constrs,
                    "ilp_file_bytes": ilp_file_bytes,
                }
            )

            print(
                f"  target={target_attr:>22s} | "
                f"mask(excl)={len(deleted_excl_target):>4d} | "
                f"cells={zone_cells:>5d} edges={zone_edges:>5d} | "
                f"time={ilp_time:.3f}s"
            )

    print(f"--> wrote: {out_csv}")


def main():
    if not GUROBI_AVAILABLE:
        print("ERROR: gurobipy not available. Baseline 3 requires Gurobi.", flush=True)
        sys.exit(2)

    dc_files = discover_dc_files(DC_DIR)
    if not dc_files:
        print(f"ERROR: No matching .py DC files found in {DC_DIR}")
        sys.exit(1)

    print(f"Found {len(dc_files)} DC files in {DC_DIR}")
    print(f"Writing CSVs to {OUT_DIR}")

    for dc_path in dc_files:
        try:
            run_one_dc_file(dc_path)
        except Exception as e:
            print(f"[ERROR] {dc_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
