#!/usr/bin/env python3
"""
min.py
"""

from __future__ import annotations
import time
import os
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple, Optional
from leakage import (
    leakage,
    dc_to_rdrs_and_weights,
    construct_hypergraph_max,
    construct_hypergraph_actual,
    compute_utility_em as compute_utility,
)
from collections import deque

try:
    import mysql.connector
except Exception:
    mysql = None

# Optional config
try:
    import config  # type: ignore
except Exception:
    config = None

LAM = 0.5

DELETION_QUERY = """
UPDATE {table_name}
SET `{column_name}` = NULL
WHERE id = {key};
"""

try:
    from gurobipy import Model, GRB, quicksum

    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False



@dataclass(frozen=True)
class CellILP:
    attribute: str
    key: int


def estimate_memory_bytes_standard(
    *,
    num_vertices: int,
    num_edges: int,
    edge_members: int,
    mask_size: int,
    stores_candidate_masks: bool,
    num_candidate_masks: int = 0,
    candidate_mask_members: int = 0,
    includes_inferable_model: bool = False,
    includes_channel_map: bool = False,
    ilp_num_cells: int = 0,
    ilp_num_vars: int = 0,
    ilp_num_constrs: int = 0,
) -> int:
    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_MASK_SET = 96

    BYTES_PER_EDGE_STRUCT = 80
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28
    BYTES_PER_CAND_MASK = 96

    BYTES_PER_ILP_CELL = 128
    BYTES_PER_ILP_VAR = 96
    BYTES_PER_ILP_CONSTR = 128

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    if includes_inferable_model:
        est += num_edges * BYTES_PER_EDGE_STRUCT
        est += num_vertices * BYTES_PER_FLOAT
        est += num_edges * BYTES_PER_FLOAT

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    if stores_candidate_masks:
        est += num_candidate_masks * BYTES_PER_CAND_MASK
        est += candidate_mask_members * BYTES_PER_MASK_MEMBER

    if ilp_num_cells or ilp_num_vars or ilp_num_constrs:
        est += ilp_num_cells * BYTES_PER_ILP_CELL
        est += ilp_num_vars * BYTES_PER_ILP_VAR
        est += ilp_num_constrs * BYTES_PER_ILP_CONSTR

    return int(est)


def get_insertion_time(cursor, table, key, attr):
    try:
        query = f"SELECT `{attr}` FROM {table}_insertiontime WHERE insertionKey = {key}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else 0
    except Exception:
        return 0


def instantiate_edges_with_time_filter(
    cursor,
    table,
    key,
    attr,
    target_time,
    hypergraph: Dict[Tuple[str, ...], float],
    *,
    touched_cells: Optional[Set[CellILP]] = None,
):
    """
    Returns instantiated edges (after insertion-time filtering).
    Additionally: if touched_cells is provided, we count ALL cells appearing
    in any relevant edge (i.e., edge contains curr attr) as "touched/considered",
    even if they do not survive the time filter.
    """
    edges = []
    for edge_attrs, weight in hypergraph.items():
        edge_attr_names = {ea.split(".")[-1] for ea in edge_attrs if isinstance(ea, str)}
        if attr not in edge_attr_names:
            continue

        # Count all cells in this relevant edge as "touched"
        if touched_cells is not None:
            for edge_attr in edge_attrs:
                touched_cells.add(CellILP(edge_attr, key))

        valid_cells = []
        for edge_attr in edge_attrs:
            attr_name = edge_attr.split('.')[-1]
            it = get_insertion_time(cursor, table, key, attr_name)
            if it >= target_time:
                valid_cells.append(edge_attr)

        if len(valid_cells) > 1:
            edges.append(set(valid_cells))

    return edges


def ilp_approach(
    cursor,
    table,
    key,
    target_attr,
    target_time,
    hypergraph: Dict[Tuple[str, ...], float],
    *,
    ilp_write_path: Optional[str] = None,
) -> Tuple[
    Set[CellILP],
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """
    CASCADING / CLOSURE SEMANTICS ILP.

    Variables:
      a_v in {0,1}: delete v
      r_v in {0,1}: v is known/reachable in inference closure

    Semantics:
      - initially known: if not deleted -> known   (r_v >= 1 - a_v)
      - for each hyperedge e and each x in e:
            if all other members are known then x becomes known
        linearized as: r_x >= sum_{y in e\{x}} r_y - (|e|-1) + 1
      - target must NOT be inferable: r_target = 0
      - target must be deleted: a_target = 1
      - objective: minimize total deletions sum(a_v)
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi required")

    start_total_ilp = time.time()

    # ---------- Phase 1: Build the relevant zone (vertices + edges) by BFS ----------
    cell_to_id: Dict[CellILP, int] = {}
    instantiated_cells: Set[CellILP] = set()
    touched_cells: Set[CellILP] = set()

    # unique edges as frozensets of CellILP
    unique_edges: Dict[frozenset[CellILP], int] = {}  # edge -> index
    edges_list: list[Set[CellILP]] = []

    cells_to_visit = deque()
    cell_to_depth: Dict[CellILP, int] = {}
    max_depth = 0

    target_cell = CellILP(f"t1.{target_attr}", key)
    cells_to_visit.append(target_cell)
    instantiated_cells.add(target_cell)
    touched_cells.add(target_cell)
    cell_to_depth[target_cell] = 0

    # Assign IDs as we discover cells
    cell_to_id[target_cell] = 0
    next_id = 1

    while cells_to_visit:
        curr = cells_to_visit.popleft()
        curr_depth = cell_to_depth[curr]
        curr_attr = curr.attribute.split(".")[-1]

        # returns list[set[str]] of "t1.attr" names, already time-filtered
        raw_edges = instantiate_edges_with_time_filter(
            cursor,
            table,
            key,
            curr_attr,
            target_time,
            hypergraph,
            touched_cells=touched_cells,
        )

        for raw_edge in raw_edges:
            # Map edge members (strings) -> CellILP
            edge_cells: Set[CellILP] = set()
            for cell_attr in raw_edge:
                c = CellILP(cell_attr, key)
                edge_cells.add(c)
                touched_cells.add(c)
                if c not in cell_to_id:
                    cell_to_id[c] = next_id
                    next_id += 1

            if len(edge_cells) < 2:
                continue

            fe = frozenset(edge_cells)
            if fe not in unique_edges:
                unique_edges[fe] = len(edges_list)
                edges_list.append(set(edge_cells))

            # BFS expansion: any cell in an edge with curr is in the zone
            for c in edge_cells:
                if c not in instantiated_cells:
                    instantiated_cells.add(c)
                    cells_to_visit.append(c)
                    cell_to_depth[c] = curr_depth + 1
                    max_depth = max(max_depth, curr_depth + 1)

    # ---------- Phase 2: Build cascading ILP ----------
    model = Model("CASCADING_ILP")
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)

    # Decision vars
    a_var: Dict[CellILP, Any] = {}  # delete
    r_var: Dict[CellILP, Any] = {}  # reachable/known

    # Create variables for all instantiated cells (zone)
    for cell, cid in cell_to_id.items():
        # deletion var a
        if cell == target_cell:
            a = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name=f"a{cid}")  # target must be deleted
        else:
            a = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"a{cid}")
        a_var[cell] = a

        # reachability var r
        r = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"r{cid}")
        r_var[cell] = r

    model.update()

    # Initial knowledge: if not deleted, then known (reachable)
    #   visible = (1 - a), so r >= 1 - a
    for cell, cid in cell_to_id.items():
        model.addConstr(r_var[cell] >= 1 - a_var[cell], name=f"visible_implies_known_{cid}")

    # Target must not be inferable in closure
    model.addConstr(r_var[target_cell] == 0, name="target_not_inferable")

    # Cascading closure constraints:
    # For each edge e and each head x in e:
    #    if all others known => x known
    # Linearization:
    #    r_x >= sum_{y != x} r_y - (k-1) + 1
    # where k = |e|
    for ei, edge in enumerate(edges_list):
        k = len(edge)
        if k < 2:
            continue
        for head in edge:
            tail = [r_var[y] for y in edge if y != head]
            # RHS becomes 1 iff all tail vars are 1; otherwise <= 0
            model.addConstr(
                r_var[head] >= quicksum(tail) - (k - 1) + 1,
                name=f"closure_e{ei}_h{cell_to_id[head]}",
            )

    # Objective: minimize number of deletions (including target is constant 1)
    obj = quicksum(a_var[cell] for cell in cell_to_id.keys())
    model.setObjective(obj, GRB.MINIMIZE)

    # Optional: write .lp before solving
    ilp_file_bytes = 0
    if ilp_write_path:
        try:
            os.makedirs(os.path.dirname(ilp_write_path) or ".", exist_ok=True)
            model.write(ilp_write_path)
            if os.path.exists(ilp_write_path):
                ilp_file_bytes = int(os.path.getsize(ilp_write_path))
        except Exception as e:
            print(f"[WARN] Failed to write ILP model to {ilp_write_path}: {e}")
            ilp_file_bytes = 0

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.lp")
        raise RuntimeError("Infeasible cascading ILP")

    # Extract solution
    to_delete: Set[CellILP] = set()
    for cell, cid in cell_to_id.items():
        if model.getVarByName(f"a{cid}").X > 0.5:
            to_delete.add(cell)

    # “Activated dependencies” no longer meaningful without b-vars.
    # Best proxy: number of edges in the instantiated zone.
    activated_dependencies_count = len(edges_list)

    try:
        ilp_num_vars = int(model.NumVars)
        ilp_num_constrs = int(model.NumConstrs)
    except Exception:
        ilp_num_vars = 0
        ilp_num_constrs = 0

    model.dispose()
    total_ilp_time = time.time() - start_total_ilp

    num_cells_instantiated = len(cell_to_id)
    num_cells_touched = len(touched_cells)

    return (
        to_delete,
        total_ilp_time,
        max_depth,
        num_cells_instantiated,
        num_cells_touched,
        activated_dependencies_count,
        ilp_num_vars,
        ilp_num_constrs,
        ilp_file_bytes,
    )

def load_parsed_dcs_for_dataset(dataset: str):
    ds = str(dataset).lower()
    mod_name = "NCVoter" if ds == "ncvoter" else ds.capitalize()
    dc_module_path = f"DCandDelset.dc_configs.top{mod_name}DCs_parsed"
    try:
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        return getattr(dc_module, "denial_constraints", []) or []
    except Exception:
        return []
def min(target: str, key: int, dataset: str, threshold: float):
    if not GUROBI_AVAILABLE:
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

    if config is None or mysql is None:
        print("[WARN] Missing deps for baseline_deletion_3 (config/mysql).")
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

    # ============================================================
    # INIT PHASE (NO InitializationManager)
    # ============================================================

    init_start = time.time()

    raw_dcs = load_parsed_dcs_for_dataset(dataset)

    class _Init:
        def __init__(self, dataset, denial_constraints):
            self.dataset = dataset
            self.denial_constraints = denial_constraints

    init_obj = _Init(dataset, raw_dcs)

    rdrs, rdr_weights = dc_to_rdrs_and_weights(init_obj)

    # ILP hyperedge dict uses "t1.attr" tokens
    hyperedge_dict: Dict[Tuple[str, ...], float] = {}
    for rdr, w in zip(rdrs, rdr_weights):
        edge = tuple(sorted({f"t1.{a}" for a in rdr}))
        hyperedge_dict[edge] = float(w)

    init_time = time.time() - init_start

    conn = None
    cursor = None


    try:
        deletion_start = time.time()

        db = config.get_database_config(dataset)
        conn = mysql.connector.connect(
            host=db['host'],
            user=db['user'],
            password=db['password'],
            database=db['database'],
            ssl_disabled=db.get('ssl_disabled', True),
        )
        cursor = conn.cursor()

        table = f"{dataset}_data"
        if dataset == "airport":
            table = "airports"

        target_time = get_insertion_time(cursor, table, key, target)

        ilp_dir = "data/ilp_models"
        ilp_path = os.path.join(ilp_dir, f"{dataset}_{table}_{target}_key{key}.lp")

        (
            to_del_cells,
            total_ilp_time,
            max_depth,
            num_cells_instantiated,
            num_cells_touched,
            activated_dependencies_count,
            ilp_num_vars,
            ilp_num_constrs,
            ilp_file_bytes,
        ) = ilp_approach(
            cursor,
            table,
            key,
            target,
            target_time,
            hyperedge_dict,
            ilp_write_path=ilp_path,
        )

        model_time = float(total_ilp_time)

        for cell in to_del_cells:
            attr = cell.attribute.split('.')[-1]
            cursor.execute(
                DELETION_QUERY.format(
                    table_name=table,
                    column_name=attr,
                    key=key,
                )
            )

        conn.commit()

        deletion_time = (time.time() - deletion_start) - model_time

        final_mask = {cell.attribute.split('.')[-1] for cell in to_del_cells}
        mask_for_leakage = set(final_mask)
        mask_for_leakage.discard(target)

        H_max = construct_hypergraph_max(target, rdrs, rdr_weights)
        H_actual = construct_hypergraph_actual(target, rdrs, rdr_weights)

        zone_size = len(H_max.vertices - {target})

        leakage_val, paths, active, blocked = leakage(
            mask_for_leakage,
            target,
            H_actual,
            return_counts=True,
        )

        utility = compute_utility(
            leakage=leakage_val,
            mask_size=len(mask_for_leakage),
            lambda_penalty=float(LAM),
            L0 = threshold,
            zone_size=zone_size,
        )

        memory_bytes = int(ilp_file_bytes)

    except Exception as e:
        print(f"Error in Baseline 3: {e}")
        import traceback
        traceback.print_exc()
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    final_mask.discard(target)
    baseline_leakage = leakage(set(), target, H_actual)

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": float(deletion_time),

        "leakage": float(leakage_val),
        "utility": float(utility),
        "mask": final_mask,
        "mask_size": int(len(final_mask)),

        "num_paths": int(paths),
        "baseline_leakage": float(baseline_leakage),
        "memory_overhead_bytes": memory_bytes,
        "num_instantiated_cells": int(zone_size),
    }

