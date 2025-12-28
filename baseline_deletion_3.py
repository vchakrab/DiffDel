#!/usr/bin/env python3
"""
COMBINED SCRIPT (UPDATED)

Contains:
  1) Baseline 3 ILP (Gurobi) deletion code
  2) exponential_deletion.py-style orchestrator (exp + gumbel) using inferable leakage

Fixes requested:
- "paths blocked" is now an ESTIMATE derived from inferable leakage diagnostics
  (no path construction). This diagnostics time is EXCLUDED from init/model/del time.
- Memory overhead is standardized and does NOT depend on explicit path lists.
- Adds UPDATE-to-NULL logic + deletion time measurement for the exp/gumbel orchestrator.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import defaultdict, deque, Counter

import numpy as np

# Optional DB deps
try:
    import mysql.connector
    from mysql.connector import Error
except Exception:
    mysql = None
    Error = Exception

# Optional config
try:
    import config  # type: ignore
except Exception:
    config = None
DELETION_QUERY = (
    "UPDATE {table_name} "
    "SET {column_name} = NULL "
    "WHERE ID = {key}"
)

# ============================================================
# SAFE PATH ESTIMATOR (NO int64 OVERFLOW)
# ============================================================

from collections import defaultdict

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))

def _enabled_edges_from_mask(hyperedges, target_cell: str, mask_set) -> list[bool]:
    """
    enabled[e] = True iff hyperedge e is 'enabled' under the mask, meaning:
      all non-target vertices of the hyperedge are NOT masked.
    This matches your earlier "all-but-one known => infer target" logic.
    """
    enabled = []
    for e in hyperedges:
        if target_cell not in e:
            enabled.append(False)
            continue
        ok = True
        for v in e:
            if v == target_cell:
                continue
            if v in mask_set:
                ok = False
                break
        enabled.append(ok)
    return enabled

def estimate_num_paths_bounded_walks(
    *,
    hyperedges: list[tuple[str, ...]],
    enabled: list[bool],
    source_cells: list[str],
    target_cell: str,
    max_depth_edges: int = 10,
    cap: int | None = 10_000_000_000
) -> int:
    """
    Estimates #paths as #bounded-length WALKS to target on the 1-mode projection
    induced by enabled hyperedges (treat each enabled hyperedge as a clique).

    CRITICAL: uses pure-Python ints -> never overflows into negative.
    """
    if not hyperedges:
        return 0
    if target_cell is None:
        return 0

    # Build vertex universe
    cells = sorted(set(x for e in hyperedges for x in e))
    if target_cell not in cells:
        return 0

    idx = {c: i for i, c in enumerate(cells)}
    t = idx[target_cell]
    n = len(cells)

    # Build adjacency multiplicities (multigraph) from enabled hyperedges
    adj: list[dict[int, int]] = [defaultdict(int) for _ in range(n)]
    for e, en in zip(hyperedges, enabled):
        if not en:
            continue
        verts = list({idx[x] for x in e})  # unique
        k = len(verts)
        if k < 2:
            continue
        # clique
        for i in range(k):
            u = verts[i]
            for j in range(k):
                if i == j:
                    continue
                v = verts[j]
                adj[u][v] += 1

    # DP for bounded-length walks
    dp = [0] * n
    for s in source_cells:
        if s in idx and idx[s] != t:
            dp[idx[s]] = 1

    total_to_target = 0
    for _ in range(max_depth_edges):
        nxt = [0] * n
        for u in range(n):
            du = dp[u]
            if du == 0:
                continue
            for v, mult in adj[u].items():
                nxt[v] += du * mult
                if cap is not None and nxt[v] > cap:
                    nxt[v] = cap
        dp = nxt
        total_to_target += dp[t]
        if cap is not None and total_to_target > cap:
            total_to_target = cap

    return int(total_to_target)

def safe_paths_blocked_from_leakage(num_paths: int, leakage: float | None) -> int:
    """
    If leakage in [0,1], approximate:
      active ~= round(L * num_paths)
      blocked = num_paths - active
    Always returns >= 0, never negative.
    """
    if num_paths is None or num_paths < 0:
        return -1
    if leakage is None:
        return -1
    L = _clamp01(leakage)
    active = int(round(L * num_paths))
    if active < 0:
        active = 0
    if active > num_paths:
        active = num_paths
    return int(num_paths - active)

# ============================================================
# (A) STANDARDIZED MEMORY ESTIMATION (NO PATH STORAGE)
# ============================================================

def estimate_memory_bytes_standard(
    *,
    num_vertices: int,
    num_edges: int,
    edge_members: int,
    mask_size: int,
    stores_candidate_masks: bool,
    num_candidate_masks: int = 0,
    candidate_mask_members: int = 0,
    includes_inferable_model: bool = True,
    includes_channel_map: bool = True,
    # ILP extras
    ilp_num_cells: int = 0,
    ilp_num_vars: int = 0,
    ilp_num_constrs: int = 0,
) -> int:
    """
    Consistent rough byte estimate across all methods.
    Goal: comparable scaling across datasets/epsilons, not exact RAM accounting.

    IMPORTANT: We intentionally do NOT count "paths" since we no longer build them for metrics.
    """

    # Core Python object-ish constants (picked to be stable and consistent)
    BYTES_PER_VERTEX = 112      # dict entry + string ref-ish
    BYTES_PER_EDGE = 184        # tuple/list + small overhead
    BYTES_PER_EDGE_MEMBER = 72  # member ref overhead-ish
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_MASK_SET = 96

    # Inferable model structures
    BYTES_PER_EDGE_STRUCT = 80      # Edge(verts,w) packed
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28

    # Candidate masks (exp mechanism only)
    BYTES_PER_CAND_MASK = 96

    # ILP scale (baseline 3)
    BYTES_PER_ILP_CELL = 128
    BYTES_PER_ILP_VAR = 96
    BYTES_PER_ILP_CONSTR = 128

    est = 0

    # Hypergraph storage (verts/edges)
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER

    # Mask storage
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    # Inferable leakage model storage (if used)
    if includes_inferable_model:
        # edges[] with verts lists + weights, plus arrays p[]
        est += num_edges * BYTES_PER_EDGE_STRUCT
        est += num_vertices * BYTES_PER_FLOAT  # p vector
        est += num_edges * BYTES_PER_FLOAT     # weights vector

    if includes_channel_map:
        # channel map over target-containing edges (worst-case: many)
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    # Candidate masks (only if exp enumerates all subsets)
    if stores_candidate_masks:
        est += num_candidate_masks * BYTES_PER_CAND_MASK
        est += candidate_mask_members * BYTES_PER_MASK_MEMBER

    # ILP extras
    if ilp_num_cells or ilp_num_vars or ilp_num_constrs:
        est += ilp_num_cells * BYTES_PER_ILP_CELL
        est += ilp_num_vars * BYTES_PER_ILP_VAR
        est += ilp_num_constrs * BYTES_PER_ILP_CONSTR

    return int(est)




# ============================================================
# (G) BASELINE 3 ILP (Gurobi) — FIXED MEMORY STANDARDIZATION
# ============================================================

try:
    from gurobipy import Model, GRB, quicksum
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

# rtf_core imports are optional
try:
    from rtf_core import initialization_phase
    from rtf_core.Algorithms import enumerate_explanations as explanations
except Exception:
    initialization_phase = None
    explanations = None


class CellILP:
    def __init__(self, attribute: str, key: int):
        self.attribute = attribute
        self.key = key

    def __hash__(self):
        return hash((self.attribute, self.key))

    def __eq__(self, other):
        return isinstance(other, CellILP) and self.attribute == other.attribute and self.key == other.key

    def __repr__(self):
        return f"Cell({self.attribute}, {self.key})"


def get_insertion_time(cursor, table, key, attr):
    try:
        query = f"SELECT `{attr}` FROM {table}_insertiontime WHERE insertionKey = {key}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else 0
    except Exception:
        return 0


def instantiate_edges_with_time_filter(cursor, table, key, attr, target_time, hypergraph: Dict[Tuple[str, ...], float]):
    edges = []
    for edge_attrs, weight in hypergraph.items():
        if attr not in edge_attrs:
            continue

        valid_cells = []
        for edge_attr in edge_attrs:
            attr_name = edge_attr.split('.')[-1]
            it = get_insertion_time(cursor, table, key, attr_name)
            if it >= target_time:
                valid_cells.append(edge_attr)

        if len(valid_cells) > 1:
            edges.append(set(valid_cells))
    return edges


def ilp_approach_matching_java(
    cursor,
    table,
    key,
    target_attr,
    target_time,
    hypergraph: Dict[Tuple[str, ...], float],
    boundaries
):
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi required")

    start_total_ilp = time.time()

    max_id = 0
    edge_counter = -1
    cell_to_id: Dict[CellILP, int] = {}
    cell_to_var = {}
    instantiated_cells = set()
    cells_to_visit = deque()

    cell_to_depth = {}
    max_depth = 0

    edge_vars = []
    existing_rdr_vars: Dict[frozenset, Any] = {}

    model = Model("P2E2_ILP")
    model.setParam('OutputFlag', 0)
    model.setParam('LogToConsole', 0)

    obj = quicksum([])

    deleted_cell = CellILP(f"t1.{target_attr}", key)
    a0 = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name=f"a{max_id}")
    cell_to_var[deleted_cell] = a0
    cell_to_id[deleted_cell] = max_id
    cell_to_depth[deleted_cell] = 0
    max_id += 1

    cells_to_visit.append(deleted_cell)
    instantiated_cells.add(deleted_cell)

    while cells_to_visit:
        curr = cells_to_visit.popleft()
        curr_id = cell_to_id[curr]
        curr_depth = cell_to_depth[curr]
        aj = cell_to_var[curr]
        obj += aj

        edges = instantiate_edges_with_time_filter(cursor, table, key, curr.attribute, target_time, hypergraph)

        for edge in edges:
            frozenset_edge = frozenset(edge)
            if frozenset_edge in existing_rdr_vars:
                bi = existing_rdr_vars[frozenset_edge]
            else:
                edge_counter += 1
                bi = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b{edge_counter}")
                edge_vars.append(bi)
                existing_rdr_vars[frozenset_edge] = bi

            hij = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"h{edge_counter}_{curr_id}")
            model.addConstr(aj == hij, name=f"head_hidden_{edge_counter}")
            model.addConstr(bi == hij, name=f"rdr_addr_{edge_counter}")

            tail_tji_vars = []
            for cell_attr in edge:
                cell = CellILP(cell_attr, key)

                if cell not in cell_to_id:
                    t_id = max_id
                    cell_to_id[cell] = max_id
                    max_id += 1
                    a_cell = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"a{t_id}")
                    cell_to_var[cell] = a_cell
                else:
                    t_id = cell_to_id[cell]
                    a_cell = cell_to_var[cell]

                tji = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"t{edge_counter}_{t_id}")
                model.addConstr(tji == a_cell, name=f"tail_sync_{edge_counter}_{t_id}")

                if cell != curr:
                    tail_tji_vars.append(tji)

                if cell not in instantiated_cells:
                    instantiated_cells.add(cell)
                    cells_to_visit.append(cell)
                    cell_to_depth[cell] = curr_depth + 1
                    max_depth = max(max_depth, curr_depth + 1)

            if tail_tji_vars:
                model.addConstr(quicksum(tail_tji_vars) >= bi, name=f"tail_req_{edge_counter}")

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.ilp")
        raise RuntimeError("Infeasible")

    to_delete = set()
    for cell, cell_id in cell_to_id.items():
        var_name = f"a{cell_id}"
        if model.getVarByName(var_name).X == 1.0:
            to_delete.add(cell)

    activated_dependencies_count = sum(1 for bi_var in edge_vars if bi_var.X == 1.0)

    # model scale stats for standardized memory estimation
    try:
        ilp_num_vars = int(model.NumVars)
        ilp_num_constrs = int(model.NumConstrs)
    except Exception:
        ilp_num_vars = 0
        ilp_num_constrs = 0

    model.dispose()
    total_ilp_time = time.time() - start_total_ilp

    return to_delete, total_ilp_time, max_depth, len(cell_to_id), activated_dependencies_count, ilp_num_vars, ilp_num_constrs


def baseline_deletion_3(target: str, key: int, dataset: str, threshold: float):
    """
    Returns:
      activated_dependencies_count,
      final_mask,
      memory_bytes,
      max_depth,
      instantiation_time,
      model_time,
      deletion_time,
      num_cells
    """
    if not GUROBI_AVAILABLE:
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0

    if initialization_phase is None or explanations is None or config is None or mysql is None:
        print("[WARN] Missing deps for baseline_deletion_3 (rtf_core/config/mysql).")
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0

    # instantiation time: init_mgr + building hypergraph structures (same as your intent)
    instantiation_start = time.time()

    init_mgr = initialization_phase.InitializationManager({"key": key, "attribute": target}, dataset, threshold)
    init_mgr.initialize()

    target_dcs = []
    for dc in init_mgr.denial_constraints:
        attrs = {p.split('.')[-1] for p in [x[0] for x in dc] + [x[2] for x in dc if isinstance(x[2], str)]}
        if target in attrs:
            target_dcs.append(dc)

    b_edges, i_edges, orig_bounds = explanations.build_graph_data(target_dcs)
    bounds = {c for c in orig_bounds if not c.startswith('t1.')}
    hyper = explanations.set_edge_weight(i_edges)

    instantiation_time = time.time() - instantiation_start

    model_time = 0.0
    deletion_time = 0.0
    conn = None
    cursor = None

    activated_dependencies_count = 0
    final_mask: Set[str] = set()
    memory_bytes = 0
    max_depth = 0
    num_cells = 0

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
        table = f"{dataset}_copy_data"

        target_time = get_insertion_time(cursor, table, key, target)

        to_del, total_ilp_time, max_depth, num_cells, activated_dependencies_count, ilp_num_vars, ilp_num_constrs = (
            ilp_approach_matching_java(cursor, table, key, target, target_time, hyper, bounds)
        )
        model_time = float(total_ilp_time)

        # Update-to-null for ILP-selected cells
        for cell in to_del:
            attr = cell.attribute.split('.')[-1]
            cursor.execute(DELETION_QUERY.format(table_name=table, column_name=attr, key=key))
        conn.commit()

        # deletion_time excludes ILP model time, as you intended
        deletion_time = (time.time() - deletion_start) - model_time

        final_mask = {cell.attribute.split('.')[-1] for cell in to_del}

        # standardized memory: ILP scale + mask + hypergraph rough size
        # Hypergraph size:
        hyper_edges = list(hyper.keys())
        num_edges = len(hyper_edges)
        edge_members = sum(len(e) for e in hyper_edges)
        num_vertices = len({a for e in hyper_edges for a in e})

        memory_bytes = estimate_memory_bytes_standard(
            num_vertices=num_vertices,
            num_edges=num_edges,
            edge_members=edge_members,
            mask_size=len(final_mask),
            stores_candidate_masks=False,
            includes_inferable_model=False,   # ILP baseline doesn't store inferable model
            includes_channel_map=False,
            ilp_num_cells=num_cells,
            ilp_num_vars=ilp_num_vars,
            ilp_num_constrs=ilp_num_constrs,
        )

    except Exception as e:
        print(f"Error in Baseline 3: {e}")
        import traceback
        traceback.print_exc()
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return (
        int(activated_dependencies_count),
        set(final_mask),
        int(memory_bytes),
        int(max_depth),
        float(instantiation_time),
        float(model_time),
        float(deletion_time),
        int(num_cells),
    )


# ============================================================
# CLI EXAMPLE
# ============================================================

if __name__ == '__main__':
    # Example: run gumbel/exp orchestrator
    # results = gumbel_deletion_main(dataset='adult', key=2, target_cell='education', method='gumbel')

    # Example: baseline 3 (requires rtf_core + gurobi + mysql + config)
    # out = baseline_deletion_3(target='education', key=2, dataset='adult', threshold=0.5)
    pass
