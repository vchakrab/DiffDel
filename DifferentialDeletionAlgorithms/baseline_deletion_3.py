#!/usr/bin/env python3
"""
BASELINE 3 (ILP) + DELEXP HYPERGRAPH + DELEXP LEAKAGE MODEL (Algorithm 2)

CLEANED + FIXED VERSION:
  - enumerate/iter_chains fixed to match EnumerateChains pseudocode AND be faster
  - rest unchanged from your uploaded script
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple
from leakage import leakage, dc_to_rdrs_and_weights, construct_hypergraph_max, construct_hypergraph_actual, compute_utility
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

try:
    from rtf_core import initialization_phase
except Exception:
    initialization_phase = None


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


def instantiate_edges_with_time_filter(cursor, table, key, attr, target_time, hypergraph: Dict[Tuple[str, ...], float]):
    edges = []
    for edge_attrs, weight in hypergraph.items():
        edge_attr_names = {ea.split(".")[-1] for ea in edge_attrs if isinstance(ea, str)}
        if attr not in edge_attr_names:
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

        curr_attr = curr.attribute.split(".")[-1]
        edges = instantiate_edges_with_time_filter(cursor, table, key, curr_attr, target_time, hypergraph)

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

    try:
        ilp_num_vars = int(model.NumVars)
        ilp_num_constrs = int(model.NumConstrs)
    except Exception:
        ilp_num_vars = 0
        ilp_num_constrs = 0

    model.dispose()
    total_ilp_time = time.time() - start_total_ilp

    return to_delete, total_ilp_time, max_depth, len(cell_to_id), activated_dependencies_count, ilp_num_vars, ilp_num_constrs


# ============================================================
# Baseline 3 wrapper + delexp leakage/utility integration
# ============================================================

def baseline_deletion_3(target: str, key: int, dataset: str, threshold: float):
    """
    Returns:
      activated_dependencies_count,
      final_mask,
      memory_bytes,
      max_depth,
      init_time,
      model_time,
      deletion_time,
      leakage,
      utility,
      num_cells_aux
    """
    if not GUROBI_AVAILABLE:
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    if initialization_phase is None or config is None or mysql is None:
        print("[WARN] Missing deps for baseline_deletion_3 (rtf_core/config/mysql).")
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    init_start = time.time()
    init_mgr = initialization_phase.InitializationManager({"key": key, "attribute": target}, dataset, threshold)


    rdrs, rdr_weights = dc_to_rdrs_and_weights(init_mgr)


    # ILP hyperedge dict uses "t1.attr" tokens
    hyperedge_dict: Dict[Tuple[str, ...], float] = {}
    for rdr, w in zip(rdrs, rdr_weights):
        edge = tuple(sorted({f"t1.{a}" for a in rdr}))
        hyperedge_dict[edge] = float(w)

    init_time = time.time() - init_start

    model_time = 0.0
    deletion_time = 0.0
    conn = None
    cursor = None

    activated_dependencies_count = 0
    final_mask: Set[str] = set()
    memory_bytes = 0
    max_depth = 0
    num_cells = 0
    leakage_val = 0.0
    utility = 0.0

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

        to_del_cells, total_ilp_time, max_depth, num_cells, activated_dependencies_count, ilp_num_vars, ilp_num_constrs = (
            ilp_approach_matching_java(cursor, table, key, target, target_time, hyperedge_dict)
        )
        model_time = float(total_ilp_time)

        for cell in to_del_cells:
            attr = cell.attribute.split('.')[-1]
            cursor.execute(DELETION_QUERY.format(table_name=table, column_name=attr, key=key))
        conn.commit()

        deletion_time = (time.time() - deletion_start) - model_time

        final_mask = {cell.attribute.split('.')[-1] for cell in to_del_cells}

        # mask should not include target
        mask_for_leakage = set(final_mask)
        mask_for_leakage.discard(target)

        H_max = construct_hypergraph_max(target, rdrs, rdr_weights)
        H_actual = construct_hypergraph_actual(target, rdrs, rdr_weights)

        zone_size = len(H_max.vertices - {target})

        # DEBUG (keep)
        E_star = [(vs, w) for (vs, w) in H_actual.edges if target in vs]
        print("\n[DEBUG leakage]")
        print("target:", target)
        print("target in vertices:", target in H_actual.vertices)
        print("num_edges:", len(H_actual.edges))
        print("num_E_star:", len(E_star))
        if E_star:
            ws = [float(w) for _, w in E_star]
            print("E* weight min/max/mean:", min(ws), max(ws), sum(ws) / len(ws))
        else:
            print("No channels into target => leakage must be 0 for all masks.")

        # ✅ FAST leakage-only call (no chains returned)
        leakage_val = leakage(mask_for_leakage, target, H_actual)

        print("Leakage:", leakage_val)

        utility = compute_utility(
            leakage=leakage_val,
            mask_size=len(mask_for_leakage),
            lam=float(LAM),
            zone_size=zone_size
        )

        num_edges = len(H_actual.edges)
        edge_members = sum(len(vs) for vs, _w in H_actual.edges)
        num_vertices = len(H_actual.vertices)

        memory_bytes = estimate_memory_bytes_standard(
            num_vertices=num_vertices,
            num_edges=num_edges,
            edge_members=edge_members,
            mask_size=len(mask_for_leakage),
            stores_candidate_masks=False,
            includes_inferable_model=False,
            includes_channel_map=False,
            ilp_num_cells=num_cells,
            ilp_num_vars=ilp_num_vars,
            ilp_num_constrs=ilp_num_constrs,
        )

    except Exception as e:
        print(f"Error in Baseline 3: {e}")
        import traceback
        traceback.print_exc()
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    num_cells_aux = int(num_cells) - 2

    return (
        int(activated_dependencies_count),
        set(final_mask),
        int(memory_bytes),
        int(max_depth),
        float(init_time),
        float(model_time),
        float(deletion_time),
        float(leakage_val),
        float(utility),
        int(num_cells_aux),
    )


if __name__ == '__main__':
    # Example:
    print(baseline_deletion_3("ProviderNumber", 500, "hospital", 0))
