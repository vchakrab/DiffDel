#!/usr/bin/env python3
"""
BASELINE 3 (ILP) + DELEXP HYPERGRAPH + DELEXP LEAKAGE MODEL (Algorithm 2)

CLEANED + FIXED VERSION:
  - enumerate/iter_chains fixed to match EnumerateChains pseudocode AND be faster
  - rest unchanged from your uploaded script
"""

from __future__ import annotations
from collections import Counter
import importlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import deque

import numpy as np

# Optional DB deps
try:
    import mysql.connector
except Exception:
    mysql = None

# Optional config
try:
    import config  # type: ignore
except Exception:
    config = None

# =========================
# USER SETTINGS
# =========================
LAM = 0.5     # set to same λ as delexp
RHO = 0.9     # delexp rho threshold (ρ-safe)
AUTO_ADJUST_RHO = True

DELETION_QUERY = """
UPDATE {table_name}
SET `{column_name}` = NULL
WHERE id = {key};
"""


# ============================================================
# STRICT WEIGHT LOADING (same naming convention as delexp)
# ============================================================

def get_dataset_weights_strict(dataset: str) -> Any:
    """
    Loads edge weights using the same convention as delexp:
      weights.weights_corrected.<dataset>_weights with a WEIGHTS object.
    """
    ds = str(dataset).lower()
    module_name = f"weights.weights_corrected.{ds}_weights"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            f"Missing edge-weight module '{module_name}'. "
            f"Expected: weights/weights_corrected/{ds}_weights.py defining WEIGHTS."
        ) from e

    if not hasattr(mod, "WEIGHTS"):
        raise FileNotFoundError(f"Module '{module_name}' exists but does not define WEIGHTS.")

    weights_obj = getattr(mod, "WEIGHTS")
    if weights_obj is None:
        raise FileNotFoundError(f"Module '{module_name}' defines WEIGHTS=None; expected actual weights.")
    return weights_obj


def map_dc_to_weight_strict(init_manager, dc, weights_obj) -> float:
    """
    delexp-style mapping: use init_manager.denial_constraints ordering.
    """
    try:
        idx = init_manager.denial_constraints.index(dc)
        attr_sets = []
        for dc in init_manager.denial_constraints:
            attrs = set()
            for a, _, b in dc:
                attrs.add(a.split(".", 1)[1])
                attrs.add(b.split(".", 1)[1])
            attr_sets.append(frozenset(attrs))

        for sets1, sets2 in zip(attr_sets, attr_sets):
            if sets1 == sets2 and not sets1.__eq__(sets2):
                print(sets1, sets2)
        counts = Counter(attr_sets)

        num_identical_dcs = sum(c for c in counts.values() if c > 1)
        num_unique_dcs = sum(c for c in counts.values() if c == 1)
    except ValueError:
        return 1.0
    try:
        print(len(weights_obj))
        print(len(dc))
        return float(weights_obj[idx])
    except Exception:
        raise RuntimeError("WEIGHTS object is not indexable by DC index; check weights module format.")


def dc_to_rdrs_and_weights_strict(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Convert denial constraints into RDRs (tuples of attribute names) + aligned weights.
    Schema-level: each DC becomes one hyperedge over the attrs it mentions.
    """
    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    weights_obj = get_dataset_weights_strict(init_manager.dataset)

    for dc in getattr(init_manager, "denial_constraints", []) or []:
        attrs: Set[str] = set()
        w = map_dc_to_weight_strict(init_manager, dc, weights_obj)

        for pred in dc:
            if not isinstance(pred, (list, tuple)) or len(pred) < 1:
                continue

            tok0 = pred[0]
            if isinstance(tok0, str) and "." in tok0:
                attrs.add(tok0.split(".")[-1])

            if len(pred) >= 3:
                tok2 = pred[2]
                if isinstance(tok2, str) and "." in tok2:
                    attrs.add(tok2.split(".")[-1])

        if len(attrs) >= 2:
            rdrs.append(tuple(sorted(attrs)))
            rdr_weights.append(float(w))

    print("RDRs:", rdrs)
    print("RDR_weights:", rdr_weights)
    return rdrs, rdr_weights


# ============================================================
# delexp: Hypergraph construction (Algorithm 1)
# ============================================================

class Hypergraph:
    def __init__(self):
        self.vertices: Set[str] = set()
        self.edges: List[Tuple[Set[str], float]] = []  # (edge_vertices, weight)

    def add_vertex(self, v: str):
        self.vertices.add(v)

    def add_edge(self, vertices: Set[str], weight: float):
        if len(vertices) >= 2:
            self.edges.append((set(vertices), float(weight)))
            self.vertices.update(vertices)


def incident_rdrs(cell: str, rdrs: List[Tuple[str, ...]]) -> List[int]:
    out: List[int] = []
    for i, rdr in enumerate(rdrs):
        if cell in rdr:
            out.append(i)
    return out


def instantiate_rdr(rdr: Tuple[str, ...], weight: float, mode: str = "MAX") -> List[Tuple[Set[str], float]]:
    verts = set(rdr)
    if len(verts) < 2:
        return []
    return [(verts, float(weight))]


def construct_local_hypergraph(
    target_cell: str,
    rdrs: List[Tuple[str, ...]],
    weights: List[float],
    mode: str = "MAX",
) -> Hypergraph:
    H = Hypergraph()
    H.add_vertex(target_cell)

    added_rdrs: Set[int] = set()
    frontier: Set[str] = {target_cell}
    seen: Set[str] = set()

    while frontier:
        next_frontier: Set[str] = set()
        for c in list(frontier):
            if c in seen:
                continue
            seen.add(c)

            for rdr_idx in incident_rdrs(c, rdrs):
                if rdr_idx in added_rdrs:
                    continue
                added_rdrs.add(rdr_idx)

                rdr = rdrs[rdr_idx]
                w = weights[rdr_idx]
                for edge_verts, edge_w in instantiate_rdr(rdr, w, mode):
                    H.add_edge(edge_verts, edge_w)
                    for v in edge_verts:
                        if v not in seen:
                            next_frontier.add(v)

        frontier = next_frontier

    return H


def construct_hypergraph_max(target_cell: str, rdrs: List[Tuple[str, ...]], weights: List[float]) -> Hypergraph:
    return construct_local_hypergraph(target_cell, rdrs, weights, mode="MAX")


def construct_hypergraph_actual(target_cell: str, rdrs: List[Tuple[str, ...]], weights: List[float]) -> Hypergraph:
    return construct_local_hypergraph(target_cell, rdrs, weights, mode="ACTUAL")


# ============================================================
# delexp: Leakage computation (Algorithm 2) — SINGLE SOURCE OF TRUTH
# ============================================================

def prod(vals: Iterable[float]) -> float:
    out = 1.0
    for x in vals:
        out *= float(x)
    return float(out)


def active(edge_verts: Set[str], x: str, K: Set[str]) -> bool:
    # Active(e, x, K): x in e and e\{x} subset K
    return (x in edge_verts) and ((edge_verts - {x}) <= K)


def hypergraph_to_edge_dict(
    H: "Hypergraph",
    *,
    tau: Optional[float] = None,
) -> Dict[str, Tuple[Set[str], float]]:
    out: Dict[str, Tuple[Set[str], float]] = {}
    for i, (verts, w) in enumerate(H.edges):
        fw = float(w)
        if tau is not None and fw > float(tau):
            continue
        out[f"e{i}"] = (set(verts), fw)
    return out


def is_edge_active_by_mask_rule(edge_verts: Set[str], mask: Set[str], target_cell: str) -> bool:
    """
    Edge ACTIVE iff it contains <2 masked verts, treating target as masked.
    """
    cnt = 0
    for v in edge_verts:
        if v == target_cell or v in mask:
            cnt += 1
            if cnt >= 2:
                return False
    return True


from collections import deque
from typing import Set, Tuple, Dict, List, Optional, Iterable

def iter_chains(mask: Set[str], target: str, edges: Dict[str, Tuple[Set[str], float]]):
    """
    EnumerateChains(M, c*, H) with *visited-state pruning*.

    Same as your current implementation, but adds:
      visited_K: Set[frozenset] to avoid re-expanding identical knowledge states K.

    NOTE: This pruning is NOT in the paper. It intentionally collapses multiple
    distinct edge-sequences that reach the same K into one expansion, which can
    significantly reduce chain counts (and will change leakage if you're summing
    over chains).
    """
    if not edges:
        return

    # Build V and O
    V = set().union(*(verts for (verts, _w) in edges.values()))
    O = V - set(mask) - {target}

    # Build incident index: vertex -> set(edge_ids)
    incident: Dict[str, Set[str]] = {}
    for eid, (verts, _w) in edges.items():
        for v in verts:
            incident.setdefault(v, set()).add(eid)

    # Length-1 chains: edges directly active for target under O
    for eid, (verts, _w) in edges.items():
        if target in verts and active(verts, target, O):
            yield [eid]

    # BFS queue of (chain_edge_ids, known_set)
    Q: deque[Tuple[List[str], Set[str]]] = deque()

    # Seed with edges that can infer some masked x directly from O
    for eid, (verts, _w) in edges.items():
        for x in (verts - O - {target}):
            if active(verts, x, O):
                Q.append(([eid], set(O) | {x}))

    # Visited-state pruning (knowledge set only)
    visited_K: Set[frozenset] = set()

    # Expand
    while Q:
        p, K = Q.popleft()

        K_key = frozenset(K)
        if K_key in visited_K:
            continue
        visited_K.add(K_key)

        used = set(p)

        # candidate edges must share a known cell (witness): e ∩ K != ∅
        cand_eids: Set[str] = set()
        for v in K:
            cand_eids |= incident.get(v, set())

        for eid in cand_eids:
            if eid in used:
                continue

            verts, _w = edges[eid]

            # Active(e, y, K) is only possible if exactly ONE vertex in e is missing from K
            missing = verts - K
            if len(missing) != 1:
                continue

            (y,) = tuple(missing)

            # Must be active given current known set
            if not active(verts, y, K):
                continue

            # Maximality: do not extend through edges already active under O
            if active(verts, y, O):
                continue

            if y == target:
                yield p + [eid]
            else:
                Q.append((p + [eid], set(K) | {y}))



def compute_leakage_delexp(
    mask: Set[str],
    target_cell: str,
    hypergraph: "Hypergraph",
    rho: float,
    *,
    tau: Optional[float] = None,
    return_counts: bool = False,
):
    """
    FAST version:
      - leakage = Noisy-OR only (streaming)
      - no chain list stored
      - returns counts if return_counts=True:
            (L, num_chains, active_chains, blocked_chains)
      - else returns L only
    """
    edge_dict = hypergraph_to_edge_dict(hypergraph, tau=tau)
    if not edge_dict:
        return (0.0, 0, 0, 0) if return_counts else 0.0

    # Precompute edge weights + edge active flags for THIS mask (O(E))
    w = {eid: float(edge_dict[eid][1]) for eid in edge_dict}
    edge_active = {
        eid: is_edge_active_by_mask_rule(edge_dict[eid][0], mask, target_cell)
        for eid in edge_dict
    }

    num_chains = 0
    active_chains = 0
    blocked_chains = 0

    # Noisy-OR product accumulator
    prod_not = 1.0
    max_chain_w = 0.0

    for ch in iter_chains(mask, target_cell, edge_dict):
        num_chains += 1

        # chain activity by your rule
        ok = True
        for eid in ch:
            if not edge_active.get(eid, True):
                ok = False
                break
        if ok:
            active_chains += 1
        else:
            blocked_chains += 1

        # chain weight + rho-safe
        cw = 1.0
        for eid in ch:
            cw *= w[eid]
        if cw > max_chain_w:
            max_chain_w = cw

        # update noisy-or
        prod_not *= (1.0 - cw)

    if num_chains == 0:
        L = 0.0
    else:
        L = 1.0 - prod_not

    if max_chain_w > float(rho):
        L = 1.0

    if return_counts:
        return float(L), int(num_chains), int(active_chains), int(blocked_chains)
    return float(L)


# ============================================================
# Utility
# ============================================================

def compute_utility_new(*, leakage: float, mask_size: int, lam: float, zone_size: int) -> float:
    """
    u(M) = -λ·L(M) - (1-λ)·|M|/(|I(c*)|-1)
    """
    denom = max(1, int(zone_size) - 1)
    norm = float(mask_size) / float(denom)
    return float(-(lam * float(leakage)) - ((1.0 - lam) * norm))


# ============================================================
# Memory estimator
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


# ============================================================
# Baseline 3 ILP (unchanged structurally)
# ============================================================

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
    init_mgr.initialize()

    rdrs, rdr_weights = dc_to_rdrs_and_weights_strict(init_mgr)

    rho = float(RHO)
    if AUTO_ADJUST_RHO and rdr_weights:
        mx = max(rdr_weights)
        if mx > rho:
            rho = min(0.999, mx + 0.01)

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
        leakage_val = compute_leakage_delexp(mask_for_leakage, target, H_actual, rho=rho)

        print("Leakage:", leakage_val)

        utility = compute_utility_new(
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
    print(baseline_deletion_3("education", 500, "adult", 0))
