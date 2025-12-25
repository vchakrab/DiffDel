#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from functools import reduce
import math

# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class Edge:
    name: str
    cells: Tuple[str, ...]
    w: float  # edge weight

def pretty_path(edges: List[Edge], path: List[int]) -> str:
    return " -> ".join("{" + edges[i].name + "}" for i in path)

def path_weight(edges: List[Edge], path: List[int]) -> float:
    w = 1.0
    for i in path:
        w *= edges[i].w
    return w

def compute_vertices(edges: List[Edge]) -> Set[str]:
    V = set()
    for e in edges:
        V.update(e.cells)
    return V

# ============================================================
# Deterministic initial known S
# - "Typical" in text: S = V \ (M ∪ {target})
# - Fig4 didactic run: S = {'Gender'} (to force multi-step chaining)
# ============================================================

def compute_initial_known_typical(edges: List[Edge], target: str, mask: Set[str]) -> Set[str]:
    V = compute_vertices(edges)
    return (V - set(mask)) - {target}

# ============================================================
# Algorithm 2 (practical version): enumerate all inference paths
# Forward-chaining DFS:
# - known starts at initial_known S
# - any edge e is applicable if exactly 1 cell in e is unknown
# - that unique unknown is inferred; recurse
# - record path if inferred == target
#
# IMPORTANT: This matches the narrative description around Alg 2:
# "repeatedly searches for applicable hyperedges ... e \\ K = {x}"
# ============================================================

def find_inference_paths_str(
    hyperedges: List[Tuple[str, ...]],
    target_cell: str,
    initial_known: Set[str],
) -> List[List[int]]:
    # This wrapper keeps your original signature,
    # but we expect the caller to build the "hyperedges" list from Edge.cells.
    all_paths: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = set()

    def dfs(known: Set[str], path: List[int]) -> None:
        progressed = False
        for ei, e in enumerate(hyperedges):
            e_set = set(e)
            unknown = e_set - known

            # applicable step: infer the unique unknown cell
            if len(unknown) == 1:
                progressed = True
                inferred = next(iter(unknown))

                new_known = set(known)
                new_known.add(inferred)
                new_path = path + [ei]

                if inferred == target_cell:
                    t = tuple(new_path)
                    if t not in seen:
                        seen.add(t)
                        all_paths.append(new_path)
                else:
                    # cycle guard: don't keep recursing if we aren't adding anything
                    dfs(new_known, new_path)

        # if no applicable edges, dead end

    dfs(set(initial_known), [])
    return all_paths

# ============================================================
# IsBlocked / filter_active_paths_str (Alg 2 helper)
# A path is blocked if at ANY step, the edge requires a masked cell
# to infer its next inferred cell.
#
# We simulate the inference along the path:
# - start known_so_far = initial_known
# - for each edge: inferred is the unique unknown in that edge
# - required = edge minus inferred
# - if required intersects mask => blocked
# ============================================================

def filter_active_paths_str(
    hyperedges: List[Tuple[str, ...]],
    paths: List[List[int]],
    mask: Set[str],
    initial_known: Set[str],
) -> List[List[int]]:
    mask = set(mask)
    active: List[List[int]] = []

    for path in paths:
        known = set(initial_known)
        blocked = False

        for ei in path:
            e = set(hyperedges[ei])
            unknown = e - known

            # path must be valid: each step infers exactly one cell
            if len(unknown) != 1:
                blocked = True
                break

            inferred = next(iter(unknown))
            required = e - {inferred}

            if required & mask:
                blocked = True
                break

            known.add(inferred)

        if not blocked:
            active.append(path)

    return active

# ============================================================
# Algorithm 3 (Leakage)
# - final edges E* are those containing the target AND which are the
#   last edge in the path (the "final(π)" in the paper)
# - group active paths by their final edge index
# - per final edge: w*_e = 1 - Π_{π in Πe} (1 - w(π))
# - overall L = 1 - Π_{e in E*} (1 - w*_e)
# ============================================================

def compute_leakage_alg3(
    edges: List[Edge],
    paths: List[List[int]],
    mask: Set[str],
    initial_known: Set[str],
    target: str,
) -> float:
    hyperedges = [e.cells for e in edges]

    # Filter to active paths
    active = filter_active_paths_str(hyperedges, paths, mask, initial_known)

    # E* = edges that can be "final" for the target:
    # i.e., edges containing target (paper), but practically: final edge in some path.
    paths_by_final: Dict[int, List[List[int]]] = {}
    for p in active:
        final_ei = p[-1]
        if target in set(hyperedges[final_ei]):
            paths_by_final.setdefault(final_ei, []).append(p)

    # compute w*_e for each final edge
    w_star: Dict[int, float] = {}
    for final_ei, plist in paths_by_final.items():
        prod = 1.0
        for p in plist:
            wp = path_weight(edges, p)
            prod *= (1.0 - wp)
        w_star[final_ei] = 1.0 - prod

    # combine across final edges containing target
    prod2 = 1.0
    for final_ei, wse in w_star.items():
        prod2 *= (1.0 - wse)

    return 1.0 - prod2

# ============================================================
# Fig 4 toy hypergraph (master)
# ============================================================

def build_fig4_edges() -> List[Edge]:
    # From the figure:
    # e_a: Gender -> Age (0.95)
    # e_b: Age -> Result (0.90)
    # e_c: Gender -> BMI (0.92)
    # e_d: BMI -> Result (0.85)
    # e_e: Age -> BMI (0.88)
    # e1 : Result -> Diag (0.95)
    # e2 : BMI -> Diag (0.85)
    #
    # In the paper’s hypergraph semantics, hyperedges are undirected sets of cells.
    return [
        Edge("e_a", ("Gender", "Age"), 0.95),
        Edge("e_b", ("Age", "Result"), 0.90),
        Edge("e_c", ("Gender", "BMI"), 0.92),
        Edge("e_d", ("BMI", "Result"), 0.85),
        Edge("e_e", ("Age", "BMI"), 0.88),
        Edge("e1",  ("Result", "Diag"), 0.95),
        Edge("e2",  ("BMI", "Diag"), 0.85),
    ]

def run_case(title: str, edges: List[Edge], target: str, initial_known: Set[str], mask: Set[str]) -> None:
    hyperedges = [e.cells for e in edges]
    V = sorted(compute_vertices(edges))

    paths = find_inference_paths_str(hyperedges, target, initial_known)
    active = filter_active_paths_str(hyperedges, paths, mask, initial_known)
    L = compute_leakage_alg3(edges, paths, mask, initial_known, target)

    print("==================================================")
    print(title)
    print(f"Vertices V: {V}")
    print(f"Target c*: {target}")
    print(f"Initial known S: {sorted(initial_known)}")
    print(f"Aux Mask M: {sorted(mask)}")
    print()
    print(f"Total paths Π: {len(paths)}")
    for p in paths:
        print(f"  - {pretty_path(edges, p)}  weight={path_weight(edges, p):.3f}")
    print()
    print(f"Active (unblocked) paths: {len(active)}")
    for p in active:
        print(f"  - {pretty_path(edges, p)}  weight={path_weight(edges, p):.3f}")
    print()
    print(f"Leakage L = {L:.3f}")
    print("==================================================\n")

if __name__ == "__main__":
    edges = build_fig4_edges()
    target = "Diag"

    # Fig 4 is effectively illustrating multi-step chaining, so use:
    S_fig4 = {"Gender"}  # <- this is why you see Gender as initial known

    run_case("Fig4 toy (Mask = ∅)", edges, target, S_fig4, mask=set())
    run_case("Fig4 toy (Mask = {Result})", edges, target, S_fig4, mask={"Result"})
    run_case("Fig4 toy (Mask = {Result, BMI})", edges, target, S_fig4, mask={"Result", "BMI"})

    # If you want the "typical" paper choice:
    # S_typical = compute_initial_known_typical(edges, target, mask=set())
    # run_case("Fig4 toy with typical S = V \\ (M ∪ {c*})", edges, target, S_typical, mask=set())
