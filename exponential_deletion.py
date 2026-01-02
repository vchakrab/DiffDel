"""
exponential_deletion.py - Hypergraph-Based Implementation

Implements Algorithm 1 (Construct Local Hypergraph) over RDRs at schema level
and Algorithm 2 (ComputeLeakage) from the paper.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple
from itertools import chain, combinations
import rtf_core.initialization_phase
from collections import deque
import weights
import numpy as np

from rtf_core import initialization_phase
from rtf_core.initialization_phase import InitializationManager


# ============================================================
# Helpers
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


import importlib


def get_dataset_weights(dataset: str):
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"

    try:
        weights_module = importlib.import_module(module_name)
        return weights_module.WEIGHTS
    except ModuleNotFoundError:
        print(f"[WARN] No weights module found for dataset: {dataset}")
        return None

    raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================
# Paper Implementation: Hypergraph Construction (Algorithm 1)
# Adapted to work at schema-level over RDRs
# ============================================================

class Hypergraph:
    """
    Represents a local hypergraph H = (V, E) around a target cell.
    Each hyperedge e has an associated weight w_e.

    At schema level:
    - Vertices are attribute names (not (tuple_id, attribute) pairs)
    - Edges connect attributes that appear together in RDRs
    """

    def __init__(self):
        self.vertices: Set[str] = set()
        self.edges: List[Tuple[Set[str], float]] = []  # (edge_vertices, weight)

    def add_vertex(self, v: str):
        self.vertices.add(v)

    def add_edge(self, vertices: Set[str], weight: float):
        """Add a hyperedge with its weight."""
        if len(vertices) >= 2:
            self.edges.append((vertices, weight))
            self.vertices.update(vertices)


def incident_rdrs(cell: str, rdrs: List[Tuple[str, ...]]) -> List[int]:
    """
    Find all RDRs that mention the given cell (attribute).
    Returns indices of RDRs.

    This is the IncidentRDRs(c, Σ) function from Algorithm 1, Line 7.
    """
    incident = []
    for idx, rdr in enumerate(rdrs):
        if cell in rdr:
            incident.append(idx)
    return incident


def instantiate_rdr(rdr: Tuple[str, ...], weight: float, mode: str = "MAX") -> List[Tuple[Set[str], float]]:
    """
    Instantiate an RDR to produce hyperedges.

    In schema-level mode (over RDRs without database):
    - MAX mode: Include all attributes in the RDR as one hyperedge
    - ACTUAL mode: Same as MAX (would filter by predicates with database)

    This corresponds to Line 8 in Algorithm 1: Instantiate(σ, c, D, mode)

    Args:
        rdr: Tuple of attributes in the RDR
        weight: Weight of this RDR
        mode: "MAX" or "ACTUAL"

    Returns:
        List of (edge_vertices, weight) tuples
    """
    # At schema level, each RDR creates one hyperedge containing all its attributes
    edge_vertices = set(rdr)

    if len(edge_vertices) < 2:
        return []

    return [(edge_vertices, weight)]


def construct_local_hypergraph(
    target_cell: str,
    rdrs: List[Tuple[str, ...]],
    weights: List[float],
    mode: str = "MAX"
) -> Hypergraph:
    """
    Algorithm 1: Construct Local Hypergraph H^{c*}

    Builds hypergraph by frontier-based exploration from target cell c*.

    Args:
        target_cell: The target attribute c*
        rdrs: List of RDRs Σ (as tuples of attributes)
        weights: Weight w_σ for each RDR
        mode: "MAX" or "ACTUAL"

    Returns:
        Hypergraph H = (V, E)
    """
    H = Hypergraph()

    # Line 1: Initialize V and E
    V = {target_cell}
    E = []
    H.add_vertex(target_cell)

    # Track which RDRs we've already added to avoid duplicates
    added_rdrs: Set[int] = set()

    # Line 2: Initialize frontier and seen
    frontier = {target_cell}
    seen = set()

    # Line 3: While frontier is not empty
    while frontier:
        # Line 4: Initialize next frontier
        next_frontier = set()

        # Line 5: For each cell c in frontier
        for c in list(frontier):
            # Check if already processed
            if c in seen:
                continue

            # Line 6: Mark as seen
            seen.add(c)

            # Line 7: For each incident RDR
            for rdr_idx in incident_rdrs(c, rdrs):
                # Skip if we've already added this RDR
                if rdr_idx in added_rdrs:
                    continue

                added_rdrs.add(rdr_idx)

                rdr = rdrs[rdr_idx]
                rdr_weight = weights[rdr_idx]

                # Line 8: Instantiate the RDR
                instantiated_edges = instantiate_rdr(rdr, rdr_weight, mode)

                # Line 9-11: For each hyperedge from instantiation
                for edge_vertices, edge_weight in instantiated_edges:
                    # Line 10: Add edge and vertices
                    H.add_edge(edge_vertices, edge_weight)

                    # Line 11: Add new vertices to next frontier
                    for v in edge_vertices:
                        if v not in seen:
                            next_frontier.add(v)

        # Line 13: Update frontier
        frontier = next_frontier

    # Line 15: Return hypergraph
    return H


def construct_hypergraph_max(
    target_cell: str,
    rdrs: List[Tuple[str, ...]],
    weights: List[float]
) -> Hypergraph:
    """
    Construct H^{c*}_max: Maximum hypergraph (includes all possible instantiations).

    This is used to define the inference zone I(c*) = V(H^{c*}_max) \ {c*}.
    """
    return construct_local_hypergraph(target_cell, rdrs, weights, mode="MAX")


def construct_hypergraph_actual(
    target_cell: str,
    rdrs: List[Tuple[str, ...]],
    weights: List[float]
) -> Hypergraph:
    """
    Construct H^{c*}_D: Actual hypergraph (only satisfied predicates).

    This is used for leakage evaluation.
    At schema level without database, this is the same as MAX.
    """
    return construct_local_hypergraph(target_cell, rdrs, weights, mode="ACTUAL")


# ============================================================
# Paper Implementation: Leakage Computation (Algorithm 2)
# ============================================================

def get_inference_chains_bfs(
    hypergraph: Hypergraph,
    target_cell: str,
    masked_cells: Set[str],
    debug: bool = False
) -> Dict[str, List[List[str]]]:
    """
    Get inference chains P_c for each masked cell c via BFS from root nodes.

    Line 4 in Algorithm 2: "Get inference chains P_c for each c ∈ M"

    A chain is a path from an observable cell to a masked cell through hyperedges.
    Root nodes are observable (visible) cells.

    Returns:
        Dict mapping masked cell -> list of chains (paths)
    """
    # Observable cells (Line 1)
    O = hypergraph.vertices - masked_cells - {target_cell}

    if debug:
        print(f"\n--- Finding Inference Chains ---")
        print(f"Observable roots: {sorted(O)}")
        print(f"Masked cells: {sorted(masked_cells)}")

    chains: Dict[str, List[List[str]]] = {c: [] for c in masked_cells}

    # BFS from each observable (root) cell
    for root in O:
        # Queue: (current_cell, path_so_far)
        queue = deque([(root, [root])])
        visited_in_this_bfs = {root}

        while queue:
            current, path = queue.popleft()

            # Explore all hyperedges containing current cell
            for edge_vertices, edge_weight in hypergraph.edges:
                if current not in edge_vertices:
                    continue

                # All other vertices in edge are reachable from current
                for next_cell in edge_vertices:
                    if next_cell == current:
                        continue

                    # Avoid cycles in path
                    if next_cell in path:
                        continue

                    new_path = path + [next_cell]

                    # If we reached a masked cell, record this chain
                    if next_cell in masked_cells:
                        chains[next_cell].append(new_path)
                        if debug:
                            print(f"  Chain to {next_cell}: {' → '.join(new_path)}")

                    # Continue BFS only through observable cells
                    if next_cell in O and next_cell not in visited_in_this_bfs:
                        visited_in_this_bfs.add(next_cell)
                        queue.append((next_cell, new_path))

    if debug:
        print(f"\nTotal chains found:")
        for cell, cell_chains in chains.items():
            print(f"  {cell}: {len(cell_chains)} chains")

    return chains


def compute_chain_weight(
    chain: List[str],
    hypergraph: Hypergraph
) -> float:
    """
    Compute chain weight w(p) from the last visible cell.

    Line 7 in Algorithm 2: w(p) ← chain weight from last visible cell

    Chain weight is the product of edge weights along the inference path.
    Uses Equation from paper for chain weight definition.
    """
    if len(chain) < 2:
        return 1.0

    # Use log-space for numerical stability
    log_weight = 0.0

    # Traverse chain and multiply edge weights
    for i in range(len(chain) - 1):
        current = chain[i]
        next_cell = chain[i + 1]

        # Find edge connecting current to next
        edge_found = False
        for edge_vertices, edge_weight in hypergraph.edges:
            if current in edge_vertices and next_cell in edge_vertices:
                if edge_weight <= 0:
                    return 0.0
                log_weight += math.log(min(1.0, edge_weight))
                edge_found = True
                break

        if not edge_found:
            # No direct edge connection (shouldn't happen in valid chain)
            return 0.0

    result = math.exp(log_weight)
    return min(1.0, max(1e-12, result))


def compute_leakage(
    mask: Set[str],
    target_cell: str,
    hypergraph: Hypergraph,
    rho: float = 0.7,
    debug: bool = False
) -> float:
    """
    Algorithm 2: ComputeLeakage(M, c*, H^{c*}_D, ρ)

    Computes leakage L(M, c*, D) for mask M on hypergraph H^{c*}_D.

    Args:
        mask: Set of masked cells M
        target_cell: Target cell c*
        hypergraph: Hypergraph H^{c*}_D
        rho: Safety threshold ρ
        debug: Print detailed diagnostics

    Returns:
        Leakage value ∈ [0, 1]
    """
    # Line 1: O ← V \ (M ∪ {c*})  (Observed/visible cells)
    O = hypergraph.vertices - mask - {target_cell}

    if debug:
        print(f"\n=== LEAKAGE COMPUTATION DEBUG ===")
        print(f"Target cell: {target_cell}")
        print(f"Mask M: {sorted(mask) if mask else '∅'}")
        print(f"Observable cells O: {sorted(O)}")

    # Lines 2-3: Initialize reachability
    # r(c) ← 1[c ∈ O]
    r: Dict[str, float] = {}
    for c in hypergraph.vertices:
        r[c] = 1.0 if c in O else 0.0

    # Line 4: Get inference chains P_c for each c ∈ M (via BFS from roots)
    chains = get_inference_chains_bfs(hypergraph, target_cell, mask, debug=debug)

    # Lines 5-9: Compute reachability for masked cells
    for c in mask:
        if c not in chains or not chains[c]:
            # No chains to this cell, reachability stays 0
            continue

        # Lines 6-7: For each chain p ∈ P_c, compute w(p)
        chain_weights = []
        for chain in chains[c]:
            w_p = compute_chain_weight(chain, hypergraph)
            chain_weights.append(w_p)

        # Line 8: r(c) ← noisy-OR aggregation (Equation for reachability)
        # r(c) = 1 - ∏_{p ∈ P_c} (1 - w(p))
        prod_fail = 1.0
        for w_p in chain_weights:
            prod_fail *= (1.0 - w_p)

        r[c] = 1.0 - prod_fail

    if debug and mask:
        print(f"\nReachabilities for masked cells:")
        for c in mask:
            print(f"  r({c}) = {r[c]:.6f}, chains: {len(chains.get(c, []))}")

    # Line 10: E* ← {e ∈ E : c* ∈ e}  (Inference channels)
    E_star = [(vertices, weight) for vertices, weight in hypergraph.edges
              if target_cell in vertices]

    if debug:
        print(f"\nInference channels (edges containing {target_cell}):")
        print(f"  Number of channels: {len(E_star)}")

    # Line 11: Initialize w*_max
    w_star_max = 0.0

    # Lines 12-14: For each channel edge, compute channel weight
    channel_weights = []

    for edge_idx, (edge_vertices, edge_weight) in enumerate(E_star):
        # Line 13: w*_e ← (∏_{c ∈ e\{c*}} r(c)) · w(e)  (Channel weight, Equation)
        prod_reach = 1.0
        prereqs = []
        for c in edge_vertices:
            if c != target_cell:
                prod_reach *= r.get(c, 0.0)
                prereqs.append((c, r.get(c, 0.0)))

        w_e_star = prod_reach * edge_weight
        channel_weights.append(w_e_star)

        if debug:
            print(f"  Channel {edge_idx}: {sorted(edge_vertices)}")
            print(f"    Prerequisites: {prereqs}")
            print(f"    Edge weight: {edge_weight:.6f}")
            print(f"    Product of reachabilities: {prod_reach:.6f}")
            print(f"    Channel weight w*_e: {w_e_star:.6f}")

        # Line 14: w*_max ← max(w*_max, w*_e)
        w_star_max = max(w_star_max, w_e_star)

    if debug:
        print(f"\nMax channel weight w*_max: {w_star_max:.6f}")
        print(f"ρ threshold: {rho:.6f}")
        print(f"Is ρ-safe? {w_star_max <= rho}")
        print(f"\nChannel weight distribution:")
        print(f"  Total channels: {len(channel_weights)}")
        if channel_weights:
            print(f"  Min channel weight: {min(channel_weights):.6f}")
            print(f"  Max channel weight: {max(channel_weights):.6f}")
            print(f"  Mean channel weight: {sum(channel_weights)/len(channel_weights):.6f}")
            print(f"  Channels with w > 0.5: {sum(1 for w in channel_weights if w > 0.5)}")
            print(f"  Channels with w > 0.3: {sum(1 for w in channel_weights if w > 0.3)}")
            print(f"  Channels with w > 0.1: {sum(1 for w in channel_weights if w > 0.1)}")

    # Line 15: Return leakage (Equation for leakage)
    if w_star_max > rho:
        # Not ρ-safe: return 1
        if debug:
            print(f"NOT ρ-safe (w*_max > ρ) → Leakage = 1.0")
        return 1.0

    # ρ-safe: compute noisy-OR aggregation using log-space to avoid underflow
    # L = 1 - ∏_{e ∈ E*} (1 - w*_e)

    # Use log-space: log(∏(1-w)) = Σlog(1-w)
    if not channel_weights:
        leakage = 0.0
    else:
        log_prod_fail = 0.0
        for w in channel_weights:
            if w >= 1.0:
                # If any channel has weight 1, leakage is 1
                leakage = 1.0
                if debug:
                    print(f"ρ-safe → Found channel with weight = 1.0 → Leakage = 1.0")
                return 1.0
            elif w > 1e-15:  # Only include non-negligible weights
                # Use log1p for numerical stability when w is small
                if w < 0.5:
                    log_prod_fail += math.log1p(-w)  # log(1-w) more stable
                else:
                    log_prod_fail += math.log(1.0 - w)

        # Convert back from log-space
        if log_prod_fail < -700:  # Underflow threshold (exp(-700) ≈ 0)
            prod_fail = 0.0
            leakage = 1.0
        else:
            prod_fail = math.exp(log_prod_fail)
            leakage = 1.0 - prod_fail

    if debug:
        print(f"ρ-safe → Computing noisy-OR over {len(channel_weights)} channels (log-space)")
        print(f"Log of product: {log_prod_fail:.6f}")
        print(f"Product of (1 - w*_e): {prod_fail:.12e}")
        print(f"Final leakage = 1 - product: {leakage:.12f}")

        # Show why it's so high
        if leakage > 0.99 and len(channel_weights) > 0:
            print(f"\n⚠️  High leakage analysis:")
            print(f"   With {len(channel_weights)} channels, even moderate weights compound:")
            avg_w = sum(channel_weights)/len(channel_weights) if channel_weights else 0
            theoretical = 1.0 - (1.0 - avg_w)**len(channel_weights)
            print(f"   Theoretical (avg weight {avg_w:.3f}): {theoretical:.6f}")
            print(f"   To reduce leakage, must mask cells to break/weaken channels")

    return max(1e-12, min(1.0, leakage))


# ============================================================
# Candidate masks + utility + exponential mechanism
# ============================================================

def compute_possible_mask_set_str(target_cell: str, hypergraph: Hypergraph) -> List[Set[str]]:
    """
    Generate all possible masks from the inference zone.

    Inference zone I(c*) = V(H^{c*}_max) \ {c*}

    The candidate mask space is the powerset of I(c*).
    """
    inference_zone = hypergraph.vertices - {target_cell}
    return [set(s) for s in powerset(sorted(inference_zone))]


def compute_utility(
        mask: Set[str],
        leakage: float,
        lam: float,
        num_candidates_minus_one: int
) -> float:
    """
    Utility function balancing leakage and mask size.

    u(M) = -λ·L(M) - (1-λ)·|M|/Z
    where Z normalizes mask size.
    """
    if num_candidates_minus_one <= 0:
        norm = 0.0
    else:
        norm = len(mask) / num_candidates_minus_one

    return float(-(lam * leakage) - ((1.0 - lam) * norm))


def exponential_mechanism_sample(
        candidates: List[Set[str]],
        *,
        target_cell: str,
        hypergraph: Hypergraph,
        epsilon: float,
        lam: float,
        rho: float = 0.9
) -> Tuple[Set[str], float]:
    """
    Sample a mask using the exponential mechanism.

    Pr[M] ∝ exp(ε·u(M) / (2·Δu))
    where Δu is the sensitivity of the utility function.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ALL {len(candidates)} CANDIDATE MASKS")
    print(f"{'='*80}")

    utilities = np.empty(len(candidates), dtype=float)
    leakages = np.empty(len(candidates), dtype=float)
    denom = max(1, len(candidates) - 1)

    sum_L = 0.0
    for i, M in enumerate(candidates):
        L = compute_leakage(M, target_cell, hypergraph, rho, debug=False)
        leakages[i] = L
        sum_L += L
        utilities[i] = compute_utility(M, L, lam, denom)

    # Print all masks and their leakages
    print(f"\nALL MASKS AND THEIR LEAKAGES:")
    print(f"{'Idx':<6} {'Mask Size':<12} {'Leakage':<12} {'Utility':<12} {'Mask Contents'}")
    print("-" * 100)

    # Sort by leakage for easier viewing
    sorted_indices = np.argsort(leakages)

    for idx in sorted_indices:
        M = candidates[idx]
        L = leakages[idx]
        U = utilities[idx]
        mask_str = f"{{{', '.join(sorted(M))}}}" if M else "∅"
        print(f"{idx:<6} {len(M):<12} {L:<12.6f} {U:<12.6f} {mask_str}")

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS:")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Average leakage: {sum_L / len(utilities):.6f}")
    print(f"  Min leakage: {np.min(leakages):.6f}")
    print(f"  Max leakage: {np.max(leakages):.6f}")
    print(f"  Leakage = 1.0 count: {np.sum(leakages >= 0.9999)}")
    print(f"  Leakage < 1.0 count: {np.sum(leakages < 0.9999)}")
    print(f"{'='*80}\n")

    # Exponential mechanism scores
    scores = (float(epsilon) * utilities) / (2.0 * max(1e-10, float(lam)))
    max_score = float(np.max(scores)) if len(scores) else 0.0
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores)

    idx = int(np.random.choice(len(candidates), p=probs))

    print(f"SELECTED MASK:")
    print(f"  Index: {idx}")
    print(f"  Mask: {sorted(candidates[idx]) if candidates[idx] else '∅'}")
    print(f"  Size: {len(candidates[idx])}")
    print(f"  Leakage: {leakages[idx]:.6f}")
    print(f"  Utility: {utilities[idx]:.6f}")
    print(f"  Selection probability: {probs[idx]:.6f}\n")

    return candidates[idx], float(utilities[idx])


# ============================================================
# "paths blocked" proxy
# ============================================================

def estimate_paths_proxy_from_channels(
        *,
        num_channel_edges: int,
        L_empty: float,
        L_mask: float
) -> Dict[str, int]:
    """
    Proxy for paths blocked based on leakage reduction.

    Estimates:
    - total_paths := number of channel edges
    - blocked := (1 - L_mask/L_empty) × total
    """
    total = int(max(0, num_channel_edges))
    if total == 0 or not np.isfinite(L_empty) or L_empty <= 1e-15 or not np.isfinite(L_mask):
        return {"num_paths_est": total, "paths_blocked_est": 0}

    frac = 1.0 - (float(L_mask) / float(L_empty))
    frac = float(max(0.0, min(1.0, frac)))
    blocked = int(round(frac * total))
    blocked = int(max(0, min(total, blocked)))
    return {"num_paths_est": total, "paths_blocked_est": blocked}


# ============================================================
# Memory estimate
# ============================================================

def estimate_memory_overhead_bytes_delexp(
        *,
        hypergraph: Hypergraph,
        mask_size: int,
        num_candidate_masks: int,
        candidate_mask_members: int,
        includes_channel_map: bool = True,
) -> int:
    """Memory overhead estimation for the exponential mechanism."""
    num_vertices = len(hypergraph.vertices)
    num_edges = len(hypergraph.edges)
    edge_members = sum(len(vertices) for vertices, _ in hypergraph.edges)

    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_SET = 96
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28
    BYTES_PER_CAND_MASK = 96

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    est += num_candidate_masks * BYTES_PER_CAND_MASK
    est += candidate_mask_members * BYTES_PER_MASK_MEMBER
    est += num_candidate_masks * 8  # utilities array

    return int(est)


# ============================================================
# Main orchestrator
# ============================================================

def load_parsed_dcs(dataset: str) -> List[List[Tuple[str, str, str]]]:
    """Load parsed denial constraints for the dataset."""
    try:
        dataset_module_name = "NCVoter" if dataset.lower() == "ncvoter" else dataset.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        return getattr(dc_module, "denial_constraints", [])
    except Exception:
        return []


def map_dc_to_weight(init_manager, dc, weights):
    """Map a denial constraint to its weight."""
    if dc in init_manager.denial_constraints:
        ind = init_manager.denial_constraints.index(dc)
        return weights[ind]
    return 1.0


def dc_to_hyperedges(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Convert denial constraints into RDRs (represented as tuples of attributes).

    Each DC becomes an RDR where:
    - Vertices are attribute names
    - Each RDR has an associated weight
    """
    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    dataset_weights = get_dataset_weights(init_manager.dataset)

    for dc in getattr(init_manager, "denial_constraints", []):
        attrs: Set[str] = set()
        weight = map_dc_to_weight(init_manager, dc, dataset_weights)

        # Extract attributes from predicates
        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                token = pred[0]
                if isinstance(token, str) and "." in token:
                    # Extract attribute name from "t1.attr" format
                    attrs.add(token.split(".")[-1])

        if len(attrs) >= 2:
            rdrs.append(tuple(sorted(attrs)))
            rdr_weights.append(weight)

    return rdrs, rdr_weights


def exponential_deletion_main(
        dataset: str,
        key: int,
        target_cell: str,
        *,
        epsilon: float = 10,
        lam: float = 0.67,
        rho: float = 0.9,
        auto_adjust_rho: bool = True
) -> Dict[str, Any]:
    """
    Main function implementing the hypergraph-based exponential deletion mechanism.

    Follows the paper's algorithms but operates at schema level over RDRs
    (without requiring database access).

    Args:
        dataset: Dataset name
        key: Record key (for initialization)
        target_cell: Target attribute to delete
        epsilon: Privacy parameter
        lam: Utility tradeoff parameter (λ)
        rho: Safety threshold (ρ) - channels with weight > ρ cause leakage = 1.0
        auto_adjust_rho: If True, automatically adjust ρ based on edge weights

    Returns:
        Dictionary with timing, leakage, utility, and other metrics
    """
    # ----------------------
    # INIT
    # ----------------------
    init_start = time.time()

    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target_cell},
        dataset,
        0
    )

    # Get RDRs (Σ) and their weights
    rdrs, rdr_weights = dc_to_hyperedges(init_manager)

    print(f"\n{'='*80}")
    print(f"RDR WEIGHTS ANALYSIS")
    print(f"{'='*80}")
    print(f"Total RDRs: {len(rdr_weights)}")
    if rdr_weights:
        print(f"Weight statistics:")
        print(f"  Min: {min(rdr_weights):.6f}")
        print(f"  Max: {max(rdr_weights):.6f}")
        print(f"  Mean: {sum(rdr_weights)/len(rdr_weights):.6f}")
        print(f"  Weights > ρ ({rho}): {sum(1 for w in rdr_weights if w > rho)}")
        print(f"  Weights ≤ ρ ({rho}): {sum(1 for w in rdr_weights if w <= rho)}")

        # Auto-adjust rho if requested and all weights exceed current rho
        if auto_adjust_rho:
            max_weight = max(rdr_weights)
            if max_weight > rho:
                # Set rho slightly above max weight to allow for variation
                suggested_rho = min(0.999, max_weight + 0.01)
                print(f"\n  ⚠️  WARNING: {sum(1 for w in rdr_weights if w > rho)} weights exceed ρ = {rho}")
                print(f"  This will cause most masks to have leakage = 1.0 (not ρ-safe)")
                print(f"  Suggested ρ: {suggested_rho:.6f} (slightly above max weight)")
                print(f"  Auto-adjusting ρ from {rho:.6f} to {suggested_rho:.6f}")
                rho = suggested_rho

    print(f"\nRDRs (Σ): {rdrs}")
    print(f"Weights: {rdr_weights}")

    instantiated_cells: Set[str] = set()
    for rdr in rdrs:
        instantiated_cells.update(rdr)
    num_instantiated_cells = len(instantiated_cells)

    init_time = time.time() - init_start

    # ----------------------
    # HYPERGRAPH CONSTRUCTION (Algorithm 1)
    # ----------------------
    model_start = time.time()

    print(f"\n=== Constructing Hypergraphs for target: {target_cell} ===")

    # Construct H^{c*}_max for defining inference zone
    H_max = construct_hypergraph_max(target_cell, rdrs, rdr_weights)
    print(H_max.edges)
    # Construct H^{c*}_D for leakage evaluation
    H_actual = construct_hypergraph_actual(target_cell, rdrs, rdr_weights)
    print(H_actual.edges)

    # Inference zone I(c*) = V(H^{c*}_max) \ {c*}
    inference_zone = H_max.vertices - {target_cell}

    print(f"Hypergraph vertices: {sorted(H_max.vertices)}")
    print(f"Number of hyperedges: {len(H_max.edges)}")
    print(f"Inference zone I(c*): {sorted(inference_zone)}")

    # ----------------------
    # CANDIDATE MASKS
    # ----------------------

    # Generate all possible masks from I(c*)
    candidates = compute_possible_mask_set_str(target_cell, H_max)
    print(f"Number of candidate masks: {len(candidates)}")

    if not candidates:
        candidates = [set()]

    # ----------------------
    # EXPONENTIAL MECHANISM
    # ----------------------

    # Sample mask using exponential mechanism
    final_mask, util_val = exponential_mechanism_sample(
        candidates,
        target_cell=target_cell,
        hypergraph=H_actual,
        epsilon=epsilon,
        lam=lam,
        rho=rho
    )

    # ----------------------
    # LEAKAGE COMPUTATION (Algorithm 2)
    # ----------------------

    print(f"\n{'='*80}")
    print("DETAILED LEAKAGE ANALYSIS - EMPTY MASK (BASELINE)")
    print(f"{'='*80}")

    # Compute leakage with empty mask (baseline)
    leakage_base = compute_leakage(set(), target_cell, H_actual, rho, debug=True)

    # Show impact of masking individual high-degree cells
    print(f"\n{'='*80}")
    print("IMPACT OF MASKING INDIVIDUAL CELLS")
    print(f"{'='*80}")

    # Find cells that appear in most channels
    cell_channel_count: Dict[str, int] = {}
    for vertices, weight in H_actual.edges:
        if target_cell in vertices:
            for cell in vertices:
                if cell != target_cell:
                    cell_channel_count[cell] = cell_channel_count.get(cell, 0) + 1

    # Sort by frequency
    sorted_cells = sorted(cell_channel_count.items(), key=lambda x: x[1], reverse=True)

    print(f"Top 5 cells by channel participation:")
    for cell, count in sorted_cells[:5]:
        leak_single = compute_leakage({cell}, target_cell, H_actual, rho, debug=False)
        print(f"  {cell}: appears in {count} channels, masking alone → L = {leak_single:.6f}")

    # Compute leakage with selected mask
    print(f"\n{'='*80}")
    print("DETAILED LEAKAGE ANALYSIS - SELECTED MASK")
    print(f"{'='*80}")
    leakage = compute_leakage(final_mask, target_cell, H_actual, rho, debug=True)

    print(f"\nLeakage (empty mask): {leakage_base:.12f}")
    print(f"Leakage (final mask): {leakage:.12f}")
    if leakage_base > 1e-12:
        print(f"Leakage reduction factor: {leakage / leakage_base:.6f}")
    print(f"Selected mask: {sorted(final_mask) if final_mask else '∅'}")

    # Count channel edges (edges containing target)
    num_channel_edges = sum(1 for vertices, _ in H_actual.edges if target_cell in vertices)

    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges=num_channel_edges,
        L_empty=leakage_base,
        L_mask=leakage
    )

    model_time = time.time() - model_start

    # ----------------------
    # MEMORY
    # ----------------------
    cand_members = sum(len(s) for s in candidates)
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hypergraph=H_actual,
        mask_size=len(final_mask),
        num_candidate_masks=len(candidates),
        candidate_mask_members=cand_members,
        includes_channel_map=True
    )

    return {
        "init_time": init_time,
        "model_time": model_time,
        "del_time": 0.0,

        "leakage": float(leakage),
        "utility": float(util_val),
        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),

        "num_paths": int(paths_proxy["num_paths_est"]),
        "paths_blocked": int(paths_proxy["paths_blocked_est"]),

        "memory_overhead_bytes": int(memory_overhead),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_channel_edges": int(num_channel_edges),
        "baseline_leakage_empty_mask": float(leakage_base),
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AIRPORT DATASET")
    print("="*60)
    result = exponential_deletion_main("airport", 500, "home_link", rho=1, auto_adjust_rho=True)
    print(f"\nFinal result summary:")
    print(f"  Leakage: {result['leakage']:.6f}")
    print(f"  Mask size: {result['mask_size']}")
    print(f"  Paths blocked: {result['paths_blocked']}/{result['num_paths']}")

    print("\n" + "="*60)
    print("ADULT DATASET")
    print("="*60)
    result = exponential_deletion_main("adult", 500, "education", rho=1, auto_adjust_rho=True)
    print(f"\nFinal result summary:")
    print(f"  Leakage: {result['leakage']:.6f}")
    print(f"  Mask size: {result['mask_size']}")
    print(f"  Paths blocked: {result['paths_blocked']}/{result['num_paths']}")

    print("\n" + "=" * 60)
    print("FLIGHT DATASET")
    print("=" * 60)
    result = exponential_deletion_main("flight", 500, "OriginCityMarketId", rho = 1, auto_adjust_rho = True)
    print(f"\nFinal result summary:")
    print(f"  Leakage: {result['leakage']:.6f}")
    print(f"  Mask size: {result['mask_size']}")
    print(f"  Paths blocked: {result['paths_blocked']}/{result['num_paths']}")

    print("\n" + "=" * 60)
    print("FLIGHT DATASET")
    print("=" * 60)
    result = exponential_deletion_main("ncvoter", 500, "c90", rho = 1,
                                       auto_adjust_rho = True)
    print(f"\nFinal result summary:")
    print(f"  Leakage: {result['leakage']:.6f}")
    print(f"  Mask size: {result['mask_size']}")
    print(f"  Paths blocked: {result['paths_blocked']}/{result['num_paths']}")

    print("\n" + "=" * 60)
    print("FLIGHT DATASET")
    print("=" * 60)
    result = exponential_deletion_main("hospital", 500, "EmergencyService", rho = 1,
                                       auto_adjust_rho = True)
    print(f"\nFinal result summary:")
    print(f"  Leakage: {result['leakage']:.6f}")
    print(f"  Mask size: {result['mask_size']}")
    print(f"  Paths blocked: {result['paths_blocked']}/{result['num_paths']}")