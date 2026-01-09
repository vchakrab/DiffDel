import importlib
import math
from collections import Counter
from typing import Tuple, List, Set, Iterable, Optional, Dict, Any

def get_dataset_weights(dataset: str) -> Any:
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


def map_dc_to_weight(init_manager, dc, weights_obj) -> float:
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


        counts = Counter(attr_sets)

        num_identical_dcs = sum(c for c in counts.values() if c > 1)
        num_unique_dcs = sum(c for c in counts.values() if c == 1)
    except ValueError:
        return 1.0
    try:
        return float(weights_obj[idx])
    except Exception:
        raise RuntimeError("WEIGHTS object is not indexable by DC index; check weights module format.")


def dc_to_rdrs_and_weights(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Convert denial constraints into RDRs (tuples of attribute names) + aligned weights.
    Schema-level: each DC becomes one hyperedge over the attrs it mentions.
    """
    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    weights_obj = get_dataset_weights(init_manager.dataset)

    for dc in getattr(init_manager, "denial_constraints", []) or []:
        attrs: Set[str] = set()
        w = map_dc_to_weight(init_manager, dc, weights_obj)

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

def compute_utility(*, leakage: float, mask_size: int, lam: float, zone_size: int) -> float:
    """
    u(M) = -λ·L(M) - (1-λ)·|M|/(|I(c*)|-1)
    """
    denom = max(1, int(zone_size) - 1)
    norm = float(mask_size) / float(denom)
    return float(-(lam * float(leakage)) - ((1.0 - lam) * norm))

def compute_utility_log(*, leakage: float, mask_size: int, lam: float, zone_size: int):
    """
    U(M) = λ · (-log(L(M) + 0.01) / log(100)) - (1-λ) · |M| / |Z_max|
    """
    leakage_term = -math.log(leakage + 0.01) / math.log(100)
    size_term = float(mask_size) / float(max(1, zone_size))
    return float((lam * leakage_term) - ((1.0 - lam) * size_term))

def compute_utility_logarithmic(*, leakage: float, mask_size: int, lam: float, zone_size: int, t: str):
    if t == "highlog":
        leakage_term = math.log(1.0 - leakage + 0.01) / math.log(100)
        size_term = float(mask_size) / float(max(1, zone_size))
        return float((lam * leakage_term) - ((1.0 - lam) * size_term))
    elif t == "efflog":
        log_term = math.log(1.0 - leakage + 0.01) / math.log(100)
        ratio_term = (1.0 - leakage) / float(mask_size + 1)

        return float((lam * log_term) + ((1.0 - lam) * ratio_term))
def compute_utility_hinge(* ,leakage, mask_size, zone_size, lam, L0 = 0.25):
    """ Cap-aware hinge utility. u(M) = - lam * max(0, L(M,c*,D) - L0) - (1-lam) * (|M|/I_max) where: - L0 is the permissible leakage cap (policy knob) - lam in (0,1) weights policy compliance vs deletion cost - |M|/I_max is normalized mask size in [0,1] Returns a finite, bounded utility in [-1, 0] (assuming L in [0,1]). """
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0,1).")
    if not (0.0 <= L0 <= 1.0):
        raise ValueError("L0 must be in [0,1].")
    violation = max(0.0, leakage - L0)
    size_term = mask_size/zone_size
    return -lam * violation - (1.0 - lam) * size_term

# ============================================================
# Leakage computation (Algorithm 2) — SINGLE SOURCE OF TRUTH
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

def iter_chains_with_masked(mask: Set[str], target: str, edges: Dict[str, Tuple[Set[str], float]]):
    """
    Like iter_chains(), but yields (chain_edge_ids, maskedCells_inferred_along_chain).

    maskedCells are the masked *intermediate* cells inferred along the chain
    (excluding the target). This matches the Algorithm-5 definition.
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
            # no intermediate inferred masked cells
            yield [eid], set()

    # BFS queue of (chain_edge_ids, known_set, inferred_masked_cells)
    Q: deque[Tuple[List[str], Set[str], Set[str]]] = deque()

    # Seed with edges that can infer some masked x directly from O
    for eid, (verts, _w) in edges.items():
        for x in (verts - O - {target}):
            if active(verts, x, O):
                inferred = set()
                if x in mask:
                    inferred.add(x)
                Q.append(([eid], set(O) | {x}, inferred))

    # Visited-state pruning (knowledge set only)
    visited_K: Set[frozenset] = set()

    while Q:
        p, K, inferred_masked = Q.popleft()

        K_key = frozenset(K)
        if K_key in visited_K:
            continue
        visited_K.add(K_key)

        used_edges = set(p)

        # candidate edges must share a known cell (witness): e ∩ K != ∅
        cand_eids: Set[str] = set()
        for v in K:
            cand_eids |= incident.get(v, set())

        for eid in cand_eids:
            if eid in used_edges:
                continue

            verts, _w = edges[eid]

            missing = verts - K
            if len(missing) != 1:
                continue

            (y,) = tuple(missing)

            if not active(verts, y, K):
                continue

            # Maximality: do not extend through edges already active under O
            if active(verts, y, O):
                continue

            new_p = p + [eid]

            if y == target:
                yield new_p, set(inferred_masked)
            else:
                new_K = set(K) | {y}
                new_inferred = set(inferred_masked)
                if y in mask:
                    new_inferred.add(y)
                Q.append((new_p, new_K, new_inferred))


def greedy_mask_disjoint(
    chains_with_info: List[Tuple[List[str], float, Set[str]]]
) -> List[Tuple[List[str], float, Set[str]]]:
    """
    Algorithm 5: GreedyMaskDisjoint

    Input: list of (chain_edge_ids, chain_weight, maskedCells)
    Output: a mask-disjoint subset maximizing weight greedily.
    """
    # sort by weight desc, then fewer masked cells first (nice tie-break)
    chains_sorted = sorted(
        chains_with_info,
        key=lambda t: (-float(t[1]), len(t[2]))
    )

    D: List[Tuple[List[str], float, Set[str]]] = []
    used: Set[str] = set()

    for ch, cw, mcells in chains_sorted:
        if mcells.isdisjoint(used):
            D.append((ch, float(cw), set(mcells)))
            used |= set(mcells)

    return D


def leakage(
    mask: Set[str],
    target_cell: str,
    hypergraph: "Hypergraph",
    *,
    tau: Optional[float] = None,
    return_counts: bool = False,
    leakage_method: str = "greedy_disjoint",  # "noisy_or" | "greedy_disjoint" | "nor_with_ie"
):
    """
    Leakage estimation options:

    1) leakage_method="noisy_or"
       L = 1 - Π_p (1 - w(p)) over ALL chains

    2) leakage_method="greedy_disjoint"
       - compute all chains + their maskedCells (inferred masked intermediates)
       - pick a mask-disjoint subset D using Algorithm 5
       - L = 1 - Π_{p in D} (1 - w(p))

    3) leakage_method="nor_with_ie"
       - compute noisy-OR over all chains
       - ALSO compute a 2nd-order inclusion–exclusion (Bonferroni) estimate:
            L_ie2 = Σ w(p) - Σ_{p<q} w(p ∪ q)
         where w(p ∪ q) is product of edge weights over the UNION of edges in the two chains
       - return min(noisy_or, L_ie2) as the "tighter" estimate (smaller leakage)
    """

    edge_dict = hypergraph_to_edge_dict(hypergraph, tau=tau)
    if not edge_dict:
        return (0.0, 0, 0, 0) if return_counts else 0.0

    # Precompute edge weights + edge active flags for THIS mask
    w = {eid: float(edge_dict[eid][1]) for eid in edge_dict}
    edge_active = {
        eid: is_edge_active_by_mask_rule(edge_dict[eid][0], mask, target_cell)
        for eid in edge_dict
    }

    num_chains = 0
    active_chains = 0
    blocked_chains = 0

    # For rho-safe (kept for compatibility with your current code structure)
    max_chain_w = 0.0

    if leakage_method in ("noisy_or", "nor_with_ie"):
        prod_not = 1.0

        # Only needed for IE
        chain_weights: List[float] = []
        chain_edge_sets: List[Set[str]] = []

        for ch in iter_chains(mask, target_cell, edge_dict):
            num_chains += 1

            ok = True
            for eid in ch:
                if not edge_active.get(eid, True):
                    ok = False
                    blocked_chains += 1
                    break
            if ok:
                active_chains += 1
            else:
                blocked_chains += 1

            cw = 1.0
            for eid in ch:
                cw *= w[eid]

            if cw > max_chain_w:
                max_chain_w = cw

            prod_not *= (1.0 - cw)

            if leakage_method == "nor_with_ie":
                chain_weights.append(float(cw))
                chain_edge_sets.append(set(ch))

        L_nor = 0.0 if num_chains == 0 else (1.0 - prod_not)

        if leakage_method == "noisy_or":
            L = L_nor
        else:
            # 2nd-order inclusion–exclusion: sum singles - sum pairwise intersections
            # Approximate P(p AND q) as product of edge weights over union(p,q).
            sum1 = float(sum(chain_weights))

            memo_union_prob: Dict[frozenset, float] = {}
            sum2 = 0.0

            m = len(chain_weights)
            for i in range(m):
                Ei = chain_edge_sets[i]
                for j in range(i + 1, m):
                    union_edges = Ei | chain_edge_sets[j]
                    key = frozenset(union_edges)
                    pu = memo_union_prob.get(key)
                    if pu is None:
                        pu = 1.0
                        for eid in union_edges:
                            pu *= w[eid]
                        memo_union_prob[key] = float(pu)
                    sum2 += float(pu)

            L_ie2 = sum1 - sum2
            # clamp to [0,1]
            if L_ie2 < 0.0:
                L_ie2 = 0.0
            elif L_ie2 > 1.0:
                L_ie2 = 1.0

            # "tighter" leakage bound => smaller estimate
            L = min(float(L_nor), float(L_ie2))

    elif leakage_method == "greedy_disjoint":
        # Collect chains with (chain, weight, maskedCells)
        chains_info: List[Tuple[List[str], float, Set[str]]] = []

        for ch, mcells in iter_chains_with_masked(mask, target_cell, edge_dict):
            num_chains += 1

            ok = True
            for eid in ch:
                if not edge_active.get(eid, True):
                    ok = False
                    break
            if ok:
                active_chains += 1
            else:
                blocked_chains += 1

            cw = 1.0
            for eid in ch:
                cw *= w[eid]

            if cw > max_chain_w:
                max_chain_w = cw

            chains_info.append((ch, float(cw), set(mcells)))

        if num_chains == 0:
            L = 0.0
        else:
            chosen = greedy_mask_disjoint(chains_info)
            prod_not = 1.0
            for _ch, cw, _mc in chosen:
                prod_not *= (1.0 - float(cw))
            L = 1.0 - prod_not

    else:
        raise ValueError(
            f"Unknown leakage_method={leakage_method!r}. "
            "Use 'noisy_or', 'greedy_disjoint', or 'nor_with_ie'."
        )

    if return_counts:
        return float(L), int(num_chains), int(active_chains), int(blocked_chains)
    return float(L)

