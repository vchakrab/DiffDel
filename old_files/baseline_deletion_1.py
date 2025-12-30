import importlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config

# =========================
# USER SETTINGS
# =========================
LAM = 0.5  # set this to the same λ you use for delexp

DELETION_QUERY = """
UPDATE {table_name}
SET `{column_name}` = NULL
WHERE id = {key};
"""


# ============================================================
# Edge weights loading (NO DEFAULTS; error if missing)
# ============================================================

def get_dataset_weights_strict(dataset: str) -> Any:
    """
    Loads edge weights using the same convention as delexp:
      weights.weights_corrected.<dataset>_weights with a WEIGHTS object.

    Raises FileNotFoundError if the module or WEIGHTS is missing.
    """
    import importlib

    ds = str(dataset).lower()
    module_name = f"weights.weights_corrected.{ds}_weights"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            f"Missing edge-weight module '{module_name}'. "
            f"Expected a file like: weights/weights_corrected/{ds}_weights.py "
            f"next to your delexp weights directory."
        ) from e

    if not hasattr(mod, "WEIGHTS"):
        raise FileNotFoundError(
            f"Module '{module_name}' exists but does not define WEIGHTS."
        )

    weights_obj = getattr(mod, "WEIGHTS")
    if weights_obj is None:
        raise FileNotFoundError(
            f"Module '{module_name}' defines WEIGHTS=None; expected actual weights."
        )
    return weights_obj





# ============================================================
# Leakage + Utility (delexp-style)
# ============================================================

def compute_utility_new(*, leakage: float, mask_size: int, lam: float, zone_size: int) -> float:
    """
    u(M) = -λ L(M) - (1-λ) * |M|/(|I(c*)|-1)
    """
    denom = max(1, int(zone_size) - 1)  # avoids division by 0 when zone_size <= 1
    norm = float(mask_size) / float(denom)
    return float(-(lam * float(leakage)) - ((1.0 - lam) * norm))


Cell = str


@dataclass(frozen=True)
class Edge:
    verts: Tuple[int, ...]
    w: float


class InferableLeakageModel:
    """
    Hypergraph leakage model:
      - observed[v]=True => adversary knows v with prob 1
      - masked cells and target are unknown initially
      - update rule:
          p[v] = 1 - Π_{edges containing v}(1 - w_e * Π_{u in e\{v}} p[u])
      - fixed point via queue relaxation
      - leakage L := p[target]
    """
    def __init__(self, hyperedges: Sequence[Iterable[Cell]], weights: Sequence[float], target: Cell):
        hyperedges = list(hyperedges)
        weights = list(weights)
        if len(hyperedges) != len(weights):
            raise ValueError("hyperedges and weights must have the same length")

        self.target = target

        verts_set: Set[Cell] = set()
        for e in hyperedges:
            verts_set |= set(e)
        verts_set.add(target)

        self.verts: List[Cell] = sorted(verts_set)
        self.vid: Dict[Cell, int] = {v: i for i, v in enumerate(self.verts)}
        self.n = len(self.verts)
        self.tid = self.vid[target]

        self.edges: List[Edge] = []
        for e, w in zip(hyperedges, weights):
            vv = tuple(sorted({self.vid[v] for v in set(e) if v in self.vid}))
            if len(vv) >= 2:
                self.edges.append(Edge(vv, float(w)))

        self.neigh: List[Set[int]] = [set() for _ in range(self.n)]
        for ed in self.edges:
            for a in ed.verts:
                for b in ed.verts:
                    if a != b:
                        self.neigh[a].add(b)

    def _recompute_pv(self, v: int, observed: List[bool], p: List[float]) -> float:
        if observed[v]:
            return 1.0

        prod_fail = 1.0
        for ed in self.edges:
            if v not in ed.verts:
                continue

            other = [u for u in ed.verts if u != v]
            if not other:
                continue

            term = 1.0
            for u in other:
                term *= float(p[u])

            infer_prob = float(ed.w) * term
            prod_fail *= (1.0 - infer_prob)

        return float(1.0 - prod_fail)

    def leakage(self, mask: Set[Cell], *, tau: float = 1e-9, max_updates: int = 1_000_000) -> float:
        observed = [True] * self.n
        observed[self.tid] = False  # target is unknown

        for m in mask:
            if m in self.vid:
                observed[self.vid[m]] = False

        p = [1.0 if observed[v] else 0.0 for v in range(self.n)]
        Q = deque(range(self.n))
        in_q = [True] * self.n
        pops = 0

        while Q and pops < max_updates:
            v = Q.popleft()
            in_q[v] = False
            pops += 1

            if observed[v]:
                continue

            new_p = self._recompute_pv(v, observed, p)
            if abs(new_p - p[v]) > tau:
                p[v] = new_p
                for u in self.neigh[v]:
                    if not in_q[u]:
                        Q.append(u)
                        in_q[u] = True

        L = float(p[self.tid])
        return float(max(0.0, min(1.0, L)))
def load_parsed_dcs(dataset: str) -> List[List[Tuple[str, str, str]]]:
    """
    Imports DCandDelset.dc_configs.top{Dataset}DCs_parsed and returns denial_constraints (or []).
    """
    try:
        dataset_module_name = "NCVoter" if dataset.lower() == "ncvoter" else dataset.capitalize()
        #dataset_module_name  = "Onlineretail" if dataset.lower() == "onlineretail" else dataset.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        return getattr(dc_module, "denial_constraints", [])
    except Exception:
        return []

def map_dc_to_weight(init_manager, dc, weights):
    return weights[init_manager.denial_constraints.index(dc)]
def get_dataset_weights(dataset: str) -> Optional[Any]:
    """
    Optional: if you have per-dataset edge weights modules, load them.
    Expected module: weights.weights_corrected.<dataset>_weights with WEIGHTS inside.
    Returns None if missing, which falls back to a constant edge_weight.
    """
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "WEIGHTS", None)
    except ModuleNotFoundError:
        return None
def dc_to_hyperedges(init_manager) -> List[Tuple[str, ...]]:
    """
    Convert init_manager.denial_constraints into hyperedges over attribute names
    (attributes only: e.g., "type", not "t1.type").
    """

    hyperedges: List[Tuple[str, ...]] = []
    hyperedge_weights: List[float] = []

    for dc in getattr(init_manager, "denial_constraints", []):
        attrs: Set[str] = set()
        weight = map_dc_to_weight(init_manager, dc, get_dataset_weights(init_manager.dataset))
        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                token = pred[0]
                if isinstance(token, str) and "." in token:
                    attrs.add(token.split(".")[-1])
        if len(attrs) >= 2:
            hyperedges.append(tuple(sorted(attrs)))
            hyperedge_weights.append(weight)
    return hyperedges, hyperedge_weights


def inference_zone_for_target(hyperedges: Sequence[Tuple[str, ...]], target: str) -> List[str]:
    zone: Set[str] = set()
    for e in hyperedges:
        if target in e:
            for v in e:
                if v != target:
                    zone.add(v)
    return sorted(zone)


def compute_leakage_and_utility_for_mask(
    *,
    init_manager,
    dataset: str,
    target: str,
    mask_cols: Set[str],
    lam: float,
) -> Tuple[float, float, int]:
    """
    Returns (leakage, utility, zone_size) for this mask under the delexp leakage model.
    Strictly requires weights file to exist.
    """
    weights: List[float]
    hyperedges: List[Tuple[str, ...]]
    hyperedges,weights = dc_to_hyperedges(init_manager)

    # Load and normalize edge weights STRICTLY (no defaults)

    model = InferableLeakageModel(hyperedges, weights, target)

    zone = inference_zone_for_target(hyperedges, target)
    zone_size = len(zone)

    L = model.leakage(set(mask_cols))
    U = compute_utility_new(leakage=L, mask_size=len(mask_cols), lam=float(lam), zone_size=zone_size)
    return float(L), float(U), int(zone_size)


# ============================================================
# Your existing memory estimator (unchanged)
# ============================================================

def measure_optimal_memory(init_manager, target, key):
    memory = 0
    cell_to_edges = {}
    all_cells = {f"t1.{target}"}

    for dc in init_manager.denial_constraints:
        head_attr = dc[0][0] if dc else None
        if not head_attr:
            continue

        head_cell = f"t1.{head_attr.split('.')[-1]}"
        all_cells.add(head_cell)

        edge = set()
        for pred in dc:
            attr = pred[0].split('.')[-1]
            cell = f"t1.{attr}"
            edge.add(cell)
            all_cells.add(cell)

        if head_cell not in cell_to_edges:
            cell_to_edges[head_cell] = []
        cell_to_edges[head_cell].append(edge)

    q = [f"t1.{target}"]
    visited = {f"t1.{target}"}

    while q:
        curr_cell_attr = q.pop(0)
        # Per cell: table_index(4) + row_index(4) + insertion_time(4) + state(1) + cost(4) = 17 bytes
        memory += 17

        if curr_cell_attr in cell_to_edges:
            for edge in cell_to_edges[curr_cell_attr]:
                # Per edge: pointers to cells (8 * edge_size) + parent pointer (8) + minCell (4) = 12 + 8 * size
                memory += 12 + len(edge) * 8
                for cell in edge:
                    if cell not in visited:
                        visited.add(cell)
                        q.append(cell)

    return memory


# ============================================================
# Main deletion function (now returns leakage + utility too)
# ============================================================

def delete_all_dependent_cells(target: str, key: int, dataset: str, threshold: float):
    """
    Returns:
      (num_constraints, num_cells_deleted, memory_bytes,
       instantiation_time, model_time, deletion_time,
       leakage, utility)
    """
    # Phase 1: INSTANTIATION
    instantiation_start = time.time()
    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target},
        dataset,
        threshold
    )
    init_manager.initialize()
    instantiation_time = time.time() - instantiation_start

    # Phase 2: MODELING
    model_start = time.time()

    cleaned_content = str(init_manager.constraint_cells).strip('{}')
    items = cleaned_content.split(', ') if cleaned_content else []

    stripped_attributes: List[str] = []
    for item in items:
        key_part = item.split('=>')[0].strip()
        dot_index = key_part.find('.')
        bracket_index = key_part.find('[')
        if dot_index != -1 and bracket_index != -1 and dot_index < bracket_index:
            attribute = key_part[dot_index + 1: bracket_index]
            stripped_attributes.append(attribute)

    constraint_cells_stripped = stripped_attributes

    memory_bytes = measure_optimal_memory(init_manager, target, key)

    # --- NEW: leakage + utility for the chosen mask (cells you delete) ---
    # You delete constraint cells AND the target cell, so include target in the mask.
    mask_set: Set[str] = set(constraint_cells_stripped)
    mask_set.add(target)

    leakage, utility, _zone_size = compute_leakage_and_utility_for_mask(
        init_manager=init_manager,
        dataset=dataset,
        target=target,
        mask_cols=mask_set,
        lam=LAM,
    )

    model_time = time.time() - model_start

    # Phase 4: UPDATE TO NULL
    deletion_start = time.time()
    conn = None
    cursor = None

    try:
        db_details = config.get_database_config(dataset)
        primary_table = dataset + "_copy_data"

        conn = mysql.connector.connect(
            host=db_details['host'],
            user=db_details['user'],
            password=db_details['password'],
            database=db_details['database'],
            ssl_disabled=db_details.get('ssl_disabled', False),
        )

        if not conn.is_connected():
            return (
                len(init_manager.denial_constraints),
                len(mask_set),
                memory_bytes,
                instantiation_time,
                model_time,
                0.0,
                leakage,
                utility,
            )

        cursor = conn.cursor()

        # Delete all constraint cells
        for constraint_cell in constraint_cells_stripped:
            cursor.execute(
                DELETION_QUERY.format(
                    table_name=primary_table,
                    column_name=constraint_cell,
                    key=key
                )
            )

        # Delete the target cell
        cursor.execute(
            DELETION_QUERY.format(
                table_name=primary_table,
                column_name=target,
                key=key
            )
        )

        conn.commit()

    except Error:
        # keep your old behavior (swallow DB errors)
        pass
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    deletion_time = time.time() - deletion_start

    return (
        len(init_manager.denial_constraints),  # num instantiated RDRs / DCs
        len(mask_set),
        mask_set,# mask size (constraint cells + target)
        memory_bytes,
        instantiation_time,
        model_time,
        deletion_time,
        leakage,                              # NEW
        utility,                              # NEW
    )

print(delete_all_dependent_cells("latitude_deg", 500, "airport", 0))
print(delete_all_dependent_cells("ProviderNumber", 500, "hospital", 0))
print(delete_all_dependent_cells("marital_status", 500, "tax", 0))
#print(delete_all_dependent_cells("voter_reg_num", 500, "ncvoter", 0))
print(delete_all_dependent_cells("education", 500, "adult", 0))
print(delete_all_dependent_cells("InvoiceNo", 500, "Onlineretail", 0))