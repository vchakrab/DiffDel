"""
Implementation of the Exponential Deletion Mechanism (Algorithm 3)
and its helper algorithms (Algorithm 1 and 2) from the paper
"Differential Inference Deletion (DID)".
"""
import math
from itertools import chain, combinations
from collections import deque
import random

# --- Helper function for loading constraints (simplified for demo) ---
# In a full implementation, this would involve database queries
# to instantiate constraints based on the current database state.
def load_instantiated_hyperedges_for_dataset(dataset_name):
    """
    Loads pre-defined *instantiated hyperedges* for a given dataset.
    Returns a list of (frozenset of cell_attributes, weight) tuples.
    Each frozenset represents a hyperedge.
    """
    # These are simplified examples of INSTANTIATED hyperedges.
    # For the test case, we assume cells like "t1.type", "t1.name" etc.
    constraints = {
        "airport": [
            (frozenset({"t1.type", "t1.name"}), 0.9), # Example: type determines name
            (frozenset({"t1.city", "t1.state"}), 1.0), # Example: city determines state
            (frozenset({"t1.type", "t1.country"}), 0.7), # Another example
        ],
        "tax": [
            (frozenset({"t1.marital_status", "t1.taxable_income"}), 0.85),
            (frozenset({"t1.occupation", "t1.taxable_income"}), 0.95),
        ],
        "hospital": [
            (frozenset({"t1.ProviderNumber", "t1.HospitalName"}), 1.0),
            (frozenset({"t1.ProviderNumber", "t1.ProviderZip"}), 0.92),
        ],
        "ncvoter": [
            (frozenset({"t1.voter_reg_num", "t1.full_name_mail"}), 1.0),
            (frozenset({"t1.voter_reg_num", "t1.res_city_desc"}), 0.88),
        ]
    }
    return constraints.get(dataset_name, [])

# --- ALGORITHM 1: Construct Hypergraph (from paper) ---
def construct_hypergraph(initial_instantiated_edges, target_cell):
    """
    Implements Algorithm 1 from the paper: Construct Hypergraph.
    
    Args:
        initial_instantiated_edges: A list of (frozenset of cell_attributes, weight) tuples
                                    representing instantiated dependencies (E_sigma_D).
        target_cell: The string name of the target cell (e.g., "t1.type").

    Returns:
        A tuple (V, E) where V is a set of vertices (cells) and E is a set of
        (hyperedge: frozenset, weight: float) tuples.
    """
    # 1: V ← {c*}, E ← ∅, frontier ← {c*}, visited ← ∅
    V = {target_cell}
    E = set() # Use a set to avoid duplicate edges
    frontier = deque([target_cell]) # Use deque for efficient BFS
    visited = set()

    # 2: while frontier ≠ ∅ do
    while frontier:
        # 3: next ← ∅
        next_frontier = set() # This will be the new frontier for the next iteration

        # 4: for each c ∈ frontier where c ∉ visited do
        current_cell = frontier.popleft()
        if current_cell in visited: # This check handles duplicates if any were added to frontier earlier
            continue

        # 5: visited ← visited ∪ {c}
        visited.add(current_cell)

        # 6: for each σ : (T, J) ⇒ h with weight w_sigma in Sigma do
        #    (Simulated using initial_instantiated_edges as E_sigma_D)
        #    Here, we iterate through the pre-instantiated edges to find ones involving 'current_cell'.
        for edge_tuple, weight in initial_instantiated_edges:
            # 7: if c is in σ then c matches some (A, ti) ∈ T ∪ {h}
            if current_cell in edge_tuple: 
                # 8: edges ← INSTANTIATE(σ, c, D, check_joins)
                #    (Simulated: initial_instantiated_edges are already instantiated, so 'edge_tuple' is 'e')
                # 9: for each hyperedge e in edges do
                e = edge_tuple # The hyperedge itself
                w_e = weight # Its weight

                # 10: E ← E ∪ {(e, w_e)}, V ← V ∪ e
                E.add((e, w_e))
                V.update(e)

                # 11: next ← next ∪ (e \ visited)
                # Add newly discovered cells from this edge to the next_frontier if not yet visited
                newly_discovered_cells = e - visited
                next_frontier.update(newly_discovered_cells)
        
        # After processing all edges for current_cell, add new discoveries to the actual frontier
        # This handles cases where multiple edges might lead to the same new cells in one pass.
        for cell_to_add in next_frontier:
            if cell_to_add not in visited and cell_to_add not in frontier:
                frontier.append(cell_to_add)

    # 12: frontier ← next (This is implicitly handled by deque's popleft and append)
    # 13: return (V, E)
    return (V, E)

# --- ALGORITHM 2: Extract Inference Paths (from paper) ---
def extract_inference_paths(target_cell, hypergraph):
    """
    Implements Algorithm 2 from the paper: Extract Inference Paths (FINDPATHS).

    Args:
        target_cell: The string name of the target cell.
        hypergraph: A tuple (V, E) where V is a set of vertices (cells) and E is a set of
                    (hyperedge: frozenset, weight: float) tuples.

    Returns:
        A list of paths. Each path is a list of (hyperedge: frozenset, weight: float) tuples.
    """
    V, E = hypergraph

    # 1: known ← V \ {c*}, Π ← ∅
    known_cells_initial = V - {target_cell} # Cells known by adversary initially
    inference_paths = [] # This will be Π

    # 2: FINDPATHSRECURSIVE (c*, known, [], Π, Ε)
    def findpaths_recursive(current_target, current_known, current_path):
        # 5: for each (e, w) ∈ E where target ∈ e do
        for edge_tuple, weight in E:
            if current_target in edge_tuple:
                # 6: unknown ← e \ known
                unknown_cells = edge_tuple - current_known

                # 7: if unknown = {target} then
                if unknown_cells == {current_target}: # Path infers the final target
                    # 8: Π ← Π ∪ {path + [(e, w)]}
                    inference_paths.append(current_path + [(edge_tuple, weight)])
                # 9: else if |unknown| = 1 then
                elif len(unknown_cells) == 1: # Path infers an intermediate cell
                    # 10: C_int ← the single element in unknown
                    intermediate_cell = unknown_cells.pop()
                    # 11: FINDPATHSRECURSIVE(target, known ∪ {C_int}, path + [(e, w)], Π, E)
                    # Recursive call: The intermediate cell is now known for further steps in this path
                    findpaths_recursive(
                        current_target, 
                        current_known | {intermediate_cell}, 
                        current_path + [(edge_tuple, weight)]
                    )

    findpaths_recursive(target_cell, known_cells_initial, [])
    # 3: return Π
    return inference_paths

# --- Helper for calculating leakage (unchanged) ---
def calculate_leakage(paths, mask):
    leakage = 1.0
    for path in paths:
        path_weight = math.prod(weight for _, weight in path)
        is_blocked = any(bool(edge & mask) for edge, _ in path)
        if not is_blocked:
            leakage *= (1 - path_weight)
    return 1 - leakage

# --- ALGORITHM 3: Exponential Deletion Mechanism (from paper) ---
def exponential_deletion(database, target_cell, raw_constraints_sigma, alpha, beta, epsilon, tau=0.0):
    """
    Implementation of Algorithm 3: Exponential Deletion Mechanism.
    
    Args:
        database: Placeholder for database D (not used in this simplified version).
        target_cell: The cell to be deleted (e.g., "t1.type").
        raw_constraints_sigma: A list of (frozenset of cell_attributes, weight) tuples
                                 representing the dependency constraints (Sigma).
        alpha, beta: Utility function parameters (leakage and deletion cost).
        epsilon: The privacy budget.
        tau: Path strength filter threshold.

    Returns:
        A sampled deletion mask (a frozenset of cell names).
    """
    # 1. Construct H_max from Sigma (simplified - we assume raw_constraints_sigma are H_max edges)
    #    (In a full implementation, H_max might be constructed differently from Sigma)
    H_max_V, H_max_E = construct_hypergraph(raw_constraints_sigma, target_cell)

    # 2. Compute inference zone I(c*) <-- REACHABLE(c*, H_max)
    #    (Simplified: Inference zone is all cells in H_max_V except target_cell)
    inference_zone = H_max_V - {target_cell}

    # 3. Construct H_D from Sigma and D (simplified: we use H_max_E as H_D_E)
    #    (In a full implementation, H_D is constructed based on actual database values)
    H_D_V, H_D_E = H_max_V, H_max_E # Using H_max as H_D for simplicity here

    # 4. Extract inference paths Hall <-- FINDPATHS(c*, H_D)
    all_paths = extract_inference_paths(target_cell, (H_D_V, H_D_E))

    # 5. Filter paths: Pi <-- {pi in Hall : w(pi) >= tau}
    paths = [p for p in all_paths if math.prod(w for _, w in p) >= tau]

    # 6. Enumerate candidate masks M <-- 2^I(c*)
    candidate_masks = []
    for i in range(len(inference_zone) + 1):
        for subset in combinations(inference_zone, i):
            candidate_masks.append(frozenset(subset))

    if not candidate_masks:
        return frozenset()

    # 7. for each M in M do
    utilities_and_masks = []
    for mask in candidate_masks:
        # 8. Pi(M) <-- {pi in Pi : M does not block pi}
        # 9. L(M) <-- 1 - product_{pi in Pi(M)} (1 - w(pi))
        leakage = calculate_leakage(paths, mask)

        # 10. u(M) <-- -alpha * L(M) - beta * |M|
        utility = -alpha * leakage - beta * len(mask)
        utilities_and_masks.append((utility, mask))

    # 11. Sample M with Pr[M] proportional to exp(epsilon * u(M) / (2 * alpha))
    # (Using Lemma 5.6 for sensitivity = alpha in the exponent numerator)
    
    # To avoid floating point issues with large exponents, we subtract the max utility.
    max_utility = max(u for u, _ in utilities_and_masks)
    
    scaled_scores = [
        (u - max_utility) * epsilon / (2 * alpha) # Paper uses 2*alpha as sensitivity
        for u, _ in utilities_and_masks
    ]
    
    probabilities = [math.exp(s) for s in scaled_scores]
    sum_probs = sum(probabilities)
    
    if sum_probs == 0: # Handle cases where all probabilities are essentially zero
        return random.choice(candidate_masks)

    probabilities = [p / sum_probs for p in probabilities]

    # Sample one mask based on the calculated probabilities
    rand_val = random.random()
    cumulative_prob = 0
    for (utility, mask), prob in zip(utilities_and_masks, probabilities):
        cumulative_prob += prob
        if rand_val < cumulative_prob:
            return mask
            
    return candidate_masks[-1] # Fallback in case of floating point quirks