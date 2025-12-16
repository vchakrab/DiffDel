"""
Implementation of the Greedy Gumbel-Max Deletion Mechanism (Algorithm 4)
from the paper "Differential Inference Deletion (DID)".
"""
import math
import random

# Import helper functions from the the exact exponential mechanism implementation
from .exponential_mechanism import (
    construct_hypergraph, # Algorithm 1
    extract_inference_paths, # Algorithm 2
    calculate_leakage,
    load_instantiated_hyperedges_for_dataset as load_constraints_for_dataset # Renamed for clarity
)

def sample_gumbel(mu=0, beta=1):
    """Sample from Gumbel(mu, beta)"""
    u = random.uniform(0, 1)
    # Ensure u is not 0 or 1 to avoid log(0) issues
    if u == 0: u = 1e-10
    if u == 1: u = 1 - 1e-10
    return mu - beta * math.log(-math.log(u))

def greedy_gumbel_max_deletion(database, target_cell, raw_constraints_sigma, alpha, beta, epsilon, K, tau=0.0):
    """
    Implementation of Algorithm 4: Greedy Gumbel-Max Deletion.

    Args:
        database: Placeholder for database D (not used).
        target_cell: The cell to be deleted.
        raw_constraints_sigma: A list of (frozenset of cell_attributes, weight) tuples
                                 representing the dependency constraints (Sigma).
        alpha, beta: Utility function parameters.
        epsilon: The privacy budget.
        K: Number of iterations (cells to select).
        tau: Path strength filter threshold.

    Returns:
        A deletion mask (a frozenset of cell names).
    """
    # 1. Construct H_max from Sigma (simplified - we assume raw_constraints_sigma are H_max edges)
    H_max_V, H_max_E = construct_hypergraph(raw_constraints_sigma, target_cell)
    inference_zone = H_max_V - {target_cell}
    
    # 2. Construct H_D and extract paths Hall <-- FINDPATHS(c*, H_D)
    #    (Simplified: Using H_max as H_D for this implementation)
    H_D_V, H_D_E = H_max_V, H_max_E
    all_paths = extract_inference_paths(target_cell, (H_D_V, H_D_E))
    paths = [p for p in all_paths if math.prod(w for _, w in p) >= tau]
    
    mask = set()
    if not inference_zone: return frozenset()
        
    # Allocate privacy budget per iteration
    epsilon_prime = epsilon / K
    # Sensitivity for marginal utility (alpha * (L_curr - L_new) - beta)
    # is alpha, because max_diff(L) is 1 (Lemma 5.3). So total sensitivity is alpha.
    # The paper uses 2*alpha in the denominator of exp for Alg 3, and 2*alpha/epsilon' for Gumbel.beta
    # So Gumbel noise scale parameter: beta_gumbel = 2 * sensitivity / epsilon_prime
    gumbel_beta_scale = 2 * alpha / epsilon_prime

    # 3. Iteratively build the mask (Algorithm 4 loop)
    for _ in range(K): # Line 5: for k = 1 to K do
        candidate_cells = list(inference_zone - mask)
        if not candidate_cells: break

        scores = {}
        # Line 7: L_curr <-- 1 - product(1 - w(pi)) for pi in Pi(M)
        leakage_current_mask = calculate_leakage(paths, frozenset(mask))
        
        for cell_candidate in candidate_cells: # Line 6: for each c in I(c*) \ M do
            # Line 8: L_new <-- 1 - product(1 - w(pi)) for pi in Pi(M U {c})
            leakage_new_mask = calculate_leakage(paths, frozenset(mask | {cell_candidate}))
            
            # Line 9: Delta_u(c) <-- alpha * (L_curr - L_new) - beta
            marginal_utility_gain = alpha * (leakage_current_mask - leakage_new_mask) - beta
            
            # Line 10: g_c <-- Gumbel(0, 2*alpha/epsilon') (where beta_gumbel = 2*alpha/epsilon')
            gumbel_noise_for_cell = sample_gumbel(mu=0, beta=gumbel_beta_scale)
            
            # Line 11: s_c <-- Delta_u(c) + g_c
            scores[cell_candidate] = marginal_utility_gain + gumbel_noise_for_cell
            
        # Add a "stop" option (Lines 12-14 in paper)
        stop_gumbel_noise = sample_gumbel(mu=0, beta=gumbel_beta_scale)
        scores["__STOP__"] = 0 + stop_gumbel_noise # Marginal utility of stopping is 0

        # Line 15: M <-- M U {arg max_c s_c}
        best_candidate = max(scores, key=scores.get)

        if best_candidate == "__STOP__": # Check for termination (Line 13-14)
            break
        
        mask.add(best_candidate)

    # Line 16: return M
    return frozenset(mask)