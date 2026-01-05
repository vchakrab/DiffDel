import math
import unittest
from typing import Dict, List, Set, Tuple
from unittest.mock import patch


# -------------------------
# Minimal Hypergraph stub
# -------------------------
class Hypergraph:
    def __init__(self):
        self.vertices: Set[str] = set()
        self.edges: List[Tuple[Set[str], float]] = []

    def add_edge(self, verts: Set[str], w: float):
        self.vertices |= set(verts)
        self.edges.append((set(verts), float(w)))


# -------------------------
# Import your functions
# -------------------------
# Change this import to match your module name:
from baseline_deletion_3 import get_inference_chains_bfs, compute_leakage_delexp

# For demonstration, you can paste your actual implementations here if needed.
# The tests assume these names exist in your module namespace.


class TestChainsAndLeakage(unittest.TestCase):
    def setUp(self):
        # Hypergraph: abcd (0.5), def (0.3), fgh (0.2)
        self.H = Hypergraph()
        self.H.add_edge(set("abcd"), 0.5)
        self.H.add_edge(set("def"), 0.3)
        self.H.add_edge(set("fgh"), 0.2)

        self.target = "h"

    # -------------------------
    # Test 1: get_inference_chains_bfs (Option 2-ish: no traversal blocking)
    # -------------------------
    def test_get_inference_chains_bfs_returns_candidate_paths_to_masked_endpoints(self):
        """
        We treat masked_cells as "endpoints of interest" (NOT as traversal blockers).
        We expect at least these candidate chains exist structurally:
          - g -> h (direct neighbor via fgh)
          - a -> d -> f (via abcd then def) if 'f' is an endpoint of interest
          - a -> d -> f -> h (via abcd, def, fgh) if 'h' is endpoint of interest

        This test does NOT assert the full list (BFS will enumerate many).
        """
        # import inside test so you only need to edit the import once above
        from baseline_deletion_3 import get_inference_chains_bfs

        chains = get_inference_chains_bfs(self.H, self.target, masked_cells={"f", "h"})

        # Must have keys for requested endpoints
        self.assertIn("f", chains)
        self.assertIn("h", chains)

        # Helper: check if a path appears in returned list
        def has_path(dst: str, path: List[str]) -> bool:
            return any(p == path for p in chains.get(dst, []))

        # Direct candidate chain to h
        self.assertTrue(has_path("h", ["g", "h"]))

        # Candidate chain to f (a -> d -> f)
        # (a,d) share edge abcd, (d,f) share edge def
        self.assertTrue(has_path("f", ["a", "d", "f"]))

        # Candidate chain to h going across all three edges
        self.assertTrue(has_path("h", ["a", "d", "f", "h"]))

    # -------------------------
    # Test 2: compute_leakage_delexp matches paper math (NOR + IE, take min)
    # -------------------------
    def test_compute_leakage_delexp_ie_and_nor_two_chains_known_overlap(self):
        """
        We force exactly two target chains:
          p1 = [g, h] uses fgh => w(p1)=0.2
          p2 = [a, d, f, h] uses abcd, def, fgh => w(p2)=0.5*0.3*0.2=0.03

        Shared hyperedge set = {fgh} with weight 0.2, so:
          L_NOR = 1 - (1-0.2)(1-0.03) = 1 - 0.8*0.97 = 0.224
          L_IE  = (0.2 + 0.03) - (0.2*0.03)/0.2 = 0.23 - 0.03 = 0.2
        Paper says L = min(1, min(L_IE, L_NOR)) => min(0.2, 0.224) = 0.2
        """
        from baseline_deletion_3 import compute_leakage_delexp

        forced_chains = {
            "h": [
                ["g", "h"],
                ["a", "d", "f", "h"],
            ]
        }

        # Stub compute_chain_weight to match your expected chain-weight semantics.
        # IMPORTANT: match your real function signature if it's keyword-only.
        def stub_compute_chain_weight(chain: List[str], hypergraph: Hypergraph, masked_cells: Set[str], target_cell: str):
            # Multiply best-edge weights along consecutive pairs (unique edges by verts-set)
            def best_edge_between(u: str, v: str) -> float:
                best = 0.0
                for ev, w in hypergraph.edges:
                    if u in ev and v in ev:
                        best = max(best, float(w))
                return best

            used_edges: Set[frozenset] = set()
            prod = 1.0
            for i in range(len(chain) - 1):
                u, v = chain[i], chain[i + 1]
                # identify the best edge's vertex-set to avoid double-counting
                best_w = -1.0
                best_ev = None
                for ev, w in hypergraph.edges:
                    if u in ev and v in ev:
                        if float(w) > best_w:
                            best_w = float(w)
                            best_ev = frozenset(ev)
                if best_ev is None:
                    return 0.0
                if best_ev not in used_edges:
                    used_edges.add(best_ev)
                    prod *= float(best_w)
            return prod

        # Patch BOTH get_inference_chains_bfs and compute_chain_weight inside your_module
        with patch("baseline_deletion_3.get_inference_chains_bfs", return_value = forced_chains), \
                patch("baseline_deletion_3.compute_chain_weight",
                      side_effect = stub_compute_chain_weight):

            L = compute_leakage_delexp(mask=set(), target_cell="h", hypergraph=self.H, rho=0.99)
            self.assertAlmostEqual(L, 0.2, places=12)

    # -------------------------
    # Test 3: rho guard still triggers (keeps your constants/behavior)
    # -------------------------
    def test_compute_leakage_delexp_rho_triggers_to_one(self):
        """
        If rho is below max chain weight, code returns 1.0 (your existing behavior).
        Here max chain weight is 0.2, so rho=0.1 triggers.
        """
        from baseline_deletion_3 import  compute_leakage_delexp

        forced_chains = {"h": [["g", "h"]]}

        def stub_compute_chain_weight(chain, hypergraph, masked_cells, target_cell):
            return 0.2  # force weight above rho in this test

        with patch("baseline_deletion_3.get_inference_chains_bfs", return_value=forced_chains), \
             patch("baseline_deletion_3.compute_chain_weight", side_effect=stub_compute_chain_weight):

            L = compute_leakage_delexp(mask=set(), target_cell="h", hypergraph=self.H, rho=0.1)
            self.assertEqual(L, 1.0)


if __name__ == "__main__":
    unittest.main()
