import unittest
from unittest.mock import patch

# IMPORTANT: adjust this import if your module filename is not baseline3.py
import baseline_deletion_3


class TestExampleFigureLeakage(unittest.TestCase):
    """
    Figure-based golden test.

    Graph:
      e_a : {Gen, Age}          w = 0.90
      e_b : {Gen, BMI}          w = 0.92
      e_c : {Age, Res}          w = 0.88
      e_1 : {Res, Diag}         w = 0.95
      e_2 : {Age, BMI, Diag}    w = 0.85

    Mask M4 = {Res, Age}
    Target = Diag

    Expected:
      p1 chain weight = 0.90*0.88*0.95 = 0.752
      p2 chain weight = 0.90*0.85      = 0.765
      Leakage ℒ (tight) = 0.878
    """

    def setUp(self):
        self.target = "Diag"

        # RDRs / hyperedges (schema-level)
        self.rdrs = [
            ("Gen", "Age"),          # e_a
            ("Gen", "BMI"),          # e_b
            ("Age", "Res"),          # e_c
            ("Res", "Diag"),         # e_1
            ("Age", "BMI", "Diag"),  # e_2
        ]
        self.weights = [0.90, 0.92, 0.88, 0.95, 0.85]

        self.H = baseline_deletion_3.construct_hypergraph_actual(self.target, self.rdrs, self.weights)

        # Mask M4 from the figure
        self.mask = {"Res", "Age"}

    def _strip_target_suffix(self, path):
        """BFS returns paths that include destination as the last vertex. Strip trailing target."""
        if path and path[-1] == self.target:
            return path[:-1]
        return path

    # ------------------------------------------------------------
    # test_inference_chains
    # ------------------------------------------------------------
    def test_inference_chains(self):
        chains = baseline_deletion_3.get_inference_chains_bfs(
            self.H,
            self.target,
            self.mask | {self.target},
        )

        target_paths = chains.get(self.target, [])
        stripped = {tuple(self._strip_target_suffix(p)) for p in target_paths}

        # Expected chains in "compute_chain_weight format" (NO target included)
        expected = {
            ("Gen", "Age", "Res"),  # corresponds to Gen->Age->Res->Diag
            ("Gen", "Age", "BMI"),  # corresponds to Gen->Age->BMI->Diag
        }

        self.assertTrue(
            expected.issubset(stripped),
            f"Missing expected chains (after stripping target).\nExpected ⊆ {stripped}",
        )

    # ------------------------------------------------------------
    # test_chain_weights
    # ------------------------------------------------------------
    def test_chain_weights(self):
        # IMPORTANT: compute_chain_weight expects chains WITHOUT the target.
        # It auto-attaches target_cell to one end that shares a hyperedge with target.
        p1 = ["Gen", "Age", "Res"]  # attaches via (Res,Diag)
        p2 = ["Gen", "Age", "BMI"]  # attaches via (Age,BMI,Diag)

        w1 = baseline_deletion_3.compute_chain_weight(
            p1, hypergraph=self.H, masked_cells=self.mask, target_cell=self.target
        )
        w2 = baseline_deletion_3.compute_chain_weight(
            p2, hypergraph=self.H, masked_cells=self.mask, target_cell=self.target
        )

        self.assertAlmostEqual(w1, 0.752, places=3)
        self.assertAlmostEqual(w2, 0.765, places=3)

    # ------------------------------------------------------------
    # test_leakage
    # ------------------------------------------------------------
    def test_leakage(self):
        """
        Your compute_leakage_delexp currently calls get_inference_chains_bfs(...)
        and then passes the returned vertex-paths directly into compute_chain_weight(...).

        But BFS paths for target end in target (e.g., [..., Diag]) while compute_chain_weight
        expects chains that do NOT include target. So leakage becomes 0.

        For this golden test, we patch get_inference_chains_bfs to return
        the same paths but with the trailing target removed for chains[target].
        """

        original_bfs = baseline_deletion_3.get_inference_chains_bfs

        def bfs_without_target_suffix(hg, target_cell, masked_cells, **kwargs):
            out = original_bfs(hg, target_cell, masked_cells, **kwargs)
            if target_cell in out:
                out[target_cell] = [
                    p[:-1] if (p and p[-1] == target_cell) else p
                    for p in out[target_cell]
                ]
            return out

        with patch.object(baseline_deletion_3, "get_inference_chains_bfs", side_effect=bfs_without_target_suffix):
            L = baseline_deletion_3.compute_leakage_delexp(
                mask=self.mask,
                target_cell=self.target,
                hypergraph=self.H,
                rho=1.0,  # disable rho cutoff for this example
            )

        self.assertAlmostEqual(L, 0.878, places=3)


if __name__ == "__main__":
    unittest.main()
