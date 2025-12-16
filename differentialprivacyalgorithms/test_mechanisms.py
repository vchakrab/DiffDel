import unittest
from .exponential_mechanism import exponential_deletion, load_instantiated_hyperedges_for_dataset, construct_hypergraph
from .gumbel_mechanism import greedy_gumbel_max_deletion

class TestDeletionMechanisms(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.db_placeholder = {} # The database state isn't used in our simplified funcs
        self.airport_constraints = load_instantiated_hyperedges_for_dataset("airport")
        self.target_cell = "t1.type"
        
        # Parameters from paper (can be adjusted)
        self.alpha = 10
        self.beta = 1
        self.epsilon = 0.5
        self.tau = 0.1
        self.K = 2 # Iterations for Gumbel

        # Pre-calculate expected inference zone for validation
        H_max_V, _ = construct_hypergraph(self.airport_constraints, self.target_cell)
        self.expected_inference_zone = H_max_V - {self.target_cell}

    def test_exponential_mechanism_runs(self):
        """Test that the exponential mechanism returns a valid mask."""
        mask = exponential_deletion(
            self.db_placeholder,
            self.target_cell,
            self.airport_constraints, # This acts as Sigma for the algorithms
            self.alpha,
            self.beta,
            self.epsilon,
            self.tau
        )
        self.assertIsInstance(mask, frozenset)
        self.assertTrue(mask.issubset(self.expected_inference_zone), 
                        f"Mask {mask} is not a subset of expected inference zone {self.expected_inference_zone}")

    def test_gumbel_mechanism_runs(self):
        """Test that the Gumbel-max mechanism returns a valid mask."""
        mask = greedy_gumbel_max_deletion(
            self.db_placeholder,
            self.target_cell,
            self.airport_constraints,
            self.alpha,
            self.beta,
            self.epsilon,
            self.K,
            self.tau
        )
        self.assertIsInstance(mask, frozenset)
        self.assertTrue(mask.issubset(self.expected_inference_zone), 
                        f"Mask {mask} is not a subset of expected inference zone {self.expected_inference_zone}")
        self.assertLessEqual(len(mask), self.K, 
                             f"Mask has size {len(mask)}, but K is {self.K}")

    def test_mechanisms_handle_no_constraints(self):
        """Test that both mechanisms handle cases with no constraints."""
        empty_constraints = []
        
        exp_mask = exponential_deletion(self.db_placeholder, self.target_cell, empty_constraints, self.alpha, self.beta, self.epsilon)
        self.assertEqual(exp_mask, frozenset())

        gum_mask = greedy_gumbel_max_deletion(self.db_placeholder, self.target_cell, empty_constraints, self.alpha, self.beta, self.epsilon, self.K)
        self.assertEqual(gum_mask, frozenset())

if __name__ == "__main__":
    unittest.main()