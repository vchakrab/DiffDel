import pandas as pd
import numpy as np
from collections import defaultdict

class PyroAFDExtractor:
    """
    AFD extraction based on Pyro algorithm (Kruse et al. VLDB 2018)
    Uses g3 error measure: fraction of tuples violating the dependency
    
    Reference: "Efficient Discovery of Approximate Dependencies"
    Pyro computes g3 by examining position list indices (PLIs)
    """
    
    def __init__(self, data, error_threshold=0.01):
        """
        Args:
            data: pandas DataFrame
            error_threshold: maximum g3 error for valid AFDs (default: 0.01)
        """
        self.data = data
        self.error_threshold = error_threshold
        self.plis = {}  # Position List Indices cache
        
    def compute_pli(self, columns):
        """
        Compute Position List Index (PLI) for given columns
        
        PLI maps each unique value combination to list of row indices
        Example: 
            Age | BMI
            30  | 25  (row 0)
            30  | 25  (row 1)
            40  | 30  (row 2)
        
        PLI[(30, 25)] = [0, 1]
        PLI[(40, 30)] = [2]
        
        This is the core data structure in Pyro for efficient error computation
        """
        if isinstance(columns, str):
            columns = [columns]
        
        key = tuple(sorted(columns))
        if key in self.plis:
            return self.plis[key]
        
        pli = defaultdict(list)
        for idx, row in self.data.iterrows():
            # Create tuple of values for these columns
            values = tuple(row[col] for col in columns)
            pli[values].append(idx)
        
        self.plis[key] = pli
        return pli
    
    def compute_g3_error(self, lhs_cols, rhs_col):
        """
        Compute g3 error for AFD: lhs_cols -> rhs_col
        
        g3 = (number of violating tuples) / (total tuples)
        
        Algorithm from Pyro paper:
        1. Compute PLI for LHS attributes
        2. For each cluster in LHS PLI:
           - Find most frequent RHS value (mode)
           - Count tuples with different RHS value (violations)
        3. g3 = sum of violations / total tuples
        
        Time complexity: O(n) where n = number of tuples
        """
        # Get PLI for LHS
        lhs_pli = self.compute_pli(lhs_cols)
        
        violations = 0
        total_tuples = len(self.data)
        
        # For each cluster of tuples with same LHS values
        for lhs_value, row_indices in lhs_pli.items():
            if len(row_indices) <= 1:
                continue  # No violations possible with single tuple
            
            # Get RHS values for this cluster
            rhs_values = [self.data.loc[idx, rhs_col] for idx in row_indices]
            
            # Count frequency of each RHS value
            from collections import Counter
            rhs_counts = Counter(rhs_values)
            
            # Most frequent value (mode)
            mode_count = rhs_counts.most_common(1)[0][1]
            
            # All tuples NOT having the mode value are violations
            cluster_violations = len(row_indices) - mode_count
            violations += cluster_violations
        
        g3_error = violations / total_tuples
        return g3_error
    
    def extract_afd_weight(self, lhs_cols, rhs_col):
        """
        Extract AFD weight using Pyro's g3 measure
        
        Weight = 1 - g3_error
        
        Interpretation:
        - w = 1.0: Perfect FD (0% violations)
        - w = 0.85: 15% of tuples violate the dependency
        - w = 0.0: Completely random (100% violations)
        
        Returns:
            dict with weight, g3_error, and metadata
        """
        g3_error = self.compute_g3_error(lhs_cols, rhs_col)
        weight = 1.0 - g3_error
        
        # Check if this is a valid AFD under threshold
        is_valid_afd = g3_error <= self.error_threshold
        
        return {
            'lhs': lhs_cols,
            'rhs': rhs_col,
            'weight': weight,
            'g3_error': g3_error,
            'is_valid_afd': is_valid_afd,
            'method': 'pyro_g3'
        }
    
    def discover_all_afds(self, max_lhs_size=3):
        """
        Discover ALL minimal AFDs (simplified version of Pyro's lattice search)
        
        Full Pyro algorithm uses:
        - Lattice traversal (bottom-up or top-down)
        - Pruning rules to skip non-minimal dependencies
        - Dynamic caching of PLIs
        
        This is a simplified version for demonstration
        """
        attributes = list(self.data.columns)
        afds = []
        
        from itertools import combinations
        
        # Try all possible LHS combinations up to max_lhs_size
        for lhs_size in range(1, max_lhs_size + 1):
            for lhs_cols in combinations(attributes, lhs_size):
                # Try each remaining attribute as RHS
                for rhs_col in attributes:
                    if rhs_col in lhs_cols:
                        continue
                    
                    result = self.extract_afd_weight(list(lhs_cols), rhs_col)
                    
                    if result['is_valid_afd']:
                        afds.append(result)
        
        return pd.DataFrame(afds)

# Example usage
def example_pyro():
    # Generate sample data
    data = pd.DataFrame({
        'Age': [30, 30, 30, 40, 40, 45, 45, 50],
        'BMI': [25, 25, 26, 30, 30, 32, 32, 28],
        'Diagnosis': ['Healthy', 'Healthy', 'Healthy', 'Diabetes', 
                      'Diabetes', 'Diabetes', 'Hypertension', 'Diabetes']
    })
    
    print("=" * 60)
    print("PYRO AFD EXTRACTION (g3 error measure)")
    print("=" * 60)
    
    extractor = PyroAFDExtractor(data, error_threshold=0.15)
    
    # Extract specific AFD
    result = extractor.extract_afd_weight(['Age', 'BMI'], 'Diagnosis')
    
    print(f"\nAFD: {result['lhs']} -> {result['rhs']}")
    print(f"g3 error: {result['g3_error']:.3f}")
    print(f"Weight: {result['weight']:.3f}")
    print(f"Valid AFD (error < 0.15): {result['is_valid_afd']}")
    
    print("\nInterpretation:")
    print(f"  - {result['g3_error']*100:.1f}% of tuples violate this dependency")
    print(f"  - {result['weight']*100:.1f}% of tuples satisfy it")
    print(f"  - For privacy: adversary has ~{result['weight']*100:.1f}% inference success rate")
    
    # Discover all AFDs
    print("\n" + "=" * 60)
    print("Discovering all AFDs with error threshold 0.15:")
    print("=" * 60)
    all_afds = extractor.discover_all_afds(max_lhs_size=2)
    print(all_afds[['lhs', 'rhs', 'weight', 'g3_error']])

example_pyro()
