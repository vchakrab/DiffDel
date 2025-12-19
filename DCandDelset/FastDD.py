import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

class FastDDExtractor:
    """
    DD/RFD extraction based on FastDD algorithm (Kuang et al. VLDB 2024)
    
    Differential Dependencies: ϕ[X](≤θ₁) → ϕ[Y](≤θ₂)
    Means: If tuples are similar on X (distance ≤ θ₁), 
           then they should be similar on Y (distance ≤ θ₂)
    
    Reference: "Efficient Differential Dependency Discovery"
    FastDD uses support as the fraction of tuple pairs satisfying the constraint
    """
    
    def __init__(self, data, min_support=0.8):
        """
        Args:
            data: pandas DataFrame
            min_support: minimum support threshold for valid DDs
        """
        self.data = data
        self.min_support = min_support
        
    def compute_distance(self, val1, val2, col_name):
        """
        Compute distance between two values
        
        For numerical: absolute difference
        For categorical: 0 if equal, 1 otherwise (edit distance could be used)
        """
        if pd.api.types.is_numeric_dtype(self.data[col_name]):
            return abs(val1 - val2)
        else:
            return 0 if val1 == val2 else 1
    
    def infer_threshold(self, col_name, percentile=25):
        """
        Infer similarity threshold from data distribution
        
        FastDD paper: "Rather than ask users to provide thresholds, 
                       thresholds can be inferred from the given instance"
        
        Method: Use percentile of pairwise distances
        - Low percentile (e.g., 25th) = strict similarity
        - High percentile (e.g., 50th) = loose similarity
        """
        if pd.api.types.is_numeric_dtype(self.data[col_name]):
            # Sample pairs for efficiency
            n_samples = min(1000, len(self.data))
            sample_indices = np.random.choice(len(self.data), size=n_samples, replace=False)
            
            distances = []
            for i in range(len(sample_indices)):
                for j in range(i + 1, len(sample_indices)):
                    idx1, idx2 = sample_indices[i], sample_indices[j]
                    dist = abs(self.data.loc[idx1, col_name] - 
                              self.data.loc[idx2, col_name])
                    distances.append(dist)
            
            threshold = np.percentile(distances, percentile)
        else:
            threshold = 0  # Exact match for categorical
        
        return threshold
    
    def compute_dd_support(self, lhs_cols, rhs_col, 
                          lhs_threshold=None, rhs_threshold=None,
                          sample_size=2000):
        """
        Compute DD support: fraction of tuple pairs satisfying the constraint
        
        Support(ϕ[X](≤θ₁) → ϕ[Y](≤θ₂)) = 
            |{(ti, tj) : d(X, ti, tj) ≤ θ₁ ∧ d(Y, ti, tj) ≤ θ₂}| / 
            |{(ti, tj) : d(X, ti, tj) ≤ θ₁}|
        
        This is the confidence measure: P(RHS similar | LHS similar)
        
        Algorithm:
        1. For each tuple pair (ti, tj):
           - Compute distance on LHS attributes
           - If distance ≤ θ₁ (similar on LHS):
               - Check if distance on RHS ≤ θ₂
               - Count successes and total
        2. Support = successes / total_similar_on_lhs
        """
        # Infer thresholds if not provided
        if lhs_threshold is None:
            if len(lhs_cols) == 1:
                lhs_threshold = self.infer_threshold(lhs_cols[0])
            else:
                # Use Euclidean distance for multiple attributes
                lhs_threshold = np.mean([self.infer_threshold(col) 
                                        for col in lhs_cols])
        
        if rhs_threshold is None:
            rhs_threshold = self.infer_threshold(rhs_col)
        
        # Sample tuple pairs for efficiency (as in FastDD paper)
        n = len(self.data)
        n_pairs = min(sample_size, n * (n - 1) // 2)
        
        pairs_similar_lhs = 0
        pairs_satisfy_dd = 0
        
        # Generate random pairs
        for _ in range(n_pairs):
            i, j = np.random.choice(n, size=2, replace=False)
            
            # Compute LHS distance
            if len(lhs_cols) == 1:
                lhs_dist = self.compute_distance(
                    self.data.loc[i, lhs_cols[0]],
                    self.data.loc[j, lhs_cols[0]],
                    lhs_cols[0]
                )
            else:
                # Euclidean distance for multiple attributes
                lhs_vals_i = [self.data.loc[i, col] for col in lhs_cols]
                lhs_vals_j = [self.data.loc[j, col] for col in lhs_cols]
                lhs_dist = euclidean(lhs_vals_i, lhs_vals_j)
            
            # Check if similar on LHS
            if lhs_dist <= lhs_threshold:
                pairs_similar_lhs += 1
                
                # Compute RHS distance
                rhs_dist = self.compute_distance(
                    self.data.loc[i, rhs_col],
                    self.data.loc[j, rhs_col],
                    rhs_col
                )
                
                # Check if similar on RHS
                if rhs_dist <= rhs_threshold:
                    pairs_satisfy_dd += 1
        
        # Support = confidence = P(RHS similar | LHS similar)
        support = pairs_satisfy_dd / pairs_similar_lhs if pairs_similar_lhs > 0 else 0
        
        return support, {
            'lhs_threshold': lhs_threshold,
            'rhs_threshold': rhs_threshold,
            'pairs_checked': n_pairs,
            'pairs_similar_lhs': pairs_similar_lhs,
            'pairs_satisfy_dd': pairs_satisfy_dd
        }
    
    def extract_dd_weight(self, lhs_cols, rhs_col, **kwargs):
        """
        Extract DD weight using FastDD's support measure
        
        Weight = support = confidence of the differential dependency
        
        Interpretation:
        - w = 1.0: All similar tuples on LHS are also similar on RHS
        - w = 0.8: 80% of similar LHS pairs have similar RHS
        - w = 0.0: LHS similarity doesn't imply RHS similarity
        
        For privacy: This is directly the inference probability
        If adversary knows tuple t1 is similar to t2 on LHS,
        they can infer t2's RHS with probability = support
        """
        if isinstance(lhs_cols, str):
            lhs_cols = [lhs_cols]
        
        support, metadata = self.compute_dd_support(lhs_cols, rhs_col, **kwargs)
        
        is_valid_dd = support >= self.min_support
        
        return {
            'lhs': lhs_cols,
            'rhs': rhs_col,
            'weight': support,
            'support': support,
            'is_valid_dd': is_valid_dd,
            'method': 'fastdd_support',
            **metadata
        }

# Example usage
def example_fastdd():
    # Generate sample data with RFD: similar Age -> similar BMI
    np.random.seed(42)
    data = pd.DataFrame({
        'PatientID': range(1, 101),
        'Age': np.random.randint(20, 80, 100),
        'BMI': np.random.uniform(18, 40, 100),
    })
    
    # Create dependency: similar Age -> similar BMI (with noise)
    data['BMI'] = data['Age'] * 0.3 + np.random.normal(0, 2, 100) + 10
    
    print("=" * 60)
    print("FASTDD EXTRACTION (support/confidence measure)")
    print("=" * 60)
    
    extractor = FastDDExtractor(data, min_support=0.7)
    
    # Extract DD: Age(≤5) -> BMI(≤2)
    result = extractor.extract_dd_weight(['Age'], 'BMI', 
                                         lhs_threshold=5, 
                                         rhs_threshold=2)
    
    print(f"\nDD: {result['lhs']}(≤{result['lhs_threshold']}) -> "
          f"{result['rhs']}(≤{result['rhs_threshold']})")
    print(f"Support (confidence): {result['support']:.3f}")
    print(f"Weight: {result['weight']:.3f}")
    print(f"Valid DD (support ≥ 0.7): {result['is_valid_dd']}")
    
    print(f"\nStatistics:")
    print(f"  - Pairs checked: {result['pairs_checked']}")
    print(f"  - Pairs similar on LHS: {result['pairs_similar_lhs']}")
    print(f"  - Pairs satisfying DD: {result['pairs_satisfy_dd']}")
    
    print("\nInterpretation for privacy:")
    print(f"  - If adversary knows two patients have similar age (±{result['lhs_threshold']} years),")
    print(f"  - They can infer BMI similarity with {result['weight']*100:.1f}% confidence")

example_fastdd()
