import pandas as pd
import numpy as np
from collections import defaultdict

class HoloCleanWeightExtractor:
    """
    Probabilistic weight extraction inspired by HoloClean (Rekatsinas et al. VLDB 2017)
    
    HoloClean uses:
    - Denial constraints for data quality
    - Probabilistic graphical models (factor graphs)
    - Confidence scores based on constraint violations and data statistics
    
    Reference: "HoloClean: Holistic Data Repairs with Probabilistic Inference"
    
    Simplified version: We compute confidence as:
    - Frequency-based: How often does the constraint hold?
    - Context-based: Given context, what's inference probability?
    """
    
    def __init__(self, data):
        self.data = data
        
    def compute_conditional_probability(self, lhs_cols, rhs_col):
        """
        Compute P(RHS | LHS) using frequency-based estimation
        
        HoloClean approach:
        1. For each unique LHS value combination
        2. Compute distribution of RHS values
        3. Estimate probability based on frequencies
        
        This gives a probabilistic weight for inference
        """
        groups = self.data.groupby(list(lhs_cols))
        
        # Store conditional probabilities
        conditional_probs = []
        supports = []
        
        for lhs_vals, group in groups:
            # Distribution of RHS values given this LHS
            rhs_dist = group[rhs_col].value_counts(normalize=True)
            
            # Maximum probability (most likely RHS value)
            max_prob = rhs_dist.iloc[0] if len(rhs_dist) > 0 else 0
            
            # Support: fraction of data with this LHS value
            support = len(group) / len(self.data)
            
            conditional_probs.append(max_prob)
            supports.append(support)
        
        # Weighted average probability
        avg_prob = np.average(conditional_probs, weights=supports)
        
        # Minimum probability (worst-case for adversary)
        min_prob = min(conditional_probs) if conditional_probs else 0
        
        return {
            'average_confidence': avg_prob,
            'worst_case_confidence': min_prob,
            'n_contexts': len(conditional_probs)
        }
    
    def compute_denial_constraint_confidence(self, predicates_func):
        """
        Compute confidence of a denial constraint
        
        Denial Constraint: NOT(condition1 AND condition2 AND ...)
        
        Confidence = fraction of tuples satisfying the constraint
        
        Example DC: NOT(Age > 65 AND Diagnosis = 'Healthy')
        Means: People over 65 shouldn't be diagnosed as 'Healthy'
        """
        satisfied = 0
        
        for idx, row in self.data.iterrows():
            if predicates_func(row):
                satisfied += 1
        
        confidence = satisfied / len(self.data)
        return confidence
    
    def extract_holoclean_weight(self, lhs_cols, rhs_col, method='average'):
        """
        Extract weight using HoloClean-style probabilistic inference
        
        Methods:
        - 'average': Expected confidence (average over contexts)
        - 'worst_case': Minimum confidence (adversarial)
        - 'entropy': Information-theoretic measure
        
        Returns weight representing inference confidence
        """
        if isinstance(lhs_cols, str):
            lhs_cols = [lhs_cols]
        
        prob_results = self.compute_conditional_probability(lhs_cols, rhs_col)
        
        if method == 'average':
            weight = prob_results['average_confidence']
        elif method == 'worst_case':
            weight = prob_results['worst_case_confidence']
        elif method == 'entropy':
            # Information gain approach
            weight = self.compute_information_gain(lhs_cols, rhs_col)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'lhs': lhs_cols,
            'rhs': rhs_col,
            'weight': weight,
            'method': f'holoclean_{method}',
            **prob_results
        }
    
    def compute_information_gain(self, lhs_cols, rhs_col):
        """
        Information-theoretic weight: How much does LHS reduce uncertainty about RHS?
        
        Weight = 1 - H(RHS | LHS) / H(RHS)
        
        Where H is Shannon entropy
        """
        from scipy.stats import entropy as scipy_entropy
        
        # Baseline entropy of RHS
        rhs_dist = self.data[rhs_col].value_counts(normalize=True)
        h_rhs = scipy_entropy(rhs_dist)
        
        # Conditional entropy H(RHS | LHS)
        groups = self.data.groupby(list(lhs_cols))
        h_rhs_given_lhs = 0
        
        for lhs_vals, group in groups:
            # Entropy within this group
            rhs_dist_given = group[rhs_col].value_counts(normalize=True)
            h_group = scipy_entropy(rhs_dist_given)
            
            # Weight by group size
            p_lhs = len(group) / len(self.data)
            h_rhs_given_lhs += p_lhs * h_group
        
        # Information gain
        if h_rhs == 0:
            weight = 1.0  # Already deterministic
        else:
            weight = 1.0 - (h_rhs_given_lhs / h_rhs)
        
        return max(0, weight)  # Ensure non-negative

# Example usage
def example_holoclean():
    # Generate sample data
    data = pd.DataFrame({
        'Zip': ['92617', '92617', '92617', '10001', '10001', '92620'],
        'City': ['Irvine', 'Irvine', 'Irvine', 'NYC', 'NYC', 'Irvine'],
        'Age': [30, 35, 40, 50, 55, 45],
        'Diagnosis': ['Healthy', 'Healthy', 'Diabetes', 'Diabetes', 'Hypertension', 'Diabetes']
    })
    
    print("=" * 60)
    print("HOLOCLEAN PROBABILISTIC EXTRACTION")
    print("=" * 60)
    
    extractor = HoloCleanWeightExtractor(data)
    
    # Extract using average confidence
    result_avg = extractor.extract_holoclean_weight(['Zip'], 'City', method='average')
    print(f"\nConstraint: {result_avg['lhs']} -> {result_avg['rhs']}")
    print(f"Average confidence: {result_avg['average_confidence']:.3f}")
    print(f"Worst-case confidence: {result_avg['worst_case_confidence']:.3f}")
    print(f"Weight (average): {result_avg['weight']:.3f}")
    
    # Extract using entropy
    result_entropy = extractor.extract_holoclean_weight(['Age'], 'Diagnosis', method='entropy')
    print(f"\nConstraint: {result_entropy['lhs']} -> {result_entropy['rhs']}")
    print(f"Weight (information gain): {result_entropy['weight']:.3f}")
    
    print("\nInterpretation:")
    print(f"  - Zip -> City has {result_avg['weight']*100:.1f}% average inference confidence")
    print(f"  - Age reduces Diagnosis uncertainty by {result_entropy['weight']*100:.1f}%")

example_holoclean()
