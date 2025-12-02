# Differential Deletion Mechanism

## Overview

This module implements the **Exponential Deletion Mechanism** (Algorithm 1) for privacy-preserving database sanitization. The algorithm uses differential privacy to select an optimal deletion mask that balances privacy protection (minimizing inferential leakage) with data utility (minimizing deletions).

## Algorithm Description

The differential deletion mechanism protects a target cell `c_t` by strategically deleting auxiliary cells to break inference paths. The algorithm:

1. **Computes the inference zone** `I(c_t)` - all cells from the same database tuple
2. **Identifies known attributes** `S` - non-null attributes except the target
3. **Extracts high-strength paths** - inference paths with weight > τ (threshold)
4. **Enumerates candidate hitting sets** - sets of cells that block all high-strength paths
5. **Samples using exponential mechanism** - selects a mask with probability proportional to its utility

### Utility Function

For a mask `M` applied to target cell `c_t`:

```
u(M, c_t) = -α · L(M, c_t) - β · |M|
```

Where:
- `α > 0`: Leakage penalty weight
- `β > 0`: Deletion cost weight
- `L(M, c_t)`: Inferential leakage (max weight of active paths)
- `|M|`: Number of cells deleted

### Exponential Mechanism

The algorithm samples a mask `M` with probability:

```
P(M) ∝ exp(ε · u(M, c_t) / (2α))
```

Where `ε` is the privacy budget.

## Installation

```bash
pip install numpy pandas mysql-connector-python
```

## Quick Start

### Basic Usage

```python
from InferenceGraph.differential_deletion import DifferentialDeletion
from fetch_row import RTFDatabaseManager

# Initialize the mechanism
dd = DifferentialDeletion(
    dataset='adult',
    alpha=1.0,      # Leakage penalty weight
    beta=0.5,       # Deletion cost weight
    epsilon=1.0,    # Privacy budget
    tau=0.1        # Path strength threshold
)

# Fetch a database row
with RTFDatabaseManager('adult') as db:
    row = db.fetch_row(key=2)

# Run differential deletion
mask = dd.exponential_deletion(
    row=row,
    key=2,
    target_attr='education'
)

# Display results
print(f"Cells to delete: {len(mask)}")
for cell in mask:
    print(f"  - {cell}")
```

### With Custom Hyperedge Weights

```python
def custom_weight_fn(hyperedge):
    """
    Custom weight function based on constraint selectivity.
    Higher weight = stronger inference capability.
    """
    # Example: weight based on number of cells
    return 1.0 / len(hyperedge)

dd = DifferentialDeletion(
    dataset='adult',
    alpha=1.0,
    beta=0.5,
    epsilon=1.0,
    tau=0.1,
    hyperedge_weight_fn=custom_weight_fn
)
```

## API Reference

### Class: `DifferentialDeletion`

Main class for the differential deletion mechanism.

#### Constructor

```python
DifferentialDeletion(
    dataset: str,
    alpha: float = 1.0,
    beta: float = 0.5,
    epsilon: float = 1.0,
    tau: float = 0.1,
    hyperedge_weight_fn = None
)
```

**Parameters:**
- `dataset` (str): Dataset name (e.g., 'adult', 'tax', 'hospital')
- `alpha` (float): Leakage penalty weight (default: 1.0)
- `beta` (float): Deletion cost weight (default: 0.5)
- `epsilon` (float): Privacy budget (default: 1.0)
- `tau` (float): Path strength threshold (default: 0.1)
- `hyperedge_weight_fn` (callable, optional): Function to compute hyperedge weights

#### Method: `exponential_deletion`

```python
exponential_deletion(
    row: Dict[str, Any],
    key: Any,
    target_attr: str
) -> Set[Cell]
```

Execute the exponential deletion mechanism.

**Parameters:**
- `row` (dict): Database row as dictionary
- `key`: Row identifier
- `target_attr` (str): Target attribute to protect

**Returns:**
- `Set[Cell]`: Deletion mask (set of cells to delete)

### Class: `InferencePath`

Represents an inference path in the hypergraph.

#### Methods

```python
compute_weight(hyperedge_weights: Dict) -> float
# Compute path weight as product of hyperedge weights

is_blocked_by(mask: Set[Cell]) -> bool
# Check if path is blocked by deletion mask
```

### Functions

#### `extract_all_paths`

```python
extract_all_paths(root: GraphNode) -> List[InferencePath]
```

Extract all paths from root to leaves in the inference graph.

#### `find_inference_paths`

```python
find_inference_paths(
    known_attrs: Set[str],
    target_attr: str,
    root: GraphNode,
    hyperedge_weights: Dict,
    tau: float
) -> List[InferencePath]
```

Extract high-strength inference paths (weight > tau).

#### `enumerate_minimal_hitting_sets`

```python
enumerate_minimal_hitting_sets(
    paths: List[InferencePath],
    inference_zone: Set[Cell],
    max_candidates: int = 100
) -> List[Set[Cell]]
```

Enumerate hitting sets using greedy and exhaustive search strategies.

#### `compute_leakage`

```python
compute_leakage(
    mask: Set[Cell],
    paths: List[InferencePath],
    hyperedge_weights: Dict
) -> float
```

Compute inferential leakage L(M, c_t) = max weight of active paths.

#### `compute_utility`

```python
compute_utility(
    mask: Set[Cell],
    target_cell: Cell,
    paths: List[InferencePath],
    hyperedge_weights: Dict,
    alpha: float,
    beta: float
) -> float
```

Compute mask utility u(M, c_t) = -α·L(M, c_t) - β·|M|.

#### `exponential_mechanism_sample`

```python
exponential_mechanism_sample(
    candidates: List[Set[Cell]],
    target_cell: Cell,
    paths: List[InferencePath],
    hyperedge_weights: Dict,
    alpha: float,
    beta: float,
    epsilon: float
) -> Set[Cell]
```

Sample a mask using exponential mechanism with differential privacy.

## Configuration

### Parameter Tuning

**Alpha (α)** - Leakage Penalty Weight:
- Higher α: Prioritizes privacy protection (lower leakage)
- Lower α: More tolerant of inference leakage
- Recommended: 0.5 - 2.0

**Beta (β)** - Deletion Cost Weight:
- Higher β: Minimizes number of deletions (preserves utility)
- Lower β: More willing to delete cells for privacy
- Recommended: 0.1 - 1.0

**Epsilon (ε)** - Privacy Budget:
- Higher ε: More deterministic selection (better utility, less privacy)
- Lower ε: More randomization (better privacy, potentially worse utility)
- Recommended: 0.5 - 2.0

**Tau (τ)** - Path Strength Threshold:
- Higher τ: Only considers strong inference paths
- Lower τ: Considers more paths (more conservative)
- Recommended: 0.05 - 0.2

### Example Configurations

**Maximum Privacy Protection:**
```python
dd = DifferentialDeletion(
    dataset='adult',
    alpha=2.0,      # High leakage penalty
    beta=0.1,       # Low deletion cost
    epsilon=0.5,    # Low privacy budget (more randomness)
    tau=0.05        # Low threshold (consider more paths)
)
```

**Balanced Privacy-Utility:**
```python
dd = DifferentialDeletion(
    dataset='adult',
    alpha=1.0,
    beta=0.5,
    epsilon=1.0,
    tau=0.1
)
```

**Maximum Utility Preservation:**
```python
dd = DifferentialDeletion(
    dataset='adult',
    alpha=0.5,      # Low leakage penalty
    beta=1.0,       # High deletion cost
    epsilon=2.0,    # High privacy budget (more deterministic)
    tau=0.2         # High threshold (fewer paths)
)
```

## Testing

### Run Unit Tests

```bash
python InferenceGraph/test_differential_deletion.py
```

This runs comprehensive tests including:
- Path extraction
- Hyperedge weight computation
- Hitting set enumeration
- Leakage computation
- Utility computation
- Exponential mechanism sampling
- End-to-end integration test

### Run with Database

```bash
python InferenceGraph/differential_deletion.py
```

This tests with real database data (requires MySQL connection).

## Performance Considerations

### Time Complexity

- **Path Extraction**: O(n·m) where n = nodes, m = branches per node
- **Hitting Set Enumeration**: O(2^k) worst case, where k = inference zone size
  - Optimized using greedy heuristics for large k
  - Configurable max_candidates limit
- **Exponential Mechanism**: O(c) where c = number of candidates

### Memory Optimization

- Paths are represented efficiently using cell references
- Hyperedge weights are cached to avoid recomputation
- Hitting sets use frozensets for deduplication

### Scalability

For large inference zones (> 20 cells):
- Use higher tau to reduce number of paths
- Limit max_candidates in hitting set enumeration
- Consider using domain-specific weight functions to prune weak paths

## Dataset Independence

The implementation is **fully dataset-agnostic**. It works with any dataset that has:
- A primary table with a key column
- Denial constraints defined in the dataset configuration
- Support from `HyperedgeBuilder` and `RTFDatabaseManager`

To use with a new dataset:

1. Add dataset configuration in `config.py`
2. Define denial constraints in `DCandDelset/dc_configs/`
3. Use the same API:

```python
dd = DifferentialDeletion(dataset='your_dataset', ...)
```

## Examples

### Example 1: Protect Education Attribute

```python
from InferenceGraph.differential_deletion import DifferentialDeletion
from fetch_row import RTFDatabaseManager

dd = DifferentialDeletion(dataset='adult', alpha=1.0, beta=0.5, epsilon=1.0, tau=0.1)

with RTFDatabaseManager('adult') as db:
    row = db.fetch_row(2)

mask = dd.exponential_deletion(row, 2, 'education')

print(f"Protected: education = {row['education']}")
print(f"Deleted {len(mask)} cells:")
for cell in mask:
    print(f"  - {cell.attribute.col} = {cell.value}")
```

### Example 2: Compare Different Strategies

```python
strategies = [
    ('High Privacy', {'alpha': 2.0, 'beta': 0.1, 'epsilon': 0.5}),
    ('Balanced', {'alpha': 1.0, 'beta': 0.5, 'epsilon': 1.0}),
    ('High Utility', {'alpha': 0.5, 'beta': 1.0, 'epsilon': 2.0}),
]

for name, params in strategies:
    dd = DifferentialDeletion(dataset='adult', tau=0.1, **params)
    mask = dd.exponential_deletion(row, 2, 'education')
    print(f"{name}: {len(mask)} cells deleted")
```

### Example 3: Batch Processing

```python
dd = DifferentialDeletion(dataset='adult', alpha=1.0, beta=0.5, epsilon=1.0, tau=0.1)

keys = [1, 2, 3, 4, 5]
target_attr = 'education'

with RTFDatabaseManager('adult') as db:
    for key in keys:
        row = db.fetch_row(key)
        mask = dd.exponential_deletion(row, key, target_attr)
        print(f"Row {key}: {len(mask)} deletions for {row[target_attr]}")
```

## Troubleshooting

### No High-Strength Paths Found

If no paths have weight > tau:
- Lower the tau threshold
- Check hyperedge weights (might all be too low)
- Verify denial constraints are loaded correctly

### Too Many Candidates

If hitting set enumeration is slow:
- Increase tau (fewer paths to hit)
- Reduce max_candidates parameter
- Use more selective hyperedge weights

### Empty Deletion Mask

If the algorithm returns an empty mask:
- The exponential mechanism may favor no deletions (high β, low α)
- Adjust parameters to prioritize privacy over utility
- Check if there are actually inference paths to block

## References

- Algorithm 1: Exponential Deletion Mechanism (research paper)
- Differential Privacy: Dwork & Roth (2014)
- Hitting Set Problem: Karp (1972)

## License

MIT License - See LICENSE file for details.

## Citation

If you use this implementation in research, please cite:

```bibtex
@software{differential_deletion_2025,
  title={Differential Deletion Mechanism for Privacy-Preserving Database Sanitization},
  author={RTF Research Group},
  year={2025},
  url={https://github.com/your-repo/DiffDel}
}
```
