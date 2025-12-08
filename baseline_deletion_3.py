"""
ILP-Based Minimal Set Deletion for P2E2 Guarantee
Based on the paper "Meaningful Data Erasure in the Presence of Dependencies"

This module implements the ILP approach (Section 4.2) to find the minimal set
of cells to delete to guarantee Pre-insertion Post-Erasure Equivalence (P2E2).
"""

import time
import sys
import random
from typing import Set, Dict, List, Tuple, FrozenSet
from collections import deque
import mysql.connector
from mysql.connector import Error


try:
    from gurobipy import Model, GRB, quicksum


    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not available. ILP approach will not work.")

# Import your existing modules
from rtf_core import initialization_phase
from rtf_core.Algorithms import enumerate_explanations as explanations
import config


DELETION_QUERY = """
            UPDATE {table_name}
            SET `{column_name}` = NULL
            WHERE id = {key};
            """


class Cell:
    """Represents a database cell with attribute and key"""

    def __init__(self, attribute: str, key: int):
        self.attribute = attribute
        self.key = key
        self.cost = 1  # Default cost

    def __hash__(self):
        return hash((self.attribute, self.key))

    def __eq__(self, other):
        return self.attribute == other.attribute and self.key == other.key

    def __repr__(self):
        return f"Cell({self.attribute}, {self.key})"


class InstantiatedRDR:
    """Represents an instantiated Relational Dependency Rule (RDR)"""

    def __init__(self, head: Cell, tail: Set[Cell]):
        self.head = head
        self.tail = tail

    def __repr__(self):
        return f"RDR({self.head} ⊥̸⊥ {self.tail})"


def build_instantiated_rdrs_from_hypergraph(
        hypergraph: Dict[FrozenSet[str], float],
        target_cell: str,
        boundary_cells: Set[str],
        key: int
) -> List[InstantiatedRDR]:
    """
    Build instantiated RDRs from hypergraph representation.

    Args:
        hypergraph: Dictionary mapping hyperedges to weights
        target_cell: The cell to be deleted (e.g., "t1.type")
        boundary_cells: Set of boundary cell attributes
        key: The row ID/key for instantiation

    Returns:
        List of instantiated RDRs
    """
    rdrs = []
    visited_edges = set()

    # BFS to find all connected hyperedges starting from target
    queue = deque([target_cell])
    visited_nodes = {target_cell}

    while queue:
        current_node = queue.popleft()

        # Find all hyperedges containing this node
        for hyperedge, weight in hypergraph.items():
            if current_node not in hyperedge:
                continue

            edge_frozen = frozenset(hyperedge)
            if edge_frozen in visited_edges:
                continue

            visited_edges.add(edge_frozen)

            # Create RDRs: for each cell in hyperedge, make it head with rest as tail
            edge_list = list(hyperedge)
            for head_attr in edge_list:
                tail_attrs = [attr for attr in edge_list if attr != head_attr]

                # Only create RDR if not all cells are boundaries
                # (we can delete from tail to prevent inference)
                if tail_attrs:
                    head_cell = Cell(head_attr, key)
                    tail_cells = {Cell(attr, key) for attr in tail_attrs}
                    rdrs.append(InstantiatedRDR(head_cell, tail_cells))

                # Add tail cells to queue for further exploration
                for attr in tail_attrs:
                    if attr not in visited_nodes and attr not in boundary_cells:
                        visited_nodes.add(attr)
                        queue.append(attr)

    return rdrs


def ilp_minimal_deletion(
        target_cell: Cell,
        rdrs: List[InstantiatedRDR],
        boundary_cells: Set[str],
        init_manager,
        target: str,
        target_denial_constraints: List,
        graph_boundary_edges: List,
        graph_internal_edges: List,
        graph_boundary_cells: Set
) -> Tuple[Set[Cell], float, int]:
    """
    Find minimal set of cells to delete using ILP approach.
    Implements the reduction described in Section 4.2 of the paper.

    Args:
        target_cell: The cell to be deleted
        rdrs: List of instantiated RDRs
        boundary_cells: Set of boundary cell attributes (should not be deleted)
        init_manager: Initialization manager (for memory calculation)
        target: Target attribute name
        target_denial_constraints: DCs involving target
        graph_boundary_edges: Boundary edges from graph
        graph_internal_edges: Internal edges from graph
        graph_boundary_cells: Boundary cells set

    Returns:
        Tuple of (cells_to_delete, solve_time, memory_bytes)
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi is required for ILP approach")

    start_time = time.time()

    # Create ILP model
    model = Model("P2E2_MinimalDeletion")
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('LogToConsole', 0)

    # Collect all cells involved
    all_cells = {target_cell}
    for rdr in rdrs:
        all_cells.add(rdr.head)
        all_cells.update(rdr.tail)

    # Create cell ID mapping
    cell_to_id = {cell: idx for idx, cell in enumerate(all_cells)}
    id_to_cell = {idx: cell for cell, idx in cell_to_id.items()}
    max_id = len(cell_to_id)

    # Binary variables for cells: a_j = 1 if cell j should be deleted
    cell_vars = {}
    for cell, cell_id in cell_to_id.items():
        # Constraint 1: Target cell must be deleted
        if cell == target_cell:
            cell_vars[cell_id] = model.addVar(
                vtype = GRB.BINARY,
                lb = 1,
                ub = 1,
                name = f"a{cell_id}"
            )
        # Don't delete boundary cells
        elif cell.attribute in boundary_cells:
            cell_vars[cell_id] = model.addVar(
                vtype = GRB.BINARY,
                lb = 0,
                ub = 0,
                name = f"a{cell_id}"
            )
        else:
            cell_vars[cell_id] = model.addVar(
                vtype = GRB.BINARY,
                name = f"a{cell_id}"
            )

    model.update()

    # Process each RDR
    edge_counter = 0
    for rdr in rdrs:
        head_id = cell_to_id[rdr.head]

        # Binary variable b_i: whether this RDR needs to be addressed
        b_var = model.addVar(vtype = GRB.BINARY, name = f"b{edge_counter}")

        # Binary variable h_ij: whether head is hidden
        h_var = model.addVar(vtype = GRB.BINARY, name = f"h{edge_counter}_{head_id}")

        # Constraint 2: If head is deleted, mark it as hidden
        # a_j = h_ij
        model.addConstr(cell_vars[head_id] == h_var, name = f"head_hidden_{edge_counter}")

        # Constraint 3: If head is hidden, RDR must be addressed
        # b_i = h_ij
        model.addConstr(b_var == h_var, name = f"rdr_addressed_{edge_counter}")

        # Binary variables t_ij: whether tail cell is deleted
        tail_vars = []
        for tail_cell in rdr.tail:
            tail_id = cell_to_id[tail_cell]
            t_var = model.addVar(vtype = GRB.BINARY, name = f"t{edge_counter}_{tail_id}")

            # Constraint 5: If cell is deleted, all tail edges involving it must reflect this
            # a_j = t_ij
            model.addConstr(t_var == cell_vars[tail_id],
                            name = f"tail_sync_{edge_counter}_{tail_id}")

            tail_vars.append(t_var)

        # Constraint 4: To prevent inference, at least one tail cell must be deleted
        # SUM(t_ij) >= b_i
        if tail_vars:
            model.addConstr(
                quicksum(tail_vars) >= b_var,
                name = f"tail_requirement_{edge_counter}"
            )

        edge_counter += 1

    # Objective: Minimize total deletions (Constraint 6)
    # W = min SUM(a_j * Cost(j))
    objective = quicksum(cell_vars[cell_id] * cell.cost
                         for cell, cell_id in cell_to_id.items())
    model.setObjective(objective, GRB.MINIMIZE)

    model.update()

    # Solve the model
    model.optimize()

    solve_time = time.time() - start_time

    # Check if solution found
    if model.status == GRB.INFEASIBLE:
        raise RuntimeError("ILP model is infeasible")

    # Extract solution: cells where a_j = 1
    cells_to_delete = set()
    for cell, cell_id in cell_to_id.items():
        if cell_vars[cell_id].X > 0.5:  # Binary variable is 1
            cells_to_delete.add(cell)

    # Calculate memory usage (matching baseline del 2 format)
    memory_bytes = calculate_ilp_memory(
        init_manager,
        target,
        target_denial_constraints,
        graph_boundary_edges,
        graph_internal_edges,
        graph_boundary_cells,
        all_cells,
        rdrs,
        model
    )

    model.dispose()

    return cells_to_delete, solve_time, memory_bytes


def calculate_ilp_memory(
        init_manager,
        target: str,
        target_denial_constraints: List,
        graph_boundary_edges: List,
        graph_internal_edges: List,
        graph_boundary_cells: Set,
        all_cells: Set[Cell],
        rdrs: List[InstantiatedRDR],
        model
) -> int:
    """
    Calculate memory usage for ILP approach matching baseline deletion 2 format.

    Per cell: 
        - table_index (4 bytes)
        - row_index (4 bytes) 
        - insertion_time (4 bytes)
        - decision variable a_j (1 byte)
        - pointer for objective (8 bytes)
        Total: 21 bytes per cell

    Per RDR edge:
        - decision variable b_i (1 byte)
        - decision variable h_ij (1 byte)
        - constraint a_j = h_ij (16 bytes)
        - constraint b_i = h_ij (16 bytes)
        - per tail cell: decision variable t_ij (1 byte) + constraint t_ij = a_j (16 bytes) + constraint sum (8 bytes)
        - pointer to constraint (8 bytes)
    """
    memory = 0

    # 1. Count all unique cells that were instantiated
    visited_cells = set()

    # Add cells from target_denial_constraints
    for dc in target_denial_constraints:
        for pred in dc:
            if len(pred) >= 1:
                visited_cells.add(pred[0])  # Left side attribute
            if len(pred) >= 3 and isinstance(pred[2], str):
                visited_cells.add(pred[2])  # Right side attribute

    # Add boundary cells
    visited_cells.update(graph_boundary_cells)

    # Add all cells from ILP model
    for cell in all_cells:
        visited_cells.add(cell.attribute)

    # Per cell: table_index(4) + row_index(4) + insertion_time(4) + decision variable(1) + objective pointer(8) = 21 bytes
    memory += len(visited_cells) * 21

    # 2. Count edges (hyperedges from denial constraints)
    for dc_tuple in target_denial_constraints:
        edge_size = len(dc_tuple)
        # Per edge: pointers to cells (8 * edge_size) + parent pointer (8) + minCell (4)
        memory += edge_size * 8 + 8 + 4

    # 3. Graph edges (boundary and internal)
    for edge in graph_boundary_edges:
        edge_size = len(edge) if hasattr(edge, '__len__') else 2
        memory += edge_size * 8 + 8 + 4

    for edge in graph_internal_edges:
        edge_size = len(edge) if hasattr(edge, '__len__') else 2
        memory += edge_size * 8 + 8 + 4
        memory += 8  # Weight (double/float)

    # 4. ILP-specific memory for RDRs
    for rdr in rdrs:
        # b_i and h_ij variables + constraints
        memory += 1 + 1 + 16 + 16

        # Per tail cell
        tail_size = len(rdr.tail)
        memory += tail_size * (1 + 16 + 8)

        # Constraint pointer
        memory += 8

    # 5. Algorithm overhead - data structures (matching baseline del 2)
    if hasattr(init_manager, 'constraint_cells'):
        memory += sys.getsizeof(init_manager.constraint_cells)
        if isinstance(init_manager.constraint_cells, list):
            for cell in init_manager.constraint_cells:
                memory += sys.getsizeof(cell)

    if hasattr(init_manager, 'denial_constraints'):
        memory += sys.getsizeof(init_manager.denial_constraints)
        for dc in init_manager.denial_constraints:
            memory += sys.getsizeof(dc)

    memory += sys.getsizeof(target_denial_constraints)
    for dc in target_denial_constraints:
        memory += sys.getsizeof(dc)

    memory += sys.getsizeof(graph_boundary_edges)
    for edge in graph_boundary_edges:
        memory += sys.getsizeof(edge)

    memory += sys.getsizeof(graph_internal_edges)
    for edge in graph_internal_edges:
        memory += sys.getsizeof(edge)

    memory += sys.getsizeof(graph_boundary_cells)

    return memory


def delete_ilp_minimal_set(target: str, key: int, dataset: str, threshold: float):
    """
    Deletes the minimal set of cells using ILP approach to guarantee P2E2.
    PLUG AND PLAY - matches the exact format of delete_one_path_dependent_cell.

    Parameters:
        target (string): Target attribute
        key (int): Gives the ID of the row in the primary table
        dataset (string): Name of the dataset that the table belongs to
        threshold (int): Threshold for deleting cells (max depth)

    Returns:
        cells_deleted (int): The total number of cells deleted
        ilp_time (float): Time spent on ILP solving
        memory_bytes (int): Memory used by the deletion algorithm in bytes
        max_depth (int): Maximum depth (always 0 for ILP - it's optimal)
        num_rdrs (int): Number of RDRs instantiated

    Side Effects:
        Deletes the minimal set of cells in the database to guarantee P2E2.
    """
    if not GUROBI_AVAILABLE:
        print("Error: Gurobi not available. Install with: pip install gurobipy")
        return 0, 0.0, 0, 0, 0

    # Initialize using same initialization phase as baseline del 2
    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target},
        dataset,
        threshold
    )
    init_manager.initialize()

    target_denial_constraints = []
    total_cells_deleted = 0
    max_depth = 0  # ILP is optimal, doesn't use depth

    # Initialize variables for memory calculation
    graph_boundary_edges = []
    graph_internal_edges = []
    graph_boundary_cells = set()

    try:
        # 1. Get Configuration and Query Details
        db_details = config.get_database_config(dataset)
        primary_table = dataset + "_copy_data"

        # 2. Establish Connection
        conn = mysql.connector.connect(
            host = db_details['host'],
            user = db_details['user'],
            password = db_details['password'],
            database = db_details['database'],
            ssl_disabled = db_details['ssl_disabled']
        )

        if not conn.is_connected():
            print("Connection failed.")
            return 0, 0.0, 0, 0, 0

        cursor = conn.cursor()

        # 3. Find denial constraints involving target attribute
        for dc in init_manager.denial_constraints:
            attrs_in_dc = set(pred.split('.')[-1] for pred in
                              [p[0] for p in dc] + [p[2] for p in dc if isinstance(p[2], str)])
            if target in attrs_in_dc:
                target_denial_constraints.append(dc)

        # 4. Build graph data (same as baseline del 2)
        graph_boundary_edges, graph_internal_edges, graph_boundary_cells = \
            explanations.build_graph_data(target_denial_constraints)

        # 5. Build hypergraph with weights
        hypergraph = explanations.set_edge_weight(graph_internal_edges)

        # 6. Create target cell
        target_cell = Cell(f"t1.{target}", key)

        # 7. Build instantiated RDRs from hypergraph
        rdrs = build_instantiated_rdrs_from_hypergraph(
            hypergraph,
            f"t1.{target}",
            graph_boundary_cells,
            key
        )

        num_rdrs = len(rdrs)

        if not rdrs:
            # No dependencies, just delete the target
            total_cells_deleted = 1
            ilp_time = 0.0
            cursor.execute(
                DELETION_QUERY.format(table_name = primary_table, column_name = target, key = key)
            )
            conn.commit()

            memory_bytes = calculate_ilp_memory(
                init_manager, target, target_denial_constraints,
                graph_boundary_edges, graph_internal_edges, graph_boundary_cells,
                {target_cell}, rdrs, None
            )

            return total_cells_deleted, ilp_time, memory_bytes, max_depth, num_rdrs

        # 8. Run ILP to find minimal deletion set
        cells_to_delete, ilp_time, memory_bytes = ilp_minimal_deletion(
            target_cell,
            rdrs,
            graph_boundary_cells,
            init_manager,
            target,
            target_denial_constraints,
            graph_boundary_edges,
            graph_internal_edges,
            graph_boundary_cells
        )

        # 9. Execute deletions in database
        total_cells_deleted = len(cells_to_delete)

        for cell in cells_to_delete:
            # Extract attribute name (remove "t1." prefix)
            attr_name = cell.attribute.split('.')[-1]
            cursor.execute(
                DELETION_QUERY.format(table_name = primary_table, column_name = attr_name,
                                      key = key)
            )

        conn.commit()

    except Error as e:
        print(f"Database error: {e}")
        return 0, 0.0, 0, 0, 0
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0.0, 0, 0, 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return total_cells_deleted, ilp_time, memory_bytes, max_depth, num_rdrs


# Example usage matching your baseline del 2 pattern
if __name__ == "__main__":
    # Test with same parameters as baseline deletion 2
    dataset = "airport"  # or "tax", "hospital", etc.
    target = "type"
    key = 6323  # or use get_random_key(dataset)
    threshold = 5.0

    if GUROBI_AVAILABLE:
        cells_deleted, ilp_time, memory, max_depth, num_rdrs = delete_ilp_minimal_set(
            target = target,
            key = key,
            dataset = dataset,
            threshold = threshold
        )

        print(f"Total cells deleted: {cells_deleted}")
        print(f"ILP solve time: {ilp_time:.4f} seconds")
        print(f"Memory used: {memory} bytes")
        print(f"Max depth: {max_depth}")
        print(f"Number of RDRs: {num_rdrs}")
    else:
        print("Gurobi not available. Install with: pip install gurobipy")
        print("You may need a Gurobi license: https://www.gurobi.com/downloads/")
