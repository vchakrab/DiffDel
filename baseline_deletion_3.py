"""
ILP Implementation Matching Java Code Exactly
Key: Build ILP model during BFS traversal of instantiated cells with time filtering
"""

import time
import sys
from typing import Set, Dict, List
from collections import deque
import mysql.connector

try:
    from gurobipy import Model, GRB, quicksum
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from rtf_core import initialization_phase
from rtf_core.Algorithms import enumerate_explanations as explanations
import config

DELETION_QUERY = "UPDATE {table_name} SET `{column_name}` = NULL WHERE id = {key};"


class Cell:
    def __init__(self, attribute: str, key: int):
        self.attribute = attribute
        self.key = key

    def __hash__(self):
        return hash((self.attribute, self.key))

    def __eq__(self, other):
        return self.attribute == other.attribute and self.key == other.key

    def __repr__(self):
        return f"Cell({self.attribute}, {self.key})"


def get_insertion_time(cursor, table, key, attr):
    """Get insertion time for a cell"""
    try:
        query = f"SELECT `{attr}` FROM {table}_insertiontime WHERE insertionKey = {key}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else 0
    except:
        return 0


def instantiate_edges_with_time_filter(cursor, table, key, attr, target_time, hypergraph):
    """
    Instantiate edges for a cell, filtering by insertion time.
    Returns list of hyperedges (sets of cell attributes that depend on this cell).
    """
    edges = []

    # Find all hyperedges containing this attribute
    for edge_attrs, weight in hypergraph.items():
        if attr not in edge_attrs:
            continue

        # Check insertion times for all cells in this edge
        valid_cells = []
        for edge_attr in edge_attrs:
            attr_name = edge_attr.split('.')[-1]
            it = get_insertion_time(cursor, table, key, attr_name)

            # Only include if inserted >= target_time
            if it >= target_time:
                valid_cells.append(edge_attr)

        # Only create edge if we have multiple valid cells
        if len(valid_cells) > 1:
            edges.append(set(valid_cells))

    return edges


def ilp_approach_matching_java(cursor, table, key, target_attr, target_time, hypergraph, boundaries):
    """
    ILP approach that exactly matches the Java implementation.
    Builds ILP model during BFS traversal.
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi required")

    start = time.time()

    # Initialize
    max_id = 0
    edge_counter = -1
    cell_to_id = {}
    cell_to_var = {}
    instantiated_cells = set()
    cells_to_visit = deque()
    
    cell_to_depth = {}  # For tracking max_depth
    max_depth = 0
    edge_vars = []  # To store bi variables for counting activated dependencies
    existing_rdr_vars = {} # Maps frozenset of attributes to its bi variable to avoid duplicates

    # Create Gurobi model
    model = Model("P2E2_ILP")
    model.setParam('OutputFlag', 0)
    model.setParam('LogToConsole', 0)

    obj = quicksum([])  # Empty objective initially

    # Start with deleted cell - must be deleted (lb=1, ub=1)
    deleted_cell = Cell(f"t1.{target_attr}", key)
    a0 = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name=f"a{max_id}")
    cell_to_var[deleted_cell] = a0
    cell_to_id[deleted_cell] = max_id
    cell_to_depth[deleted_cell] = 0 # Initial depth
    max_id += 1

    cells_to_visit.append(deleted_cell)
    instantiated_cells.add(deleted_cell)

    # BFS traversal building ILP model
    ilp_model_build_start_time = time.time()
    memory_size = 0
    while cells_to_visit:
        curr = cells_to_visit.popleft()
        memory_size += 21  # From Java for cell properties
        curr_id = cell_to_id[curr]
        curr_depth = cell_to_depth[curr] # Get current cell's depth
        aj = cell_to_var[curr]

        # Add to objective
        obj += aj

        # Get edges for this cell (with time filtering)
        edges = instantiate_edges_with_time_filter(
            cursor, table, key, curr.attribute, target_time, hypergraph
        )

        if edges:
            for edge in edges:
                memory_size += 42 + len(edge) * 25  # From Java for edge properties
                frozenset_edge = frozenset(edge)
                if frozenset_edge in existing_rdr_vars:
                    bi = existing_rdr_vars[frozenset_edge]
                else:
                    # Create b_i variable (whether this RDR needs addressing)
                    edge_counter += 1
                    bi = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b{edge_counter}")
                    edge_vars.append(bi) # Store bi for later analysis
                    existing_rdr_vars[frozenset_edge] = bi

                # Create h_ij variable (whether head is hidden)
                hij = model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                  name=f"h{edge_counter}_{curr_id}")

                # Constraint: aj == hij (if head deleted, mark as hidden)
                model.addConstr(aj == hij, name=f"head_hidden_{edge_counter}")

                # Constraint: bi == hij (if head hidden, RDR must be addressed)
                model.addConstr(bi == hij, name=f"rdr_addr_{edge_counter}")

                # Process tail cells
                tail_tji_vars = []
                for cell_attr in edge:
                    # Create cell object
                    cell = Cell(cell_attr, key)

                    # Get or create cell variable
                    if cell not in cell_to_id:
                        t_id = max_id
                        cell_to_id[cell] = max_id
                        max_id += 1
                        a_cell = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"a{t_id}")
                        cell_to_var[cell] = a_cell
                    else:
                        t_id = cell_to_id[cell]
                        a_cell = cell_to_var[cell]

                    # Create t_ji variable (whether tail cell j in edge i is deleted)
                    tji = model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                      name=f"t{edge_counter}_{t_id}")

                    # Constraint: tji == a_cell (sync tail edge with cell deletion)
                    model.addConstr(tji == a_cell, name=f"tail_sync_{edge_counter}_{t_id}")

                    # If this is NOT the head cell, add its var to the tail summation
                    if cell != curr:
                        tail_tji_vars.append(tji)

                    # Add to queue if not visited
                    if cell not in instantiated_cells:
                        instantiated_cells.add(cell)
                        cells_to_visit.append(cell)
                        cell_to_depth[cell] = curr_depth + 1 # Set depth for new cells
                        max_depth = max(max_depth, curr_depth + 1) # Update max depth

                # Constraint: sum of TAIL deletions >= bi (if RDR addressed, delete at least one TAIL)
                if tail_tji_vars:
                    model.addConstr(quicksum(tail_tji_vars) >= bi,
                                  name=f"tail_req_{edge_counter}")
    
    instantiation_time = time.time() - ilp_model_build_start_time # Time for model building before optimize

    # Set objective and optimize
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    model_time = time.time() - start - instantiation_time # Optimization time only

    # Check status
    if model.status == GRB.INFEASIBLE:
        print("ERROR: Model infeasible")
        model.computeIIS()
        model.write("infeasible.ilp")
        raise RuntimeError("Infeasible")

    # Extract solution
    to_delete = set()
    for cell, cell_id in cell_to_id.items():
        var_name = f"a{cell_id}"
        if model.getVarByName(var_name).X == 1.0:
            to_delete.add(cell)
            
    # Count activated dependencies (bi == 1)
    activated_dependencies_count = sum(1 for bi_var in edge_vars if bi_var.X == 1.0)

    model.dispose()

    return to_delete, model_time, instantiation_time, max_depth, len(cell_to_id), activated_dependencies_count, memory_size


def baseline_deletion_3(target: str, key: int, dataset: str, threshold: float):
    """Main function matching Java ILP approach"""
    if not GUROBI_AVAILABLE:
        return 0, 0.0, 0, 0, 0

    print(f"\n{'='*70}")
    print(f"ILP DELETION - JAVA STYLE")
    print(f"Dataset: {dataset} | Target: {target} | Key: {key}")
    print(f"{'='*70}\n")

    # Initialize
    init_mgr = initialization_phase.InitializationManager(
        {"key": key, "attribute": target}, dataset, threshold
    )
    init_mgr.initialize()

    try:
        db = config.get_database_config(dataset)
        conn = mysql.connector.connect(
            host=db['host'], user=db['user'],
            password=db['password'], database=db['database'],
            ssl_disabled=db['ssl_disabled']
        )
        cursor = conn.cursor()
        table = f"{dataset}_copy_data"

        # Get target insertion time
        target_time = get_insertion_time(cursor, table, key, target)
        print(f"Target insertion time: {target_time}")

        # Get denial constraints with target
        target_dcs = []
        for dc in init_mgr.denial_constraints:
            attrs = set(p.split('.')[-1] for p in
                       [x[0] for x in dc] + [x[2] for x in dc if isinstance(x[2], str)])
            if target in attrs:
                target_dcs.append(dc)

        print(f"Denial constraints: {len(target_dcs)}")

        # Build graph
        b_edges, i_edges, orig_bounds = explanations.build_graph_data(target_dcs)
        bounds = {c for c in orig_bounds if not c.startswith('t1.')}

        print(f"Boundary cells: {len(bounds)}")

        # Build hypergraph
        hyper = explanations.set_edge_weight(i_edges)
        print(f"Hypergraph edges: {len(hyper)}")

        # Run ILP (Java style - builds model during traversal)
        print("\nRunning ILP (Java style)...\n")
        to_del, model_time, instantiation_time, max_depth, num_cells, activated_dependencies_count, memory_bytes = ilp_approach_matching_java(
            cursor, table, key, target, target_time, hyper, bounds
        )

        print(f"ILP complete: {len(to_del)} cells to delete")
        print(f"  - Model building (instantiation) time: {instantiation_time:.4f}s")
        print(f"  - Model optimization time: {model_time:.4f}s")
        print(f"  - Total cells instantiated: {num_cells}")
        print(f"  - Max depth traversed: {max_depth}")
        print(f"  - Activated dependencies (RDRs addressed): {activated_dependencies_count}")
        print(f"  - Memory usage (bytes): {memory_bytes}\n")

        # Execute deletions and measure time
        start_deletion_time = time.time()
        for cell in to_del:
            attr = cell.attribute.split('.')[-1]
            cursor.execute(DELETION_QUERY.format(table_name=table, column_name=attr, key=key))

        conn.commit()
        deletion_time = time.time() - start_deletion_time
        print(f"Deletions committed in {deletion_time:.4f}s\n")

        cursor.close()
        conn.close()

        # Return all metrics as per P2E2 evaluation methodology
        return (
            activated_dependencies_count,  # len(found_explanations)
            len(to_del),                  # total_cells_deleted
            memory_bytes,                 # memory_bytes (accurate)
            max_depth,                    # Maximum depth of explanation paths
            instantiation_time,           # Phase 1: Instantiation time
            model_time,                   # Phase 2+3: Modeling + Optimization time
            deletion_time                 # Phase 4: Update to NULL time
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, 0.0, 0.0, 0.0


if __name__ == "__main__":
    if not GUROBI_AVAILABLE:
        print("Gurobi required")
        sys.exit(1)

    dataset = "airport"
    target = "type"
    key = 6323
    threshold = 5.0

    activated_dependencies_count, total_cells_deleted, memory_bytes, max_depth, instantiation_time, model_time, deletion_time = baseline_deletion_3(target, key, dataset, threshold)

    print(f"{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"  Cells deleted: {total_cells_deleted}")
    print(f"  Total time: {instantiation_time + model_time + deletion_time:.4f}s")
    print(f"  Memory usage: {memory_bytes} bytes")
    print(f"  Max depth: {max_depth}")
    print(f"  Activated dependencies: {activated_dependencies_count}")
    print(f"{'='*70}")
