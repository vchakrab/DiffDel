"""
ILP with Full Gurobi Debug Output
Prints all variables, constraints, and model details for inspection
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
        return f"{self.attribute}"


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
    """Instantiate edges for a cell, filtering by insertion time"""
    edges = []

    for edge_attrs, weight in hypergraph.items():
        if attr not in edge_attrs:
            continue

        valid_cells = []
        for edge_attr in edge_attrs:
            attr_name = edge_attr.split('.')[-1]
            it = get_insertion_time(cursor, table, key, attr_name)

            if it >= target_time:
                valid_cells.append(edge_attr)

        if len(valid_cells) > 1:
            edges.append(set(valid_cells))

    return edges


def ilp_debug_approach(cursor, table, key, target_attr, target_time, hypergraph, boundaries):
    """ILP with full debug output"""
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi required")

    start = time.time()

    # Initialize
    cell_to_id = {}
    cell_to_var = {}
    instantiated_cells = set()
    cells_to_visit = deque()

    all_cells_list = []
    all_edges = []

    # Create Gurobi model
    model = Model("P2E2_Debug")
    model.setParam('OutputFlag', 0)

    # Start with target
    deleted_cell = Cell(f"t1.{target_attr}", key)
    cells_to_visit.append(deleted_cell)
    instantiated_cells.add(deleted_cell)
    all_cells_list.append(deleted_cell)

    print(f"\n{'=' * 80}")
    print(f"PHASE 1: DISCOVERING CELLS AND EDGES")
    print(f"{'=' * 80}")
    print(f"Target: {deleted_cell} (time: {target_time})")
    print(f"Boundaries: {boundaries if boundaries else 'NONE'}\n")

    # BFS traversal
    iteration = 0
    while cells_to_visit:
        curr = cells_to_visit.popleft()
        iteration += 1

        print(f"[Iteration {iteration}] Processing: {curr}")

        edges = instantiate_edges_with_time_filter(
            cursor, table, key, curr.attribute, target_time, hypergraph
        )

        if edges:
            print(f"  Found {len(edges)} hyperedge(s):")
            for edge_idx, edge in enumerate(edges):
                edge_list = sorted(list(edge), key = lambda x: x)
                print(f"    Edge {edge_idx}: {{{', '.join(str(e) for e in edge_list)}}}")

                # Create RDRs from this edge
                for head_attr in edge_list:
                    tail_attrs = [a for a in edge_list if a != head_attr]

                    if tail_attrs:
                        head_cell = Cell(head_attr, key)
                        tail_cells = {Cell(a, key) for a in tail_attrs}

                        all_edges.append((head_cell, tail_cells))

                        if head_cell not in instantiated_cells:
                            instantiated_cells.add(head_cell)
                            all_cells_list.append(head_cell)
                            cells_to_visit.append(head_cell)
                            print(f"      New cell discovered: {head_cell}")

                        for t_cell in tail_cells:
                            if t_cell not in instantiated_cells:
                                instantiated_cells.add(t_cell)
                                all_cells_list.append(t_cell)
                                if t_cell.attribute not in boundaries:
                                    cells_to_visit.append(t_cell)
                                    print(f"      New cell discovered: {t_cell}")
        else:
            print(f"  No edges found (leaf node)")
        print()

    print(f"{'=' * 80}")
    print(f"DISCOVERY COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total cells discovered: {len(all_cells_list)}")
    print(f"Total RDRs (edges) discovered: {len(all_edges)}\n")

    # Show all discovered RDRs
    print(f"{'=' * 80}")
    print(f"ALL DISCOVERED RDRs:")
    print(f"{'=' * 80}")
    for idx, (head, tail) in enumerate(all_edges):
        tail_str = ', '.join(sorted([str(t) for t in tail]))
        print(f"RDR {idx}: {head} <- {{{tail_str}}}")
    print()

    # Create cell ID mapping
    print(f"{'=' * 80}")
    print(f"PHASE 2: CREATING VARIABLES")
    print(f"{'=' * 80}")
    for idx, cell in enumerate(all_cells_list):
        cell_to_id[cell] = idx

    # Create variables
    for cell in all_cells_list:
        cell_id = cell_to_id[cell]

        if cell == deleted_cell:
            var = model.addVar(lb = 1, ub = 1, vtype = GRB.BINARY, name = f"a{cell_id}")
            cell_to_var[cell] = var
            print(f"a{cell_id} = {cell} [FORCED TO 1 - target cell]")
        elif cell.attribute in boundaries:
            var = model.addVar(lb = 0, ub = 0, vtype = GRB.BINARY, name = f"a{cell_id}")
            cell_to_var[cell] = var
            print(f"a{cell_id} = {cell} [FORCED TO 0 - boundary cell]")
        else:
            var = model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = f"a{cell_id}")
            cell_to_var[cell] = var
            print(f"a{cell_id} = {cell} [FREE - ILP decides]")

    model.update()
    print()

    # Add constraints
    print(f"{'=' * 80}")
    print(f"PHASE 3: CREATING CONSTRAINTS")
    print(f"{'=' * 80}")
    constraints_added = 0

    for rdr_idx, (head_cell, tail_cells) in enumerate(all_edges):
        if head_cell == deleted_cell:
            print(f"RDR {rdr_idx}: SKIP (head is target)")
            continue

        head_var = cell_to_var[head_cell]
        head_id = cell_to_id[head_cell]

        tail_vars = []
        tail_ids = []
        for t in tail_cells:
            if t.attribute not in boundaries:
                tail_vars.append(cell_to_var[t])
                tail_ids.append(cell_to_id[t])

        if tail_vars:
            # Constraint: sum(a_tail) >= 1 - a_head
            model.addConstr(
                quicksum(tail_vars) >= 1 - head_var,
                name = f"rdr_{rdr_idx}"
            )

            tail_str = ' + '.join([f"a{tid}" for tid in tail_ids])
            print(f"RDR {rdr_idx}: {tail_str} >= 1 - a{head_id}")
            print(
                f"  Meaning: If {head_cell} NOT deleted, delete at least one of: {{{', '.join(str(t) for t in tail_cells if t.attribute not in boundaries)}}}")
            constraints_added += 1
        else:
            if head_cell.attribute not in boundaries:
                model.addConstr(head_var == 1, name = f"rdr_force_{rdr_idx}")
                print(f"RDR {rdr_idx}: a{head_id} = 1 (FORCED - no deletable tail cells)")
                constraints_added += 1
            else:
                print(f"RDR {rdr_idx}: SKIP (head is boundary, no deletable tails)")
        print()

    print(f"Total constraints added: {constraints_added}\n")

    # Set objective
    print(f"{'=' * 80}")
    print(f"OBJECTIVE FUNCTION")
    print(f"{'=' * 80}")
    obj_terms = []
    for cell in all_cells_list:
        cell_id = cell_to_id[cell]
        obj_terms.append(f"a{cell_id}")

    print(f"minimize: {' + '.join(obj_terms)}")
    print(f"(Minimize total cells deleted)\n")

    obj = quicksum(cell_to_var[cell] for cell in all_cells_list)
    model.setObjective(obj, GRB.MINIMIZE)

    model.update()

    # Write model to file for inspection
    model.write("debug_model.lp")
    print(f"Model written to: debug_model.lp\n")

    # Optimize
    print(f"{'=' * 80}")
    print(f"PHASE 4: SOLVING")
    print(f"{'=' * 80}")
    model.setParam('OutputFlag', 1)
    model.optimize()

    solve_time = time.time() - start

    if model.status == GRB.INFEASIBLE:
        print("\nERROR: Model is INFEASIBLE")
        model.computeIIS()
        model.write("infeasible.ilp")
        raise RuntimeError("Infeasible")
    elif model.status != GRB.OPTIMAL:
        print(f"\nWARNING: Model status {model.status}")

    # Extract solution
    print(f"\n{'=' * 80}")
    print(f"SOLUTION")
    print(f"{'=' * 80}")
    to_delete = set()
    for cell in all_cells_list:
        cell_id = cell_to_id[cell]
        var_value = cell_to_var[cell].X
        if var_value > 0.5:
            to_delete.add(cell)
            print(f"a{cell_id} = 1  DELETE: {cell}")
        else:
            print(f"a{cell_id} = 0  KEEP:   {cell}")

    print(f"\nObjective value: {model.objVal}")
    print(f"Cells to delete: {len(to_delete)}")

    model.dispose()

    return to_delete, solve_time, len(all_cells_list)


def delete_ilp_debug(target: str, key: int, dataset: str, threshold: float):
    """Main function with full debug output"""
    if not GUROBI_AVAILABLE:
        print("Error: Gurobi not available")
        return 0, 0.0, 0, 0, 0

    print(f"\n{'#' * 80}")
    print(f"# ILP DELETION WITH FULL DEBUG OUTPUT")
    print(f"# Dataset: {dataset} | Target: {target} | Key: {key}")
    print(f"{'#' * 80}")

    init_mgr = initialization_phase.InitializationManager(
        {"key": key, "attribute": target}, dataset, threshold
    )
    init_mgr.initialize()

    try:
        db = config.get_database_config(dataset)
        conn = mysql.connector.connect(
            host = db['host'], user = db['user'],
            password = db['password'], database = db['database'],
            ssl_disabled = db['ssl_disabled']
        )
        cursor = conn.cursor()
        table = f"{dataset}_copy_data"

        target_time = get_insertion_time(cursor, table, key, target)

        target_dcs = []
        for dc in init_mgr.denial_constraints:
            attrs = set(p.split('.')[-1] for p in
                        [x[0] for x in dc] + [x[2] for x in dc if isinstance(x[2], str)])
            if target in attrs:
                target_dcs.append(dc)

        b_edges, i_edges, orig_bounds = explanations.build_graph_data(target_dcs)
        bounds = {c for c in orig_bounds if not c.startswith('t1.')}
        hyper = explanations.set_edge_weight(i_edges)

        to_del, ilp_time, num_cells = ilp_debug_approach(
            cursor, table, key, target, target_time, hyper, bounds
        )

        print(f"\n{'=' * 80}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 80}")
        print(f"Cells deleted: {len(to_del)}")
        print(f"Solve time: {ilp_time:.4f}s")
        print(f"Total cells: {num_cells}")
        print(f"{'=' * 80}\n")

        # Execute deletions
        for cell in to_del:
            attr = cell.attribute.split('.')[-1]
            cursor.execute(DELETION_QUERY.format(table_name = table, column_name = attr, key = key))

        conn.commit()
        cursor.close()
        conn.close()

        return len(to_del), ilp_time, num_cells * 20, 0, num_cells

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0.0, 0, 0, 0


if __name__ == "__main__":
    if not GUROBI_AVAILABLE:
        print("ERROR: Gurobi required")
        sys.exit(1)

    dataset = "airport"
    target = "type"
    key = 6323
    threshold = 5.0

    cells, ilp_time, mem, depth, num = delete_ilp_debug(target, key, dataset, threshold)

    print(f"\nCheck the file 'debug_model.lp' to see the exact LP formulation Gurobi solved")