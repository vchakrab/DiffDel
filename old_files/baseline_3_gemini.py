import gurobipy as gp
from gurobipy import GRB
import time
from collections import deque
from typing import Set, Dict, List, TYPE_CHECKING, Any


# --- Mock Utils for Time Tracking (Matching the original Java context) ---
class Utils:
    ilpTimes = [0.0] * 5  # Index 1: Instantiation, 2: Model Build, 3: Optimization
    ilpCounts = [0] * 4  # Index 1: Instantiations/Cells added
    measureMemory = False


# --- Core Data Structure: Cell ---
class Cell:
    def __init__(self, table: str, key: str):
        self.table = table
        self.key = key

    def __hash__(self):
        # Hashable by its unique identifier (table and key)
        return hash((self.table, self.key))

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.table == other.table and self.key == other.key

    def __repr__(self):
        return f"C({self.table}:{self.key})"


# --- Graph Structure: InstantiatedModel ---
class InstantiatedModel:
    def __init__(self):
        # Maps Cell (head) to a list of its dependent hyperedges (tail: list of Cells)
        self.cell2Edge: Dict[Cell, List[List[Cell]]] = {}
        self.instantiationTime: Dict[Cell, float] = {}
        self.modelConstructionTime: float = 0.0  # Time tracking placeholder


# --- Setup for the ILP function (from previous response, included for completeness) ---
# NOTE: The actual ilp_approach_gurobi function is assumed to be defined above.
# We'll call it now.
# ---

# Mock environment setup (assuming it's passed or defined globally like in Java)
# Replace 'Utils' with your actual tracking mechanism if different.
# ---

def ilp_approach_gurobi(model_instance: InstantiatedModel, deleted: Cell, env: gp.Env) -> Set[Cell]:
    """
    Translates the single-cell ILP deletion approach from Java (using GRB)
    to Python using the gurobipy library.

    The ILP formulation is a Minimum Hitting Set / Set Cover problem to find
    the smallest set of cells (nodes) to delete to cover all induced dependencies (hyperedges).

    :param model_instance: InstantiatedModel object containing cell2Edge graph.
    :param deleted: The starting Cell that must be deleted (initial violation).
    :param env: The Gurobi environment object (GRBEnv in Java).
    :return: A set of Cell objects marked for deletion by the ILP solver.
    :raises gp.GurobiError: If the model is infeasible.
    """
    # Assuming model_instance.modelConstructionTime is set before calling this function
    # by the Python equivalent of InstantiatedModel constructor.
    # Utils.ilpTimes[2] += model_instance.modelConstructionTime # Removed: Model construction time should be tracked outside or accounted for by the loop below

    start_build_time = time.perf_counter_ns()

    # --- Initialization ---
    max_id = 0
    edge_counter = -1
    cell_to_id: Dict[Cell, int] = {}
    cell_to_var: Dict[Cell, gp.Var] = {}
    instantiated_cells: Set[Cell] = set()
    to_delete: Set[Cell] = set()
    cells_to_visit: deque[Cell] = deque()

    # Create the Gurobi model within the provided environment
    grb_model = gp.Model("CellDeletionILP", env = env)

    # Suppress Gurobi output for clean console (equivalent to Java's set(GRB.IntParam.OutputFlag, 0))
    grb_model.Params.OutputFlag = 0

    # Objective function expression
    obj = gp.LinExpr()

    # --- Initial Deletion (a_0) Constraint ---
    # The starting cell must be marked for deletion.
    # Add variable a0: must be 1 (lower and upper bound set to 1)
    a_deleted = grb_model.addVar(lb = 1, ub = 1, vtype = GRB.BINARY, name = "a0")
    cell_to_var[deleted] = a_deleted
    cell_to_id[deleted] = max_id
    max_id += 1
    cells_to_visit.append(deleted)
    instantiated_cells.add(deleted)

    # --- Build Constraints via Graph Traversal (BFS) ---
    while cells_to_visit:
        curr = cells_to_visit.popleft()
        curr_id = cell_to_id[curr]
        a_j = cell_to_var[curr]

        # 1. Add current cell to the objective (Minimize sum(a_j))
        obj += a_j

        # Track instantiation time (mocked)
        Utils.ilpTimes[1] += model_instance.instantiationTime.get(curr, 0.0)

        edges = model_instance.cell2Edge.get(curr)
        if edges is not None:
            for edge in edges:
                edge_counter += 1

                # 2. Add Edge Variables (bi, hij)
                # bi: Edge cover variable
                # hij: Link variable (a_j AND bi)
                b_i = grb_model.addVar(vtype = GRB.BINARY, name = f"b{edge_counter}")
                h_ij = grb_model.addVar(vtype = GRB.BINARY, name = f"h{edge_counter}_{curr_id}")

                # Constraints to enforce a_j = b_i (if curr is deleted, the edge is activated/must be covered)
                # Equivalent to: a_j = h_ij AND b_i = h_ij
                grb_model.addConstr(a_j == h_ij, name = f"aj_eq_hij_{edge_counter}_{curr_id}")
                grb_model.addConstr(b_i == h_ij, name = f"bi_eq_hij_{edge_counter}_{curr_id}")

                t_ji_vars: List[gp.Var] = []

                # 3. Process Cells in the Edge (The 'tail' cells of the rule)
                for cell_l in edge:  # cell_l is a Cell object
                    t_id: int
                    a_cell: gp.Var

                    # Ensure the cell variable 'a_cell' exists. If not, create it.
                    if cell_l not in cell_to_id:
                        t_id = max_id
                        cell_to_id[cell_l] = max_id
                        max_id += 1
                        a_cell = grb_model.addVar(vtype = GRB.BINARY, name = f"a{t_id}")
                        cell_to_var[cell_l] = a_cell
                    else:
                        t_id = cell_to_id[cell_l]
                        a_cell = cell_to_var[cell_l]

                    # Variable t_li: links the deletion of cell_l to the edge cover sum
                    t_li = grb_model.addVar(vtype = GRB.BINARY, name = f"t{edge_counter}{t_id}")
                    t_ji_vars.append(t_li)

                    # Constraint: t_li = a_cell (Equivalent to Java's GRB.EQUAL constraint)
                    grb_model.addConstr(t_li == a_cell,
                                        name = f"tji_eq_aCell_{edge_counter}_{t_id}")

                    # If this cell is new, add it to the queue to process its outgoing dependencies
                    if cell_l not in instantiated_cells:
                        cells_to_visit.append(cell_l)
                        instantiated_cells.add(cell_l)

                # 4. Set Cover Constraint
                # Sum(t_li) >= b_i: If the edge is activated (b_i=1), at least one cell in the tail must be deleted.
                grb_model.addConstr(gp.quicksum(t_ji_vars) >= b_i,
                                    name = f"Set_Cover_{edge_counter}")

    # --- Update Model Construction Time ---
    stop_build_time = time.perf_counter_ns()
    Utils.ilpTimes[2] += (stop_build_time - start_build_time) / 1e9  # Convert ns to seconds

    # --- Optimize ---
    grb_model.setObjective(obj, GRB.MINIMIZE)

    start_optim_time = time.perf_counter_ns()
    grb_model.optimize()
    stop_optim_time = time.perf_counter_ns()

    # --- Check Status and Extract Solution ---
    if grb_model.status == GRB.INFEASIBLE:
        # Check if the model is infeasible (status code 3 in Gurobi)
        raise gp.GurobiError(f"Infeasible model. Status code: {grb_model.status}")

    # Extract the solution by checking the value of all 'a' variables
    for cell, var in cell_to_var.items():
        # GRB.DoubleAttr.X is the solution value
        if var.X > 0.5:  # Checks if the binary variable is 1
            to_delete.add(cell)

    # Clean up Gurobi model (equivalent to grbModel.dispose() in Java)
    grb_model.dispose()

    # Update optimization and counts (mocked)
    Utils.ilpTimes[3] += (stop_optim_time - start_optim_time) / 1e9
    Utils.ilpCounts[1] += len(
        cell_to_id) - 1  # Total cells instantiated - 1 (the initial deleted cell)

    # NOTE: Memory measurement (measureILPMemory) is highly platform/language specific.
    # The Java code implements a heuristic based on object sizes. You would need to
    # implement a Python-specific memory tracking function if needed.

    return to_delete


# --- 3. Execution Code ---

# 1. Create Cells
CA = Cell("T1", "k1")
CB = Cell("T2", "k2")
CC = Cell("T3", "k3")
CD = Cell("T4", "k4")
CE = Cell("T5", "k5")

# 2. Populate the InstantiatedModel (The Dependency Graph)
model = InstantiatedModel()

# Rule 1: CA -> {CB, CC}
model.cell2Edge[CA] = [[CB, CC]]

# Rule 2: CB -> {CD, CE}
model.cell2Edge[CB] = [[CD, CE]]

# Rule 3: CC -> {CD}
if CC not in model.cell2Edge:
    model.cell2Edge[CC] = []
model.cell2Edge[CC].append([CD])

# 3. Create Gurobi Environment and Run Solver
# NOTE: This requires a valid Gurobi license to run.
try:
    # Use a temporary environment for the test
    gurobi_env = gp.Env(empty = True)
    gurobi_env.start()

    print("--- Starting ILP Solver ---")
    start_time = time.time()

    # Run the function
    deleted_cells = ilp_approach_gurobi(model, CA, gurobi_env)

    end_time = time.time()
    print("--- Solver Finished ---")

    # 4. Print Output and Metrics
    print(f"\n✅ Optimal Deletion Set (Size {len(deleted_cells)}):")
    for cell in sorted(list(deleted_cells), key = lambda c: c.key):
        print(f"   -> {cell}")

    print("\n--- Summary Statistics ---")
    print(f"Total time: {end_time - start_time:.4f} seconds")
    print(f"Model Build Time: {Utils.ilpTimes[2]:.4f} seconds")
    print(f"Optimization Time: {Utils.ilpTimes[3]:.4f} seconds")
    print(
        f"Total Cells Instantiated/Added to ILP: {Utils.ilpCounts[1] + 1}")  # +1 for the initial deleted cell

except gp.GurobiError as e:
    print(f"\n❌ Gurobi Error during execution: {e.message}")
    print("Please ensure your Gurobi license is valid and configured correctly.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
finally:
    # Ensure the environment is shut down
    if 'gurobi_env' in locals() and gurobi_env.getAttr("IsStarted") == 1:
        gurobi_env.dispose()