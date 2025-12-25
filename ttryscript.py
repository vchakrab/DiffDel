#!/usr/bin/env python3
from typing import List, Tuple, Set

# -------------------------
# Helpers
# -------------------------
def he(s: str) -> Tuple[str, ...]:
    return tuple(s)

def pretty_path(hyperedges, path: List[int]) -> str:
    return " -> ".join(["{" + "".join(hyperedges[i]) + "}" for i in path])

def compute_vertices(hyperedges: List[Tuple[str, ...]]) -> Set[str]:
    V = set()
    for e in hyperedges:
        V.update(e)
    return V

# ------------------------------------------------------------
# Deterministic initial known set S (MASK-INDEPENDENT):
# paper-style "adversary observes everything except target"
# S = V \ {target}
# ------------------------------------------------------------
def compute_initial_known(hyperedges: List[Tuple[str, ...]],
                          target_cell: str) -> Set[str]:
    V = compute_vertices(hyperedges)
    return V - {target_cell}

# ------------------------------------------------------------
# Path extraction (forward-chaining DFS):
# Start from initial_known, repeatedly apply any hyperedge e where
# exactly 1 cell is unknown; infer that cell, recurse.
# Record a path when inferred == target_cell.
#
# Output: paths as list of edge indices, in inference order.
# ------------------------------------------------------------
def find_inference_paths_str(hyperedges: List[Tuple[str, ...]],
                             target_cell: str,
                             initial_known: Set[str]) -> List[List[int]]:
    all_paths: List[List[int]] = []
    seen_paths = set()

    def dfs(known: Set[str], path: List[int], used_edges: Set[int]):
        for edge_idx, edge in enumerate(hyperedges):
            if edge_idx in used_edges:
                continue

            edge_set = set(edge)
            unknown = edge_set - known

            if len(unknown) == 1:
                inferred = next(iter(unknown))

                new_known = set(known)
                new_known.add(inferred)

                new_path = path + [edge_idx]
                new_used = set(used_edges)
                new_used.add(edge_idx)

                if inferred == target_cell:
                    t = tuple(new_path)
                    if t not in seen_paths:
                        seen_paths.add(t)
                        all_paths.append(new_path)
                    continue

                dfs(new_known, new_path, new_used)

    dfs(set(initial_known), [], set())
    return all_paths

# ------------------------------------------------------------
# Active path filtering (IsBlocked-style):
# A path is blocked if at ANY step the edge requires a masked cell
# to infer the next cell.
#
# IMPORTANT: known_so_far starts from initial_known (mask-independent)
# ------------------------------------------------------------
def filter_active_paths_str(hyperedges: List[Tuple[str, ...]],
                            paths: List[List[int]],
                            mask: Set[str],
                            initial_known: Set[str]) -> List[List[int]]:
    mask = set(mask)
    active_paths: List[List[int]] = []

    for path in paths:
        known_so_far = set(initial_known)
        is_blocked = False

        for edge_idx in path:
            edge = set(hyperedges[edge_idx])
            unknown_in_edge = edge - known_so_far

            # If this edge isn't actually applicable at this step, path is invalid
            if len(unknown_in_edge) != 1:
                is_blocked = True
                break

            inferred = next(iter(unknown_in_edge))
            required = edge - {inferred}

            # IsBlocked: if any required cell is masked, edge can't fire
            if required & mask:
                is_blocked = True
                break

            known_so_far.add(inferred)

        if not is_blocked:
            active_paths.append(path)

    return active_paths

# -------------------------
# Toy runner
# -------------------------
def run_case(hyperedges: List[Tuple[str, ...]], target: str, mask: Set[str]):
    V = sorted(compute_vertices(hyperedges))
    S = compute_initial_known(hyperedges, target)          # <-- fixed, mask-independent
    total_paths = find_inference_paths_str(hyperedges, target, S)
    active_paths = filter_active_paths_str(hyperedges, total_paths, mask, S)

    print("==================================================")
    print(f"Hyperedges: {[ ''.join(e) for e in hyperedges ]}")
    print(f"Vertices V: {V}")
    print(f"Target: {target}")
    print(f"Initial known S = V \\ {{target}}: {sorted(S)}")
    print(f"Mask M: {sorted(mask)}")
    print()
    print(f"Total paths (mask-independent): {len(total_paths)}")
    for p in total_paths:
        print("  -", pretty_path(hyperedges, p), "| idx:", p)
    print()
    print(f"Active paths (depends on mask): {len(active_paths)}")
    for p in active_paths:
        print("  -", pretty_path(hyperedges, p), "| idx:", p)
    print("==================================================\n")

if __name__ == "__main__":
    # Your sanity check: "abcd" and "def", target = "c"
    hyperedges = [he("abcd"), he("def")]

    # No mask
    run_case(hyperedges, target="c", mask=set())

    # Mask that blocks the final inference step for c (edge abcd requires a,b,d)
    run_case(hyperedges, target="c", mask={"d", "f"})
    run_case(hyperedges, target="c", mask={"d"})
    run_case(hyperedges, target="c", mask={"f"})
