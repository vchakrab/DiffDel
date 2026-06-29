"""
Builds a BFS spanning tree of the attribute co-occurrence graph for each
dataset's denial constraints (rooted at the dataset's target attribute),
then prints rho1 = |E| / |V| for each dataset.

  |V| = number of unique attributes appearing in any DC predicate
  |E| = number of unique unordered attribute pairs that co-occur
        in at least one DC (i.e. edges of the dependency graph)

A small hand-verifiable "example" dataset (attributes a/b/c/d/e) is included
first so you can sanity-check that build_bfs_tree() is working correctly
before trusting it on the real datasets.

Usage:
    python3 rho1_bfs.py
"""

import ast
import re
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# TINY HAND-VERIFIABLE EXAMPLE
# ---------------------------------------------------------------------------
# Edges implied by these DCs (each DC -> a clique over its attributes,
# but every DC below only has 2 attributes, so each DC = 1 edge):
#
#   DC1: a == a  ,  b == b   ->  edge a-b
#   DC2: a == a  ,  c == c   ->  edge a-c
#   DC3: b == b  ,  d == d   ->  edge b-d
#   DC4: c == c  ,  e == e   ->  edge c-e
#   DC5: d == d  ,  e == e   ->  edge d-e   (creates a cycle b-d-e-c-a-b)
#
# Graph:
#         a
#        / \
#       b   c
#       |   |
#       d---e
#
# V = {a,b,c,d,e}            -> |V| = 5
# E = {a-b,a-c,b-d,c-e,d-e}  -> |E| = 5
# rho1 = |E|/|V| = 5/5 = 1.0
#
# BFS spanning tree rooted at 'a' (neighbors visited in sorted order):
#   a [ROOT]
#     b
#       d [leaf]
#     c
#       e [leaf]
#
# Note: edge d-e exists in the graph (counted in |E|) but is NOT a tree
# edge, since both d and e are already reached via b and c respectively
# by the time BFS would consider it. This is the kind of thing this
# example lets you check by eye.
EXAMPLE_DCS = [
    [("t1.a", "==", "t2.a"), ("t1.b", "==", "t2.b")],
    [("t1.a", "==", "t2.a"), ("t1.c", "==", "t2.c")],
    [("t1.b", "==", "t2.b"), ("t1.d", "==", "t2.d")],
    [("t1.c", "==", "t2.c"), ("t1.e", "==", "t2.e")],
    [("t1.d", "==", "t2.d"), ("t1.e", "==", "t2.e")],
]
EXAMPLE_TARGET = "a"

# ---------------------------------------------------------------------------
# CONFIG: map each dataset name to its parsed-DC file and its target attribute
# ---------------------------------------------------------------------------
FILES = {
    "airport":  "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topAirportDCs_parsed.py",
    "hospital": "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topHospitalDCs_parsed.py",
    "tax":      "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topTaxDCs_parsed.py",
    "adult":    "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topAdultDCs_parsed.py",
    "flight":   "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topFlightDCs_parsed.py",
}

TARGET_ATTR = {
    "airport":  "continent",
    "hospital": "ProviderNumber",
    "tax":      "city",
    "adult":    "education",
    "flight":   "FlightDate",
}


def extract_attr(predicate):
    """Strip the 't1.'/'t2.' prefix off a predicate's column reference."""
    return predicate[0].split(".")[1]


def load_dcs(path):
    """Read a *_parsed.py file and return its denial_constraints list."""
    with open(path) as f:
        src = f.read()
    match = re.search(r"denial_constraints\s*=\s*(\[.*\])", src, re.DOTALL)
    return ast.literal_eval(match.group(1))


def build_bfs_tree(dcs, target):
    """
    Build the attribute co-occurrence graph from the DC list, then run BFS
    from `target` to build a spanning tree.

    Returns: (V, E, children, depth, visited)
      V        = total number of unique attributes (whole graph)
      E        = total number of unique co-occurrence edges (whole graph)
      children = dict: node -> list of child nodes in the BFS tree
      depth    = dict: node -> depth in the BFS tree
      visited  = set of nodes reachable from target (i.e. in the tree)
    """
    all_attrs = set()
    for dc in dcs:
        for pred in dc:
            all_attrs.add(extract_attr(pred))

    # Build undirected adjacency from co-occurrence within each DC
    edge_set = set()
    adj = defaultdict(set)
    for dc in dcs:
        attrs_in_dc = list({extract_attr(p) for p in dc})
        for i in range(len(attrs_in_dc)):
            for j in range(i + 1, len(attrs_in_dc)):
                a, b = sorted([attrs_in_dc[i], attrs_in_dc[j]])
                edge_set.add((a, b))
                adj[a].add(b)
                adj[b].add(a)

    V = len(all_attrs)
    E = len(edge_set)

    # BFS spanning tree rooted at target
    visited = {target}
    children = defaultdict(list)
    depth = {target: 0}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for nbr in sorted(adj[node]):  # sorted -> deterministic tree
            if nbr not in visited:
                visited.add(nbr)
                children[node].append(nbr)
                depth[nbr] = depth[node] + 1
                queue.append(nbr)

    return V, E, children, depth, visited


def print_tree(node, children, target, depth, indent=0):
    tag = " [ROOT]" if node == target else (" [leaf]" if not children[node] else "")
    # print("  " * indent + node + tag)
    for child in children[node]:
        print_tree(child, children, target, depth, indent + 1)


def run_example():
    """Run build_bfs_tree on the tiny a/b/c/d/e example and print everything
    needed to verify it by hand against the diagram in the comments above."""
    # print("=" * 60)
    # print("EXAMPLE (a/b/c/d/e) -- sanity check")
    # print("=" * 60)

    V, E, children, depth, visited = build_bfs_tree(EXAMPLE_DCS, EXAMPLE_TARGET)

    # print("children:", dict(children))
    # print("depth:", depth)
    # print("visited:", visited)
    # print(f"|V| = {V}, |E| = {E}")
    # print(f"rho1 = {E}/{V} = {E / V:.6f}")
    # print()
    # print("BFS tree:")
    print_tree(EXAMPLE_TARGET, children, EXAMPLE_TARGET, depth)
    # print()
    # print("Expected: |V|=5, |E|=5, rho1=1.0, tree = a->[b,c], b->[d], c->[e]")
    # print("=" * 60)
    # print()


def main():
    run_example()

    results = {}
    for name, path in FILES.items():
        dcs = load_dcs(path)
        target = TARGET_ATTR[name]
        V, E, children, depth, visited = build_bfs_tree(dcs, target)

        rho1 = E / V
        # print("children", target, children)
        # print("depth", target, depth)
        # print("target", target, visited)

        # Uncomment to also see the BFS tree structure for this dataset:
        # print(f"--- {name} (root={target}) ---")
        # print_tree(target, children, target, depth)
        # print()

        results[name] = rho1

    # Output: just rho1 for each dataset
    for name, rho1 in results.items():
        pass
        # print(f"{name}: rho1 = {rho1:.6f}")


if __name__ == "__main__":
    main()