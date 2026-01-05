import unittest

from baseline_deletion_3 import *

# class Hypergraph:
#     def __init__(self, edges):
#         # edges: List[Tuple[Set[str], float]]
#         self.edges = edges
#
# class TestBaseline(unittest.TestCase):
#     edges = [
#         ({"a", "b", "c", "d"}, 0.5),  # connects target c to chain[-1] = d
#         ({"d", "e", "f"}, 0.2),
#         ({"f", "g", "h"}, 0.3),
#     ]
#
#     hg = Hypergraph(edges)
#     target_cell = "c"
#     def test_compute_chain_weight(self):
#     # Hyperedges
#         chain = ["a", "b", "d"]
#         masked_cells = set()
#
#         edges = [
#         ({"a", "b", "c", "d"}, 0.5),  # connects target c to chain[-1] = d
#         ({"d", "e", "f"}, 0.2),
#         ({"f", "g", "h"}, 0.3),
#         ]
#
#         hg = Hypergraph(edges)
#         target_cell = "c"
#
#         self.assertEqual(compute_chain_weight(
#             chain,
#             hypergraph=hg,
#             masked_cells=masked_cells,
#             target_cell=target_cell), 0.5)
#
#     def test_compute_simple(self):
#         edges = [
#             ({"a", "b", "c", "d"}, 0.5),  # connects target c to chain[-1] = d
#             ({"d", "e", "f"}, 0.2),
#             ({"f", "g", "h"}, 0.3),
#         ]
#
#         hg = Hypergraph(edges)
#         target_cell = "c"
#         chain = ["a", "b", "d"]
#         masked_cells = {"a"}
#         self.assertEqual(compute_chain_weight(
#             chain,
#             hypergraph = hg,
#             masked_cells = masked_cells,
#             target_cell = target_cell), 0.5)
#
#     def test_compute_harder(self):
#         edges = [
#             ({"a", "b", "c", "d"}, 0.5),  # connects target c to chain[-1] = d
#             ({"d", "e", "f"}, 0.2),
#             ({"f", "g", "h"}, 0.3),
#         ]
#
#         hg = Hypergraph(edges)
#         target_cell = "c"
#         chain = ["a", "b", "d", "e", "f", "g", "h"]
#         masked_cells = {"a", "d", "f"}
#         self.assertEqual(compute_chain_weight(
#             chain,
#             hypergraph = hg,
#             masked_cells = masked_cells,
#             target_cell = target_cell), 0.03)
# test_construct_local_hypergraph.py
#
# Run with:
#   python -m unittest -v test_construct_local_hypergraph.py
#
# IMPORTANT:
# - Set MODULE_UNDER_TEST to the module where your function lives.
# - Your module must define: construct_local_hypergraph, incident_rdrs, instantiate_rdr, Hypergraph
#
# These tests heavily patch incident_rdrs/instantiate_rdr to focus on *graph construction logic*
# (frontier expansion, deduping by rdr_idx, termination on cycles, etc.).

import unittest
from unittest.mock import patch, call
# baseline3_test.py
#
# Run with:
#   pytest -q -s baseline3_test.py
#   python -m unittest -v baseline3_test.py
#
# Assumes baseline_deletion_3 defines:
#   - construct_local_hypergraph(target_cell, rdrs, weights, mode="MAX") -> Hypergraph
#   - incident_rdrs(cell, rdrs) -> List[int]
#   - instantiate_rdr(rdr, w, mode) -> Iterable[Tuple[Iterable[str], float]]
#   - Hypergraph object exposes:
#       H.vertices : Set[str]
#       H.edges    : List[Tuple[Set[str], float]]

import importlib
import unittest
from unittest.mock import patch

MODULE_UNDER_TEST = "baseline_deletion_3"


class TestConstructLocalHypergraph(unittest.TestCase):
    def _import_under_test(self):
        return importlib.import_module(MODULE_UNDER_TEST)

    # --------------------------
    # Pretty printing
    # --------------------------
    def print_case(self, title, rdrs, weights, H):
        print(f"\n=== {title} ===")

        print("\nRDRs (index -> rdr, weight):")
        if not rdrs:
            print("  (none)")
        else:
            for i, (r, w) in enumerate(zip(rdrs, weights)):
                print(f"  {i}: rdr={r}  w={w}")

        print("\nHypergraph:")
        verts = sorted(getattr(H, "vertices", []))
        print("  Vertices:", verts)

        print("  Edges:")
        edges = getattr(H, "edges", [])
        if not edges:
            print("    (none)")
        else:
            # Sort for stable output
            def edge_key(e):
                ev, ew = e
                return (ew, tuple(sorted(ev)))
            for ev, ew in sorted(edges, key=edge_key):
                print(f"    w={ew}  verts={sorted(list(ev))}")

        print("=============================\n")

    def _mk_debug_incident(self, fn, label=""):
        """Wrap an incident_rdrs side_effect to print calls/returns."""
        def _wrapped(c, _rdrs):
            out = fn(c, _rdrs)
            print(f"[incident_rdrs{(':'+label) if label else ''}] cell={c} -> rdr_idxs={out}")
            return out
        return _wrapped

    def _mk_debug_instantiate(self, fn, label=""):
        """Wrap an instantiate_rdr side_effect to print calls/returns."""
        def _wrapped(rdr, w, mode):
            edges = fn(rdr, w, mode)
            pretty = [(list(v), ew) for (v, ew) in edges]
            print(f"[instantiate_rdr{(':'+label) if label else ''}] rdr={rdr} w={w} mode={mode} -> {pretty}")
            return edges
        return _wrapped

    # --------------------------
    # Helpers (normalize asserts)
    # --------------------------
    def assertHasEdge(self, H, verts, w):
        self.assertTrue(hasattr(H, "edges"), "Hypergraph must expose .edges")
        self.assertIn((set(verts), w), H.edges,
                      msg=f"Expected edge {(set(verts), w)} in H.edges={H.edges}")

    def assertVertices(self, H, verts):
        self.assertTrue(hasattr(H, "vertices"), "Hypergraph must expose .vertices")
        self.assertEqual(H.vertices, set(verts))

    def print_unreachable_summary(self, title, rdrs, weights, H, used_rdr_idxs):
        print(f"\n=== {title} ===\n")

        print("RDRs (index -> rdr, weight):")
        for i, (r, w) in enumerate(zip(rdrs, weights)):
            print(f"  {i}: rdr={r}  w={w}")

        used = set(used_rdr_idxs)
        ignored = set(range(len(rdrs))) - used

        print("\nUsed RDRs:")
        if not used:
            print("  (none)")
        else:
            for i in sorted(used):
                print(f"  {i} -> {rdrs[i][0]}")

        print("\nIgnored (unreachable) RDRs:")
        if not ignored:
            print("  (none)")
        else:
            for i in sorted(ignored):
                print(f"  {i} -> {rdrs[i][0]}")

        print("\nHypergraph:")
        print("  Vertices:", sorted(H.vertices))

        print("\n  Edges:")
        if not H.edges:
            print("    (none)")
        else:
            for verts, w in sorted(H.edges, key = lambda e: (e[1], sorted(e[0]))):
                print(f"    w={w}  verts={sorted(verts)}")

        print("\n========================================\n")

    # --------------------------
    # Tests
    # --------------------------
    def test_no_rdrs(self):
        m = self._import_under_test()

        target = "C"
        rdrs = []
        weights = []

        with patch.object(m, "incident_rdrs", autospec=True, return_value=[]), \
             patch.object(m, "instantiate_rdr", autospec=True) as inst_mock:
            H = m.construct_local_hypergraph(target, rdrs, weights)

            inst_mock.assert_not_called()

            self.print_case("test_no_rdrs", rdrs, weights, H)

            self.assertVertices(H, {target})
            self.assertEqual(len(H.edges), 0)

    def test_single_rdr_one_edge(self):
        m = self._import_under_test()

        target = "C"
        rdrs = [("dummy_rdr_0",)]
        weights = [0.5]

        def incident_logic(c, _rdrs):
            return [0] if c == target else []

        def inst_logic(rdr, w, mode):
            self.assertEqual(rdr, rdrs[0])
            self.assertEqual(w, 0.5)
            self.assertEqual(mode, "MAX")
            return [
                (("A", target), w),
            ]

        with patch.object(m, "incident_rdrs", autospec=True,
                          side_effect=self._mk_debug_incident(incident_logic, "single")), \
             patch.object(m, "instantiate_rdr", autospec=True,
                          side_effect=self._mk_debug_instantiate(inst_logic, "single")):
            H = m.construct_local_hypergraph(target, rdrs, weights, mode="MAX")

            self.print_case("test_single_rdr_one_edge", rdrs, weights, H)

            self.assertVertices(H, {"A", target})
            self.assertEqual(len(H.edges), 1)
            self.assertHasEdge(H, {"A", target}, 0.5)

    def test_nested_rdr_expansion_two_hops(self):
        m = self._import_under_test()

        target = "C"
        rdrs = [("rdr0",), ("rdr1",)]
        weights = [0.5, 0.2]

        def incident_logic(c, _rdrs):
            if c == target:
                return [0]
            if c == "A":
                return [1]
            return []

        def inst_logic(rdr, w, mode):
            if rdr == rdrs[0]:
                return [(("A", target), w)]
            if rdr == rdrs[1]:
                return [(("B", "A"), w)]
            raise AssertionError("Unexpected RDR passed to instantiate_rdr")

        with patch.object(m, "incident_rdrs", autospec=True,
                          side_effect=self._mk_debug_incident(incident_logic, "nested")), \
             patch.object(m, "instantiate_rdr", autospec=True,
                          side_effect=self._mk_debug_instantiate(inst_logic, "nested")):
            H = m.construct_local_hypergraph(target, rdrs, weights)

            self.print_case("test_nested_rdr_expansion_two_hops", rdrs, weights, H)

            self.assertVertices(H, {target, "A", "B"})
            self.assertEqual(len(H.edges), 2)
            self.assertHasEdge(H, {"A", target}, 0.5)
            self.assertHasEdge(H, {"B", "A"}, 0.2)

    def test_multiple_rdrs_for_same_cell(self):
        m = self._import_under_test()

        target = "C"
        rdrs = [("rdr0",), ("rdr1",)]
        weights = [0.5, 0.3]

        def incident_logic(c, _rdrs):
            return [0, 1] if c == target else []

        def inst_logic(rdr, w, mode):
            if rdr == rdrs[0]:
                return [(("A", target), w)]
            if rdr == rdrs[1]:
                return [(("B", target, "D"), w)]  # 3-vertex hyperedge
            raise AssertionError("Unexpected RDR")

        with patch.object(m, "incident_rdrs", autospec=True,
                          side_effect=self._mk_debug_incident(incident_logic, "multi")), \
             patch.object(m, "instantiate_rdr", autospec=True,
                          side_effect=self._mk_debug_instantiate(inst_logic, "multi")):
            H = m.construct_local_hypergraph(target, rdrs, weights)

            self.print_case("test_multiple_rdrs_for_same_cell", rdrs, weights, H)

            self.assertVertices(H, {target, "A", "B", "D"})
            self.assertEqual(len(H.edges), 2)
            self.assertHasEdge(H, {"A", target}, 0.5)
            self.assertHasEdge(H, {"B", target, "D"}, 0.3)

    def test_dedup_same_rdr_idx_even_if_returned_multiple_times(self):
        m = self._import_under_test()

        target = "C"
        rdrs = [("rdr0",), ("rdr1",)]
        weights = [0.5, 0.2]

        def incident_logic(c, _rdrs):
            return [0, 0, 1] if c == target else []

        def inst_logic(rdr, w, mode):
            if rdr == rdrs[0]:
                return [(("A", target), w)]
            if rdr == rdrs[1]:
                return [(("B", target), w)]
            raise AssertionError("Unexpected RDR")

        with patch.object(m, "incident_rdrs", autospec=True,
                          side_effect=self._mk_debug_incident(incident_logic, "dedup")), \
             patch.object(m, "instantiate_rdr", autospec=True,
                          side_effect=self._mk_debug_instantiate(inst_logic, "dedup")) as inst_mock:
            H = m.construct_local_hypergraph(target, rdrs, weights)

            self.print_case("test_dedup_same_rdr_idx_even_if_returned_multiple_times", rdrs, weights, H)

            self.assertEqual(inst_mock.call_count, 2)

            got = [(args[0], args[1], args[2]) for (args, _kwargs) in inst_mock.call_args_list]
            self.assertCountEqual(
                got,
                [
                    (rdrs[0], 0.5, "MAX"),
                    (rdrs[1], 0.2, "MAX"),
                ],
            )

            self.assertVertices(H, {target, "A", "B"})
            self.assertEqual(len(H.edges), 2)
            self.assertHasEdge(H, {"A", target}, 0.5)
            self.assertHasEdge(H, {"B", target}, 0.2)

    def test_cycle_terminates_and_does_not_infinite_loop(self):
        m = self._import_under_test()

        target = "C"
        rdrs = [("rdr0",), ("rdr1",)]
        weights = [0.5, 0.4]

        def incident_logic(c, _rdrs):
            if c == target:
                return [0]
            if c == "A":
                return [1]
            return []

        def inst_logic(rdr, w, mode):
            if rdr == rdrs[0]:
                return [(("A", target), w)]
            if rdr == rdrs[1]:
                return [(("A", target), w)]  # closes cycle
            raise AssertionError("Unexpected RDR")

        with patch.object(m, "incident_rdrs", autospec=True,
                          side_effect=self._mk_debug_incident(incident_logic, "cycle")), \
             patch.object(m, "instantiate_rdr", autospec=True,
                          side_effect=self._mk_debug_instantiate(inst_logic, "cycle")):
            H = m.construct_local_hypergraph(target, rdrs, weights)

            self.print_case("test_cycle_terminates_and_does_not_infinite_loop", rdrs, weights, H)

            self.assertVertices(H, {target, "A"})
            self.assertEqual(len(H.edges), 2)
            self.assertHasEdge(H, {"A", target}, 0.5)
            self.assertHasEdge(H, {"A", target}, 0.4)

    def test_unreachable_rdrs_are_not_used(self):
        m = self._import_under_test()

        target = "C"
        rdrs = [("rdr0",), ("rdr1",), ("rdr2_unreachable",), ("rdr3_unreachable",)]
        weights = [0.5, 0.2, 0.9, 0.8]

        def incident_logic(c, _rdrs):
            if c == target:
                return [0]
            if c == "A":
                return [1]
            return []

        def inst_logic(rdr, w, mode):
            if rdr == rdrs[0]:
                return [(("A", target), w)]
            if rdr == rdrs[1]:
                return [(("B", "A"), w)]
            raise AssertionError(f"instantiate_rdr called for unreachable rdr: {rdr}")

        with patch.object(m, "incident_rdrs", autospec = True, side_effect = incident_logic), \
                patch.object(m, "instantiate_rdr", autospec = True,
                             side_effect = inst_logic) as inst_mock:

            H = m.construct_local_hypergraph(target, rdrs, weights)

            # ✅ CLEAN OUTPUT (no spam)
            self.print_unreachable_summary(
                title = "test_unreachable_rdrs_are_not_used",
                rdrs = rdrs,
                weights = weights,
                H = H,
                used_rdr_idxs = {0, 1},
            )

            # instantiate_rdr should be called ONLY for rdr0 and rdr1
            self.assertEqual(inst_mock.call_count, 2)

            self.assertVertices(H, {target, "A", "B"})
            self.assertEqual(len(H.edges), 2)
            self.assertHasEdge(H, {"A", target}, 0.5)
            self.assertHasEdge(H, {"A", "B"}, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
