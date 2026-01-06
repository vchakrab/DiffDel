# test_figure_example_edgechains.py
import pytest

# Change this import to wherever you put the functions
# (active, enumerate_chains, leakage)
import global_functions as m


def build_figure_edges():
    # edge_id -> (verts_set, weight)
    return {
        "ea": ({"Gen", "Age"}, 0.90),
        "eb": ({"Gen", "BMI"}, 0.92),
        "ec": ({"Age", "Res"}, 0.88),
        "e1": ({"Res", "Diag"}, 0.95),
        "e2": ({"Age", "BMI", "Diag"}, 0.85),
    }


TARGET = "Diag"

FIGURE_MASKS = [
    ("M1", set()),
    ("M2", {"Res"}),
    ("M3", {"Age"}),
    ("M4", {"Res", "Age"}),
    ("M5", {"Gen"}),
    ("M6", {"Age", "BMI"}),
    ("M7", {"Res", "Age", "BMI"}),
    ("M8", {"Gen", "Age", "BMI", "Res"}),
]

# These match the output of YOUR enumerate_chains()+leakage() code exactly.
EXPECTED_L = {
    "M1": 0.9925,
    "M2": 0.8500,
    "M3": 0.98825,
    "M4": 0.87786,
    "M5": 0.9925,
    "M6": 0.9500,
    "M7": 0.7524,
    "M8": 0.0000,
}


def _chain_weight(chain, edges):
    out = 1.0
    for eid in chain:
        out *= float(edges[eid][1])
    return out


@pytest.mark.parametrize("name,mask", FIGURE_MASKS)
def test_figure_example_print_all(name, mask):
    edges = build_figure_edges()

    print("\n" + "=" * 80)
    print(f"{name}: mask={sorted(mask)} target={TARGET}")

    chains = m.enumerate_chains(mask=mask, target=TARGET, edges=edges)

    # Sort chains for stable printing: by length then lexicographically
    chains_sorted = sorted(chains, key=lambda c: (len(c), tuple(c)))

    print(f"Chains ({len(chains_sorted)}):")
    for i, c in enumerate(chains_sorted):
        w = _chain_weight(c, edges)
        print(f"  {i:02d}: {c}  weight={w:.6f}")

    L = float(m.leakage(mask=mask, target=TARGET, edges=edges))
    print(f"Leakage L={L:.6f}")

    assert L == pytest.approx(EXPECTED_L[name], abs=1e-6)
