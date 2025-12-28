"""
PLOTS (Set 1): Core tradeoff plots for your inferable-mode Exponential Mechanism.

Generates:
  (A) Leakage–Deletion tradeoff curve (points from epsilon sweep)
  (B1) Leakage vs epsilon
  (B2) |M| vs epsilon
  (Optional) same plots using surprisal:  -log(1 - Leakage)

Assumptions:
- You provide:
    hyperedges: List[Set[cell]]
    weights:    List[float]  (parallel)
    target:     cell
- Inference zone I(c*) = union of hyperedges minus target (as you requested).
- EM enumerates ALL subsets of I(c*): candidates = 2^{|I|}.
  (Feasible only for small |I|.)

Outputs:
- PNG files saved to ./plots/
- A CSV summary saved to ./plots/epsilon_sweep_summary.csv

Usage:
  1) Fill in hyperedges/weights/target in main()
  2) Run: python plot_set1_em.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import deque
import os
import math
import random
import pandas as pd
import matplotlib.pyplot as plt

Cell = Any


# -----------------------------
# Model (inferable semantics) + leakage
# -----------------------------

def inference_zone_union(target: Cell, hyperedges: Sequence[Iterable[Cell]]) -> List[Cell]:
    z: Set[Cell] = set()
    for e in hyperedges:
        z |= set(e)
    z.discard(target)
    return sorted(z, key=lambda x: repr(x))


@dataclass(frozen=True)
class Edge:
    verts: Tuple[int, ...]
    w: float


class InferableLeakageModel:
    def __init__(self, hyperedges: Sequence[Iterable[Cell]], weights: Sequence[float], target: Cell):
        if len(hyperedges) != len(weights):
            raise ValueError("hyperedges and weights must have the same length")

        V: Set[Cell] = {target}
        hedges: List[Set[Cell]] = []
        for e in hyperedges:
            s = set(e)
            if len(s) < 2:
                raise ValueError(f"Each hyperedge must contain >=2 cells, got {s}")
            hedges.append(s)
            V |= s

        ordered = sorted(V, key=lambda x: repr(x))
        self.cell_to_id: Dict[Cell, int] = {c: i for i, c in enumerate(ordered)}
        self.id_to_cell: List[Cell] = ordered
        self.n = len(ordered)
        self.target = target
        self.tid = self.cell_to_id[target]

        self.edges: List[Edge] = []
        for i, (s, w) in enumerate(zip(hedges, weights)):
            w = float(w)
            if not (0.0 < w <= 1.0):
                raise ValueError(f"Edge {i} weight must be in (0,1], got {w}")
            verts = tuple(sorted(self.cell_to_id[c] for c in s))
            self.edges.append(Edge(verts=verts, w=w))

        self.inc: List[List[int]] = [[] for _ in range(self.n)]
        for ei, e in enumerate(self.edges):
            for v in e.verts:
                self.inc[v].append(ei)

        neigh_sets: List[Set[int]] = [set() for _ in range(self.n)]
        for e in self.edges:
            vs = e.verts
            for v in vs:
                neigh_sets[v].update(vs)
        for v in range(self.n):
            neigh_sets[v].discard(v)
        self.neigh: List[Tuple[int, ...]] = [tuple(sorted(s)) for s in neigh_sets]

        self.channel_edges: List[int] = [ei for ei, e in enumerate(self.edges) if self.tid in e.verts]
        self.zone: List[Cell] = inference_zone_union(target, hyperedges)

    def _attempt(self, e: Edge, infer_v: int, p: List[float]) -> float:
        prod = 1.0
        for u in e.verts:
            if u == infer_v:
                continue
            prod *= p[u]
            if prod == 0.0:
                return 0.0
        a = e.w * prod
        if a <= 0.0:
            return 0.0
        if a >= 1.0:
            return 1.0
        return a

    def _recompute_pv(self, v: int, observed: List[bool], p: List[float]) -> float:
        if observed[v]:
            return 1.0
        prod_fail = 1.0
        for ei in self.inc[v]:
            e = self.edges[ei]
            a = self._attempt(e, v, p)
            if a == 0.0:
                continue
            if a == 1.0:
                return 1.0
            prod_fail *= (1.0 - a)
            if prod_fail == 0.0:
                return 1.0
        return 1.0 - prod_fail

    def _compute_L(self, p: List[float]) -> float:
        t = self.tid
        prod_fail = 1.0
        for ei in self.channel_edges:
            e = self.edges[ei]
            prod = 1.0
            for u in e.verts:
                if u == t:
                    continue
                prod *= p[u]
                if prod == 0.0:
                    break
            q = e.w * prod
            if q <= 0.0:
                continue
            if q >= 1.0:
                return 1.0
            prod_fail *= (1.0 - q)
            if prod_fail == 0.0:
                return 1.0
        return 1.0 - prod_fail

    def leakage(self, mask: Set[Cell], *, tau: float = 1e-10, max_updates: int = 2_000_000) -> float:
        mask_ids = {self.cell_to_id[c] for c in mask}
        if self.tid in mask_ids:
            raise ValueError("mask includes target")
        observed = [True] * self.n
        observed[self.tid] = False
        for v in mask_ids:
            observed[v] = False

        p = [1.0 if observed[v] else 0.0 for v in range(self.n)]
        Q = deque(range(self.n))
        in_q = [True] * self.n
        pops = 0
        while Q and pops < max_updates:
            v = Q.popleft()
            in_q[v] = False
            pops += 1
            if observed[v]:
                continue
            new_p = self._recompute_pv(v, observed, p)
            if abs(new_p - p[v]) > tau:
                p[v] = new_p
                for u in self.neigh[v]:
                    if not in_q[u]:
                        Q.append(u)
                        in_q[u] = True

        return self._compute_L(p)


# -----------------------------
# EM expectations over ALL subsets of I(c*)
# -----------------------------

def precompute_all_mask_leakages(model: InferableLeakageModel) -> Tuple[Dict[int, float], int]:
    zone = model.zone
    k = len(zone)
    leak_by_bits: Dict[int, float] = {}
    for bits in range(1 << k):
        M = {zone[i] for i in range(k) if (bits >> i) & 1}
        leak_by_bits[bits] = model.leakage(M)
    return leak_by_bits, k


def em_expectations_all_subsets(
    model: InferableLeakageModel,
    leak_by_bits: Dict[int, float],
    *,
    alpha: float,
    beta: float,
    epsilon: float,
) -> Tuple[float, float, int]:
    """
    Utility: u(M) = -alpha*L(M) - beta*|M|
    Sensitivity: Δu = alpha
    EM sampling weights: exp(epsilon*u/(2*alpha))

    Returns:
      E[L], E[|M|], MAP_bits
    """
    zone = model.zone
    k = len(zone)
    logw = [0.0] * (1 << k)
    Ls = [0.0] * (1 << k)
    sizes = [0] * (1 << k)

    for bits in range(1 << k):
        L = leak_by_bits[bits]
        s = bits.bit_count()
        u = -alpha * L - beta * s
        Ls[bits] = L
        sizes[bits] = s
        logw[bits] = (epsilon * u) / (2.0 * alpha)  # Δu=alpha

    m = max(logw)
    ws = [math.exp(x - m) for x in logw]
    Z = sum(ws)
    probs = [w / Z for w in ws]

    expL = sum(p * l for p, l in zip(probs, Ls))
    expS = sum(p * s for p, s in zip(probs, sizes))
    map_bits = max(range(1 << k), key=lambda b: probs[b])
    return expL, expS, map_bits


# -----------------------------
# Plot helpers
# -----------------------------

def surprisal(leak: float) -> float:
    # -log(1-leak), safe at boundaries
    leak = min(max(leak, 0.0), 1.0)
    if leak >= 1.0:
        return float("inf")
    return -math.log(max(1e-18, 1.0 - leak))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_set1(df: pd.DataFrame, outdir: str) -> None:
    # (B1) Leakage vs epsilon
    plt.figure()
    plt.plot(df["epsilon"], df["E_Leakage"], marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("Expected Leakage  E[L(M)]")
    plt.title("Leakage vs epsilon (Exponential Mechanism, inferable mode)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "leakage_vs_epsilon.png"), dpi=200)
    plt.close()

    # (B2) Mask size vs epsilon
    plt.figure()
    plt.plot(df["epsilon"], df["E_MaskSize"], marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("Expected Mask Size  E[|M|]")
    plt.title("Mask size vs epsilon (Exponential Mechanism, inferable mode)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "masksize_vs_epsilon.png"), dpi=200)
    plt.close()

    # (A) Tradeoff: mask size vs leakage (epsilon-labeled)
    plt.figure()
    plt.plot(df["E_Leakage"], df["E_MaskSize"], marker="o")
    for _, r in df.iterrows():
        plt.annotate(str(r["epsilon"]), (r["E_Leakage"], r["E_MaskSize"]))
    plt.xlabel("Expected Leakage  E[L(M)]")
    plt.ylabel("Expected Mask Size  E[|M|]")
    plt.title("Leakage–Deletion tradeoff (points labeled by epsilon)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tradeoff_masksize_vs_leakage.png"), dpi=200)
    plt.close()

    # Optional: surprisal versions (good if leakage saturates near 1)
    if "E_Surprisal" in df.columns:
        plt.figure()
        plt.plot(df["epsilon"], df["E_Surprisal"], marker="o")
        plt.xlabel("epsilon")
        plt.ylabel("Expected Surprisal  E[-log(1-L)]")
        plt.title("Surprisal vs epsilon (expands high-leakage region)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "surprisal_vs_epsilon.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(df["E_Surprisal"], df["E_MaskSize"], marker="o")
        for _, r in df.iterrows():
            plt.annotate(str(r["epsilon"]), (r["E_Surprisal"], r["E_MaskSize"]))
        plt.xlabel("Expected Surprisal  E[-log(1-L)]")
        plt.ylabel("Expected Mask Size  E[|M|]")
        plt.title("Mask size vs surprisal (epsilon-labeled tradeoff)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "tradeoff_masksize_vs_surprisal.png"), dpi=200)
        plt.close()


# -----------------------------
# Main runner
# -----------------------------

def main():
    # -------------------------
    # TODO: Replace with YOUR hypergraph
    # -------------------------
    G, A, B, R, D = "Gender", "Age", "BMI", "Res", "Diag"
    target = D
    hyperedges = [
        {G, A},
        {A, R},
        {G, B},
        {B, R},
        {R, D},      # channel
        {A, B, D},   # channel
    ]
    weights = [0.95, 0.90, 0.92, 0.85, 0.95, 0.85]
    # -------------------------

    alpha = 1.0
    beta = 0.05
    eps_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    model = InferableLeakageModel(hyperedges, weights, target=target)
    zone = model.zone
    k = len(zone)
    print(f"Inference zone size |I| = {k}")
    print("Inference zone:", zone)

    # Guard: all-subsets enumeration is exponential
    if k > 24:
        raise RuntimeError(f"|I|={k} too large for all-subsets EM. "
                           f"Use this script only for small inference zones.")

    leak_by_bits, _ = precompute_all_mask_leakages(model)

    rows = []
    for eps in eps_vals:
        expL, expS, map_bits = em_expectations_all_subsets(
            model,
            leak_by_bits,
            alpha=alpha,
            beta=beta,
            epsilon=eps,
        )
        map_mask = [zone[i] for i in range(k) if (map_bits >> i) & 1]
        rows.append({
            "epsilon": eps,
            "E_Leakage": expL,
            "E_MaskSize": expS,
            "E_Surprisal": surprisal(expL),
            "MAP_Mask": str(map_mask),
        })

    df = pd.DataFrame(rows)
    outdir = "plots"
    ensure_dir(outdir)
    df.to_csv(os.path.join(outdir, "epsilon_sweep_summary.csv"), index=False)

    plot_set1(df, outdir)

    print(f"\nSaved plots to ./{outdir}/")
    print("Saved summary to ./plots/epsilon_sweep_summary.csv")
    print("\nPreview:\n", df)


if __name__ == "__main__":
    main()
