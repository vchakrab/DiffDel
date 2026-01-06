from collections import deque
from itertools import combinations

def active(edge_verts, x, K):
    return (x in edge_verts) and (set(edge_verts) - {x} <= K)

def enumerate_chains(mask, target, edges):
    """
    edges: dict[eid] = (set(verts), weight)
    Returns list of chains, each chain is a list of edge-ids ending at target.
    """
    V = set().union(*(verts for (verts, _w) in edges.values()))
    O = V - set(mask) - {target}

    chains = []

    # -----------------------------
    # Direct-to-target chains (len 1)
    # -----------------------------
    direct_target_witness = set()
    for eid, (verts, _w) in edges.items():
        if target in verts and active(verts, target, O):
            chains.append([eid])
            direct_target_witness |= (set(verts) - {target})

    # BFS: queue holds (chain_edge_ids, known_set)
    Q = deque()

    # --------------------------------------------------------
    # Seed with e that can infer some masked x directly from O
    # BUT: do not seed using a witness that already enables
    # a direct target inference (matches the figure; removes ec→e2 in M3).
    # --------------------------------------------------------
    for eid, (verts, _w) in edges.items():
        verts_set = set(verts)

        # candidate inferred cells x are masked/non-observed and not the target
        for x in (verts_set - O - {target}):
            if not active(verts_set, x, O):
                continue

            witness = verts_set - {x}

            # key restriction: forbid "detour" seeds that use direct-to-target witness
            if witness & direct_target_witness:
                continue

            Q.append(([eid], set(O) | {x}))

    while Q:
        p, K = Q.popleft()
        used = set(p)

        for eid, (verts, _w) in edges.items():
            if eid in used:
                continue
            verts_set = set(verts)
            if not (verts_set & K):
                continue  # must share a known witness cell

            # try to infer some new y using this edge
            for y in (verts_set - K):
                if not active(verts_set, y, K):
                    continue
                # maximality: y should NOT be inferable from O directly via this edge
                if active(verts_set, y, O):
                    continue

                if y == target:
                    chains.append(p + [eid])
                else:
                    Q.append((p + [eid], set(K) | {y}))

    return chains

def leakage(mask, target, edges):
    chains = enumerate_chains(mask, target, edges)
    if not chains:
        return 0.0

    w = {eid: float(edges[eid][1]) for eid in edges}
    chain_w = [prod(w[eid] for eid in p) for p in chains]
    chain_sets = [set(p) for p in chains]

    # noisy-or
    L_nor = 1.0
    for wp in chain_w:
        L_nor *= (1.0 - wp)
    L_nor = 1.0 - L_nor

    # IE up to 3rd order
    s1 = sum(chain_w)
    s2 = 0.0
    for i, j in combinations(range(len(chains)), 2):
        inter = chain_sets[i] & chain_sets[j]
        w_inter = prod(w[e] for e in inter) if inter else 1.0
        s2 += (chain_w[i] * chain_w[j]) / w_inter

    s3 = 0.0
    for i, j, k in combinations(range(len(chains)), 3):
        ij = chain_sets[i] & chain_sets[j]
        ik = chain_sets[i] & chain_sets[k]
        jk = chain_sets[j] & chain_sets[k]
        ijk = ij & chain_sets[k]
        w_ij  = prod(w[e] for e in ij)  if ij  else 1.0
        w_ik  = prod(w[e] for e in ik)  if ik  else 1.0
        w_jk  = prod(w[e] for e in jk)  if jk  else 1.0
        w_ijk = prod(w[e] for e in ijk) if ijk else 1.0
        s3 += (chain_w[i] * chain_w[j] * chain_w[k] * w_ijk) / (w_ij * w_ik * w_jk)

    L_ie = max(0.0, min(1.0, s1 - s2 + s3))
    return min(1.0, min(L_nor, L_ie))

def prod(vals):
    out = 1.0
    for x in vals:
        out *= x
    return out

print(enumerate_chains({"d", "f"}, "a", {"H1": ({'a', 'b', 'c', 'd'}, 0.5), "H2": ({'f', 'e', 'd'}, 0.2), "H3": ({'f', 'g', 'h'}, 0.3), "H4": ({'f', 'i', 'j'}, 0.2)}))


