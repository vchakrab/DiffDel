\ Model CASCADING_ILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  a0 + a1 + a2 + a3
Subject To
 visible_implies_known_0: a0 + r0 >= 1
 visible_implies_known_1: a1 + r1 >= 1
 visible_implies_known_2: a2 + r2 >= 1
 visible_implies_known_3: a3 + r3 >= 1
 target_not_inferable: r0 = 0
 closure_e0_h1: - r0 + r1 >= 0
 closure_e0_h0: r0 - r1 >= 0
 closure_e1_h1: r1 - r2 >= 0
 closure_e1_h2: - r1 + r2 >= 0
 closure_e2_h2: r2 - r3 >= 0
 closure_e2_h3: - r2 + r3 >= 0
Bounds
 a0 = 1
Binaries
 a0 r0 a1 r1 a2 r2 a3 r3
End
