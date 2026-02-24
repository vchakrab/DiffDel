\ Model CASCADING_ILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  a0 + a1 + a2 + a3 + a4 + a5 + a6
Subject To
 visible_implies_known_0: a0 + r0 >= 1
 visible_implies_known_1: a1 + r1 >= 1
 visible_implies_known_2: a2 + r2 >= 1
 visible_implies_known_3: a3 + r3 >= 1
 visible_implies_known_4: a4 + r4 >= 1
 visible_implies_known_5: a5 + r5 >= 1
 visible_implies_known_6: a6 + r6 >= 1
 target_not_inferable: r0 = 0
 closure_e0_h2: - r0 - r1 + r2 >= -1
 closure_e0_h1: - r0 + r1 - r2 >= -1
 closure_e0_h0: r0 - r1 - r2 >= -1
 closure_e1_h2: r2 - r3 >= 0
 closure_e1_h3: - r2 + r3 >= 0
 closure_e2_h2: r2 - r4 >= 0
 closure_e2_h4: - r2 + r4 >= 0
 closure_e3_h2: - r1 + r2 - r5 >= -1
 closure_e3_h1: r1 - r2 - r5 >= -1
 closure_e3_h5: - r1 - r2 + r5 >= -1
 closure_e4_h2: - r1 + r2 - r6 >= -1
 closure_e4_h1: r1 - r2 - r6 >= -1
 closure_e4_h6: - r1 - r2 + r6 >= -1
 closure_e5_h4: - r1 + r4 >= 0
 closure_e5_h1: r1 - r4 >= 0
 closure_e6_h5: - r3 + r5 >= 0
 closure_e6_h3: r3 - r5 >= 0
Bounds
 a0 = 1
Binaries
 a0 r0 a1 r1 a2 r2 a3 r3 a4 r4 a5 r5 a6 r6
End
