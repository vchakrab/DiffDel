\ Model CASCADING_ILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10
Subject To
 visible_implies_known_0: a0 + r0 >= 1
 visible_implies_known_1: a1 + r1 >= 1
 visible_implies_known_2: a2 + r2 >= 1
 visible_implies_known_3: a3 + r3 >= 1
 visible_implies_known_4: a4 + r4 >= 1
 visible_implies_known_5: a5 + r5 >= 1
 visible_implies_known_6: a6 + r6 >= 1
 visible_implies_known_7: a7 + r7 >= 1
 visible_implies_known_8: a8 + r8 >= 1
 visible_implies_known_9: a9 + r9 >= 1
 visible_implies_known_10: a10 + r10 >= 1
 target_not_inferable: r0 = 0
 closure_e0_h0: r0 - r1 >= 0
 closure_e0_h1: - r0 + r1 >= 0
 closure_e1_h0: r0 - r2 >= 0
 closure_e1_h2: - r0 + r2 >= 0
 closure_e2_h0: r0 - r3 >= 0
 closure_e2_h3: - r0 + r3 >= 0
 closure_e3_h4: - r0 + r4 >= 0
 closure_e3_h0: r0 - r4 >= 0
 closure_e4_h5: - r1 + r5 >= 0
 closure_e4_h1: r1 - r5 >= 0
 closure_e5_h6: - r1 + r6 >= 0
 closure_e5_h1: r1 - r6 >= 0
 closure_e6_h1: r1 - r2 >= 0
 closure_e6_h2: - r1 + r2 >= 0
 closure_e7_h1: r1 - r3 >= 0
 closure_e7_h3: - r1 + r3 >= 0
 closure_e8_h4: - r1 + r4 >= 0
 closure_e8_h1: r1 - r4 >= 0
 closure_e9_h3: - r2 + r3 >= 0
 closure_e9_h2: r2 - r3 >= 0
 closure_e10_h4: - r2 + r4 >= 0
 closure_e10_h2: r2 - r4 >= 0
 closure_e11_h6: - r2 + r6 >= 0
 closure_e11_h2: r2 - r6 >= 0
 closure_e12_h7: - r2 + r7 >= 0
 closure_e12_h2: r2 - r7 >= 0
 closure_e13_h4: - r3 + r4 >= 0
 closure_e13_h3: r3 - r4 >= 0
 closure_e14_h6: - r3 + r6 >= 0
 closure_e14_h3: r3 - r6 >= 0
 closure_e15_h7: - r3 + r7 >= 0
 closure_e15_h3: r3 - r7 >= 0
 closure_e16_h8: - r3 + r8 - r9 >= -1
 closure_e16_h9: - r3 - r8 + r9 >= -1
 closure_e16_h3: r3 - r8 - r9 >= -1
 closure_e17_h10: - r3 - r8 + r10 >= -1
 closure_e17_h8: - r3 + r8 - r10 >= -1
 closure_e17_h3: r3 - r8 - r10 >= -1
 closure_e18_h10: - r3 - r9 + r10 >= -1
 closure_e18_h3: r3 - r9 - r10 >= -1
 closure_e18_h9: - r3 + r9 - r10 >= -1
 closure_e19_h10: - r3 - r6 + r10 >= -1
 closure_e19_h6: - r3 + r6 - r10 >= -1
 closure_e19_h3: r3 - r6 - r10 >= -1
 closure_e20_h4: r4 - r6 >= 0
 closure_e20_h6: - r4 + r6 >= 0
 closure_e21_h4: r4 - r7 >= 0
 closure_e21_h7: - r4 + r7 >= 0
 closure_e22_h10: - r4 - r9 + r10 >= -1
 closure_e22_h4: r4 - r9 - r10 >= -1
 closure_e22_h9: - r4 + r9 - r10 >= -1
 closure_e23_h4: r4 - r8 - r9 >= -1
 closure_e23_h8: - r4 + r8 - r9 >= -1
 closure_e23_h9: - r4 - r8 + r9 >= -1
 closure_e24_h5: r5 - r7 >= 0
 closure_e24_h7: - r5 + r7 >= 0
 closure_e25_h5: r5 - r6 >= 0
 closure_e25_h6: - r5 + r6 >= 0
 closure_e26_h6: r6 - r7 >= 0
 closure_e26_h7: - r6 + r7 >= 0
 closure_e27_h8: r8 - r9 >= 0
 closure_e27_h9: - r8 + r9 >= 0
 closure_e28_h10: - r8 - r9 + r10 >= -1
 closure_e28_h8: r8 - r9 - r10 >= -1
 closure_e28_h9: - r8 + r9 - r10 >= -1
Bounds
 a0 = 1
Binaries
 a0 r0 a1 r1 a2 r2 a3 r3 a4 r4 a5 r5 a6 r6 a7 r7 a8 r8 a9 r9 a10 r10
End
