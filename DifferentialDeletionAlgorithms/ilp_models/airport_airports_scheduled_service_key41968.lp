\ Model P2E2_ILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  a0 + a1 + a2 + a3 + a4 + a5 + a6
Subject To
 head_hidden_0: a0 - h0_0 = 0
 rdr_addr_0: b0 - h0_0 = 0
 tail_sync_0_0: - a0 + t0_0 = 0
 tail_sync_0_1: - a1 + t0_1 = 0
 tail_req_0: - b0 + t0_1 >= 0
 head_hidden_1: a0 - h1_0 = 0
 rdr_addr_1: b1 - h1_0 = 0
 tail_sync_1_0: - a0 + t1_0 = 0
 tail_sync_1_2: - a2 + t1_2 = 0
 tail_req_1: - b1 + t1_2 >= 0
 head_hidden_2: a0 - h2_0 = 0
 rdr_addr_2: b2 - h2_0 = 0
 tail_sync_2_3: - a3 + t2_3 = 0
 tail_sync_2_0: - a0 + t2_0 = 0
 tail_sync_2_4: - a4 + t2_4 = 0
 tail_req_2: - b2 + t2_3 + t2_4 >= 0
 head_hidden_3: a0 - h3_0 = 0
 rdr_addr_3: b3 - h3_0 = 0
 tail_sync_3_4: - a4 + t3_4 = 0
 tail_sync_3_0: - a0 + t3_0 = 0
 tail_sync_3_5: - a5 + t3_5 = 0
 tail_req_3: - b3 + t3_4 + t3_5 >= 0
 head_hidden_4: a0 - h4_0 = 0
 rdr_addr_4: b4 - h4_0 = 0
 tail_sync_4_4: - a4 + t4_4 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_6: - a6 + t4_6 = 0
 tail_req_4: - b4 + t4_4 + t4_6 >= 0
 head_hidden_5: a1 - h5_1 = 0
 rdr_addr_5: b5 - h5_1 = 0
 tail_sync_5_5: - a5 + t5_5 = 0
 tail_sync_5_1: - a1 + t5_1 = 0
 tail_req_5: - b5 + t5_5 >= 0
 head_hidden_5: a1 - h5_1 = 0
 rdr_addr_5: b0 - h5_1 = 0
 tail_sync_5_0: - a0 + t5_0 = 0
 tail_sync_5_1: - a1 + t5_1 = 0
 tail_req_5: - b0 + t5_0 >= 0
 head_hidden_5: a2 - h5_2 = 0
 rdr_addr_5: b1 - h5_2 = 0
 tail_sync_5_0: - a0 + t5_0 = 0
 tail_sync_5_2: - a2 + t5_2 = 0
 tail_req_5: - b1 + t5_0 >= 0
 head_hidden_6: a2 - h6_2 = 0
 rdr_addr_6: b6 - h6_2 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_sync_6_2: - a2 + t6_2 = 0
 tail_req_6: - b6 + t6_4 >= 0
 head_hidden_6: a3 - h6_3 = 0
 rdr_addr_6: b2 - h6_3 = 0
 tail_sync_6_3: - a3 + t6_3 = 0
 tail_sync_6_0: - a0 + t6_0 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_req_6: - b2 + t6_0 + t6_4 >= 0
 head_hidden_6: a4 - h6_4 = 0
 rdr_addr_6: b2 - h6_4 = 0
 tail_sync_6_3: - a3 + t6_3 = 0
 tail_sync_6_0: - a0 + t6_0 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_req_6: - b2 + t6_3 + t6_0 >= 0
 head_hidden_6: a4 - h6_4 = 0
 rdr_addr_6: b3 - h6_4 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_sync_6_0: - a0 + t6_0 = 0
 tail_sync_6_5: - a5 + t6_5 = 0
 tail_req_6: - b3 + t6_0 + t6_5 >= 0
 head_hidden_6: a4 - h6_4 = 0
 rdr_addr_6: b4 - h6_4 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_sync_6_0: - a0 + t6_0 = 0
 tail_sync_6_6: - a6 + t6_6 = 0
 tail_req_6: - b4 + t6_0 + t6_6 >= 0
 head_hidden_6: a4 - h6_4 = 0
 rdr_addr_6: b6 - h6_4 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_sync_6_2: - a2 + t6_2 = 0
 tail_req_6: - b6 + t6_2 >= 0
 head_hidden_6: a5 - h6_5 = 0
 rdr_addr_6: b5 - h6_5 = 0
 tail_sync_6_5: - a5 + t6_5 = 0
 tail_sync_6_1: - a1 + t6_1 = 0
 tail_req_6: - b5 + t6_1 >= 0
 head_hidden_6: a5 - h6_5 = 0
 rdr_addr_6: b3 - h6_5 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_sync_6_0: - a0 + t6_0 = 0
 tail_sync_6_5: - a5 + t6_5 = 0
 tail_req_6: - b3 + t6_4 + t6_0 >= 0
 head_hidden_6: a6 - h6_6 = 0
 rdr_addr_6: b4 - h6_6 = 0
 tail_sync_6_4: - a4 + t6_4 = 0
 tail_sync_6_0: - a0 + t6_0 = 0
 tail_sync_6_6: - a6 + t6_6 = 0
 tail_req_6: - b4 + t6_4 + t6_0 >= 0
Bounds
 a0 = 1
Binaries
 a0 b0 h0_0 t0_0 a1 t0_1 b1 h1_0 t1_0 a2 t1_2 b2 h2_0 a3 t2_3 t2_0 a4 t2_4
 b3 h3_0 t3_4 t3_0 a5 t3_5 b4 h4_0 t4_4 t4_0 a6 t4_6 b5 h5_1 t5_5 t5_1 h5_1
 t5_0 t5_1 h5_2 t5_0 t5_2 b6 h6_2 t6_4 t6_2 h6_3 t6_3 t6_0 t6_4 h6_4 t6_3
 t6_0 t6_4 h6_4 t6_4 t6_0 t6_5 h6_4 t6_4 t6_0 t6_6 h6_4 t6_4 t6_2 h6_5 t6_5
 t6_1 h6_5 t6_4 t6_0 t6_5 h6_6 t6_4 t6_0 t6_6
End
