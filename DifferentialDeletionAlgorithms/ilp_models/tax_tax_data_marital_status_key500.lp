\ Model P2E2_ILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  a0 + a1 + a2 + a3
Subject To
 head_hidden_0: a0 - h0_0 = 0
 rdr_addr_0: b0 - h0_0 = 0
 tail_sync_0_0: - a0 + t0_0 = 0
 tail_sync_0_1: - a1 + t0_1 = 0
 tail_sync_0_2: - a2 + t0_2 = 0
 tail_req_0: - b0 + t0_1 + t0_2 >= 0
 head_hidden_1: a0 - h1_0 = 0
 rdr_addr_1: b1 - h1_0 = 0
 tail_sync_1_3: - a3 + t1_3 = 0
 tail_sync_1_0: - a0 + t1_0 = 0
 tail_sync_1_2: - a2 + t1_2 = 0
 tail_req_1: - b1 + t1_3 + t1_2 >= 0
 head_hidden_2: a0 - h2_0 = 0
 rdr_addr_2: b2 - h2_0 = 0
 tail_sync_2_3: - a3 + t2_3 = 0
 tail_sync_2_0: - a0 + t2_0 = 0
 tail_sync_2_1: - a1 + t2_1 = 0
 tail_req_2: - b2 + t2_3 + t2_1 >= 0
 head_hidden_3: a0 - h3_0 = 0
 rdr_addr_3: b3 - h3_0 = 0
 tail_sync_3_3: - a3 + t3_3 = 0
 tail_sync_3_0: - a0 + t3_0 = 0
 tail_sync_3_1: - a1 + t3_1 = 0
 tail_sync_3_2: - a2 + t3_2 = 0
 tail_req_3: - b3 + t3_3 + t3_1 + t3_2 >= 0
 head_hidden_4: a1 - h4_1 = 0
 rdr_addr_4: b4 - h4_1 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b4 + t4_2 >= 0
 head_hidden_4: a1 - h4_1 = 0
 rdr_addr_4: b0 - h4_1 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b0 + t4_0 + t4_2 >= 0
 head_hidden_4: a1 - h4_1 = 0
 rdr_addr_4: b2 - h4_1 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_req_4: - b2 + t4_3 + t4_0 >= 0
 head_hidden_4: a1 - h4_1 = 0
 rdr_addr_4: b3 - h4_1 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b3 + t4_3 + t4_0 + t4_2 >= 0
 head_hidden_4: a2 - h4_2 = 0
 rdr_addr_4: b4 - h4_2 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b4 + t4_1 >= 0
 head_hidden_4: a2 - h4_2 = 0
 rdr_addr_4: b0 - h4_2 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b0 + t4_0 + t4_1 >= 0
 head_hidden_4: a2 - h4_2 = 0
 rdr_addr_4: b1 - h4_2 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b1 + t4_3 + t4_0 >= 0
 head_hidden_4: a2 - h4_2 = 0
 rdr_addr_4: b3 - h4_2 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b3 + t4_3 + t4_0 + t4_1 >= 0
 head_hidden_4: a3 - h4_3 = 0
 rdr_addr_4: b1 - h4_3 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b1 + t4_0 + t4_2 >= 0
 head_hidden_4: a3 - h4_3 = 0
 rdr_addr_4: b2 - h4_3 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_req_4: - b2 + t4_0 + t4_1 >= 0
 head_hidden_4: a3 - h4_3 = 0
 rdr_addr_4: b3 - h4_3 = 0
 tail_sync_4_3: - a3 + t4_3 = 0
 tail_sync_4_0: - a0 + t4_0 = 0
 tail_sync_4_1: - a1 + t4_1 = 0
 tail_sync_4_2: - a2 + t4_2 = 0
 tail_req_4: - b3 + t4_0 + t4_1 + t4_2 >= 0
Bounds
 a0 = 1
Binaries
 a0 b0 h0_0 t0_0 a1 t0_1 a2 t0_2 b1 h1_0 a3 t1_3 t1_0 t1_2 b2 h2_0 t2_3
 t2_0 t2_1 b3 h3_0 t3_3 t3_0 t3_1 t3_2 b4 h4_1 t4_1 t4_2 h4_1 t4_0 t4_1
 t4_2 h4_1 t4_3 t4_0 t4_1 h4_1 t4_3 t4_0 t4_1 t4_2 h4_2 t4_1 t4_2 h4_2 t4_0
 t4_1 t4_2 h4_2 t4_3 t4_0 t4_2 h4_2 t4_3 t4_0 t4_1 t4_2 h4_3 t4_3 t4_0 t4_2
 h4_3 t4_3 t4_0 t4_1 h4_3 t4_3 t4_0 t4_1 t4_2
End
