import enumerate_explanations as ee
import random
import time
def compute_random_for_num_edges(hypergraph):
    weighted_hypergraph_dict = {}
    for edge_set in hypergraph:
        weighted_hypergraph_dict[frozenset(edge_set)] = random.random()
    return weighted_hypergraph_dict

def collect_data(dataset, attrs, filename):
    boundary_edges, internal_edges, boundary_cells = ee.build_graph_data(dataset)
    hypergraph = compute_random_for_num_edges(internal_edges)
    for attribute in attrs:
        explanations_values = ee.find_all_weighted_explanations_weighted(hypergraph, "t1." + attribute, boundary_cells, 5)
        print(explanations_values)
        cell_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        exp_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        count = 0
        threshold = 0.0
        while threshold <= 1.0:
            for explanation_weight_set in explanations_values:
                if explanation_weight_set[1] > threshold:
                    cell_count[count] += len(explanation_weight_set[0])
                    exp_count[count] += 1

                    print(f'{attribute}', explanation_weight_set[0])
            count += 1
            threshold += 0.1
        with open(filename, 'a') as f:
            f.write(attribute + ',' +  str(cell_count) + ',' + str(exp_count) + '\n')
    return boundary_edges, internal_edges, boundary_cells, hypergraph

print(collect_data(ee.airport_constraints, ee.airport_attributes, 'airport_exps_data.csv'))

# def collect_time_based_data(dataset, attrs, filename):
#     boundary_edges, internal_edges, boundary_cells = ee.build_graph_data(dataset)
#     hypergraph = compute_random_for_num_edges(internal_edges)
#     # choose random 100 cells and then count time it takes to run those for a particular threshhold value
#     # update threshold value and do the same on the rest and collect that data. data we need to collect is time based only.
#     with open(filename, 'a') as f:
#         f.write("Hypergraph: " + str(hypergraph) + '\n')
#         f.write("Boundary edges: " + str(boundary_edges) + '\n')
#         f.write("Internal edges: " + str(internal_edges) + '\n')
#         f.write("Boundary cells: " + str(boundary_cells) + '\n')
#         f.write("time,threshold" + '\n')
#
#
#     attr_set = []
#     for i in range(100):
#         attr_set.append(random.choice(attrs))
#     with open(filename, 'a') as f:
#         f.write("random_choices: " + str(attr_set) + '\n')
#     threshold = 0.0
#     while threshold <= 1.0:
#         for i in range(100):
#             start = time.time()
#             attr = attr_set[i]
#             explanations_values = ee.find_all_weighted_explanations_weighted(hypergraph,
#                                                                              "t1." + attr,
#                                                                              boundary_cells, 10)
#             cell_count = 0
#             exp_count = 0
#             count = 0
#             for explanation_weight_set in explanations_values:
#                 if explanation_weight_set[1] > threshold:
#                     cell_count += len(explanation_weight_set[0])
#                     exp_count += 1
#                 count += 1
#             end_time = time.time() - start
#             with open(filename, 'a') as f:
#                 f.write(str(end_time) + "," + str(threshold) + "," + str(exp_count) + '\n')
#         threshold += 0.1
# collect_time_based_data(ee.airport_constraints, ee.airport_attributes, 'airport_time_data_alg1.csv')

