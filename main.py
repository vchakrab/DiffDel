import collect_data, graph

if __name__ == '__main__':
    print('Collecting data...')
    collect_data.run_all_experiments()
    print('Constructing graphs...')
    graph().graph_all_experiments()
    