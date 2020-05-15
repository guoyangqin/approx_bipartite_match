from functions.ApproxEuclideanBipartiteMatch import ApproxEuclideanBipartiteMatch
import numpy as np
import networkx as nx

N = 100
epsilon = 0.90  # [0,1], The bigger the longer expected time, but more accurate

node_set = [np.random.uniform(0, 10, (N, 2)), np.random.uniform(0, 10, (N, 2))]

aebm = ApproxEuclideanBipartiteMatch(node_set, epsilon=epsilon, C=10)

alpha, i_star, S, T = aebm.compute_alpha()
component = list(nx.connected_component_subgraphs(S))

print(alpha, i_star, len(component))

if 0:
    pos = nx.spring_layout(T)
    nx.draw_networkx_nodes(T, pos, node_size=5)
    nx.draw_networkx_edges(T, pos)


    for c in component:
        pos = nx.spring_layout(c)
        nx.draw_networkx_edges(c, pos)
