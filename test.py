from functions.ExactEuclideanBipartiteMatch import *
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = "arial"
plt.rcParams['svg.fonttype'] = 'none'

figure, axes = plt.subplots(1, 2, figsize=(16, 8))

# Generate nodes
N = 5000
node_set = [np.random.uniform(0, 10, (N, 2)), np.random.uniform(0, 10, (N, 2))]

for n, c in zip(node_set, ['r', 'b']):
    x, y = n[:, 0], n[:, 1]
    for ax in axes:
        ax.scatter(x, y, 1, color=c)

# ==== 1. Exact algorithms ====
start_time = time.time()
eebm = ExactEuclideanBipartiteMatch(node_set)
bipartite_distance_array, avg_distance, ind_1, ind_2 = \
    eebm.bipartite_match(islonlat=False, distance_category='euclidean')
duration = time.time() - start_time

ax = axes[0]
ax.set_title('Exact Euclidean bipartite matching\n avg_min_cost=%0.2f, runtime=%0.2fs' % (avg_distance, duration))

node_pair = zip(node_set[0][ind_1, :], node_set[1][ind_2, :])
for r, b in node_pair:
    x, y = [r[0], b[0]], [r[1], b[1]]
    ax.plot(x, y, color='k', linewidth=0.5)

# ==== 2. Approx algorithms ====
start_time = time.time()
eebm = ExactEuclideanBipartiteMatch(node_set)
bipartite_distance_array, avg_distance, ind_1, ind_2 = \
    eebm.bipartite_match(islonlat=False, distance_category='euclidean')
duration = time.time() - start_time

ax = axes[1]
ax.set_title('Approximate Euclidean bipartite matching\n avg_min_cost=%0.2f, runtime=%0.2fs' % (avg_distance, duration))

node_pair = zip(node_set[0][ind_1, :], node_set[1][ind_2, :])
for r, b in node_pair:
    x, y = [r[0], b[0]], [r[1], b[1]]
    ax.plot(x, y, color='k', linewidth=0.5)
