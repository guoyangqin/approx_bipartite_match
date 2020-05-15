from functions.ExactEuclideanBipartiteMatch import ExactEuclideanBipartiteMatch
from functions.ApproxEuclideanBipartiteMatch import ApproxEuclideanBipartiteMatch
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = "arial"
plt.rcParams['svg.fonttype'] = 'none'

figure, axes = plt.subplots(1, 2, figsize=(16, 8))

# Generate nodes
N = 50
epsilon = 0.90  # [0,1], The bigger the longer expected time, but more accurate

node_set = [np.random.uniform(0, 10, (N, 2)), np.random.uniform(0, 10, (N, 2))]

for n, c in zip(node_set, ['r', 'b']):
    x, y = n[:, 0], n[:, 1]
    for ax in axes:
        ax.scatter(x, y, 1, color=c)

# ==== 1. Exact algorithms ====
start_time = time.time()
print('Exact...')
bipartite_distance_array, avg_distance, ind_1, ind_2 = ExactEuclideanBipartiteMatch(node_set).match()
duration = time.time() - start_time

ax = axes[0]
ax.set_title(
    'Exact Euclidean bipartite matching\n N=%d, eps=%0.2f, avg_min_cost=%0.2f, runtime=%0.2fs' % (
        N, epsilon, avg_distance, duration))

node_pair = zip(node_set[0][ind_1, :], node_set[1][ind_2, :])
for r, b in node_pair:
    x, y = [r[0], b[0]], [r[1], b[1]]
    ax.plot(x, y, color='k', linewidth=0.5)

# ==== 2. Approx algorithms ====
start_time = time.time()
print('Approximate...')
avg_distance0 = avg_distance

bipartite_distance_array, avg_distance, ind_1, ind_2 = \
    ApproxEuclideanBipartiteMatch(node_set, epsilon=epsilon, C=10).match()
duration = time.time() - start_time

extra_avg_distance = (avg_distance - avg_distance0) / avg_distance0 * 100

ax = axes[1]
ax.set_title(
    'Approximate Euclidean bipartite matching\n N=%d, eps=%0.2f, avg_min_cost=%0.2f (+%0.2f%%), runtime=%0.2fs' % (
        N, epsilon, avg_distance, extra_avg_distance, duration))

node_pair = zip(node_set[0][ind_1, :], node_set[1][ind_2, :])

for r, b in node_pair:
    x, y = [r[0], b[0]], [r[1], b[1]]
    ax.plot(x, y, color='k', linewidth=0.5)

plt.show()
