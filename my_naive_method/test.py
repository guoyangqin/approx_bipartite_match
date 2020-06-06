import numpy as np
import time
from my_naive_method.functions.ExactEuclideanBipartiteMatch import *
from my_naive_method.functions.ApproxEuclideanBipartiteMatch import *

# Generate nodes
N = 5000
cell_size = N / 1e4
node_set = [np.random.uniform(0, 10, (N, 2)), np.random.uniform(0, 10, (N, 2))]

# === 0. Accurate ===
start = time.time()
eebm = ExactEuclideanBipartiteMatch(node_set)
_, avg_distance, ind_1, ind_2 = eebm.match(distance_category='manhattan')
print('[%0.4fs] EXACT avg_dist = %0.4f' % (time.time() - start, avg_distance))

# === 1. Centralized ===
start = time.time()
aebm = ApproxEuclideanBipartiteMatch(node_set, cell_size=cell_size)
_, avg_distance, ind_1, ind_2 = aebm.match(distance_category='manhattan')
print('[%0.4fs] APPROX avg_dist = %0.4f' % (time.time() - start, avg_distance))
