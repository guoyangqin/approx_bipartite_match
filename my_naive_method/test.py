import time
from my_naive_method.functions.ExactEuclideanBipartiteMatch import *
from my_naive_method.functions.ApproxEuclideanBipartiteMatch import *

# Generate nodes
N = 5000
cell_size = 2
print('Cell size=%0.2f' % cell_size)
node_set = [np.random.uniform(0, 10, (N, 2)), np.random.uniform(0, 10, (N, 2))]

# === 0. Accurate ===
start = time.time()
eebm = ExactEuclideanBipartiteMatch(node_set)
_, avg_distance_1, ind_1, ind_2 = eebm.match(distance_category='manhattan')
t1 = time.time() - start
print('[%0.2fs] EXACT avg_dist = %0.4f' % (t1, avg_distance_1))

# === 1. Centralized ===
start = time.time()
aebm = ApproxEuclideanBipartiteMatch(node_set, cell_size=cell_size)
_, avg_distance_2, ind_1, ind_2 = aebm.match(distance_category='manhattan')
t2 = time.time() - start
t2_1 = (t2 - t1) / t1 * 100
avg_distance_2_1 = (avg_distance_2 - avg_distance_1) / avg_distance_1 * 100
print('[%0.2fs (%0.1f%%)] APPROX avg_dist = %0.4f (+%0.1f%%)'
      % (t2, t2_1, avg_distance_2, avg_distance_2_1))
