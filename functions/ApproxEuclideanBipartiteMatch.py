import lapsolve

class ApproxEuclideanBipartiteMatch:
    def __init__(self):
        pass

    def approx_match(self, N, S):
        pass

    def sub_approx_match(self):
        pass

    def exact_match(pos_1, pos_2, distance_category='manhattan'):
        # pos_1 and pos_2 are both two-column numpy array
        # denoting the longitude and latitude coordinates
        # distance_category = ['manhattan', 'euclidean']

        dist_mat = get_dist_mat(pos_1, pos_2, how=distance_category)

        if not lap_imported:
            # =================================
            # lapsolver (much faster than `scipy.optimize.linear_sum_assignment`)
            # https://github.com/cheind/py-lapsolver
            # Works well for float32 type of cost matrix
            # Scale distances up to make integer different
            dist_mat_float = dist_mat.astype('float32')
            ind_1, ind_2 = solve_dense(dist_mat_float)

            bipartite_distance_array = dist_mat[ind_1, ind_2]
            avg_distance = np.mean(bipartite_distance_array)
            # print(avg_distance)
            # =================================
        else:
            # lap (faster than `lapsolver`)
            # https://github.com/gatagat/lap
            # Works well for int type of cost matrix
            # Scale distances up to make integer different
            dist_mat_int = (100 * dist_mat).astype('int')
            ind_2, _ = lap.lapjv(dist_mat_int, return_cost=False, extend_cost=True)
            ind_1 = np.arange(len(ind_2))

            bipartite_distance_array = dist_mat[ind_1, ind_2]
            avg_distance = np.mean(bipartite_distance_array)
            # print(avg_distance)

        return bipartite_distance_array, avg_distance, ind_1, ind_2