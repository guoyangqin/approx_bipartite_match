import pandas as pd
import numpy as np
from my_naive_method.functions.ExactEuclideanBipartiteMatch import *


class ApproxEuclideanBipartiteMatch:
    def __init__(self, node_set, region_dim=[[0, 10], [0, 10]], cell_size=1, approx_thres=200):
        self.node_set = node_set
        self.region_dim = region_dim
        self.cell_size = cell_size
        self.approx_thres = approx_thres  # If both parties are less than this number, do exact matching

    def match(self, islonlat=False, distance_category='euclidean'):
        pos_1, pos_2 = self.node_set

        if max(len(pos_1), len(pos_2)) <= self.approx_thres:
            bipartite_distance_array, avg_distance, ind_1, ind_2 \
                = ExactEuclideanBipartiteMatch(self.node_set).match(distance_category='manhattan')
        else:
            bipartite_distance_array, avg_distance, ind_1, ind_2 = \
                self.approx_match(islonlat, distance_category)

        return bipartite_distance_array, avg_distance, ind_1, ind_2

    def approx_match(self, islonlat=False, distance_category='euclidean'):
        node_set_cell_id = self.assign_cell_id()

        # === Overall parameters
        pos_1, pos_2 = self.node_set
        dist_mat = get_dist_mat(pos_1, pos_2, islonlat=islonlat, how=distance_category)

        # === Match within cell locally (randomly)
        node_set_counts = self.find_within_cell_nodes(node_set_cell_id)

        # Iterate each cell and match
        local_match_result = []
        for i, row in node_set_counts.iterrows():
            p1_index, p2_index = row.p1, row.p2
            if len(p1_index) * len(p2_index) != 0:
                node_set_local = [self.node_set[0][p1_index, :], self.node_set[1][p2_index, :]]

                _, _, ind_1, ind_2 \
                    = ExactEuclideanBipartiteMatch(node_set_local).match(distance_category='manhattan')
                local_match_result += [[p1_index[i], p2_index[j]] for (i, j) in zip(ind_1, ind_2)]

        local_match_result = np.array(local_match_result)

        # === Match among cells globally (optimally)
        # Get nodes that are not locally matched
        global_match_index_p1 = np.array([i for i in range(len(self.node_set[0])) if i not in local_match_result[:, 0]])
        global_match_index_p2 = np.array([i for i in range(len(self.node_set[1])) if i not in local_match_result[:, 1]])

        if len(global_match_index_p1) * len(global_match_index_p2) != 0:
            node_set_global = [self.node_set[0][global_match_index_p1, :], self.node_set[1][global_match_index_p2, :]]

            _, _, ind_1, ind_2 \
                = ExactEuclideanBipartiteMatch(node_set_global).match(distance_category='manhattan')

            global_match_result = np.concatenate((global_match_index_p1[ind_1].reshape(-1, 1),
                                                  global_match_index_p2[ind_2].reshape(-1, 1)), axis=1)
            # === Combine match result
            match_result = np.concatenate((local_match_result, global_match_result), axis=0)
        else:
            match_result = local_match_result

        match_result = match_result[match_result[:, 0].argsort(), :]

        ind_1, ind_2 = match_result[:, 0], match_result[:, 1]
        bipartite_distance_array = dist_mat[ind_1, ind_2]
        avg_distance = np.mean(bipartite_distance_array)

        return bipartite_distance_array, avg_distance, ind_1, ind_2

    def assign_cell_id(self):
        def get_cell_id(pos):
            min_pos = np.min(self.region_dim, axis=1)
            cell_id = np.int_((pos - min_pos) / self.cell_size)
            return cell_id

        df = [pd.DataFrame(
            np.concatenate(
                (self.node_set[i], get_cell_id(self.node_set[i])),
                axis=1),
            columns=['x', 'y', 'x_id', 'y_id']).reset_index()
              for i in range(len(self.node_set))]

        # Add id for each party
        df[0]['p'], df[1]['p'] = 'p1', 'p2'

        node_set_df = df[0].append(df[1])

        return node_set_df

    def find_within_cell_nodes(self, node_set_df):
        node_set_counts = node_set_df.groupby(['x_id', 'y_id', 'p'])['index'].apply(list).unstack().reset_index()
        node_set_counts.p1 = node_set_counts.p1.apply(lambda d: d if isinstance(d, list) else [])  # fill NaN with []
        node_set_counts.p2 = node_set_counts.p2.apply(lambda d: d if isinstance(d, list) else [])

        return node_set_counts
