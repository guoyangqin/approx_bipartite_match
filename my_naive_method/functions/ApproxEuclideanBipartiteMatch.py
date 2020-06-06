import pandas as pd
import numpy as np
from my_naive_method.functions.ExactEuclideanBipartiteMatch import *


class ApproxEuclideanBipartiteMatch:
    def __init__(self, node_set, region_dim=[[0, 10], [0, 10]], cell_size=1):
        self.node_set = node_set
        self.region_dim = region_dim
        self.cell_size = cell_size

    def match(self, islonlat=False, distance_category='euclidean'):
        node_set_cell_id = self.assign_cell_id()

        # === Overall parameters
        pos_1, pos_2 = self.node_set
        dist_mat = get_dist_mat(pos_1, pos_2, islonlat=islonlat, how=distance_category)

        # === Match within cell locally (randomly)
        local_match_index, global_match_index = self.find_within_cell_nodes(node_set_cell_id)

        # Iterate each cell and match
        local_match_result = []
        for cell_group_id in np.unique(local_match_index[:, -1]):
            cell_local_match_index = local_match_index[local_match_index[:, -1] == cell_group_id, :2]

            node_set_local = [n[cell_local_match_index[:, i], :] for i, n in enumerate(self.node_set)]

            _, _, ind_1, ind_2 \
                = ExactEuclideanBipartiteMatch(node_set_local).match(distance_category='manhattan')
            local_match_result += [
                [cell_local_match_index[i, 0], cell_local_match_index[j, 1]]
                for (i, j) in zip(ind_1, ind_2)]

        local_match_result = np.array(local_match_result)

        # === Match among cells globally (optimally)
        node_set_global = [n[global_match_index[:, i], :] for i, n in enumerate(self.node_set)]

        _, _, ind_1, ind_2 \
            = ExactEuclideanBipartiteMatch(node_set_global).match(distance_category='manhattan')
        global_match_result = np.concatenate((global_match_index[ind_1, 0].reshape(-1, 1),
                                              global_match_index[ind_2, 1].reshape(-1, 1)), axis=1)
        # === Combine match result
        match_result = np.concatenate((local_match_result, global_match_result), axis=0)
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

        node_set_counts.p1.apply(np.random.shuffle)
        node_set_counts['num_within_cell'] = np.minimum(node_set_counts.p1.apply(len), node_set_counts.p2.apply(len))

        # Randomly pick within nodes to match locally
        node_set_counts['p1_local'] = node_set_counts.apply(lambda x: x.p1[:x.num_within_cell], axis=1)
        node_set_counts['p1_global'] = node_set_counts.apply(lambda x: x.p1[x.num_within_cell:], axis=1)

        node_set_counts['p2_local'] = node_set_counts.apply(lambda x: x.p2[:x.num_within_cell], axis=1)
        node_set_counts['p2_global'] = node_set_counts.apply(lambda x: x.p2[x.num_within_cell:], axis=1)

        # Get local and global matching index pairs
        local_cell_group_id = node_set_counts.reset_index().apply(lambda x: [x['index']] * x.num_within_cell, axis=1)
        # p1 index, p2 index, & their cell group id
        local_match_index = np.concatenate((np.hstack(node_set_counts.p1_local).reshape((-1, 1)),
                                            np.hstack(node_set_counts.p2_local).reshape((-1, 1)),
                                            np.hstack(local_cell_group_id).reshape((-1, 1))), axis=1).astype(int)
        # p1 index & p2 index
        global_match_index = np.concatenate((np.hstack(node_set_counts.p1_global).reshape((-1, 1)),
                                             np.hstack(node_set_counts.p2_global).reshape((-1, 1))), axis=1).astype(int)

        return local_match_index, global_match_index
