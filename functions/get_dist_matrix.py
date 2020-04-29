import numpy as np


def approximate_lonlat_distance(dlon, dlat, lat0, how='manhattan'):
    deg_len = 110.25  # km
    dlat = dlat * np.cos(lat0 * np.pi / 180)

    dist_mat = None

    if how == 'manhattan':
        dist_mat = (np.abs(dlon) + np.abs(dlat)) * deg_len
    elif how == 'euclidean':
        dist_mat = np.sqrt(dlon ** 2 + dlat ** 2) * deg_len

    return dist_mat


def get_distance(dx, dy, how='manhattan'):
    if how == 'manhattan':
        dist_mat = np.abs(dx) + np.abs(dy)
    elif how == 'euclidean':
        dist_mat = np.sqrt(dx ** 2 + dy ** 2)

    return dist_mat


def get_dist_mat(pos_1, pos_2, islonlat, how='manhattan'):
    pos_diff_list = []
    for i in range(2):
        pos = np.meshgrid(pos_2[:, i], pos_1[:, i])
        pos_diff_list.append(np.diff(pos, axis=0)[0])  # pos[1] - pos[0]

        if i == 1:
            lat0 = np.sum(pos, axis=0)[0] / 2

    if islonlat:
        dist_mat = approximate_lonlat_distance(pos_diff_list[0], pos_diff_list[1], lat0, how)
    else:
        dist_mat = get_distance(pos_diff_list[0], pos_diff_list[1], how)

    return dist_mat
