import matplotlib.pyplot as plt

plt.rcParams['font.family'] = "arial"
plt.rcParams['svg.fonttype'] = 'none'


def plot_match_result(nodes, indices, ax_id=0, figaxes=None):
    if figaxes:
        [fig, axes] = figaxes
    else:
        fig, axes = plt.subplots(1, 2)

    ax = axes[ax_id]

    # Plot all nodes
    [p1, p2] = nodes
    ax.scatter(p1[:, 0], p1[:, 1], s=1, marker='o', color='r')
    ax.scatter(p2[:, 0], p2[:, 1], s=1, marker='o', color='b')

    # Plot matching links
    [p1_indices, p2_indices] = indices
    for [p1_index, p2_index] in zip(p1_indices, p2_indices):
        p1_, p2_ = p1[p1_index, :], p2[p2_index, :]
        ax.plot([p1_[0], p2_[0]], [p1_[1], p2_[1]], color='k')

    return [fig, axes]
