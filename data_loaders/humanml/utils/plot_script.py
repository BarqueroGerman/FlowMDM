import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion_mix(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, vis_mode='default', lengths = []):
    matplotlib.use('Agg')
    data = joints.copy().reshape(len(joints), -1, 3)
    frames_number = data.shape[0]
    frame_colors = []
    if vis_mode == 'alternate':
        for i, length in enumerate(lengths):
            frame_colors += ['purple'] * length if i % 2 == 1 else ['orange'] * length
    elif vis_mode == 'gt':
        frame_colors = ['blue'] * frames_number
    else:
        raise NotImplementedError
    plot_3d_motion_multicolor(save_path, kinematic_tree, joints, title, dataset, figsize=figsize, fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors)


def plot_3d_motion_multicolor(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, vis_mode="default", frame_colors=[]):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    if type(title) == str:
        title = ["\n".join(wrap(title, 35))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 35)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == "humanml":
        data *= 1.3  # scale for visualization
    elif dataset =='babel':
        data *= 1.3


    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0].copy()

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 2], MAXS[2] - trajec[index, 2])

        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)

    ani.save(save_path, fps=fps)

    plt.close()


def plot_3d_motion_multicolor_pdf(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, vis_mode="default", frame_colors=[]):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    if type(title) == str:
        title = ["\n".join(wrap(title, 35))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 35)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == "humanml":
        data *= 1.3  # scale for visualization
    elif dataset =='babel':
        data *= 1.3


    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0].copy()

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    for index in range(0, frame_number, 2):
        if title[index] not in ["sit down", "stand up"]:
            continue
        fig = plt.figure(figsize=figsize)
        plt.tight_layout()
        ax = p3.Axes3D(fig)
        init()

        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        #plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 2], MAXS[2] - trajec[index, 2])

        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        import os
        os.makedirs(save_path.replace(".mp4", "/pdf"), exist_ok=True)
        plt.savefig(save_path.replace(".mp4", "/pdf") + str(index) + ".pdf")
        # clear the figure
        plt.clf()
        plt.close()


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title_max_width = len(title) // 3 # max 3 lines
    title_size = 10 if title_max_width < 25 else 6
    title = '\n'.join(wrap(title, title_max_width))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=title_size)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

        """
        # add point to origin
        ax.scatter([0], [0], [0], color='black', s=10)
        # plot 3 arrows for axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='black', length=1.0, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='red', length=1.0, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='blue', length=1.0, arrow_length_ratio=0.1)
        """

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset in ['humanml', ]:
        data *= 1.3  # scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_left_hand = ["#000000", ] * 5
    colors_right_hand = ["#6a329f", ] * 5 
    colors_face = ["#00008B"]
    colors = colors_orange + colors_left_hand + colors_right_hand + colors_face
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        #ax.view_init(elev=0, azim=0)
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        while len(kinematic_tree) > len(used_colors):
            used_colors += used_colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()
