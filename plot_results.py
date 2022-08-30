#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results
Author: Matthias Hirschmanner
"""
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def plot_dominant_plane(pcd: o3d.geometry.PointCloud,
                        inliers: np.ndarray,
                        plane_eq: np.ndarray) -> None:
    """ Plot the inlier points in red and the rest of the pointcloud as is. A coordinate frame is drawn on the plane

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.array

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.array [a,b,c,d]

    :return: None
    """

    # Filter the inlier points and color them red
    inlier_indices = np.nonzero(inliers)[0]
    inlier_cloud = pcd.select_by_index(inlier_indices)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    # Create a rotation matrix according to the plane equation.
    # Detailed explanation of the approach can be found here: https://math.stackexchange.com/q/1957132
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    u2 = np.cross(normal_vector, u1)
    rot_mat = np.c_[u1, u2, normal_vector]

    # Create a coordinate frame and transform it to a point on the plane and with its z-axis in the same direction as
    # the normal vector of the plane
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.rotate(rot_mat, center=(0, 0, 0))
    if any(inlier_indices):
        coordinate_frame.translate(np.asarray(inlier_cloud.points)[-1])
        coordinate_frame.scale(0.3, np.asarray(inlier_cloud.points)[-1])

    geometries = [inlier_cloud, outlier_cloud, coordinate_frame]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.3, 0.32, -0.9])
    vc.set_lookat([-0.13, -0.15, 0.92])
    vc.set_up([0.22, -0.89, -0.39])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def plot_multiple_planes(plane_eqs: List[np.ndarray],
                         plane_pcds: List[o3d.geometry.PointCloud],
                         filtered_pcd: o3d.geometry.PointCloud) -> None:
    """ Plot the pointclouds in plane_pcds in different colors and keep colors of filtered_pcd.

    :param plane_eqs: List of np.ndarrays each holding the coefficient of a plane equation for one of the planes
    :type plane_eqs: List of np.ndarray each with shape (4,)

    :param plane_pcds: List of pointclouds with each element holding the inliers of one plane
    :type plane_pcds: List of o3d.geometry.PointClouds

    :param filtered_pcd: Pointcloud of all points which are not part of any of the planes
    :type filtered_pcd: o3d.geometry.PointCloud

    :return: None
    """
    colormap = plt.cm.get_cmap("gist_rainbow", len(plane_pcds))

    # Color the individual plane pointclouds in different colors
    for i, plane_pcd in enumerate(plane_pcds):
        plane_pcd.paint_uniform_color(colormap(i)[0:3])

    geometries = plane_pcds + [filtered_pcd]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.0, 0.0, -1])
    vc.set_lookat([0, -0.0, 1])
    vc.set_up([0, -1, 0])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def draw_plane_mesh(pcd: o3d.geometry.PointCloud,
                    inliers: np.ndarray,
                    plane_eq: np.ndarray) -> None:
    """ Plot the dominant plane as a mesh, inliers in dark blue, outliers in light blue

    :param pcd: The (down-sampled) pointcloud
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.ndarray

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.ndarray [a,b,c,d]

    :return: None
    """
    # Filter the inlier points and color them red
    inlier_indices = np.nonzero(inliers)[0]
    inlier_pcd = pcd.select_by_index(inlier_indices)
    inlier_pcd.paint_uniform_color([91 / 255., 95 / 255., 151 / 255.])

    outlier_pcd = pcd.select_by_index(inlier_indices, invert=True)
    outlier_pcd.paint_uniform_color([184 / 255., 184 / 255., 209 / 255.])

    # Create a rotation matrix according to the plane equation.
    # Detailed explanation of the approach can be found here: https://math.stackexchange.com/q/1957132
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    if np.linalg.norm(u1) != 0:
        u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal_vector, u1)
    if np.linalg.norm(u2) != 0:
        u2 = u2 / np.linalg.norm(u2)
    rot_mat = np.c_[u1, u2, normal_vector]
    points = np.asarray(inlier_pcd.points)
    points_transformed = rot_mat.T @ points.T
    min_tf = np.min(points_transformed, axis=1)
    max_tf = np.max(points_transformed, axis=1)
    mean_tf = np.mean(points_transformed, axis=1)
    std_tf = np.std(points_transformed, axis=1)
    extrema_transformed = np.array([[min_tf[0], min_tf[1], mean_tf[2]],
                                    [min_tf[0], max_tf[1], mean_tf[2]],
                                    [max_tf[0], min_tf[1], mean_tf[2]],
                                    [max_tf[0], max_tf[1], mean_tf[2]]])

    extrema = rot_mat @ extrema_transformed.T
    extrema_o3d = o3d.utility.Vector3dVector(extrema.T)
    triangles = np.array([[0, 1, 2], [2, 1, 3], [1, 0, 2], [1, 2, 3]])

    triangles_o3d = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(vertices=extrema_o3d, triangles=triangles_o3d)
    mesh.paint_uniform_color([255 / 255., 107 / 255., 108 / 255.])

    geometries = [mesh] + [inlier_pcd] + [outlier_pcd]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.0, 0.0, -1])
    vc.set_lookat([0, -0.0, 1])
    vc.set_up([0, -1, 0])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def plot_sac_comparison(data: List[np.ndarray],
                        title: str,
                        x_label: str,
                        save_fig: bool = False,
                        violin_plot: bool = False,
                        figure_size: Tuple[int, int] = (8, 4)) -> None:
    """ Plot different metrics for the three error functions (RANSAC, MSAC, MLESAC)

    :param data: Each list entry holds the generated data for one error function in the order [RANSAC, MSAC, MLESAC]
    :type data: List[np.ndarray]. List has a length of 3 and each np.ndarray has the shape (num_of_iterations,)

    :param title: Title of the plot
    :type title: str

    :param x_label: Label of the x-axis of the plot
    :type x_label: str

    :param save_fig: If set to True, the plot will be saved as a png with the 'title' as filename
    :type save_fig: bool

    :param violin_plot: If set to True, a violinplot will be generated. If set to False, a boxplot will be generated.
    :type violin_plot: bool

    :param figure_size: The size of the resulting figure as defined in Matplotlib
    :type figure_size: Tuple[int, int]

    :return: None
    """

    fig, ax = plt.subplots(figsize=figure_size)

    if violin_plot:
        violin_parts = ax.violinplot(data,
                                     showmeans=False,
                                     showmedians=True,
                                     vert=False)

        # If you want to add some color to your violinplot, you can play with the parameters below
        # colors = ['pink', 'lightblue', 'lightgreen']
        # for patch, color in zip(violin_parts['bodies'], colors):
        #     patch.set_facecolor(color)
        # for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        #     vp = violin_parts[partname]
        #     vp.set_edgecolor('grey')
    else:
        box_parts = ax.boxplot(data,
                               vert=False)

    ax.set_title(title)
    ax.set_yticks([y + 1 for y in range(len(data))])
    ax.set_yticklabels(['RANSAC', 'MSAC', 'MLESAC'])
    ax.set_xlabel(x_label)
    ax.set_ylabel('')

    if save_fig:
        plt.savefig(title.replace(" ", "_") + ".png", )

    plt.show()
