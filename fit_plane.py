#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

"""

from typing import List, Tuple, Callable

import copy

import numpy as np
import open3d as o3d


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (best_plane, best_inliers, num_iterations)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray, int)
    """
    ######################################################

    points = np.asarray(pcd.points)
    N = len(points)
    m = 3
    eta_0 = 1-confidence
    k, eps, error_star = 0, m/N, np.inf
    I = 0
    best_inliers = np.full(shape=(N,),fill_value=0.)
    best_plane = np.full(shape=(4,), fill_value=-1.)
    while pow((1-pow(eps, m)),k) >= eta_0:
        p1, p2, p3 = points[np.random.randint(N)], points[np.random.randint(N)], points[np.random.randint(N)]
        if np.linalg.norm(p1-p2) < min_sample_distance or np.linalg.norm(p2-p3) < min_sample_distance or np.linalg.norm(p1-p3) < min_sample_distance:
            continue
        n = np.cross(p2-p1,p3-p1)
        n = n/np.linalg.norm(n) ### normalization
        if n[2] < 0: ### positive z direction
            n = -n
        d = -np.dot(n,p1) ### parameter d
        distances = np.abs(np.dot(points, n)+d)
        error, inliers = error_func(pcd, distances, inlier_threshold)
        if error < error_star:
            I = np.sum(inliers)
            eps = I/N
            best_inliers = inliers
            error_star = error
        k = k + 1
    A = points[best_inliers]
    y = np.full(shape=(len(A),),fill_value=1.)
    best_plane[0:3] = np.linalg.lstsq(A, y, rcond=-1)[0]
    if best_plane[2] < 0:  ### positive z direction
        best_plane = -best_plane
    return best_plane, best_inliers, k


def filter_planes(pcd: o3d.geometry.PointCloud,
                  min_points_prop: float,
                  confidence: float,
                  inlier_threshold: float,
                  min_sample_distance: float,
                  error_func: Callable) -> Tuple[List[np.ndarray],
                                                 List[o3d.geometry.PointCloud],
                                                 o3d.geometry.PointCloud]:
    """ Find multiple planes in the input pointcloud and filter them out.

    Find multiple planes by applying the detect_plane function multiple times. If a plane is found in the pointcloud,
    the inliers of this pointcloud are filtered out and another plane is detected in the remaining pointcloud.
    Stops if a plane is found with a number of inliers < min_points_prop * number of input points.

    :param pcd: The (down-sampled) pointcloud in which to detect planes
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param min_points_prop: The proportion of points of the input pointcloud which have to be inliers of a plane for it
        to qualify as a valid plane.
    :type min_points_prop: float

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers for each plane.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from a plane to be considered an inlier (in meters).
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (plane_eqs, plane_pcds, filtered_pcd)
        plane_eqs is a list of np.arrays each holding the coefficient of a plane equation for one of the planes
        plane_pcd is a list of pointclouds with each holding the inliers of one plane
        filtered_pcd is the remaining pointcloud of all points which are not part of any of the planes
    :rtype: (List[np.ndarray], List[o3d.geometry.PointCloud], o3d.geometry.PointCloud)
    """
    ######################################################
    
    filtered_pcd = copy.deepcopy(pcd)
    filtered_points = np.asarray(filtered_pcd.points)
    N = len(filtered_points)
    plane_eqs = []
    plane_pcds = []
    while True:
        best_plane, best_inliers, num_iterations = fit_plane(filtered_pcd, confidence, inlier_threshold, min_sample_distance, error_func)
        if np.sum(best_inliers)/N <= min_points_prop:
            break
        plane_eqs.append(best_plane)
        plane_pcds.append(filtered_pcd.select_by_index(np.nonzero(best_inliers)[0]))
        filtered_pcd = filtered_pcd.select_by_index(np.nonzero(best_inliers)[0], invert=True)

    return plane_eqs, plane_pcds, filtered_pcd
