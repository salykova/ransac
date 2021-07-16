#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script to compare different error functions for plane fitting
"""

import numpy as np
import open3d as o3d

from fit_plane import *
from error_funcs import *
from plot_results import *

if __name__ == '__main__':

    #########################################################
    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.02
    min_sample_distance = 0.8
    #########################################################

    #########################################################
    # Parameters to generate the test data
    a, b, c, d = 0, 0.2, 1, -2  # Plane parameters: a*x + b*y + c*z + d = 0
    normal_prop = 0.96  # To what proportion of the generated points we add noise in z-direction
    normal_std = 0.1  # The standard deviation of the Gaussian noise we add to normal_prop of the points
    num_of_tests = 500  # Number of tests: How often we try to fit the plane to the generated data
    plane_dimensions = 5  # The x,y-coordinates are uniformly sampled from the range [-plane_dimension, plane_dimension]
    num_of_generated_points = 10000
    #########################################################

    # Generate x and y coordinates in the defined range
    points = np.random.uniform(-plane_dimensions,
                               plane_dimensions,
                               size=(num_of_generated_points, 2))

    # Calculate the z values in a way, that the points are in the plane defined by the plane parameters
    plane_eq = np.array([a, b, c, d])
    z = -(a * points[..., 0] + b * points[..., 1] + d) / c

    # Add noise according to the noise parameters define above.
    # Add Gaussian noise (mu = 0, sigma = normal_std) to a proportion of the points (normal_prop)
    noisy_z = z + np.random.binomial(n=1, p=normal_prop, size=z.shape) * np.random.normal(loc=0, scale=normal_std,
                                                                                          size=z.shape)
    noisy_points = np.c_[points, noisy_z.reshape((-1, 1))]

    # Convert numpy data to a o3d.Pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(noisy_points)

    # Create empty arrays for the test results
    planes_ransac = np.empty((num_of_tests, 4))
    planes_msac = np.empty((num_of_tests, 4))
    planes_mlesac = np.empty((num_of_tests, 4))
    ransac_steps = np.empty(num_of_tests)
    msac_steps = np.empty(num_of_tests)
    mlesac_steps = np.empty(num_of_tests)
    angle_error_ransac = np.empty(num_of_tests)
    angle_error_msac = np.empty(num_of_tests)
    angle_error_mlesac = np.empty(num_of_tests)

    # Estimate a plane with each error function for num_of_tests times
    for i in range(num_of_tests):
        # Apply plane-fitting algorithms and store results
        best_plane_ransac, _, num_steps_ransac = fit_plane(pcd=pcd,
                                                           confidence=confidence,
                                                           inlier_threshold=inlier_threshold,
                                                           min_sample_distance=min_sample_distance,
                                                           error_func=ransac_error)
        planes_ransac[i] = best_plane_ransac.copy()
        ransac_steps[i] = num_steps_ransac

        best_plane_msac, _, num_steps_msac = fit_plane(pcd=pcd,
                                                       confidence=confidence,
                                                       inlier_threshold=inlier_threshold,
                                                       min_sample_distance=min_sample_distance,
                                                       error_func=msac_error)
        planes_msac[i] = best_plane_msac.copy()
        msac_steps[i] = num_steps_msac

        best_plane_mlesac, best_inliers_mlesac, num_steps_mlesac = fit_plane(pcd=pcd,
                                                                             confidence=confidence,
                                                                             inlier_threshold=inlier_threshold,
                                                                             min_sample_distance=min_sample_distance,
                                                                             error_func=mlesac_error)

        planes_mlesac[i] = best_plane_mlesac.copy()
        mlesac_steps[i] = num_steps_mlesac

    # The angle-error is the angle between the normal vector of the estimated and the correct plane
    angle_error_ransac = np.arccos(np.dot(planes_ransac[..., 0:3], plane_eq[0:3]) / (
            np.linalg.norm(plane_eq[0:3]) * np.linalg.norm(planes_ransac[..., 0:3], axis=1)))
    angle_error_msac = np.arccos(np.dot(planes_msac[..., 0:3], plane_eq[0:3]) / (
            np.linalg.norm(plane_eq[0:3]) * np.linalg.norm(planes_msac[..., 0:3], axis=1)))
    angle_error_mlesac = np.arccos(np.dot(planes_mlesac[..., 0:3], plane_eq[0:3]) / (
            np.linalg.norm(plane_eq[0:3]) * np.linalg.norm(planes_mlesac[..., 0:3], axis=1)))

    angle_error_data = [angle_error_ransac * 180 / np.pi,
                        angle_error_msac * 180 / np.pi,
                        angle_error_mlesac * 180 / np.pi]

    plot_sac_comparison(angle_error_data,
                        title="Angle Error",
                        x_label="angle error in degree",
                        violin_plot=False,
                        save_fig=True)
    print("RANSAC Angle Error in Degree (Mean +/- Std):", np.mean(angle_error_ransac) * 180 / np.pi,
          "+/-", np.std(angle_error_ransac))
    print("MSAC Angle Error in Degree (Mean +/- Std):", np.mean(angle_error_msac) * 180 / np.pi,
          "+/-", np.std(angle_error_msac))
    print("MLESAC Angle Error in Degree (Mean +/- Std):", np.mean(angle_error_mlesac) * 180 / np.pi,
          "+/-", np.std(angle_error_mlesac))

    # The z-shift is the shift in z-direction at x = 0, y = 0.
    # If it is positive, the estimated plane has a larger z-value than the correct plane equation
    # for x = 0, y = 0 -> plane_eq: c * z + d = 0 -> z-offset at x = 0, y = 0: z = -d / c
    z_shift_ransac = -((planes_ransac[..., 3] / planes_ransac[..., 2]) - plane_eq[3] / plane_eq[2])
    z_shift_msac = -((planes_msac[..., 3] / planes_msac[..., 2]) - plane_eq[3] / plane_eq[2])
    z_shift_mlesac = -((planes_mlesac[..., 3] / planes_mlesac[..., 2]) - plane_eq[3] / plane_eq[2])

    # The z-error is the absolute shift
    z_error_data = [np.abs(z_shift_ransac),
                    np.abs(z_shift_msac),
                    np.abs(z_shift_mlesac)]
    plot_sac_comparison(z_error_data,
                        title="Absolute Error in z-direction at x=0, y=0",
                        x_label="error in z direction in meter",
                        save_fig=True)

    # Plot the needed iterations for the different error functions
    iterations_data = [ransac_steps,
                       msac_steps,
                       mlesac_steps]
    plot_sac_comparison(iterations_data,
                        title="Number of Iterations",
                        x_label="iterations needed",
                        save_fig=True)

    # Plot one plane
    #draw_plane_mesh(pcd, best_inliers_mlesac, best_plane_mlesac)
