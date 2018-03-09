"""
  Copyright:
  2016 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
  The copyright of this software source code is the property of HHI.
  This software may be used and/or copied only with the written permission
  of HHI and in accordance with the terms and conditions stipulated
  in the agreement/contract under which the software has been supplied.
  The software distributed under this license is distributed on an "AS IS" basis,
  WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
"""
## @package mvutils
# Library of utility functions for operation with motion vectors.

import numpy as np

from hhi_stmrftracking import imgutils

# Predefined Constants
SMALL_VEC_THRESHOLD = 0.25
SMALLEST_BLOCK_SIZE = 4
MV_SCALE = 4

## Given a vector and a matrix of row-vectors, determine if the matrix contain
# that one vector in its rows.
#
# @type vector  2-ndarray
# @param vector 2-dimensional vector to look for in the collection of vectors
# @type collection  Nx2-ndarray, where N is the number of vectors in this
#                   collection
# @param collection Matrix with two columns and N rows, containing N
#                   2-dimensional vectors
# @return   bool, True if `vector` is in `collection`, False otherwise
def is_in(vector, collection, begin=0):
    return ((collection[begin:,0] == vector[0]) \
        & (collection[begin:,1] == vector[1])).any()

def filter_nonzero(vectors):
    return (vectors[:,0] != 0) | (vectors[:,1] != 0)

def filter_nonsmall(vectors):
    return vectors[:,0]**2 + vectors[:,1]**2 > SMALL_VEC_THRESHOLD

## Given a motion vector frame project it backwards by **adding** each MV
# to its respective coordinate after the transformation into the "prediction
# block space". For example, given an input frame A with A[i,j] = [di,dj], the
# result matrix B has in this coordinate the value
# B[i,j] = [i,j] + [di,dj] / BLK_SIDE, where BLK_SIDE tells how many prediction
# blocks there are in the side of a macro block (MB), e.g. assuming that every
# MB has 16 prediction blocks arranged in a 4x4 block we have BLK_SIDE = 4.
#
# @type mv_frame    HxWx2-ndarray, where H is the height and W the width of the
#                   frame
# @param mv_frame   The current frame composed of motion vectors
# @return   HxWx2-ndarray of type float, where H is the height and W the width
#           of the frame
def project_backwards(mv_frame):
    coord_grid = imgutils.coordinates_matrix(mv_frame.shape)
    return coord_grid + mv_frame/(SMALLEST_BLOCK_SIZE * MV_SCALE)

## Calculate the polar vector median of the given list of vectors. The result
# is a vector in Cartesian coordinate system.
#
# @type vectors     Nx2-ndarray, where N is the number of vectors of the input
# @param vectors    Array of vectors whose polar vector media is to be
#                   calculated
# @type vectorfilter    Nx2-ndarray -> N-ndarray of type bool
# @param vectorfilter   Function for filtering the input. The function receives
#                       the list of vectors (as an ndarray) and returns a
#                       boolean ndarray, where the entry is True iff the
#                       respective vector is not to be excluded from the angle
#                       calculations.
# @return   2-ndarray of type float
def polar_vector_median(vectors, vectorfilter=filter_nonzero):
    # Filter the input
    vectors_nonzero = vectors[vectorfilter(vectors)]
    n = vectors_nonzero.shape[0]
    if n == 0:
        return np.zeros(2, dtype=np.float)
    else:
        # Transform the input to polar coordinates and sort the angles in [-pi,+pi]
        theta_sort = np.arctan2(vectors_nonzero[:,1], vectors_nonzero[:,0])
        theta_sort.sort()
        radius = (vectors**2).sum(axis=1)

        if n <= 2:
            idx_narrowest = 0
            num_nonoutliers = n
        else:
            # Calculate the difference in angles between adjacent vectors
            theta_sort_diff = np.diff(np.concatenate((theta_sort, [0])))
            # The difference of the last to the first is the angle that
            # complements the circle (2pi)
            theta_sort_diff[-1] = 2*np.pi - (theta_sort[-1] - theta_sort[0])
            # Performs the cumulative sum of those differences (in place)
            theta_diff_cumsum = theta_sort_diff.cumsum(out=theta_sort_diff)
            # Define the number M of non-outlier vectors
            # num_nonoutliers = (n+1) // 2 (Should be like this by the paper)
            num_nonoutliers = n // 2
            if num_nonoutliers == 1:
                # Get the one in the middle
                idx_narrowest = num_nonoutliers
            else:
                # Sum up each adjacent sequence of M-1 differences of the angles
                sum_adjacent_angles = theta_diff_cumsum[num_nonoutliers-2:] \
                    - np.concatenate(([0],theta_diff_cumsum[:-(num_nonoutliers-1)]))
                # The result index is the minimal sum-up
                idx_narrowest = sum_adjacent_angles.argmin()

        # Calculate the median of the narrowest beam and of all vector norms
        idx_median = idx_narrowest + num_nonoutliers//2
        if idx_median == theta_sort.shape[0]: # Rare case
            idx_median = 0
        pvm_theta = theta_sort[idx_median] if num_nonoutliers % 2 \
            else (theta_sort[idx_median-1] + theta_sort[idx_median]) / 2
        pvm_radius = np.sqrt(np.median(radius, overwrite_input=True))

        # Return the median vector in Cartesian coordinates
        return pvm_radius * np.array([np.cos(pvm_theta), np.sin(pvm_theta)])

## Given an array of two-dimensional vectors, calculate the angle theta, which
# is the angle of the polar vector median (in the polar coordinate system).
#
# @type vectors     Nx2-ndarray, where N is the number of vectors to calculate
#                   polar vector angle theta
# @param vectors    Array of vectors whose polar vector medium angle theta is
#                   to be calculated.
# @return   2-ndarray of type float
def compute_pvm_theta(vectors):
    vectors_nonzero = vectors[filter_nonzero(vectors)]
    n = vectors_nonzero.shape[0]
    if n == 0:
        pvm_theta = 0
    else:
        # Transform the input to polar coordinates and sort the angles in [-pi,+pi]
        theta_sort = np.arctan2(vectors_nonzero[:,1], vectors_nonzero[:,0])
        theta_sort.sort()

        if n <= 2:
            idx_narrowest = 0
            num_nonoutliers = n
        else:
            # Calculate the difference in angles between adjacent vectors
            theta_sort_diff = np.diff(np.concatenate((theta_sort, [0])))
            # The difference of the last to the first is the angle that
            # complements the circle (2pi)
            theta_sort_diff[-1] = 2*np.pi - (theta_sort[-1] - theta_sort[0])
            # Performs the cumulative sum of those differences (in place)
            theta_diff_cumsum = theta_sort_diff.cumsum(out=theta_sort_diff)
            # Define the number M of non-outlier vectors
            # num_nonoutliers = (n+1) // 2 (Should be like this by the paper)
            num_nonoutliers = n // 2
            if num_nonoutliers == 1:
                # Get the one in the middle
                idx_narrowest = num_nonoutliers
            else:
                # Sum up each adjacent sequence of M-1 differences of the angles
                sum_adjacent_angles = theta_diff_cumsum[num_nonoutliers-2:] \
                    - np.concatenate(([0],theta_diff_cumsum[:-(num_nonoutliers-1)]))
                # The result index is the minimal sum-up
                idx_narrowest = sum_adjacent_angles.argmin()

        # Calculate the median of the narrowest beam and of all vector norms
        idx_median = idx_narrowest + num_nonoutliers//2
        if idx_median == theta_sort.shape[0]: # Rare case
            idx_median = 0
        pvm_theta = theta_sort[idx_median] if num_nonoutliers % 2 \
            else (theta_sort[idx_median-1] + theta_sort[idx_median]) / 2
    return pvm_theta

## Vectorized (faster) version of function `polar_vector_median`, which
# calculates the polar vector median of the given vectors. The results are
# vectors in Cartesian coordinate system.
#
# @type vectors     MxNx2-ndarray, where M is the number of lists for which to
#                   compute PVM, and N is the number of vectors in each list
# @param vectors    Array of Array of vectors whose polar vector media is to be
#                   calculated. The outer array has M Nx2-matrices, for each
#                   one of those, the PVM is calculated and returned in the
#                   respective position of the returned array.
# @return   Mx2-ndarray of type float
def vectorized_pvm(vectors):
    # Calculate the radius of the resulting vectors
    radius = (vectors**2).sum(axis=2)
    pvm_radius = np.median(radius, axis=1, overwrite_input=True)
    # Calculate the angle of the resulting vectors only for those whose median
    # is not zero (case it is zero, then pvm_theta gets zero respectively)
    pvm_theta = [radius and compute_pvm_theta(vecs) \
        for radius,vecs in zip(pvm_radius, vectors)]
    return (np.sqrt(pvm_radius) * np.array((np.cos(pvm_theta), np.sin(pvm_theta)))).T
