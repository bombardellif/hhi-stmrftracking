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
## @package stmrf_model
# Library of functions for the spatial-temporal Markov Random Field estimation
# model.

import numpy as np
import scipy.spatial.distance
import math

from hhi_stmrftracking import imgutils
from hhi_stmrftracking import mvutils

# Predefined Constants
NUM_ICM_ITERATIONS = 6
TEMPORAL_WEIGHT = 4
CONTEXT_WEIGHT = 3/2
COMPACTNESS_WEIGHT = 1
ICM_COST_THRESHOLD = -TEMPORAL_WEIGHT/2 - COMPACTNESS_WEIGHT/2
COMPACTNESS_ALPHA = 1/6
COMPACTNESS_BETA = 1/12
COHERENCE_STDDEV_LOW = 0.5
BBOX_BORDER = 2
C1 = 1/128
C2 = 1
C3 = 2

## Given the previous mask and the observed motion vectors, compute the
# temporal continuity component of the objective function for the current mask.
#
# @type curr_mask   HxW-ndarray of type bool, where H is the height and W the
#                   width of the mask
# @param curr_mask  Assumed that this function is called in a loop in the
#                   optimization procedure, this argument is the current
#                   solution of the optimization problem, i.e. the current mask.
#
# @type prev_mask   HxW-ndarray of type bool, where H is the height and W the
#                   width of the mask
# @param prev_mask   The previous mask, which is used to predict the next one
#
# @type back_proj_grid  HxWx2-ndarray of type float, where H is the height and W
#                       the width of the frame
# @param back_proj_grid A matrix, where each entry has the coordinates of this
#                       block projected back into the previous frame according
#                       to the respective motion vector.
# @return   HxW-ndarray of type float, where H is the height and W the width of
#           the mask
def temporal_continuity(curr_mask, prev_mask, back_proj_grid):
    bottom_lim,right_lim,_ = back_proj_grid.shape
    # Determine the indices of the object's bounding box
    bbox_top, bbox_bottom, bbox_left, bbox_right = imgutils.bounding_box(
        curr_mask,
        BBOX_BORDER)
    bbox_row,bbox_col = [
        np.ravel(coords) for coords in np.meshgrid(
            np.arange(bbox_top, bbox_bottom),
            np.arange(bbox_left, bbox_right),
            indexing='ij')
        ]
    # Get the coordinates of each block of the current object's bounding box
    # projected backwards
    coords_back_proj_bound_box = back_proj_grid[
        bbox_top:bbox_bottom,
        bbox_left:bbox_right].reshape(-1,2)
    # Split the coordinates into integer and fractional parts
    int_part_rows, frac_part_rows = np.divmod(coords_back_proj_bound_box[:,0], 1)
    int_part_rows = int_part_rows.astype(np.int16)
    int_part_cols, frac_part_cols = np.divmod(coords_back_proj_bound_box[:,1], 1)
    int_part_cols = int_part_cols.astype(np.int16)
    # Since the block projected backwards can be partitioned in four sub-blocks,
    # where each sub-block in inside exactly one block of the previous frame,
    # we determine the coordinates and area of each one of these sub-blocks
    coords_subblock = [
        (int_part_rows, int_part_cols),
        (int_part_rows, int_part_cols+1),
        (int_part_rows+1, int_part_cols),
        (int_part_rows+1, int_part_cols+1)
    ]
    areas_subblock = [
        (1-frac_part_rows) * (1-frac_part_cols),
        (1-frac_part_rows) * frac_part_cols,
        frac_part_rows * (1-frac_part_cols),
        frac_part_rows * frac_part_cols
    ]

    # The loop below runs four times, one time for each sub-block. For each one,
    # it selects the blocks that can be accessed (i.e., whose projection is
    # inside the frame limits) and calculates the contribution of the sub-block
    # to the overall result by multiplying its area with the binary value of
    # the mask at the previous frame. Value are negatively accumulated.
    temp_cont_frame = np.zeros(curr_mask.shape)
    for coord_sb,area_sb in zip(coords_subblock, areas_subblock):
        # Determine which blocks have the current sub-block within the frame
        idx_block_inside = (coord_sb[0] >= 0) & (coord_sb[0] < bottom_lim) \
            & (coord_sb[1] >= 0) & (coord_sb[1] < right_lim)
        # Compute for each sub-block of the current mask the area with label
        # True at the projection on the previous mask
        temp_cont_frame[bbox_row[idx_block_inside],
                        bbox_col[idx_block_inside]] \
            -= prev_mask[coord_sb[0][idx_block_inside],
                         coord_sb[1][idx_block_inside]] \
               * area_sb[idx_block_inside]

    # Perform an average of each block with its first order Neighbors
    neighborhood_values = imgutils.sum_neighbors(
        temp_cont_frame,
        imgutils.IDX_FST_ORD_NEIGHBORS)
    neighborhood_values /= 4
    temp_cont_frame += neighborhood_values
    temp_cont_frame /= 2

    # The final result is the negative value of the areas
    return temp_cont_frame

## Given the current mask compute the compactness component of the objective
# function.
#
# @type curr_mask   HxW-ndarray, where H is the height and W the width of mask
# @param curr_mask  Assumed that this function is called in a loop in the
#                   optimization procedure, this argument is the current
#                   solution of the optimization problem, i.e. the current mask.
# @return   HxW-ndarray of type float, where H is the height and W the width of
#           the mask
def compactness(curr_mask):
    # First Order Neighbors
    compactness_frame_fst_ord = imgutils.subtract_neighbors(
        curr_mask,
        imgutils.IDX_FST_ORD_NEIGHBORS)
    compactness_frame_fst_ord *= COMPACTNESS_ALPHA
    # Second Order Neighbors
    compactness_frame_snd_ord = imgutils.subtract_neighbors(
        curr_mask,
        imgutils.IDX_SND_ORD_NEIGHBORS)
    compactness_frame_snd_ord *= COMPACTNESS_BETA
    return compactness_frame_fst_ord + compactness_frame_snd_ord

## Given the current mask and the observer motion vectors, compute the
# context coherence component of the objective function.
#
# @type curr_mask   HxW-ndarray of type bool, where H is the height and W the
#                   width of the mask
# @param curr_mask  Assumed that this function is called in a loop in the
#                   optimization procedure, this argument is the current
#                   solution of the optimization problem, i.e. the current mask.
#
# @type mv_frame    HxWx2-ndarray, where H is the height and W the width of the
#                   frame
# @param mv_frame   The current frame composed of motion vectors
# @return   HxW-ndarray of type float, where H is the height and W the width of
#           the mask
def context_coherence(curr_mask, mv_frame, motion_effect):
    top, bottom, left, right = imgutils.bounding_box(curr_mask, BBOX_BORDER)
    bbox_frame = mv_frame[top:bottom, left:right]
    bbox_curr_mask = curr_mask[top:bottom, left:right]
    mask_static_blk = (bbox_frame[:,:,0] == 0) & (bbox_frame[:,:,1] == 0)
    # Select only non zero vectors for the array of object vectors
    bbox_obj_mask = bbox_curr_mask & ~mask_static_blk
    objects_vectors = bbox_frame[bbox_obj_mask]
    # Determine the object's representative vector
    representative_vec = np.rint(mvutils.polar_vector_median(
        objects_vectors,
        vectorfilter=mvutils.filter_nonsmall)).astype(objects_vectors.dtype)
    # Determine the distance of the vectors in the object's bounding box to the
    # representative vector
    # flat_bbox_vectors is an array of vectors
    flat_bbox_vectors = bbox_frame.reshape(-1,2)
    flat_bbox_dists = scipy.spatial.distance.cdist(
        flat_bbox_vectors,
        representative_vec[np.newaxis])
    # Get the distances of the object's MVs only
    obj_distances = flat_bbox_dists[bbox_curr_mask.ravel()]
    # Remove outliers and determine the standard deviation of the distribution
    if len(obj_distances) == 0 or obj_distances.max() <= 1:
        nonout_2ndmom = COHERENCE_STDDEV_LOW
    else:
        second_moment = math.sqrt(np.mean(obj_distances ** 2))
        threshold = 2*max(second_moment, 1)
        nonoutliers_distance = obj_distances[obj_distances <= threshold]
        nonout_2ndmom = COHERENCE_STDDEV_LOW if nonoutliers_distance.max() <= 1\
            else math.sqrt(np.mean(nonoutliers_distance ** 2))
    # Normalize the distances (in place). Note that all distances are in [-1,1]
    np.minimum(flat_bbox_dists/(2*nonout_2ndmom) - 1, 1, out=flat_bbox_dists)
    # Force a higher value (>=1) for static blocks (in place)
    idx_static = mask_static_blk.ravel()
    flat_bbox_dists[idx_static] = np.maximum(flat_bbox_dists[idx_static], 1)
    flat_bbox_dists /= motion_effect
    # Return the whole frame with the energies of each block
    # By default, blocks outside the bounding box get value 1
    context_coherence_frame = np.ones(curr_mask.shape)
    bbox_height, bbox_width = bottom-top, right-left
    context_coherence_frame[top:bottom, left:right] = \
        flat_bbox_dists.reshape(bbox_height, bbox_width)
    return context_coherence_frame

## Estimate the new mask based on the previous one and on the current
# observation (i.e. current frame).
#
# The function estimates the mask by maximizing the maximum-a-posteriori
# probability, what is to say, by minimizing the model's energy function.
#
# @type prev_mask   HxW-ndarray of type bool, where H is the height and W the
#                   width of the mask
# @param prev_mask   The previous mask, which is used to predict the next one
#
# @type mv_frame    HxWx2-ndarray, where H is the height and W the width of the
#                   frame
# @param mv_frame   The current frame composed of motion vectors
# @return   HxW-ndarray, where H is the height and W the width of the mask
def estimate_mask(prev_mask, initialized_mask, mv_frame, gmc_mv_frame, gm_params):
    back_proj_grid = mvutils.project_backwards(mv_frame)
    costs = np.empty(3, dtype=np.object)
    # Determine motion effect
    np.absolute(gm_params, out=gm_params)
    d = C1*(C2*(gm_params[2] + gm_params[5])**C3
        + abs(1-gm_params[0]) + abs(1-gm_params[4]) + gm_params[1] + gm_params[3])
    motion_effect = 2 - math.exp(-d)

    # import matplotlib.pyplot as plt
    # def debug_show(newm):
    #     plt.figure(figsize=(18, 18))
    #     plt.subplot(2,2,1)
    #     plt.imshow(costs[0], cmap='gray')
    #     plt.colorbar()
    #     plt.subplot(2,2,2)
    #     plt.imshow(costs[1], cmap='gray')
    #     plt.colorbar()
    #     plt.subplot(2,2,3)
    #     plt.imshow(costs[2], cmap='gray')
    #     plt.colorbar()
    #     plt.subplot(2,2,4)
    #     # plt.imshow(newm, cmap='gray')
    #     x,y = np.meshgrid(range(mv_frame.shape[1]), range(mv_frame.shape[0]))
    #     # Multiply column component  by -1 because of invert_yaxis
    #     plt.quiver(x.flatten(), y.flatten(), mv_frame[:,:,1].flatten(),
    #        -mv_frame[:,:,0].flatten(), scale=4, units='xy')
    #     plt.gca().invert_yaxis()
    #     plt.show()

    new_mask = initialized_mask
    for i in range(NUM_ICM_ITERATIONS):
        # If the mask is not empty, then calculate the energy components
        if new_mask.any():
            costs[0] = temporal_continuity(new_mask, prev_mask, back_proj_grid)
            costs[0] *= TEMPORAL_WEIGHT
            costs[1] = context_coherence(new_mask, gmc_mv_frame, motion_effect)
            costs[1] *= CONTEXT_WEIGHT
            costs[2] = compactness(new_mask)
            costs[2] *= COMPACTNESS_WEIGHT

            new_mask = costs.sum(axis=0) < ICM_COST_THRESHOLD

    # debug_show(new_mask)

    return new_mask
