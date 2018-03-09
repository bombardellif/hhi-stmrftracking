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
## @package mincut_model
# Library of functions for the Markov Random Field inference model using the
# technique of minimum-s-t-cut.

import numpy as np
import scipy.ndimage
import cv2

from hhi_stmrftracking import imgutils
from hhi_stmrftracking import mvutils
from hhi_stmrftracking.ioutils import MV_MAT_TYPE
from mincut import mincut

# Predefined Constants
BBOX_BORDER = 5
ERODE_KERNEL = np.ones((6,6), np.uint8)
LAPLACE_CORREC = 1
SMALL_MASK_SIZE = 20
GAUSSIAN_SIGMA = 1

# Reset procedure used by evaluation notebook
def set_params(laplace_correc, temp_weight):
    global LAPLACE_CORREC
    LAPLACE_CORREC = laplace_correc
    mincut.set_params(0.5, temp_weight)

## Given the previous mask and the coordinates of the backward (in time)
# projection of the current frame's motion vectors of the current frame, compute
# the temporal energy (or temporal continuity) of the current frame.
#
# @type prev_mask   HxW-ndarray of type bool, where H is the height and W the
#                   width of the mask
# @param prev_mask   The previous mask, which is used to predict the next one
# @type back_proj_grid  HxWx2-ndarray of type float, where H is the height and W
#                       the width of the frame
# @param back_proj_grid A matrix, where each entry has the coordinates of this
#                       block projected back into the previous frame according
#                       to the respective motion vector.
# @return   HxW-ndarray of type float, where H is the height and W the width of
#           the mask
def temporal_energy(curr_mask, prev_mask, back_proj_grid):
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
    temp_cont_frame = np.zeros(prev_mask.shape)
    for coord_sb,area_sb in zip(coords_subblock, areas_subblock):
        # Determine which blocks have the current sub-block within the frame
        idx_block_inside = (coord_sb[0] >= 0) & (coord_sb[0] < bottom_lim) \
            & (coord_sb[1] >= 0) & (coord_sb[1] < right_lim)
        # Compute for each sub-block of the current mask the area with label
        # True at the projection on the previous mask
        temp_cont_frame[bbox_row[idx_block_inside],
                        bbox_col[idx_block_inside]] \
            += prev_mask[coord_sb[0][idx_block_inside],
                         coord_sb[1][idx_block_inside]] \
               * area_sb[idx_block_inside]

    # The final result is a matrix with real entries between 0 and 1. They are
    # 0 if the whole respective block is projected on a background  area, and 1
    # if the whole block is projected on a foreground area. There may be values
    # in between
    return temp_cont_frame

def round_away_from_zero(data):
    result = np.copysign(.5, data)
    result += data
    np.trunc(result, out=result)
    return result.astype(MV_MAT_TYPE)

class MincutEstimator:
    def __init__(self, mask_shape):
        self.solver = mincut.MincutSolver()
        # To avoid allocation of memory in every cycle, we use always the same
        # memory space for some variables used in the class
        self.high_pass_frame = np.empty((2, mask_shape[0], mask_shape[1]),
                                        dtype=MV_MAT_TYPE)

    ## Estimate the new mask based on the previous one and on the current
    # observation (i.e. current frame).
    #
    # The function estimates the mask by maximizing the maximum-a-posteriori
    # probability, through optimization of minimum s-t-cut.
    #
    # @type prev_mask   HxW-ndarray of type bool, where H is the height and W the
    #                   width of the mask
    # @param prev_mask   The previous mask, which is used to predict the next one
    #
    # @type mv_frame    HxWx2-ndarray, where H is the height and W the width of the
    #                   frame
    # @param mv_frame   The current frame composed of motion vectors
    # @return   HxW-ndarray of type bool, where H is the height and W the width of
    #           the mask
    def estimate_mask(self, prev_mask, initialized_mask, mv_frame, gmc_frame):
        if prev_mask.any() and initialized_mask.any() and ~initialized_mask.all():
            # Calculate the temporal energy by projecting the MVs onto last mask
            back_proj_grid = mvutils.project_backwards(mv_frame)
            temp_energy = temporal_energy(initialized_mask, prev_mask,
                                          back_proj_grid)
            # Sample vectors from the area of the last mask
            eroded_mask = cv2.erode(initialized_mask.view(np.uint8), ERODE_KERNEL)\
                .astype(np.bool, copy=False)
            if np.count_nonzero(eroded_mask) >= SMALL_MASK_SIZE:
                temp_energy = scipy.ndimage.filters.gaussian_filter(temp_energy, GAUSSIAN_SIGMA)
            else:
                eroded_mask = initialized_mask

            # Collect the vectors of current frame under the last mask. This is
            # a rough estimate of the object, which is used to estimate the
            # probability of a vector to belong to the foreground.
            object_vectors = gmc_frame[eroded_mask]
            min_obj,max_obj = object_vectors.min(axis=0),object_vectors.max(axis=0)
            bins_obj = np.absolute(max_obj - min_obj)*2 + 2
            xedges_obj = np.linspace(min_obj[0], max_obj[0] + .5, bins_obj[0])
            yedges_obj = np.linspace(min_obj[1], max_obj[1] + .5, bins_obj[1])
            H_obj,_,_ = np.histogram2d(object_vectors[:,0], object_vectors[:,1],
                                      bins=(xedges_obj,yedges_obj))
            pad_H_obj = np.zeros((H_obj.shape[1]+1, H_obj.shape[0]+1))
            pad_H_obj[:-1,:-1] = H_obj.T
            pad_H_obj += LAPLACE_CORREC
            pad_H_obj *= 1 / (object_vectors.shape[0]
                             + LAPLACE_CORREC*pad_H_obj.shape[0]*pad_H_obj.shape[1])

            # Estimate background's probability distribution parameters
            bg_vectors = gmc_frame[~initialized_mask]
            min_bg,max_bg = bg_vectors.min(axis=0),bg_vectors.max(axis=0)
            bins_bg = np.absolute(max_bg - min_bg)*2 + 2
            xedges_bg = np.linspace(min_bg[0], max_bg[0] + .5, bins_bg[0])
            yedges_bg = np.linspace(min_bg[1], max_bg[1] + .5, bins_bg[1])
            H_bg,_,_ = np.histogram2d(bg_vectors[:,0], bg_vectors[:,1],
                                      bins=(xedges_bg,yedges_bg))
            pad_H_bg = np.zeros((H_bg.shape[1]+1, H_bg.shape[0]+1))
            pad_H_bg[:-1,:-1] = H_bg.T
            pad_H_bg += LAPLACE_CORREC
            pad_H_bg *= 1 / (bg_vectors.shape[0]
                            + LAPLACE_CORREC*pad_H_bg.shape[0]*pad_H_bg.shape[1])

            # Apply high pass filter on MV field to highlight edges
            scipy.ndimage.filters.laplace(round_away_from_zero(gmc_frame[:,:,0]),
                                          output=self.high_pass_frame[0],
                                          mode='constant')
            scipy.ndimage.filters.laplace(round_away_from_zero(gmc_frame[:,:,1]),
                                          output=self.high_pass_frame[1],
                                          mode='constant')

            top,bottom,left,right = imgutils.bounding_box(initialized_mask,
                                                          BBOX_BORDER)
            mask_outside_bbox = np.ones_like(prev_mask)
            mask_outside_bbox[top:bottom, left:right] = False
            # Estimate the new mask by applying graph cuts
            return self.solver.solve_mincut(gmc_frame,
                np.transpose(self.high_pass_frame, (1,2,0)),
                temp_energy,
                pad_H_bg,
                xedges_bg,
                yedges_bg,
                pad_H_obj,
                xedges_obj,
                yedges_obj,
                mask_outside_bbox)\
            .astype(np.bool, copy=False)
        else:
            return prev_mask
