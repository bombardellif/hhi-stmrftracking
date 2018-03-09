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
## @package tracking
# Library for tracking objects in video sequences.

import numpy as np
import cv2

from hhi_stmrftracking import mvutils
from hhi_stmrftracking import preprocessing
from hhi_stmrftracking import mincut_model
from hhi_stmrftracking import stmrf_model
from hhi_stmrftracking import globalmotion
from hhi_stmrftracking.ioutils import MV_MAT_TYPE

set_params = mincut_model.set_params

CLOSE_KERNEL = np.ones((3,3), np.uint8)
# Method to use: 0 for STMRF and 1 for minimum s-t cut
METHOD = 1

def init_mask(mask, gm_params):
    bottom_lim,right_lim = mask.shape
    block_size = mvutils.SMALLEST_BLOCK_SIZE
    # Get coordinates of the center of each block of the object
    coord_i,coord_j = mask.nonzero()
    coord_i *= block_size
    coord_i += block_size // 2
    coord_j *= block_size
    coord_j += block_size // 2
    # Perform the transformation
    C = (gm_params[1] * gm_params[3] - gm_params[0] * gm_params[4])
    new_i = -(coord_i*gm_params[4] - coord_j*gm_params[1]
            - gm_params[2]*gm_params[4] + gm_params[1]*gm_params[5]) / C
    new_j = (coord_i*gm_params[3] - coord_j*gm_params[0]
            - gm_params[2]*gm_params[3] + gm_params[0]*gm_params[5]) / C
    new_i = np.rint((new_i+1)/block_size).astype(np.int16) - 1
    new_j = np.rint((new_j+1)/block_size).astype(np.int16) - 1
    inside_frame = (new_i >= 0) & (new_i < bottom_lim) \
        & (new_j >= 0) & (new_j < right_lim)
    # Create new mask and initialize it
    new_mask = np.zeros_like(mask)
    new_mask[new_i[inside_frame], new_j[inside_frame]] = True
    closed_mask = cv2.morphologyEx(new_mask.view(np.uint8), cv2.MORPH_CLOSE,
                                  CLOSE_KERNEL).view(np.bool)
    return closed_mask

def round_away_from_zero(data):
    result = np.copysign(.5, data)
    result += data
    np.trunc(result, out=result)
    return result.astype(MV_MAT_TYPE)

## Given a mask at the first frame determining the object to be tracked in the
# frame sequence return a sequence of masks describing the object in the
# respective frame.
# It assumes that each entry of `object_mask` corresponds to a 4x4 prediction
# block of a frame in the video stream.
#
# @type object_mask    HxW-ndarray, where H is the height and W the Width of
#                       the frame
# @param object_mask   Binary matrix describing the object in first frame
#
# @type mv_frame_stream     Iterator Nx4-ndarray, where N is the number of
#                           motion vectors in the respective frame
# @param mv_frame_stream    The stream of motion vector frames. It should be a
#                           list where each element represents a frame and each
#                           element is a two-dimensional array. This array has
#                           shape (N,4), where N is the number of motion
#                           vectors in the frame. Each line of this array is of
#                           the kind [x,y,dx,dy], where x,y are the coordinates,
#                           and dx,dy are the values of a motion vector.
# @return   Iterator of HxW-ndarrays, where each element of the list is a mask
def track(object_mask, mv_frame_stream):
    # Get the stream of preprocessed frames
    preprocessed_frames = preprocessing.transform_to_grid(
        mv_frame_stream,
        object_mask.shape)

    if METHOD == 1:
        estimator = mincut_model.MincutEstimator(object_mask.shape)

    curr_mask = object_mask
    # The given object's mask is always returned as the first tracked mask
    yield curr_mask
    for (frame,mvs_split) in preprocessed_frames:
        if frame.size:
            if METHOD == 1:
                adjusted_frame = frame / mvutils.MV_SCALE
            else:
                adjusted_frame = round_away_from_zero(frame / mvutils.MV_SCALE)

            # Apply global motion compensation
            gm_params = globalmotion.estimation(curr_mask, mvs_split)
            if (gm_params == globalmotion.NO_MOTION_GMP).all():
                initialized_mask = curr_mask
            else:
                initialized_mask = init_mask(curr_mask, gm_params)
                if METHOD == 1:
                    globalmotion.compensation(adjusted_frame, gm_params,
                                              out=adjusted_frame)
                else:
                    adjusted_frame = round_away_from_zero(
                        globalmotion.compensation(adjusted_frame, gm_params,
                                                  out=None))

            if METHOD == 1:
                # Estimate object's mask through Graph Cuts
                curr_mask = estimator.estimate_mask(
                    curr_mask,
                    initialized_mask,
                    frame,
                    adjusted_frame)
            else:
                # Estimate object's mask through ST-MRF Model
                curr_mask = stmrf_model.estimate_mask(
                    curr_mask,
                    initialized_mask,
                    frame,
                    adjusted_frame,
                    gm_params)
        yield curr_mask
