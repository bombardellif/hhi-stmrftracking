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
## @package preprocessing
# Library of functions for frame preprocessing before the actual tracking takes
# place.

import numpy as np
import cv2

from hhi_stmrftracking import mvutils
from utils import mvutils as fast_mvutils

# Predefined Constants
VIDEO_MB_SIZE = 16
VIDEO_MB_HALF_SIZE = VIDEO_MB_SIZE // 2
VIDEO_MB_QUARTER_SIZE = VIDEO_MB_SIZE // 4
VIDEO_MB_SIZE_IN_PB = VIDEO_MB_SIZE // mvutils.SMALLEST_BLOCK_SIZE
INTRAPREDICTION_ITERATIONS = 6

## Given a frame in the raw format as input, transform it by assigning the
# vectors to the previously allocated `out_grid`.
#
# It is assumed that the `out_grid` is large enough to fit the motion vectors.
# *NOTE*: This function has side-effects on the parameter `out_grid`, whose
# value is changed by the function.
#
# @type mv_frame_raw    Nx4-ndarray, where N is the number of motion vectors to
#                       fit into the grid.
# @param mv_frame_raw  Sequence of motion vectors to fit into the grid, where
#                       each row of the matrix is of the kind [x,y,dx,dy], where
#                       x,y are the coordinates, and dx,dy are the values of a
#                       motion vector.
#
# @type out_grid    HxWx2-ndarray of type int, where H is the height and W is
#                   the width of the frame, and every element of this grid is a
#                   two-dimensional vector.
# @param out_grid   A previously allocated ndarray which receives the MVs.
# @return   HxWx2-ndarray of type int, which is a reference to `out_grid`
def fit_to_grid(mv_frame_raw, out_grid, value=None):
    bottom_lim,right_lim = out_grid.shape[:2]
    # Place the given vectors in the matrix at their respective indices. Note
    # that the motion vectors are flipped in order to get a transformation from
    # (x,y) to (row,column) coordinate system
    idx_inside = (mv_frame_raw[:,1] < bottom_lim) \
        & (mv_frame_raw[:,0] < right_lim)
    valid_mvs = mv_frame_raw if idx_inside.all() else mv_frame_raw[idx_inside,:]

    out_grid[valid_mvs[:,1], valid_mvs[:,0]] = np.fliplr(valid_mvs[:,2:]) \
        if value is None else value
    return out_grid

## Return an array with the neighbor vectors of `gridcoords` indicated by the
# parameter `directions`.
#
# @type mv_frame_grid   HxWx2-ndarray of type int, where H is the height and
#                       W is the width of the frame, and every element of this
#                       grid is a two-dimensional vector.
# @param mv_frame_grid  The partial grid of motion vectors, where the decoded
#                       and the already-intrapredicted motion vectors are in
#                       their respective coordinates and the rest of the matrix
#                       has the value zero.
# @type gridcoords      Nx2-ndarray of type int, where N is the number of MB
#                       coordinates
# @param gridcoords     Each vector in this array has the index of the top-left
#                       element of a macroblock in `mv_frame_grid`.
# @type directions      string
# @param directions     String telling which neighbors to collect (North, South,
#                       east and west) in respect to each index `gridcoords`.
# @return  Mx2-ndarray of type int, where M and is the total number of vectors
#          collected, M depends on the parameter `directions` and is a multiple
#          of N.
def collect_neighbors(mv_frame_grid, gridcoords, directions='NWSE'):
    mb_size = VIDEO_MB_SIZE_IN_PB
    # How many neighbors to collect will determine the number of vectors per MB
    num_neighbors = len(directions)
    n = gridcoords.shape[0]
    output_shape = (n, num_neighbors * mb_size, 2)
    if n == 0:
        return np.empty(output_shape, dtype=gridcoords.dtype)
    idx_shape = (n * num_neighbors, mb_size)
    rows,cols = np.empty(idx_shape, dtype=gridcoords.dtype), \
        np.empty(idx_shape, dtype=gridcoords.dtype)
    arange_mb_size = np.arange(mb_size, dtype=gridcoords.dtype)
    i,m = 0,n
    # Collect vectors in the North
    if 'N' in directions:
        rows[i:m] = gridcoords[:,0:1] - 1
        cols[i:m] = gridcoords[:,1:2] + arange_mb_size
        i += n
        m += n
    # Collect vectors in the West
    if 'W' in directions:
        rows[i:m] = gridcoords[:,0:1] + arange_mb_size
        cols[i:m] = gridcoords[:,1:2] - 1
        i += n
        m += n
    # Collect vectors in the South
    if 'S' in directions:
        rows[i:m] = gridcoords[:,0:1] + mb_size
        cols[i:m] = gridcoords[:,1:2] + arange_mb_size
        i += n
        m += n
    # Collect vectors in the East
    if 'E' in directions:
        rows[i:m] = gridcoords[:,0:1] + arange_mb_size
        cols[i:m] = gridcoords[:,1:2] + mb_size
    vector_idx = rows.reshape(-1), cols.reshape(-1)
    return mv_frame_grid[vector_idx].reshape(output_shape)

## Fill each 4x4 submatrix of `mv_frame_grid` indicated by the parameter
# `gridcoords` with the respective vectors of parameter `vectors`.
#
# @type mv_frame_grid   HxWx2-ndarray of type int, where H is the height and
#                       W is the width of the frame, and every element of this
#                       grid is a two-dimensional vector.
# @param mv_frame_grid  The partial grid of motion vectors, where the decoded
#                       and the already-intrapredicted motion vectors are in
#                       their respective coordinates and the rest of the matrix
#                       has the value zero.
# @type gridcoords      Nx2-ndarray of type int, where N is the number of MB
#                       coordinates
# @param gridcoords     Each vector in this array has the index of the top-left
#                       element of a macroblock in `mv_frame_grid`.
# @type vectors         Nx2-ndarray, where N is the number of MB coordinates
# @param vectors        Each row in this array has the motion vector respective
#                       to the macroblock of same coordinate in `gridcoords`.
# @return   HxWx2-ndarray of type int, which is a reference to `mv_frame_grid`
def fill_macroblocks(mv_frame_grid, gridcoords, vectors):
    mb_size = VIDEO_MB_SIZE_IN_PB
    n = gridcoords.shape[0]
    if n:
        idx_shape = (n, mb_size)
        padded_vectors = cv2.resize(
            vectors,
            (0,0),
            fx=1,
            fy=mb_size,
            interpolation=cv2.INTER_NEAREST)
        rows,cols = np.empty(idx_shape, dtype=gridcoords.dtype), \
            np.empty(idx_shape, dtype=gridcoords.dtype)
        arange_mb_size = np.arange(mb_size, dtype=gridcoords.dtype)
        # Fill the MBs in a row-wise manner. First get the indices that fill the
        # first row of every MB, then iterates four times incrementing the row
        rows[:] = gridcoords[:,0:1]
        cols[:] = gridcoords[:,1:2] + arange_mb_size
        rows_idx, cols_idx = rows.reshape(-1), cols.reshape(-1)
        for i in range(mb_size):
            mv_frame_grid[rows_idx, cols_idx] = padded_vectors
            rows_idx += 1
    return mv_frame_grid

## Perform intraprediction of the macroblocks (MB) indicated with True by the
# respective element in the matrix `padded_intracoded_grid`.
#
# Intraprediction is performed based on the prediction blocks that are neighbor
# of the given MBs. In this implementation the neighbor considered are those
# located immediately above, on the right, below, and on the left (North, East,
# South, and West, respectively).
# The resulting motion vector of each intra-coded MBs is calculated by the
# Polar Vector Median of the above mentioned neighbor prediction blocks.
# *NOTE*: This function has side-effects on both parameters `mv_frame_grid` and
# `padded_intracoded_grid` whose values are changed by the function.
#
# @type mv_frame_grid   HxWx2-ndarray of type int, where H is the height and
#                       W is the width of the frame, and every element of this
#                       grid is a two-dimensional vector.
# @param mv_frame_grid  The partial grid of motion vectors, where the decoded
#                       motion vectors are in their respective coordinates and
#                       the rest of the matrix has the value zero.
#
# @type padded_intracoded_grid  H'xW'-ndarray of type bool, where H' is the
#                               height and W' the width of the matrix, which
#                               each element corresponds to a macroblock.
# @param padded_intracoded_grid Each element has the value True if the
#                               corresponding MB is intracoded and False
#                               otherwise. The elements in the most outer border
#                               must have value True in order to simplify the
#                               computation (see for example numpy.pad).
# @return   HxWx2-ndarray of type int, which is a reference to `mv_frame_grid`
def intrapredict(mv_frame_grid, padded_intracoded_grid):
    # Get masks for North, West, South, East MBs on `intracoded_grid`
    intracoded_grid = padded_intracoded_grid[1:-1, 1:-1]
    mask_N = padded_intracoded_grid[:-2, 1:-1]
    mask_W = padded_intracoded_grid[1:-1, :-2]
    mask_S = padded_intracoded_grid[2:, 1:-1]
    mask_E = padded_intracoded_grid[1:-1, 2:]
    # Repeat many times predicting motion vectors from neighbor Macro Blocks
    for i in range(INTRAPREDICTION_ITERATIONS):
        predicted_coords = []
        predicted_vectors = []
        intracoded_coords = np.transpose(intracoded_grid.nonzero())
        if intracoded_coords.size == 0:
            return mv_frame_grid
        # Get the intrapredicted MB coordinates in "prediction-block space"
        intracoded_gridcoords = intracoded_coords * VIDEO_MB_SIZE_IN_PB
        # Get the number of intra-coded neighbors for every intra-coded MB
        coord_idx_ic_N = mask_N[intracoded_grid]
        coord_idx_ic_W = mask_W[intracoded_grid]
        coord_idx_ic_S = mask_S[intracoded_grid]
        coord_idx_ic_E = mask_E[intracoded_grid]
        num_ic_neighbors = coord_idx_ic_N.view(np.int8) \
            + coord_idx_ic_W.view(np.int8) \
            + coord_idx_ic_S.view(np.int8) \
            + coord_idx_ic_E.view(np.int8)
        # Get the MBs with no intra-coded MB as neighbor and compute PVM
        no_ic_neighbor = num_ic_neighbors == 0
        pred_coords_no_ic = intracoded_gridcoords[no_ic_neighbor]
        predicted_coords.append(pred_coords_no_ic)
        neighbor_vectors = collect_neighbors(mv_frame_grid, pred_coords_no_ic)
        predicted_vectors.append(fast_mvutils.fast_vectorized_pvm(neighbor_vectors))
        # Get the MBs with only one intra-coded MB as neighbor and compute PVM
        one_ic_neighbor = num_ic_neighbors == 1
        pred_coords_ic_north = intracoded_gridcoords[coord_idx_ic_N \
                                                    & one_ic_neighbor]
        pred_coords_ic_west = intracoded_gridcoords[coord_idx_ic_W \
                                                    & one_ic_neighbor]
        pred_coords_ic_south = intracoded_gridcoords[coord_idx_ic_S \
                                                    & one_ic_neighbor]
        pred_coords_ic_east = intracoded_gridcoords[coord_idx_ic_E \
                                                    & one_ic_neighbor]
        predicted_coords.extend((
            pred_coords_ic_north,
            pred_coords_ic_west,
            pred_coords_ic_south,
            pred_coords_ic_east))
        neighbor_vectors = np.vstack((
            collect_neighbors(mv_frame_grid, pred_coords_ic_north, directions='WSE'),
            collect_neighbors(mv_frame_grid, pred_coords_ic_west, directions='NSE'),
            collect_neighbors(mv_frame_grid, pred_coords_ic_south, directions='NWE'),
            collect_neighbors(mv_frame_grid, pred_coords_ic_east, directions='NWS')
            ))
        predicted_vectors.append(fast_mvutils.fast_vectorized_pvm(neighbor_vectors))
        if i > 0:
            # Get the MBs with two intra-coded MB as neighbor and compute PVM
            two_ic_neighbor = num_ic_neighbors == 2
            pred_coords_ic_NW = intracoded_gridcoords[coord_idx_ic_N \
                                                    & coord_idx_ic_W \
                                                    & two_ic_neighbor]
            pred_coords_ic_NS = intracoded_gridcoords[coord_idx_ic_N \
                                                    & coord_idx_ic_S \
                                                    & two_ic_neighbor]
            pred_coords_ic_NE = intracoded_gridcoords[coord_idx_ic_N \
                                                    & coord_idx_ic_E \
                                                    & two_ic_neighbor]
            pred_coords_ic_WE = intracoded_gridcoords[coord_idx_ic_W \
                                                    & coord_idx_ic_E \
                                                    & two_ic_neighbor]
            pred_coords_ic_WS = intracoded_gridcoords[coord_idx_ic_W \
                                                    & coord_idx_ic_S \
                                                    & two_ic_neighbor]
            pred_coords_ic_SE = intracoded_gridcoords[coord_idx_ic_S \
                                                    & coord_idx_ic_E \
                                                    & two_ic_neighbor]
            predicted_coords.extend((
                pred_coords_ic_NW, pred_coords_ic_NS, pred_coords_ic_NE,
                pred_coords_ic_WE, pred_coords_ic_WS, pred_coords_ic_SE))
            neighbor_vectors = np.vstack((
                collect_neighbors(mv_frame_grid, pred_coords_ic_NW, directions='SE'),
                collect_neighbors(mv_frame_grid, pred_coords_ic_NS, directions='WE'),
                collect_neighbors(mv_frame_grid, pred_coords_ic_NE, directions='WS'),
                collect_neighbors(mv_frame_grid, pred_coords_ic_WE, directions='NS'),
                collect_neighbors(mv_frame_grid, pred_coords_ic_WS, directions='NE'),
                collect_neighbors(mv_frame_grid, pred_coords_ic_SE, directions='NW')))
            predicted_vectors.append(fast_mvutils.fast_vectorized_pvm(neighbor_vectors))
        # Assign the predicted vectors to the respective MBs in the grid
        fill_macroblocks(
            mv_frame_grid,
            np.vstack(predicted_coords),
            np.vstack(predicted_vectors).round())
        # Update `intracoded_grid`
        predicted_coords_idx = no_ic_neighbor | one_ic_neighbor
        if i > 0:
            predicted_coords_idx |= two_ic_neighbor
        row_idx, col_idx = intracoded_coords[predicted_coords_idx].T
        intracoded_grid[row_idx, col_idx] = False
    return mv_frame_grid

## Split the stream of motion vectors in raw format into a 7-tuple, according to
# their block type, where each element of the tuple is an ndarray of the same
# kind containing the respective motion vectors. The order of the block types in
# the result tuple is as follows: (16x16,8x16,16x8,8x8,4x8,8x4,4x4), where NxM
# means N rows and M columns.
#
# @type mv_frame_raw    Nx4-ndarray, where N is the number of motion vectors in
#                       the frame
# @param mv_frame_raw  Sequence of motion vectors from the raw format, where
#                       each row of the matrix is of the kind [x,y,dx,dy], where
#                       x,y are the coordinates, and dx,dy are the values of a
#                       motion vector.
# @return  7-tuple of Nx4-ndarray, explained above.
def split_by_blocktype(mv_frame_raw):
    # Classify MVs according to whether their positions are multiple of 8 or not
    coord_multiple_of_8 = mv_frame_raw[:,:2] % VIDEO_MB_HALF_SIZE == 0
    coord_not_multiple_of_8 = ~coord_multiple_of_8
    row_multiple_of_8 = coord_multiple_of_8[:,1]
    col_multiple_of_8 = coord_multiple_of_8[:,0]
    row_not_multiple_of_8 = coord_not_multiple_of_8[:,1]
    col_not_multiple_of_8 = coord_not_multiple_of_8[:,0]

    # Apply the same logic to finer MVs, classify whether they divide by 4
    finer_mvs_raw = mv_frame_raw[row_not_multiple_of_8 & col_not_multiple_of_8, :]
    coord_multiple_of_4 = finer_mvs_raw[:,:2] % VIDEO_MB_QUARTER_SIZE == 0
    coord_not_multiple_of_4 = ~coord_multiple_of_4
    row_multiple_of_4 = coord_multiple_of_4[:,1]
    col_multiple_of_4 = coord_multiple_of_4[:,0]
    row_not_multiple_of_4 = coord_not_multiple_of_4[:,1]
    col_not_multiple_of_4 = coord_not_multiple_of_4[:,0]
    return (
        mv_frame_raw[row_multiple_of_8 & col_multiple_of_8, :],
        mv_frame_raw[row_not_multiple_of_8 & col_multiple_of_8, :],
        mv_frame_raw[row_multiple_of_8 & col_not_multiple_of_8, :],
        finer_mvs_raw[row_multiple_of_4 & col_multiple_of_4, :],
        finer_mvs_raw[row_not_multiple_of_4 & col_multiple_of_4, :],
        finer_mvs_raw[row_multiple_of_4 & col_not_multiple_of_4, :],
        finer_mvs_raw[row_not_multiple_of_4 & col_not_multiple_of_4, :]
    )

## Given a tuple which splits the stream of MVs according to their block types
# as described in \ref split_by_blocktype, scale their coordinates in order to
# fit the respective sub-sampled grid. For example, the MVs in MBs of kind 16x16
# have their coordinates divided by 16, the MVs in 8x16 blocks have the
# x-coordinate divided by 16 and the y-coordinate divided by 8, etc.
#
# *NOTE*: This function has side-effects on the parameter `mv_frame_split`,
# whose value is changed by the function.
#
# @type mv_frame_split  7-tuple of Nx4-ndarray
# @param mv_frame_split A tuple with the sequence of motion vectors split by
#                       the function \ref split_by_blocktype.
# @return 7-tuple of Nx4-ndarray, which is a reference to `mv_frame_split`
def transform_coordinates(mv_frame_split):
    # 16x16 Macro block
    mv_frame_split[0][:,:2] //= VIDEO_MB_SIZE
    # Horizontal 8x16 Sub Macro block (8 lines 16 columns)
    mv_frame_split[1][:,0] //= VIDEO_MB_SIZE
    mv_frame_split[1][:,1] //= VIDEO_MB_HALF_SIZE
    # Vertical 16x8 Sub Macro block
    mv_frame_split[2][:,0] //= VIDEO_MB_HALF_SIZE
    mv_frame_split[2][:,1] //= VIDEO_MB_SIZE
    # 8x8 Sub Macro Block
    mv_frame_split[3][:,:2] //= VIDEO_MB_HALF_SIZE
    # Horizontal 4x8 Sub Macro Block
    mv_frame_split[4][:,0] //= VIDEO_MB_HALF_SIZE
    mv_frame_split[4][:,1] //= VIDEO_MB_QUARTER_SIZE
    # Vertical 8x4 Sub Macro Block
    mv_frame_split[5][:,0] //= VIDEO_MB_QUARTER_SIZE
    mv_frame_split[5][:,1] //= VIDEO_MB_HALF_SIZE
    # 4x4 Sub Macro Block
    mv_frame_split[6][:,:2] //= VIDEO_MB_QUARTER_SIZE
    return mv_frame_split

def inverse_transform_coordinates(mv_frame_split):
    # 16x16 Macro block
    mv_frame_split[0][:,:2] *= VIDEO_MB_SIZE
    mv_frame_split[0][:,:2] += VIDEO_MB_HALF_SIZE
    # Horizontal 8x16 Sub Macro block (8 lines 16 columns)
    mv_frame_split[1][:,0] *= VIDEO_MB_SIZE
    mv_frame_split[1][:,0] += VIDEO_MB_HALF_SIZE
    mv_frame_split[1][:,1] *= VIDEO_MB_HALF_SIZE
    mv_frame_split[1][:,1] += VIDEO_MB_QUARTER_SIZE
    # Vertical 16x8 Sub Macro block
    mv_frame_split[2][:,0] *= VIDEO_MB_HALF_SIZE
    mv_frame_split[2][:,0] += VIDEO_MB_QUARTER_SIZE
    mv_frame_split[2][:,1] *= VIDEO_MB_SIZE
    mv_frame_split[2][:,1] += VIDEO_MB_HALF_SIZE
    # 8x8 Sub Macro Block
    mv_frame_split[3][:,:2] *= VIDEO_MB_HALF_SIZE
    mv_frame_split[3][:,:2] += VIDEO_MB_QUARTER_SIZE
    return mv_frame_split

def transform_coords_macroblock_space(mv_frame_split):
    result = [mv_frame_split[0]]

    result.append(mv_frame_split[1][:,:2].copy())
    result[1][:,1] //= 2
    result.append(mv_frame_split[2][:,:2].copy())
    result[2][:,0] //= 2
    result.append(mv_frame_split[3][:,:2] // 2)

    result.append(mv_frame_split[4][:,:2].copy())
    result[4][:,0] //= 2
    result[4][:,1] //= 4

    result.append(mv_frame_split[5][:,:2].copy())
    result[5][:,0] //= 4
    result[5][:,1] //= 2

    result.append(mv_frame_split[6][:,:2] // 4)
    return result

## Transform the array of motion vectors from a slice-like format into an array
# of motion vectors in a grid-like format, performing intraprediction of motion
# vectors.
#
# It is assumed that the coordinate of every vector corresponds to the middle of
# its respective prediction block. For example, a prediction block of the kind
# 16x16px at the top-left corner has a MV with coordinate (8,8); a prediction
# block of the kind 8x16px at the top-left corner has a MV with coordinate
# (x,y) = (8,4), and so forth.
# It is also assumed that the `frame_shape` is large enough to fit the motion
# vector frame.
#
# @type mv_frame_raw    Nx4-ndarray, where N is the number of motion vectors in
#                       the respective frame.
# @param mv_frame_raw   The sequence of motion vectors. This array has shape
#                       (N,4), where N is the number of motion vectors in the
#                       frame. Each line of this array is of the kind
#                       [x,y,dx,dy], where x,y are the coordinates, and dx,dy
#                       are the values of a motion vector.
#
# @type frame_shape     2-tuple
# @param frame_shape    Shape of the resulting grid in the form (rows,cols)
# @return   HxWx2-ndarray of type int, where H is the height and W is the width
#           of the frame, and every element of this grid is a two-dimensional
#           vector.
def fit_and_fill_grid(mv_frame_raw, frame_shape):
    # mv_frame_raw[:,2:] = np.rint(mv_frame_raw[:,2:] / 4).astype(mv_frame_raw.dtype)
    # Split the MVs according to their block types (e.g. 16x16, 8x16, etc.)
    # and transform their coordinates to a smaller resolution respectively
    mv_frame_split = transform_coordinates(
        split_by_blocktype(mv_frame_raw))
    grid_mb_shape = (
        frame_shape[0] // VIDEO_MB_SIZE_IN_PB,
        frame_shape[1] // VIDEO_MB_SIZE_IN_PB,
        2
    )
    # Start assigning the 16x16 MVs to the first grid, then proceed by
    # resizing this grid into a larger one and assigning the finer MVs 8x16
    # and 16x8)
    grid_mb = fit_to_grid(
        mv_frame_split[0],
        np.zeros(grid_mb_shape, dtype=mv_frame_split[0].dtype))
    grid_mb_8x16 = fit_to_grid(
        mv_frame_split[1],
        cv2.resize(grid_mb, (0,0), fx=1, fy=2,
                   interpolation=cv2.INTER_NEAREST))
    grid_mb_16x8 = fit_to_grid(
        mv_frame_split[2],
        cv2.resize(grid_mb, (0,0), fx=2, fy=1,
                   interpolation=cv2.INTER_NEAREST))
    # Combine the horizontal and vertical partitions of size 8x16 and 16x8
    grid_mb_8x8 = cv2.resize(
            grid_mb_8x16, (0,0), fx=2, fy=1, interpolation=cv2.INTER_NEAREST) \
        | cv2.resize(
            grid_mb_16x8, (0,0), fx=1, fy=2, interpolation=cv2.INTER_NEAREST)
    # Assign the MVs which correspond to blocks 8x8
    grid_mb_8x8 = fit_to_grid(mv_frame_split[3], grid_mb_8x8)
    # Do the analogous step, now for blocks of size 4x8, and 8x4
    grid_mb_4x8 = fit_to_grid(
        mv_frame_split[4],
        cv2.resize(grid_mb_8x8, (0,0), fx=1, fy=2,
                   interpolation=cv2.INTER_NEAREST))
    grid_mb_8x4 = fit_to_grid(
        mv_frame_split[5],
        cv2.resize(grid_mb_8x8, (0,0), fx=2, fy=1,
                   interpolation=cv2.INTER_NEAREST))
    # Combine the horizontal and vertical partitions of size 4x8 and 8x4
    grid_mb_4x4 = cv2.resize(
            grid_mb_4x8, (0,0), fx=2, fy=1, interpolation=cv2.INTER_NEAREST) \
        | cv2.resize(
            grid_mb_8x4, (0,0), fx=1, fy=2, interpolation=cv2.INTER_NEAREST)
    # Finally, assign the MVs of blocks 4x4 into the grid
    grid_mb_4x4 = fit_to_grid(mv_frame_split[6], grid_mb_4x4)
    # Find out intra-coded blocks and apply intraprediction. First, create a
    # matrix full of True, then assign False to those MBs where there are
    # motion vectors.
    pad_shape = grid_mb_shape[0]+2, grid_mb_shape[1]+2
    padded_grid_intracoded = np.ones(pad_shape, dtype=np.bool)
    # Note that grid_intracoded is a masked array (pointer)
    grid_intracoded = padded_grid_intracoded[1:-1, 1:-1]
    mv_frame_split_mb_space = transform_coords_macroblock_space(mv_frame_split)
    for mv_category in mv_frame_split_mb_space:
        fit_to_grid(mv_category, grid_intracoded, False)
    return (
        intrapredict(grid_mb_4x4, padded_grid_intracoded),
        mv_frame_split)

## Transform the stream of motion vectors from a slice-like format into a stream
# of motion vectors in a grid-like format, performing intraprediction of motion
# vectors. If the input frame is empty, yield the same empty ndarray.
#
# It is assumed that the coordinate of every vector corresponds to the middle of
# its respective prediction block. For example, a prediction block of the kind
# 16x16px at the top-left corner has a MV with coordinate (8,8); a prediction
# block of the kind 8x16px at the top-left corner has a MV with coordinate
# (x,y) = (8,4), and so forth.
# It is also assumed that the `frame_shape` is large enough to fit the motion
# vector frame.
#
# @type mv_frame_stream     Iterator Nx4-ndarray, where N is the number of
#                           motion vectors in the respective frame.
# @param mv_frame_stream    The stream of motion vector frames. It should be a
#                           list where each element represents a frame and each
#                           element is a two-dimensional array. This array has
#                           shape (N,4), where N is the number of motion
#                           vectors in the frame. Each line of this array is of
#                           the kind [x,y,dx,dy], where x,y are the coordinates,
#                           and dx,dy are the values of a motion vector.
#
# @type frame_shape     2-tuple
# @param frame_shape    Shape of the resulting grid in the form (rows,cols)
# @return   Iterator of HxWx2-ndarray of type int, where H is the height and
#           W is the width of the frame, and every element of this grid is a
#           two-dimensional vector.
def transform_to_grid(mv_frame_stream, frame_shape):
    return (
        fit_and_fill_grid(mvs_raw, frame_shape) \
            if mvs_raw.size else (mvs_raw,mvs_raw)
        for mvs_raw in mv_frame_stream
    )
