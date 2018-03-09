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
## @package imgutils
# Library of utility functions for processing images

import numpy as np

# Predefined Constants
IDX_FST_ORD_NEIGHBORS = [
    ((0,-1,0,None),(1,None,0,None)),    # North
    ((0,None,0,-1),(0,None,1,None)),    # West
    ((1,None,0,None),(0,-1,0,None)),    # South
    ((0,None,1,None),(0,None,0,-1))    # East
]
IDX_SND_ORD_NEIGHBORS = [
    ((0,-1,0,-1),(1,None,1,None)),    # Northwest
    ((1,None,0,-1),(0,-1,1,None)),    # Southwest
    ((1,None,1,None),(0,-1,0,-1)),    # Southeast
    ((0,-1,1,None),(1,None,0,-1))    # Northeast
]

## Given a mask return a new array that aggregates each block of pixels of size
# block_size by block_size in the mask into a single entry in the new array,
# which has the number of pixels with value True, thus aggregating the True
# values of the mask by a sum into square blocks of size block_size.
#
# Block are aligned at the top left corner of the mask, if a block extrapolates
# the mask's limits the pixels outside the image are considered to be False.
#
# @type mask    HxW-ndarray, where H is the height and W the Width of the mask
# @param mask   Binary mask to be aggregated into blocks
#
# @type block_size  int
# @param block_size The size of the side of the squared block
# @return   H`xW`-ndarray, where H` is the new picture's height which equals
#           ceil(H/block_size), and W`is the new picture's width which equals
#           ceil(W/block_size)
def blockify(mask, block_size):
    mask_shape = np.array(mask.shape)
    # Determine how much the image has to be "padded" on the right side and at
    # the bottom so that the blocks fill the whole matrix
    super_block = block_size ** 2
    gap = (super_block - mask_shape % super_block) % super_block
    # Pad the matrix with zeros
    padded_mask = np.pad(
        mask,
        ((0, gap[0]),(0,gap[1])),
        'constant',
        constant_values=0)
    padded_mask_rows = padded_mask.shape[0]
    # Accumulate the values of the image in squared blocks. This is done in two
    # phases: (i) Compute the sums in the rows of every block
    partial_accum = padded_mask.reshape(-1, block_size).sum(axis=1) \
        .reshape(padded_mask_rows, -1)
    partial_accum_columns = partial_accum.shape[1]
    # (ii) Compute the sums in the columns of every block
    final_accum = partial_accum.T.reshape(-1, block_size).sum(axis=1) \
        .reshape(partial_accum_columns, -1).T

    return final_accum

## Reduce image depth to two (True or False) by applying the integer division
# of the pixel values by `threshold`
#
# @type img ndarray of type int
# @param img    Image to be discretized
#
# @type threshold   int
# @param threshold  The threshold which determines whether the pixel is True or
#                   False
# @return  ndarray of same size as `img` but of type bool
def discretize_bw(img, threshold):
    return (img // threshold).astype(np.bool)

## Given the shape of a two dimensional matrix, return a matrix with the same
# size for the two first dimensions (i.e. same number of rows and columns),
# where every element is a two dimensional vector and its value equals its
# index in this matrix. Thus, the result matrix A, has the property that
# A[i,j] = [i,j] for all i in range(shape[0]), j in range(shape[1]).
#
# @type shape   n-tuple of int, where n >= 2
# @param shape  The tuple which determines, by its two first values, the shape
#               of the first two dimensions of the result matrix
# @return       HxWx2-ndarray, where H = shape[0] and W = shape[1]
def coordinates_matrix(shape):
    rows_cols = np.indices(shape[:2])
    return np.transpose(rows_cols, axes=(1,2,0))

## Return a new matrix of same shape as `source`, where each entry has the sum
# of its neighbors (adjacent entries). The exact adjacent indices are defined
# by the argument `indices`.
#
# As the computation is performed as matrix multiplications, the argument
# `indices` defines the range indices of each one of the neighbor directions,
# thus its length equals the number of neighbor directions to be considered,
# e.g. 4 for North, West, South, East. Each element is then a pair (N,B),
# whose first coordinate has the range index for the source (so that one can
# access the source as `source[N[0]:N[1], N[2]:N[3]]`), and the second has the
# range index for the result (for accessing as `result[B[0]:B[1], B[2]:B[3]]`).
# Predefined indices are defined as constants IDX_FST_ORD_NEIGHBORS, and
# IDX_SND_ORD_NEIGHBORS in this module.
#
# @type source  ndarray
# @param source The source to obtain the neighbor values
# @type indices     list of 2-tuples of 4-tuples of int
# @param indices    Indices of the neighbors for matrix multiplication, see the
#                   documentation above.
# @return ndarray
def sum_neighbors(source, indices):
    result = np.zeros(source.shape)
    for neighbor,block in indices:
        result[block[0]:block[1], block[2]:block[3]] \
            += source[neighbor[0]:neighbor[1], neighbor[2]:neighbor[3]]
    return result

## Same as \ref sum_neighbors, but instead of sum the subtraction with each
# neighbor is performed (or the negative accumulation, as you may prefer).
#
# @see sum_neighbors
# @type source  ndarray
# @param source The source to obtain the neighbor values
# @type indices     list of 2-tuples of 4-tuples of int
# @param indices    Indices of the neighbors for matrix multiplication, see the
#                   documentation above.
# @return ndarray
def subtract_neighbors(source, indices):
    result = np.zeros(source.shape)
    for neighbor,block in indices:
        result[block[0]:block[1], block[2]:block[3]] \
            -= source[neighbor[0]:neighbor[1], neighbor[2]:neighbor[3]]
    return result

## Determine the indices of the object's bounding box with a margin of one as a
# 4-tuple (top, bottom, left, right). In a matrix A, the bounding box should be
# accessed as A[top:bottom, left:right].
#
# @type object_mask     HxW-ndarray of type bool, where H is the height and W
#                       the width of the mask
# @param object_mask    The mask has value True in the coordinates where the
#                       the object is in the frame.
# @type border  int
# @param border Extra border for the  bounding box
# @return  4-tuple of int (top, bottom, left, right)
def bounding_box(object_mask, border=0):
    bottom_lim,right_lim = object_mask.shape
    object_row,object_col = object_mask.nonzero()

    if object_row.size and object_col.size:
        bbox_top = max(object_row.min()-border, 0)
        bbox_bottom = min(object_row.max()+border+1, bottom_lim)
        bbox_left = max(object_col.min()-border, 0)
        bbox_right = min(object_col.max()+border+1, right_lim)
        return bbox_top, bbox_bottom, bbox_left, bbox_right
    else:
        return None
