## @package globalmotion
# Library for computing global motion estimation and compensation on motion
# vector fields.

import numpy as np
import cv2
from hhi_stmrftracking import preprocessing
from hhi_stmrftracking import mvutils

# Definitions
TOL = 1e-3
MAXITER = 20
NUM_PARAMS = 6
W_TUNE_CONST = 1.5
MIN_VECTORS_ESTIMATION = 5
NO_MOTION_GMP = np.array([1., 0., 0., 0., 1., 0.])

def collect_global_motion_vectors(mask, mvs_split):
    mb_size = preprocessing.VIDEO_MB_SIZE_IN_PB
    redim_sizes = [
        (mb_size,mb_size),
        (mb_size//2,mb_size),
        (mb_size,mb_size//2),
        (mb_size//2,mb_size//2)
    ]
    selected_blocks = []
    for k,redim_size in enumerate(redim_sizes):
        grid_shape = (
            mask.shape[0] // redim_size[0],
            mask.shape[1] // redim_size[1]
        )
        # Create a bool matrix with true where there are blocks of the current
        # block type k
        bool_grid_blocks = preprocessing.fit_to_grid(
            mvs_split[k],
            np.zeros(grid_shape, dtype=np.bool),
            True)
        # Set blocks inside the area of the object's mask to False in order to
        # exclude these from global motion estimation
        redim_mask = cv2.resize(mask.view(np.uint8), grid_shape[::-1])\
            .view(np.bool)
        bool_grid_blocks[redim_mask] = False
        # Select those blocks which have value True in the respective entry
        coord_i = mvs_split[k][:,1]
        coord_j = mvs_split[k][:,0]
        selected_blocks.append(mvs_split[k][bool_grid_blocks[coord_i, coord_j]])
    result = np.vstack(
        preprocessing.inverse_transform_coordinates(selected_blocks))
    # Convert MVs to pixel scale
    result[:,2:] = np.rint(result[:,2:] / mvutils.MV_SCALE).astype(result.dtype)
    return (
        np.fliplr(result[:,:2]),  # Coordinates as (i,j)
        np.fliplr(result[:,2:])   # Motion vectors as (vi,vj)
    )

def create_H(mv_coord, N):
    rows = mv_coord.shape[0]
    padded_mv_coord = np.hstack(
        (mv_coord, np.ones((rows,1), dtype=mv_coord.dtype)))
    H = np.hstack((
        padded_mv_coord,
        np.zeros((rows, N), dtype=mv_coord.dtype),
        padded_mv_coord
    )).reshape(2*rows, N)
    return H

def estimate_error(x, mv, mv_coord, H):
    estimate = H.dot(x).reshape(-1, 2) - mv_coord
    distance = np.absolute(mv - estimate).sum(axis=1)
    mean_error = distance.mean()
    return (distance, mean_error)

def create_W_diag(per_mv_error, mean_error):
    c_mu = W_TUNE_CONST * mean_error
    W_estim = np.maximum(1 - (per_mv_error / c_mu)**2, 0)**2 \
        if abs(c_mu) != 0 else np.zeros_like(per_mv_error)
    W_diag = np.vstack((W_estim,W_estim)).T.reshape(-1)
    return W_diag

def optimize_parameters(mv, mv_coord):
    # Initial solution: no motion
    x = NO_MOTION_GMP
    if mv.shape[0] < MIN_VECTORS_ESTIMATION:
        return x
    N = NUM_PARAMS
    mv_coord_flat = mv_coord.reshape(-1)
    mv_flat = mv.reshape(-1)
    # Define vector v in R^N
    v = mv_flat + mv_coord_flat
    # Define matrix H in R^NxN
    H = create_H(mv_coord, N)
    # Initialize W as the identity matrix
    W_diag = np.ones(v.shape[0])
    last_mean_error,k,continue_criterium = 0,0,True
    while k < MAXITER and continue_criterium:
        # Define A := H'WH in R^NxN
        A = H.T.dot((W_diag * H.T).T)
        if np.linalg.det(A) == 0:
            return x
        # Define b := H'WV in R^N
        b = H.T.dot(W_diag * v)
        # Solve the linear system Ax = b
        x = np.linalg.solve(A, b)
        # Calculate prediction error
        per_mv_error,mean_error = estimate_error(x, mv, mv_coord, H)
        # calculate next W from current solution
        W_diag = create_W_diag(per_mv_error, mean_error)
        # Loop stops if mean error is near zero or if it did not change much
        continue_criterium = mean_error > TOL \
            and abs(mean_error - last_mean_error) > TOL
        last_mean_error = mean_error
        k += 1
    return x

def estimation(mask, mvs_split):
    # Collect MVs which will determine global motion (non-object)
    mv_coord,mv = collect_global_motion_vectors(mask, mvs_split)
    # Determine solution of the optimization problem
    return optimize_parameters(mv, mv_coord)

def compensation(mv_field, params, out=None):
    rows,cols,_ = mv_field.shape
    coords = np.mgrid[:rows,:cols].reshape(2, -1).T
    # Transform to pixel domain coordinates
    coords *= preprocessing.VIDEO_MB_SIZE_IN_PB
    coords += preprocessing.VIDEO_MB_SIZE_IN_PB // 2
    H = create_H(coords, params.size)
    # global_motion_x_estim = (x,y,1).dot((a1,a2,a3)))
    # global_motion_y_estim = (x,y,1).dot((a4,a5,a6)))
    global_motion_estimate = (H.dot(params).reshape(-1, 2) - coords)\
        .reshape(rows, cols, 2)
    return np.subtract(mv_field, global_motion_estimate, out=out)
