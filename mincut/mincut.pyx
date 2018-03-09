# distutils: language=c++
# cython: profile=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.limits cimport INT_MAX
from libcpp cimport bool

cimport ibfs
from ibfs cimport IBFSInitMode
from ibfs cimport IBFSGraph

# Defines
cdef enum StEdgeType:
    S_EDGE=0,
    T_EDGE=1
cdef enum GridEdgeType:
    HORIZ_EDGE=0,
    VERT_EDGE=1
# Constants
cdef double GAMMA_GRID = 0.5
cdef double TEMPORAL_WEIGHT = 6
cdef int PRECISION_DECIMAL = 3
# Pre-calculated values
cdef double GAMMA_GRID_SQR = GAMMA_GRID**2
cdef int SCALE_PRECISION = 10**PRECISION_DECIMAL

def set_params(gamma_grid, temp_weight):
    global GAMMA_GRID,\
        GAMMA_GRID_SQR,\
        TEMPORAL_WEIGHT
    GAMMA_GRID = gamma_grid
    GAMMA_GRID_SQR = GAMMA_GRID**2
    TEMPORAL_WEIGHT = temp_weight

cdef class PyIBFSGraph:
    cdef IBFSGraph* c_ibfs
    cdef bool first
    cdef unsigned int source
    cdef unsigned int sink
    cdef int [:] s_capacities
    cdef int [:] t_capacities
    cdef int [:,:] grid_horiz_capacities
    cdef int [:,:] grid_vert_capacities

    ## Constructor
    def __cinit__(self,
            const unsigned int grid_size,
            const unsigned int num_edges):
        self.first = True
        # Extern IBFS
        self.c_ibfs = new IBFSGraph(IBFSInitMode.IB_INIT_FAST)
        self.c_ibfs.initSize(grid_size + 2, num_edges)
        # Add Source and sink to graph (other nodes don't need initialization)
        self.source = 0
        self.sink = grid_size + 1
        self.c_ibfs.addNode(self.source, INT_MAX, 0)
        self.c_ibfs.addNode(self.sink, 0, INT_MAX)

    ## Destructor
    def __dealloc__(self):
        del self.c_ibfs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_st_edge(self,
            const StEdgeType st_edge_type,
            cnp.ndarray[cnp.float_t, ndim=1] new_capacities):
        cdef unsigned int k
        cdef int capacity
        cdef cnp.ndarray[cnp.int32_t, ndim=1] increment
        cdef cnp.ndarray[cnp.int32_t, ndim=1] rounded_new_capacities \
             = np.rint(SCALE_PRECISION * new_capacities).astype(np.int32)
        if self.first:
            if st_edge_type == S_EDGE:
                for k in range(rounded_new_capacities.size):
                    capacity = rounded_new_capacities[k]
                    self.c_ibfs.addEdge(self.source, k+1, capacity, 0)
                self.s_capacities = rounded_new_capacities
            else:
                for k in range(rounded_new_capacities.size):
                    capacity = rounded_new_capacities[k]
                    self.c_ibfs.addEdge(k+1, self.sink, capacity, 0)
                self.t_capacities = rounded_new_capacities
        else:
            if st_edge_type == S_EDGE:
                increment = rounded_new_capacities - self.s_capacities
                for k in range(increment.size):
                    if increment[k] != 0:
                         self.c_ibfs.incEdge(self.source, k+1, increment[k], 0)
                self.s_capacities = rounded_new_capacities
            else:
                increment = rounded_new_capacities - self.t_capacities
                for k in range(increment.size):
                    if increment[k] != 0:
                        self.c_ibfs.incEdge(k+1, self.sink, increment[k], 0)
                self.t_capacities = rounded_new_capacities

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_grid_edge(self,
            const GridEdgeType grid_edge_type,
            cnp.ndarray[cnp.float_t, ndim=2] new_capacities):
        cdef unsigned int i,j,k
        cdef int capacity, inc
        cdef unsigned int rows = new_capacities.shape[0]
        cdef unsigned int cols = new_capacities.shape[1]
        cdef cnp.ndarray[cnp.int32_t, ndim=2] increment
        cdef cnp.ndarray[cnp.int32_t, ndim=2] rounded_new_capacities \
             = np.rint(SCALE_PRECISION * new_capacities).astype(np.int32)
        if self.first:
            if grid_edge_type == HORIZ_EDGE:
                k = 1
                for i in range(rows):
                    for j in range(cols):
                        capacity = rounded_new_capacities[i,j]
                        self.c_ibfs.addEdge(k, k+1, capacity, capacity)
                        k += 1
                    # At the last column there is no edge, so just increment k
                    k += 1
                self.grid_horiz_capacities = rounded_new_capacities
            else:
                k = 1
                for i in range(rows):
                    for j in range(cols):
                        capacity = rounded_new_capacities[i,j]
                        self.c_ibfs.addEdge(k, k+cols, capacity, capacity)
                        k += 1
                self.grid_vert_capacities = rounded_new_capacities
        else:
            if grid_edge_type == HORIZ_EDGE:
                increment = rounded_new_capacities - self.grid_horiz_capacities
                k = 1
                for i in range(rows):
                    for j in range(cols):
                        inc = increment[i,j]
                        if inc != 0:
                            self.c_ibfs.incEdge(k, k+1, inc, inc)
                        k += 1
                    k += 1
                self.grid_horiz_capacities = rounded_new_capacities
            else:
                increment = rounded_new_capacities - self.grid_vert_capacities
                k = 1
                for i in range(rows):
                    for j in range(cols):
                        inc = increment[i,j]
                        if inc != 0:
                            self.c_ibfs.incEdge(k, k+cols, inc, inc)
                        k += 1
                self.grid_vert_capacities = rounded_new_capacities

    cdef inline void computeMaxFlow(self):
        if self.first:
            # Initialize intern structures
            self.c_ibfs.initGraph()
            self.first = False
        # Run optimization algorithm
        self.c_ibfs.computeMaxFlow(True)

cdef cnp.ndarray[cnp.float_t, ndim=1] source_arc_weights(
        cnp.ndarray mv,
        cnp.ndarray minus_log_hist,
        cnp.ndarray xedges,
        cnp.ndarray yedges,
        cnp.ndarray temporal_energy):
    xedgeright = np.digitize(mv[:,:,0].reshape(-1), xedges)
    yedgeright = np.digitize(mv[:,:,1].reshape(-1), yedges)
    return minus_log_hist[yedgeright-1,xedgeright-1] \
        + TEMPORAL_WEIGHT * temporal_energy.reshape(-1)

cdef cnp.ndarray[cnp.float_t, ndim=1] sink_arc_weights(
        cnp.ndarray mv,
        cnp.ndarray minus_log_hist,
        cnp.ndarray xedges,
        cnp.ndarray yedges,
        cnp.ndarray temporal_energy):
    xedgeright = np.digitize(mv[:,:,0].reshape(-1), xedges)
    yedgeright = np.digitize(mv[:,:,1].reshape(-1), yedges)
    return minus_log_hist[yedgeright-1,xedgeright-1] \
        + TEMPORAL_WEIGHT * (1-temporal_energy).reshape(-1)

cdef cnp.ndarray[cnp.float_t, ndim=2] grid_arc_weights(
        cnp.ndarray mv1,
        cnp.ndarray mv2):
    cdef cnp.ndarray sqr_difference = np.square(mv1 - mv2, dtype=np.int32)
    cdef cnp.ndarray sqr_distance = sqr_difference[:,:,0]
    sqr_distance += sqr_difference[:,:,1]
    cdef cnp.ndarray result = sqr_distance + GAMMA_GRID_SQR
    result *= np.sqrt(result)
    np.divide(GAMMA_GRID, result, out=result)
    return result

cdef class MincutSolver:
    cdef PyIBFSGraph ibfs_graph

    def __cinit__(self):
        self.ibfs_graph = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def solve_mincut(self,
                    cnp.ndarray frame,
                    cnp.ndarray high_pass_frame,
                    cnp.ndarray[cnp.float_t, ndim=2] temporal_energy,
                    cnp.ndarray[cnp.float_t, ndim=2] background_hist,
                    cnp.ndarray[cnp.float_t, ndim=1] bg_xedges,
                    cnp.ndarray[cnp.float_t, ndim=1] bg_yedges,
                    cnp.ndarray[cnp.float_t, ndim=2] foreground_hist,
                    cnp.ndarray[cnp.float_t, ndim=1] fg_xedges,
                    cnp.ndarray[cnp.float_t, ndim=1] fg_yedges,
                    cnp.ndarray mask_outside_bbox):
        # Some variables used throughout the function
        cdef unsigned int rows = frame.shape[0]
        cdef unsigned int cols = frame.shape[1]
        cdef unsigned int grid_size = rows*cols
        cdef unsigned int num_edges
        # Initialize the s-t-graph only once
        if self.ibfs_graph is None:
            num_edges = (rows-1)*cols \
                + rows*(cols-1) \
                + 2*grid_size
            self.ibfs_graph = PyIBFSGraph(grid_size, num_edges)

        # Add the arcs connecting source to grid and grid to sink
        cdef cnp.ndarray minus_log_bg_hist = np.maximum(-np.log(background_hist), 0.)
        cdef cnp.ndarray source_capacities = source_arc_weights(
            frame,
            minus_log_bg_hist,
            bg_xedges,
            bg_yedges,
            temporal_energy)
        # Force outside of bounding box to be background
        source_capacities.reshape((rows,cols))[mask_outside_bbox] = 0.

        self.ibfs_graph.set_st_edge(S_EDGE, source_capacities)
        cdef cnp.ndarray minus_log_fg_hist = np.maximum(-np.log(foreground_hist), 0.)
        cdef cnp.ndarray sink_capacities = sink_arc_weights(
            frame,
            minus_log_fg_hist,
            fg_xedges,
            fg_yedges,
            temporal_energy)
        self.ibfs_graph.set_st_edge(T_EDGE, sink_capacities)
        # Add the arcs of the grid (between neighbors in 4-neighborhood)
        cdef cnp.ndarray horiz_capacities = grid_arc_weights(high_pass_frame[:,:-1],
                                                             high_pass_frame[:,1:])
        self.ibfs_graph.set_grid_edge(HORIZ_EDGE, horiz_capacities)
        cdef cnp.ndarray vert_capacities = grid_arc_weights(high_pass_frame[:-1,:],
                                                            high_pass_frame[1:,:])
        self.ibfs_graph.set_grid_edge(VERT_EDGE, vert_capacities)

        # Compute max-flow/min-cut
        self.ibfs_graph.computeMaxFlow()
        # Construct result mask
        cdef cnp.ndarray[cnp.npy_bool] mask = np.zeros(grid_size, dtype=np.uint8)
        cdef unsigned int k
        for k in range(grid_size):
            mask[k] = self.ibfs_graph.c_ibfs.isNodeOnSrcSide(k+1, 0)
        return mask.reshape((rows,cols))
