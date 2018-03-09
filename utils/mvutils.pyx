
from libc.stddef cimport size_t
from libc.stdint cimport int16_t
import numpy as np
cimport numpy as cnp

from utils cimport arraywrapper

cdef extern from "cpp/fast_mvutils.h":
    double* c_fast_vectorized_pvm(int16_t (*vectors)[2], int shape[2])

def fast_vectorized_pvm(cnp.ndarray vectors):
    cdef double* c_result
    cdef int shape[2]
    shape[0],shape[1] = vectors.shape[:2]
    c_result = c_fast_vectorized_pvm(<int16_t(*)[2]>vectors.data, shape)
    # Result array has shape[0] lines and two columns
    shape[1] = 2
    return arraywrapper.create_ndarray(<void*>c_result, shape, cnp.NPY_DOUBLE)
