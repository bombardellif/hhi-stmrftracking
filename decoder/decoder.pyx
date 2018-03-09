from libcpp cimport bool
from libc.stddef cimport size_t
from libc.stdint cimport int16_t
from cpython cimport pycapsule
import numpy as np
cimport numpy as cnp

cimport mvextract
from streambuffer cimport StreamBuffer
from utils cimport arraywrapper

# Global mapping of id to stream buffer pointer
buffer_map = {}
EMPTY_ARRAY = np.empty(0, dtype=np.int16)

def init():
    mvextract.init()

def read_videostream(tracking_id, py_url):
    cdef StreamBuffer* buff
    cdef bytes py_byte_url = py_url.encode()
    cdef const char* url = py_byte_url
    ret = False
    if tracking_id not in buffer_map:
        buff = mvextract.read_videostream(url)
        if buff is not NULL:
            buffer_map[tracking_id] = pycapsule.PyCapsule_New(<void*>buff,
                                                              NULL, NULL)
            ret = True
    return ret

def stop(tracking_id):
    cdef StreamBuffer* buff
    if tracking_id in buffer_map:
        buff = <StreamBuffer*>pycapsule.PyCapsule_GetPointer(
            buffer_map[tracking_id],
            NULL)
        mvextract.stop(buff)

def get_next_frame(tracking_id):
    cdef StreamBuffer* buff
    cdef int16_t (*data)[4]
    cdef size_t size
    cdef int shape[2]
    cdef bool status = False
    cdef cnp.ndarray array = EMPTY_ARRAY
    if tracking_id in buffer_map:
        buff = <StreamBuffer*>pycapsule.PyCapsule_GetPointer(
            buffer_map[tracking_id],
            NULL)
        status = mvextract.get_next_frame(buff, &data, &size)
        shape[0],shape[1] = (<int>size),4
        if status and data is not NULL:
            array = arraywrapper.create_ndarray(<void*>data, shape,
                                                cnp.NPY_INT16)
    return status, array

def destroy(tracking_id):
    cdef StreamBuffer* buff
    if tracking_id in buffer_map:
        buff = <StreamBuffer*>pycapsule.PyCapsule_GetPointer(
            buffer_map[tracking_id],
            NULL)
        mvextract.destroy(buff)
        del buffer_map[tracking_id]
