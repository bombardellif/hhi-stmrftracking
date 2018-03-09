
from libcpp cimport bool
from libc.stddef cimport size_t
from libc.stdint cimport int16_t

from streambuffer cimport StreamBuffer

cdef extern from "mvextract/mvextract.h":
    bool init()

    StreamBuffer* read_videostream(const char url[])

    void stop(StreamBuffer* buff)

    bool get_next_frame(StreamBuffer* buff, int16_t (**out)[4], size_t* size)

    void destroy(StreamBuffer* buff)
