
from libc.stdlib cimport malloc,free
from cpython cimport PyObject, Py_INCREF
import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef class ArrayWrapper:
    """ Implementation of wrapper class by Gael Varoquaux
        http://gael-varoquaux.info/programming/cython-example-of-exposing-c-
        computed-arrays-in-python-without-data-copies.html """
    cdef void *data
    cdef cnp.npy_intp* shape
    cdef int ndim
    cdef int typenum

    cdef set_data(self, void* data, const int[:] shape, const int typenum):
        """ This cannot be done in the constructor as it must recieve C-level
        arguments.
        """
        cdef unsigned int i
        self.data = data
        self.shape = <cnp.npy_intp*>malloc(sizeof(cnp.npy_intp) * shape.size)
        for i in range(shape.size):
            self.shape[i] = shape[i]
        self.ndim = shape.size
        self.typenum = typenum

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        ndarray = cnp.PyArray_SimpleNewFromData(self.ndim, self.shape,
                                                self.typenum, self.data)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(self.data)
        free(<void*>self.shape)

cdef create_ndarray(void* data, const int[:] shape, const int typenum):
    cdef cnp.ndarray ndarray
    wrapper = ArrayWrapper()
    wrapper.set_data(data, shape, typenum)
    ndarray = np.array(wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*>wrapper
    # Increment the reference count, as the above assignment was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(wrapper)
    return ndarray
