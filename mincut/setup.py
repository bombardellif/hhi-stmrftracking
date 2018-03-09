from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("mincut",
        sources = [
            "mincut.pyx",
            "ibfs/ibfs.cpp"
        ],
        language="c++"
    )
]))
