from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("arraywrapper",
        sources = ["arraywrapper.pyx"]),
    Extension("mvutils",
        sources = [
            "mvutils.pyx",
            "cpp/fast_mvutils.cpp"
        ],
        language="c++",
        include_dirs = ["../libs/opencv/include",
            "../."],
        library_dirs = ["../libs/opencv/build/lib"],
        libraries = [
            "m",
            "opencv_core"
        ],
        extra_compile_args=["-std=c++11"]
    )])
)
