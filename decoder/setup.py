from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("decoder",
        sources = [
            "decoder.pyx",
            "mvextract/concurrentqueue.c",
            "mvextract/streambuffer.c",
            "mvextract/mvextract.c"
        ],
        include_dirs = [
            "../libs/FFmpeg/ffmpeg_build/include",
            "../."],
        library_dirs = ["../libs/FFmpeg/ffmpeg_build/lib"],
        libraries = [
            "avutil",
            "avformat",
            "avcodec",
            "swresample",
            "z",
            "m",
            "pthread"],
        extra_compile_args=["-std=c11"]
    )
]))
