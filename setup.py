from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [Extension(
                "fast_lpca",
                sources=["fast_lpca.pyx"]
            )]

setup(
    ext_modules = cythonize(extensions)
)

