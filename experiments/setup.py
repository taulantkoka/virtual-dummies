from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="helpers",                 # import name in Python
        sources=["helpers.pyx"],         # your Cython file
        include_dirs=[np.get_include()], # NumPy headers
        language="c++"                   # needed if you use C++
    )
]

setup(
    name="helpers",
    ext_modules=cythonize(extensions, language_level="3"),
)