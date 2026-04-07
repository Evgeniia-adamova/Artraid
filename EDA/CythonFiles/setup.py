from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import pandas as pd

extensions = [
    Extension(
        name="corrFuncs",
        sources=["corrFuncs.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions)
)