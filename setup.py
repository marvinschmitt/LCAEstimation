from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np


ext_modules = [
    Extension(r'generative_models', [r'generative_models.pyx'])
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)
