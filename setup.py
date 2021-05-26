try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(r'generative_models', [r'generative_models.pyx'])
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)
