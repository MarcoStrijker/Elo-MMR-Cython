from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("elo_mmr_cython\*.pyx")
)
