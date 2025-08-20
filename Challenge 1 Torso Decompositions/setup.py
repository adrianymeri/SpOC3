from setuptools import setup
from Cython.Build import cythonize
import numpy

def build():
    setup(
        ext_modules=cythonize("solver_cython.pyx", quiet=True),
        include_dirs=[numpy.get_include()],
        script_args=['build_ext', '--inplace']
    )
