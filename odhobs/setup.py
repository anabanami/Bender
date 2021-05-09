from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

extension = Extension("ho", sources=["ho.pyx"], include_dirs=[np.get_include()])
setup(cmdclass={"build_ext": build_ext}, ext_modules=[extension])
