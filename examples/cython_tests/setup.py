from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension
from Cython.Build import cythonize

setup(
    name="mqa",
    ext_modules=cythonize(
        Extension(
            "mqa",
            sources=["mqa.pyx"],
            language="c++",
        )
    ),
    cmdclass={"build_ext": BuildExtension},
)
