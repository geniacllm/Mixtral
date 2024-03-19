from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("helpers", ["helpers.cpp"]),
]

setup(
    name="helpers",
    version="1.0",
    author="Your Name",
    description="Pybind11 example plugin",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)