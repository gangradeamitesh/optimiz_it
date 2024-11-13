from setuptools import setup, find_packages , Extension
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

cython_files = [
    "optimiz/stochastic_gradient_descent",
    "optimiz/gradient_descent",
    "optimiz/facility_location"
]

extensions = [
    #Extension("optimiz.stochastic_gradient_descent", ["optimiz/stochastic_gradient_descent.pyx"]),
    Extension(name.replace("/" , ".") , [name+".pyx"]) for name in cython_files
]

setup(
    name="optimiz",
    version="1.0.1",
    author="Amitesh Gangrade",
    author_email="gangradeamitesh@gmail.com",
    description="A simple optimization library for machine learning",
    long_description="A simple Optimization Library for Machine Learning Algorithms ",
    long_description_content_type="text/markdown",
    url="https://github.com/gangradeamitesh/optimiz_it.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    zip_safe = False
)