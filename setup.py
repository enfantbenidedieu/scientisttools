# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Setting up
setup(
    name = "scientisttools",
    version = "0.2.0post1",
    author = "Duverier DJIFACK ZEBAZE",
    author_email = "djifacklab@gmail.com",
    description = "Python library for multidimensional analysis, classification - clustering analysis and multidimensional analysis",
    long_description_content_type = "text/markdown",
    long_description = open("README.md", "r",encoding="utf-8").read(),
    url = "https://github.com/enfantbenidedieu/scientisttools",
    packages = find_packages(),
    classifiers = [
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    python_requires = ">=3.11",
    install_requires = [
            "numpy>=1.21",
            "pandas>=1.4",
            "scikit-learn>=1.2",
            "statsmodels>=0.14.6",
            "plotnine>=0.10.1",
            "openpyxl>=3.1.5",
            "adjustText>=0.8.2",
            "pyreadr>=0.5.4",
            "mizani>=0.14.4",
            "tabulate>=0.9.0"
        ],
    include_package_data = True,
    package_data = {"": ["data/"]},
    keywords = "multidimensional analysis, clustering analysis, data visualization, statistical analysis, data science, machine learning",
    project_urls = {
        "Bug Reports": "https://github.com/enfantbenidedieu/scientisttools/issues",
        "Source": "https://github.com/enfantbenidedieu/scientisttools",
        "Documentation": "https://scientisttools.readthedocs.io",
    }
)