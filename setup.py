from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'Python library for multidimensional analysis'
LONG_DESCRIPTION = 'A python package dedicated to multivariate Exploratory Data Analysis'

# Setting up
setup(
    name="scientisttools",
    version=VERSION,
    author="Duverier DJIFACK ZEBAZE",
    author_email="duverierdjifack@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy>=1.11.0",
                      "matplotlib>=2.0.0",
                      "scikit-learn>=0.18.0",
                      "pandas>=0.19.0"],
    python_requires=">=3",
    package_data={"": ["*.txt"]},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)