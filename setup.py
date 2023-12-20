import setuptools

with open("README.md", "r",encoding="utf-8") as fh:
    long_description = fh.read()

# Setting up
setuptools.setup(
    name="scientisttools",
    version="0.1.3",
    author="Duverier DJIFACK ZEBAZE",
    author_email="duverierdjifack@gmail.com",
    description="Python library for multidimensional analysis, classification - clustering analysis and discriminant analysis.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.26.2",
                      "matplotlib>=3.5.3",
                      "scikit-learn>=1.2.2",
                      "pandas>=1.5.3",
                      "mapply>=0.1.21",
                      "plotnine>=0.10.1",
                      "plydata>=0.4.3",
                      "pingouin>=0.5.3",
                      "scientistmetrics>=0.0.3",
                      "ggcorrplot>=0.0.2"],
    python_requires=">=3.10",
    include_package_data=True,
    package_data={"": ["data/*.xlsx",
                       "data/*.xls",
                       "data/*.txt",
                       "data/*.csv",
                       "data/*.rda"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)