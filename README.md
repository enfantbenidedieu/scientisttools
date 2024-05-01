# scientisttools : Python library for multidimensional analysis

## About scientisttools

scientisttools is a Python package dedicated to multivariate Exploratory Data Analysis and clustering analysis.

## Why use scientisttools?

* It performs **classical principal component methods** : 
    * Principal Components Analysis (PCA)
    * Principal Components Analysis with partial correlation matrix (PartialPCA)
    * Exploratory Factor Analysis (EFA)
    * Classical Multidimensional Scaling (CMDSCALE)
    * Metric and Non - Metric Multidimensional Scaling (MDS)
    * Correspondence Analysis (CA)
    * Multiple Correspondence Analysis (MCA)
    * Factor Analysis of Mixed Data (FAMD)
    * Multiple Factor Analysis (MFA)
    * Multiple Factor Analysis for qualitatives/categoricals variables (MFAQUAL)
    * Multiple Factor Analysis of Mixed Data (MFAMIX)
    * Multiple Factor Analysis of Contingence Tables (MFACT)

* In some methods, it allowed to **add supplementary informations** such as supplementary individuals and/or variables.
* It provides a geometrical point of view, a lot of graphical outputs.
* It provides efficient implementations, using a scikit-learn API.

Those statistical methods can be used in two ways :
* as descriptive methods ("datamining approach")
* as reduction methods in scikit-learn pipelines ("machine learning approach")

scientisttools also performs clustering analysis

* **Clustering analysis**:
    * Hierarchical Clustering on Principal Components (HCPC)
    * Variables Hierarchical Clustering Analysis (VARHCA)
    * Variables Hierarchical Clustering Analysis on Principal Components (VARHCPC)
    * Categorical Variables Hierarchical Clustering Analysis (CATVARHCA)


Notebooks are availabled.

## Installation

### Dependencies

scientisttools requires 

```
Python >=3.10
numpy >=1.26.4
matplotlib >=3.8.4
scikit-learn >=1.2.2
pandas >=2.2.2
polars >=0.19.2
mapply >=0.1.21
plotnine >=0.10.1
pingouin >=0.5.4
scientistmetrics >=0.0.4
```

### User installation

You can install scientisttools using `pip` :

```
pip install scientisttools
```

Tutorial are available

## Author

Duvérier DJIFACK ZEBAZE ([duverierdjifack@gmail.com](duverierdjifack@gmail.com))
