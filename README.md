# scientisttools : Python library for multidimensional analysis

## About scientisttools

**scientisttools** is a `Python` package dedicated to multivariate Exploratory Data Analysis.

## Why use scientisttools?

* It performs **classical principal component methods** : 
    * Principal Components Analysis (PCA)
    * Principal Components Analysis with partial correlation matrix (PPCA)
    * Weighted Principal Components Analysis (WPCA)
    * Expectation-Maximization Principal Components Analysis (EMPCA)
    * Exploratory Factor Analysis (EFA)
    * Classical Multidimensional Scaling (CMSCALE)
    * Metric and Non - Metric Multidimensional Scaling (MDS)
    * Correspondence Analysis (CA)
    * Multiple Correspondence Analysis (MCA)
    * Factor Analysis of Mixed Data (FAMD)
* In some methods, it allowed to **add supplementary informations** such as supplementary individuals and/or variables.
* It provides a geometrical point of view, a lot of graphical outputs.
* It provides efficient implementations, using a scikit-learn API.

## Installation

### Dependencies

scientisttools requires 

```{python}
Python 3
Numpy >= 1.24.3
Matplotlib >= 3.5.3
Scikit-learn >=  1.2.2
Pandas >= 2.0.0
Plotnine >= 0.10.1
Plydata >= 0.4.3
```

### User installation

You can install scientisttools using `pip` :

```
pip install scientisttools
```

Tutorial are available

````
https://github.com/enfantbenidedieu/scientisttools/blob/master/pca_example.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/partial_pca.ipynb

````
