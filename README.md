# scientisttools : Python library for multidimensional analysis

## About scientisttools

**scientisttools** is a `Python` package dedicated to multivariate Exploratory Data Analysis, clustering analysis and discriminant analysis.

## Why use scientisttools?

* It performs **classical principal component methods** : 
    * Principal Components Analysis (PCA)
    * Principal Components Analysis with partial correlation matrix (PPCA)
    * Weighted Principal Components Analysis (WPCA)
    * Expectation-Maximization Principal Components Analysis (EMPCA)
    * Exploratory Factor Analysis (EFA)
    * Classical Multidimensional Scaling (CMDSCALE)
    * Metric and Non - Metric Multidimensional Scaling (MDS)
    * Correspondence Analysis (CA)
    * Multiple Correspondence Analysis (MCA)
    * Factor Analysis of Mixed Data (FAMD)
    * Multiple Factor Analysis (MFA)
* In some methods, it allowed to **add supplementary informations** such as supplementary individuals and/or variables.
* It provides a geometrical point of view, a lot of graphical outputs.
* It provides efficient implementations, using a scikit-learn API.

Those statistical methods can be used in two ways :
* as descriptive methods ("datamining approach")
* as reduction methods in scikit-learn pipelines ("machine learning approach")

`scientisttools` also performs some algorithms such as `clustering analysis` and `discriminant analysis`.

* **Clustering analysis**:
    * Hierarchical Clustering on Principal Components (HCPC)
    * Variables Hierarchical Clustering Analysis (VARHCA)
    * Variables Hierarchical Clustering Analysis on Principal Components (VARHCPC)
    * Categorical Variables Hierarchical Clustering Analysis (CATVARHCA)
* **Discriminant Analysis**
    * Canonical Discriminant Analysis (CANDISC)
    * Linear Discriminant Analysis (LDA)
    * Discriminant with qualitatives variables (DISQUAL)
    * Discriminant Correspondence Analysis (DISCA)
    * Discriminant with mixed data (DISMIX)
    * Stepwise Discriminant Analysis (STEPDISC) (only `backward` elimination is available).

Notebooks are availabled.

## Installation

### Dependencies

scientisttools requires 

```
numpy>=1.26.2
matplotlib>=3.5.3
scikit-learn>=1.2.2
pandas>=1.5.3
mapply>=0.1.21
plotnine>=0.10.1
plydata>=0.4.3
pingouin>=0.5.3
scientistmetrics>=0.0.3
ggcorrplot>=0.0.2
```

### User installation

You can install scientisttools using `pip` :

```
pip install scientisttools
```

Tutorial are available

````
https://github.com/enfantbenidedieu/scientisttools/blob/master/ca_example2.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/classic_mds.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/efa_example.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/famd_example.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/ggcorrplot.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/mca_example.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/mds_example.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/partial_pca.ipynb
https://github.com/enfantbenidedieu/scientisttools/blob/master/pca_example.ipynb
````

## Author

Duvérier DJIFACK ZEBAZE ([duverierdjifack@gmail.com](duverierdjifack@gmail.com))
