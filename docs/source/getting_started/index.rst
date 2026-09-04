.. _getting_started:

===============
Getting started
===============

`scientisttools <https://github.com/enfantbenidedieu/scientisttools>`_ is an open source python library dedicated to exploratory multivariate data analysis, 
clustering analysis and multidimensional scaling distributed under the MIT Licence.

The purpose of this guide is to illustrate some of the main features of ``scientisttools``. It assumes basic working knowledge of `scikit-learn <https://scikit-learn.org/stable/>`_ practices.

Fitting : estimator basics
--------------------------

As scikit-learn, scientisttools provides models called `estimators <https://scikit-learn.org/stable/glossary.html#term-estimators>`_. 
Each estimator can be fitted to some data using its `fit <https://scikit-learn.org/stable/glossary.html#term-fit>`_ method.

Here is a simple example where we fit a :class:`~scientisttools.PCA` to :class:`~scientisttools.datasets.decathlon` data:

.. code:: python
  
  >>> from scientisttools.datasets import decathlon
  >>> from scientisttools import PCA
  >>> clf = PCA(ind_sup=range(41,46),sup_var=(10,11,12))
  >>> clf.fit(decathlon.data)
  PCA(ind_sup=range(41,46),sup_var=(10,11,12))
  
The ``fit`` method generally accepts 2 inputs:

- The samples `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ (or design matrix) ``X``. The size of ``X``
  is typically ``(n_samples, n_features)``, which means that samples are
  represented as rows and features are represented as columns.
- The target values ``y`` which are true lables for ``X``. ``y`` is a pandas Series with categorical terms, but for unsupervised learning tasks, ``y`` does not need to be specified.

Transform
---------

Once the estimator is fitted, it can be used for projecting new data in the first principal components previously extracted from a training set. 
You don't need to re-train the estimator:

.. code:: python

  >>> # new data
  >>> newdata = decathlon.ind_sup 
  >>> # projection of new data in the first principal components
  >>> clf.transform(newdata).head() 