# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp

from .pca import PCA
from .mca import MCA
from .famd import FAMD
from .eta2 import eta2
from .splitmix import splitmix

def coeffLg(X=None,Y=None,first_eigen_X=None,first_eigen_Y=None) -> np:
    """
    Calculate the Lg coefficients between groups
    --------------------------------------------

    Description
    -----------
    Performs Lg coefficients between two groups X and Y

    Usage
    -----
    ```python
    >>> function_lg(X=None,Y=None,first_eigen_X=None,first_eigen_Y=None)
    ```

    Parameters
    ----------
    `X` : pandas dataframe of shape (n_samples, n_columns)

    `Y` : pandas dataframe of shape (n_samples, n_columns)

    'first_eigen_X' : a float specifying the first eigenvalue of X (by default = None)

    `first_eigen_Y` : a float specifying the first eigenvalue of Y (by default = None)

    Returns
    -------
    a numpy array of shape (2,2) where lg[i,j] is the Lg coefficients between group i and group j (i,j = 1,2)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Check if Y is an instance of pd.DataFrame class
    if not isinstance(Y,pd.DataFrame):
        raise TypeError(
        f"{type(Y)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # If first_eigen_X is None
    if first_eigen_X is None:
        # Principal component analysis (PCA)
        if all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            first_eigen_X = PCA(standardize=True).fit(X).eig_.iloc[0,0]
        # Multiple correspondence analysis (MCA)
        elif all(pd.api.types.is_string_dtype(X[col]) for col in X.columns):
            first_eigen_X = MCA().fit(X).eig_.iloc[0,0]
        # Factor analysis of mixed data (FAMD)
        else:
            first_eigen_X = FAMD().fit(X).eig_.iloc[0,0]
    
    # If first_eigen_Y is None
    if first_eigen_Y is None:
        # Principal component analysis (PCA)
        if all(pd.api.types.is_numeric_dtype(Y[col]) for col in Y.columns):
            first_eigen_Y = PCA(standardize=True).fit(Y).eig_.iloc[0,0]
        # Multiple correspondence analysis (MCA)
        elif all(pd.api.types.is_string_dtype(Y[col]) for col in Y.columns):
            first_eigen_Y = MCA().fit(Y).eig_.iloc[0,0]
        # Factor analysis of mixed data (FAMD)
        else:
            first_eigen_Y = FAMD().fit(Y).eig_.iloc[0,0]

    def Lg(X=None,Y=None,first_eigen_X=None,first_eigen_Y=None):
        # If x and y are quantitative columns
        if (all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns) and all(pd.api.types.is_numeric_dtype(Y[col]) for col in Y.columns)):
            # Sum of square coefficient of correlation
            sum_corr2 = np.array([(np.corrcoef(X[col1],Y[col2])[0,1])**2 for col1 in X.columns for col2 in Y.columns]).sum()
            # Weighted the sum using the eigenvalues of each group
            value = (1/(first_eigen_X*first_eigen_Y))*sum_corr2
        # If x and y are qualitative columns
        elif (all(pd.api.types.is_string_dtype(X[col]) for col in X.columns) and all(pd.api.types.is_string_dtype(Y[col]) for col in Y.columns)):
            # Sum of chi-squared
            sum_chi2 = np.array([sp.stats.chi2_contingency(pd.crosstab(X[col1],Y[col2]),correction=False).statistic for col1 in X.columns for col2 in Y.columns]).sum()
            # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
            value = (1/(X.shape[0]*X.shape[1]*Y.shape[1]*first_eigen_X*first_eigen_Y))*sum_chi2
        # If x is qualitative and y is quantitative
        elif (all(pd.api.types.is_string_dtype(X[col]) for col in X.columns) and all(pd.api.types.is_numeric_dtype(Y[col]) for col in Y.columns)):
            # Sum of square correlation ratio
            sum_eta2 = np.array([eta2(X[col1],Y[col2],digits=10)["Eta2"] for col1 in X.columns for col2 in Y.columns]).sum()
            # Weighted the sum using eigenvalues and number of categoricals variables
            value = (1/(X.shape[1]*first_eigen_X*first_eigen_Y))*sum_eta2
        # If x is quantitative and y is qualitative
        elif (all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns) and all(pd.api.types.is_string_dtype(Y[col]) for col in Y.columns)):
            # Sum of square correlation ratio
            sum_eta2 = np.array([eta2(Y[col2],X[col1],digits=10)["Eta2"] for col1 in X.columns for col2 in Y.columns]).sum()
            # Weighted the sum using eigenvalues and number of categoricals variables
            value = (1/(Y.shape[1]*first_eigen_X*first_eigen_Y))*sum_eta2
        else:
            # Split X
            X_quanti = splitmix(X=X)["quanti"]
            X_quali = splitmix(X=X)["quali"]
            # Split Y
            Y_quanti = splitmix(X=Y)["quanti"]
            Y_quali = splitmix(X=Y)["quali"]

            # If quantitative columns in group 1
            if X_quanti is not None:
                # 
                if Y_quanti is not None:
                    # Pearson correlation coefficient
                    value1 = np.array([(np.corrcoef(X_quanti[col1],Y_quanti[col2])[0,1])**2 for col1 in X_quanti.columns for col2 in Y_quanti.columns]).sum()
                else:
                    value1 = 0.0
                
                if Y_quali is not None:
                    # Square correlation ratio
                    value2 = np.array([eta2(Y_quali[col2],X_quanti[col1],digits=10)["Eta2"] for col1 in X_quanti.columns for col2 in Y_quali.columns]).sum()/Y_quali.shape[1]
                else:
                    value2 = 0.0
                value_X = value1 + value2
            else:
                value_X = 0
            
            # If qualitative in group 1
            if X_quali is not None:
                # If quantitative variable in group 2
                if Y_quanti is not None:
                    value3 = np.array([eta2(X_quali[col1],Y_quanti[col2],digits=10)["Eta2"] for col1 in X_quali.columns for col2 in Y_quanti.columns]).sum()/X_quali.shape[1]
                else:
                    value3 = 0.0
                
                if Y_quali is not None:
                    value4 = np.array([sp.stats.chi2_contingency(pd.crosstab(X_quali[col1],Y_quali[col2]),correction=False).statistic for col1 in X_quali.columns for col2 in Y_quali.columns]).sum()
                    value4 = (1/(X.shape[0]*X_quali.shape[1]*Y_quali.shape[1]))*value4
                else:
                    value4 = 0.0
                value_Y = value3 + value4
            else:
                value_Y = 0.0

            value = (value_X + value_Y)/(first_eigen_X*first_eigen_Y)

        return value
    
    # Compute
    lg = np.zeros((4,4))
    lg[0,0] = Lg(X=X,Y=X,first_eigen_X=first_eigen_X,first_eigen_Y=first_eigen_X)
    lg[1,1] = Lg(X=Y,Y=Y,first_eigen_X=first_eigen_Y,first_eigen_Y=first_eigen_Y)
    lg[0,1] = Lg(X=X,Y=Y,first_eigen_X=first_eigen_X,first_eigen_Y=first_eigen_Y)
    lg[1,0] = Lg(X=Y,Y=X,first_eigen_X=first_eigen_Y,first_eigen_Y=first_eigen_X)
    return lg