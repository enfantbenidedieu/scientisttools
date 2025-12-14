# -*- coding: utf-8 -*-
from numpy import zeros
from pandas import DataFrame
from pandas.api.types import is_string_dtype
from collections import namedtuple
from typing import NamedTuple

def predictMCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Multiple Correspondence Analysis
    ----------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin for new individuals with Multiple Correspondence Analysis.

    Usage
    -----
    ```python
    >>> predictMCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Returns
    -------
    a namedtuple of pandas DataFrames/Series containing all the results for the new individuals, including:
    
    `coord`: coordinates of the new individuals,

    `cos2`: squared cosines of the new individuals,

    `dist2`: squared distance to origin of the new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_canines
    >>> from scientisttools import PCA, predictMCA
    >>> canines = load_canines()
    >>> res_mca = MCA(ind_sup=range(27,33),sup_var=(6,7))
    >>> res_mca.fit(canines)
    >>> #statistics for new individuals
    >>> ind_sup = load_canines("ind_sup")
    >>> predict = predictMCA(res_mca,X=ind_sup)
    >>> predict.coord.head() #coordinates of new individuals
    >>> predict.cos2.head() #cos2 of new individuals
    >>> predict.dist2.head() #dist2 of new individuals
    ```
    """
    if self.model_ != "mca": #check if self is an object of class MCA
        raise TypeError("'self' must be an object of class MCA")

    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")

    if not all(is_string_dtype(X[q]) for q in X.columns): #check if all columns are categoricals
        raise TypeError("All columns in `X` must be categoricals")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect of columns
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the MCA result")
    X = X.loc[:,self.call_.X.columns] #reorder columns
    
    #create disjunctive table for new individuals
    dummies_new = DataFrame(zeros((X.shape[0],self.call_.dummies.shape[1])),columns=self.call_.dummies.columns,index=X.index)
    for i in range(X.shape[0]):
        values = [X.iloc[i,j] for j in range(X.shape[1])]
        for k in range(self.call_.dummies.shape[1]):
            if self.call_.dummies.columns[k] in values:
                dummies_new.iloc[i,k] = 1
    #proportion avec actif levels
    p_k = self.call_.dummies.mul(self.call_.ind_weights,axis=0).sum(axis=0)
    #standardization: z_ik = (x_ik/pk)-1
    Z = dummies_new.div(p_k,axis=1).sub(1)
    #coordinates of the new individuals
    coord = Z.mul(self.call_.levels_weights,axis=1).dot(self.svd_.V)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 of the new individuals
    sqdisto = Z.pow(2).mul(self.call_.levels_weights,axis=1).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #cos2 of the new individuals
    sqcos = coord.pow(2).div(sqdisto,axis=0)
    #convert to namedtuple
    return namedtuple("predictMCAResult",["coord","cos2","dist2"])(coord,sqcos,sqdisto)