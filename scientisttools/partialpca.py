# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import pingouin as pg
import statsmodels.api as sm

from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

from .pca import PCA
from .kmo import global_kmo_index, per_item_kmo_index
from .revaluate_cat_variable import revaluate_cat_variable
from .function_eta2 import function_eta2

class PartialPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Component Analysis (PartialPCA)
    -------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Partial Principal Component Analysis with supplementary individuals

    Usage
    -----
    ```python
    >>> PartialPCA(standardize = True, n_components = 5, partial = None, ind_weights = None, var_weights = None, ind_sup = None, parallelize = False)
    ```

    Parameters:
    -----------
    `standardize` : a boolean, if True (value set by default) then data are scaled to unit variance

    `n_components` : number of dimensions kept in the results (by default 5)

    `partiel` : a list of string specifying the name of the partial variables

    `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); the weights are given only for the active individuals
    
    `var_weights` : an optional variables weights (by default, uniform column weights); the weights are given only for the active variables
    
    `ind_sup` : list/tuple indicating the indexes of the supplementary individuals

    parallelize : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_`  : dictionary of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)
    
    `ind_` : dictionary of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    `ind_sup_` : dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `call_` : dictionary with some statistics

    `others_` : dictionary of others statistics

    `model_` : string specifying the model fitted = 'partialpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    A. Boudou (1982), Analyse en composantes principales partielle, Statistique et analyse des données, tome 7, n°2 (1982), p. 1-21

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_partialpca_ind, get_partialpca_var, get_partialpca, summaryPartialPCA, fviz_partialpca_ind, fviz_partialpca_var, fviz_partialpca_biplot, predictPartialPCA, supvarPartialPCA

    Examples
    --------
    ```python
    >>> # load cars2006 dataset
    >>> from scientisttools import load_cars2006
    >>> D = load_cars2006(which="actif")
    >>> from scientisttools import PartialPCA
    >>> res_partialpca = PartialPCA(n_components=None,standardize=True,partial=["CYL"],parallelize=False)
    >>> res_partialpca.fit(D)
    ```
    """
    def __init__(self,
                 standardize = True,
                 n_components = 5,
                 partial = None,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.standardize = standardize
        self.partial = partial
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas/polars DataFrame of float, shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        `y` : None
            y is ignored

        Returns
        -------
        `self` : object
            Returns the instance itself
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is a pandas Dataframe
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns:
                X[col] = X[col].astype("object")
        
        ####################################
        if self.partial is None:
            raise ValueError("'partial' must be assigned.")
        
        # check if partial is set
        if isinstance(self.partial,str):
            partial = [self.partial]
        elif ((isinstance(self.partial,list) or isinstance(self.partial,tuple)) and len(self.partial)>=1):
            partial = [str(x) for x in self.partial]
        
        # Check if supplementary qualitatives variables
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup]
        else:
            quali_sup_label = None

        #  Check if supplementary quantitatives variables
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup]
        else:
            quanti_sup_label = None
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None
             
        ###### Store initial data
        Xtot = X.copy()

        # Drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        # Drop supplementary quantitative variables
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        # Drop supplementary individuals
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)
        
        ################## Summary quantitatives variables ####################
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti
        
        # KMO index
        global_kmo = global_kmo_index(X)
        per_var_kmo = per_item_kmo_index(X)

        self.others_ = {"corr" : X.corr(method="pearson"),"partial_corr" : X.pcorr(),"global_kmo" : global_kmo,"kmo_per_var" : per_var_kmo}

        ############# Compute average mean and standard deviation
        d1 = DescrStatsW(X,weights=self.ind_weights,ddof=0)

        # Standardization
        means = d1.mean
        if self.standardize:
            std = d1.std
        else:
            std = np.ones(X.shape[1])
        # Z = (X - mu)/sigma
        Z = (X - means.reshape(1,-1))/std.reshape(1,-1)

        # Store
        means = pd.Series(means,index=X.columns,name="average")
        std = pd.Series(std,index=X.columns,name="scale")

        # Extract partial
        y = X[partial]

        # Drop partial columns
        x = X.drop(columns = partial)

        # Normalized coeffcients
        normalized_coef = pd.DataFrame(np.zeros((x.shape[1],len(partial))),columns = self.partial,index=x.columns)
        for col in x.columns:
            normalized_coef.loc[col,:] = sm.OLS(endog=Z[col],exog=sm.add_constant(Z[partial])).fit().params[1:]

        # Ordinary least squares models
        ols_results = pd.DataFrame(np.zeros((x.shape[1],len(partial)+4)),columns = [*["intercept"],*partial,*["R2","Adj. R2","RMSE"]],index=x.columns)
        resid = pd.DataFrame(index=x.index.tolist(),columns=x.columns).astype("float")
        model = {}
        for col in x.columns:
            res = sm.OLS(endog=x[col],exog=sm.add_constant(y)).fit()
            ols_results.loc[col,:] = [*res.params.values.tolist(),*[res.rsquared,res.rsquared_adj,mean_squared_error(x[col],res.fittedvalues,squared=False)]]
            resid.loc[:,col] = res.resid
            model[col] = res
        
        #### Store separate model
        self.separate_model_ = model
        
        # Principal Components Analysis with resid
        global_pca = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights).fit(resid)

        ##############################################################################################################################################
        #                                        Compute supplementrary individuals statistics
        ###############################################################################################################################################
        # Statistics for supplementary individuals
        if self.ind_sup is not None:
            # Transform to float
            X_ind_sup = X_ind_sup.astype("float")

            # Apply regression to compute Residuals
            X_ind_resid = pd.DataFrame().astype("float")
            for col in X.columns:
                X_ind_resid[col] = X_ind_sup[col] - model[col].predict(X_ind_sup[partial])
            
            # Concatenate the two datasets
            X_ind_resid = pd.concat((resid,X_ind_resid),axis=0)

            # PCA with supplementary individuals
            global_pca = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights,ind_sup=ind_sup).fit(X_ind_resid)
            # Extract supplementary individuals informations
            self.ind_sup_ = global_pca.ind_sup_
        
        ###############################################################################################################################
        #                               Compute supplementary quantitatives variables statistics
        ###############################################################################################################################
        # Statistics for supplementary quantitatives variables
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            # Transform to float
            X_quanti_sup = X_quanti_sup.astype("float")

            # Summary statistics
            self.summary_quanti_.insert(0,"group","active")
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            summary_quanti_sup.insert(0,"group","sup")

            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

            # Compute resid
            ols2_results = pd.DataFrame(np.zeros((X_quanti_sup.shape[1],len(partial)+4)),columns = [*["intercept"],*partial,*["R2","Adj. R2","RMSE"]],index=X_quanti_sup.columns)
            X_quanti_sup_resid = pd.DataFrame(columns=X_quanti_sup.columns,index=X_quanti_sup.index.tolist()).astype("float")
            for col in X_quanti_sup.columns:
                res = sm.OLS(endog=X_quanti_sup[col],exog=sm.add_constant(y)).fit()
                ols2_results.loc[col,:] = [*res.params.values.tolist(),*[res.rsquared,res.rsquared_adj,mean_squared_error(X_quanti_sup[col],res.fittedvalues,squared=False)]]
                X_quanti_sup_resid.loc[:,col] = res.resid

            # Concatenate
            X_quanti_sup_resid = pd.concat((resid,X_quanti_sup_resid),axis=1)
            # Find index
            index1 = [X_quanti_sup_resid.columns.tolist().index(x) for x in X_quanti_sup_resid.columns if x in X_quanti_sup.columns]

            # PCA with supplementary quantitatives variables
            global_pca = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights,quanti_sup=index1).fit(X_quanti_sup_resid)
            # Extract supplementary individuals informations
            self.quanti_sup_ = global_pca.quanti_sup_
            # Add ols results
            self.quanti_sup_["ols"] = ols2_results

        ##############################################################################################################################################
        # Compute supplementary qualitatives variables statistics
        ###############################################################################################################################################
        # Statistics for supplementary qualitatives variables
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            
            # Concatenate
            X_quali_sup_resid = pd.concat((resid,X_quali_sup),axis=1)

            # Find new index labels
            index2 = [X_quali_sup_resid.columns.tolist().index(x) for x in X_quali_sup_resid.columns if x in X_quali_sup.columns]
            
            # PCA with supplementary qualitatives variables
            global_pca = PCA(standardize=self.standardize,n_components=self.n_components,quali_sup=index2).fit(X_quali_sup_resid)
            
            # Extract supplementary qualitatives variables informations
            self.quali_sup_ = global_pca.quali_sup_
            self.summary_quali_ = global_pca.summary_quali_
        
        # Update number of components
        n_components = global_pca.call_["n_components"]
        var_weights = global_pca.call_["var_weights"]
        ind_weights = global_pca.call_["ind_weights"]

        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot":Xtot,
                      "X" : x,
                      "Z" : Z,
                      "y" : y,
                      "ind_sup" : ind_sup_label,
                      "quanti_sup" : quanti_sup_label,
                      "quali_sup" : quali_sup_label,
                      "means" : means,
                      "std" : std,
                      "ind_weights" : ind_weights,
                      "var_weights" : var_weights,
                      "n_components" : n_components,
                      "standardize" : self.standardize,
                      "partial" : partial}

        # Store global PCA method
        self.global_pca_ = global_pca

        # Store singular values decomposition
        self.svd_ = global_pca.svd_

        # Eigen values informations
        self.eig_ = global_pca.eig_

        # Individuals informations
        self.ind_ = global_pca.ind_

        # Variables informations
        self.var_ = global_pca.var_

        # Add others
        self.others_ = {**self.others_,**{"ols" : ols_results, "normalized_coef" : normalized_coef}}
     
        # store model
        self.model_ = "partialpca"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y` : None.
            y is ignored.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_["coord"]

    def transform(self,X):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """ 
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # Check if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Transform to float
        X = X.astype("float")
        # Residuals
        resid = pd.DataFrame().astype("float")
        for col in self.separate_model_.keys():
            resid[col] = X[col] - self.separate_model_[col].predict(sm.add_constant(X[self.call_["partial"]]))
        
        # Apply PCA transform to resid
        coord = self.global_pca_.transform(resid)
        return coord

def predictPartialPCA(self,X=None):
    """
    Predict projection for new individuals with Partial Principal Component Analysis (PartialPCA)
    ---------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and square distance to origin of new individuals with Partial Principal Component Analysis (PartialPCA)

    Usage
    -----
    ```python
    >>> predictPartialPCA(self,X)
    ```

    Parameters
    ----------
    `self` : an object of class PartialPCA

    `X` : pandas/polars dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    dictionary of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : square cosinus of the new individuals

    `dist` : square distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load cars2006 dataset
    >>> from scientisttools import load_cars2006
    >>> D = load_cars2006(which="actif")
    >>> from scientisttools import PartialPCA
    >>> res_partialpca = PartialPCA(n_components=None,standardize=True,partial=["CYL"],parallelize=False)
    >>> res_partialpca.fit(D)
    >>> # Load supplementary individuals
    >>> ind_sup = load_cars2006(which="indsup")
    >>> from scientisttools import predictPartialPCA
    >>> predict = predictPartialPCA(res_partialpca,X=ind_sup)
    ```
    """
    # Check if self is an object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    
    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()
    
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    means = self.global_pca_.call_["means"].values # Average
    std = self.global_pca_.call_["std"].values # Standard deviation
    var_weights = self.call_["var_weights"].values  # Variables weights
    n_components = self.call_["n_components"] # number of components

    # Transform to float
    X = X.astype("float")
    # Residuals
    resid = pd.DataFrame().astype("float")
    for col in self.separate_model_.keys():
        # Model residuals
        resid[col] = X[col] - self.separate_model_[col].predict(sm.add_constant(X[self.call_["partial"]]))
    
    #### Standardize residuals
    Z = (resid - means.reshape(1,-1))/std.reshape(1,-1)

    # Apply PCA projection on residuals
    coord = mapply(Z,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
    coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
    coord.index = X.index.tolist()

    #  New data square distance to origin
    dist2 = mapply(Z,lambda  x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    dist2.name = "Sq. Dist."
    dist2.index.name = None

    # New data square cosinus
    cos2 = mapply(coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)

    # Store all informations
    res = {"coord" : coord, "cos2" : cos2,"dist" : dist2}
    return res

def supvarPartialPCA(self,X_quanti_sup=None,X_quali_sup=None):
    """
    Supplementary variables in Partial Principal Components Analysis (PartialPCA)
    -----------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Partial Principal Components Analysis (PartialPCA)

    Usage
    -----
    ```python
    >>> supvarPartialPCA(self,X_quanti_sup=None,X_quali_sup=None)
    ```

    Parameters
    ----------
    `self` : an object of class PartialPCA

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives variables (default = None)

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives variables (default = None)

    Returns
    -------
    dictionary of dictionary containing the results for supplementary variables including : 

    `quanti` : dictionary containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : dictionary containing the results of the supplementary qualitatives/categories variables including :
        * coord : factor coordinates of the supplementary categories
        * cos2 : square cosinus of the supplementary categories
        * vtest : value-test of the supplementary categories
        * dist : square distance to origin of the supplementary categories
        * eta2 : square correlation ratio of the supplementary qualitatives variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
     ```python
    >>> # load cars2006 dataset
    >>> from scientisttools import load_cars2006
    >>> D = load_cars2006(which="actif")
    >>> from scientisttools import PartialPCA
    >>> res_partialpca = PartialPCA(n_components=None,standardize=True,partial=["CYL"],parallelize=False)
    >>> res_partialpca.fit(D)
    >>> # Supplementary quantitatives variables
    >>> X_quanti_sup = load_cars2006(which="varquantsup")
    >>> # Supplementary qualitatives variables
    >>> X_quali_sup = load_cars2006(which="varqualsup")
    >>> from scientisttools import supvarPartialPCA
    >>> sup_var_predict = supvarPartialPCA(res_partialpca,X_quanti_sup=X_quanti_sup,X_quali_sup=X_quali_sup)
    ``` 
    """
    # Check if self is and object of class PCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1

    # Extract
    ind_weights = self.call_["ind_weights"].values
    n_components = self.call_["n_components"]

    ########################################################################################################################
    #                                          For supplementary quantitatives variables
    #########################################################################################################################
    # Supplementary quantitatives variables statistics
    if X_quanti_sup is not None:
        # Transform to float
        X_quanti_sup = X_quanti_sup.astype("float")

        # Extract
        partial = self.call_["partial"]
        y = self.call_["y"]

        # Compute resid
        ols2_results = pd.DataFrame(np.zeros((X_quanti_sup.shape[1],len(partial)+4)),columns = [*["intercept"],*partial,*["R2","Adj. R2","RMSE"]],index=X_quanti_sup.columns)
        X_quanti_sup_resid = pd.DataFrame(columns=X_quanti_sup.columns,index=X_quanti_sup.index.tolist()).astype("float")
        for col in X_quanti_sup.columns:
            res = sm.OLS(endog=X_quanti_sup[col],exog=sm.add_constant(y)).fit()
            ols2_results.loc[col,:] = [*res.params.values.tolist(),*[res.rsquared,res.rsquared_adj,mean_squared_error(X_quanti_sup[col],res.fittedvalues,squared=False)]]
            X_quanti_sup_resid.loc[:,col] = res.resid

        # Extract coefficients and intercept
        ols2_results = pd.DataFrame(np.zeros((X_quanti_sup.shape[1],len(partial)+4)),columns = [*["intercept"],*partial,*["R2","Adj. R2","RMSE"]],index=X_quanti_sup.columns)
        X_quanti_sup_resid = pd.DataFrame(columns=X_quanti_sup.columns,index=X_quanti_sup.index.tolist()).astype("float")
        for col in X_quanti_sup.columns:
            res = sm.OLS(endog=X_quanti_sup[col],exog=sm.add_constant(y)).fit()
            ols2_results.loc[col,:] = [*res.params.values.tolist(),*[res.rsquared,res.rsquared_adj,mean_squared_error(X_quanti_sup[col],res.fittedvalues,squared=False)]]
            X_quanti_sup_resid.loc[:,col] = res.resid
        
        # Standardize
        d2 = DescrStatsW(X_quanti_sup_resid,weights=ind_weights,ddof=0)

        # Standardization
        means_sup = d2.mean
        if self.standardize:
            std_sup = d2.std
        else:
            std_sup = np.ones(X_quanti_sup.shape[1])
        
        # Z = (X - mu)/sigma
        Z_quanti_sup_resid = (X_quanti_sup_resid - means_sup.reshape(1,-1))/std_sup.reshape(1,-1)

        # Supplementary quantitatives variables coordinates
        quanti_sup_coord = mapply(Z_quanti_sup_resid,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(self.svd_["U"][:,:n_components])
        quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Supplementary quantitatives variables squared distance to origin
        quanti_sup_cor = mapply(Z_quanti_sup_resid,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        quanti_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_sup_cor)

        # Supplementary quantitatives variables square cosine
        quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2)/np.sqrt(quanti_sup_dist2),axis=0,progressbar=False,n_workers=n_workers)    

        # Store supplementary quantitatives informations
        quanti_sup =  {"coord":quanti_sup_coord, "cor" : quanti_sup_coord, "cos2" : quanti_sup_cos2, "ols" : ols2_results}
    else:
        quanti_sup = None
    
    ###########################################################################################################################
    #                                                   For supplementary qualitatives variables
    ###########################################################################################################################
    # Supplementary qualitatives variables statistics
    if X_quali_sup is not None:
        # check if X_quali_sup is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Variables weights
        var_weights = self.global_pca_.call_["var_weights"].values
        means = self.global_pca_.call_["means"].values
        std = self.global_pca_.call_["std"].values

        # Square correlation ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

        # Barycenter
        barycentre = pd.DataFrame().astype("float")
        n_k = pd.Series().astype("float")
        for col in X_quali_sup.columns:
            vsQual = X_quali_sup[col]
            modalite, counts = np.unique(vsQual, return_counts=True)
            n_k = pd.concat([n_k,pd.Series(counts,index=modalite)],axis=0)
            bary = pd.DataFrame(index=modalite,columns=self.global_pca_.call_["X"].columns)
            for mod in modalite:
                idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
                bary.loc[mod,:] = np.average(self.global_pca_.call_["X"].iloc[idx,:],axis=0,weights=ind_weights[idx])
            barycentre = pd.concat((barycentre,bary),axis=0)
        
        # Supplementary qualitatives squared distance
        bary = (barycentre - means.reshape(1,-1))/std.reshape(1,-1)
        quali_sup_dist2  = mapply(bary, lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        quali_sup_dist2.name = "Sq. Dist."

        # Supplementary qualitatives coordinates
        quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Supplementary qualiatives square cosine
        quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # Supplementary qualitatives value-test
        quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/self.svd_["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)
        quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((n_rows-n_k[k])/((n_rows-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

        # Supplementary categories informations
        quali_sup = {"coord" : quali_sup_coord, "cos2" : quali_sup_cos2, "vtest" : quali_sup_vtest, "dist" : quali_sup_dist2,"eta2" : quali_sup_eta2}
    else:
        quali_sup = None

    # Store all informations
    res = {"quanti" : quanti_sup, "quali" : quali_sup}
    return res