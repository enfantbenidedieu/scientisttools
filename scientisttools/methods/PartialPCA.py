# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import pingouin as pg
import statsmodels.api as sm
from typing import NamedTuple
from collections import namedtuple, OrderedDict
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

#intern functions
from scientisttools.methods.PCA import PCA
from scientisttools.others.kmo import kmo_index
from scientisttools.others.predict_sup import predict_ind_sup, predict_quanti_sup
from scientisttools.others.recodecont import recodecont
from scientisttools.others.revaluate_cat_variable import revaluate_cat_variable
from scientisttools.others.function_eta2 import function_eta2
from scientisttools.others.conditional_average import conditional_average

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

    `partiel` : an integer or a list/tuple of string specifying the name of the partial variables

    `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); the weights are given only for the active individuals
    
    `var_weights` : an optional variables weights (by default, uniform column weights); the weights are given only for the active variables
    
    `ind_sup` : list/tuple indicating the indexes of the supplementary individuals

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_`  : namedtuple of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)
    
    `ind_` : namedtuple of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    `ind_sup_` : namedtuple of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `call_` : namedtuple with some statistics

    `others_` : namedtuple of others statistics

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
    >>> res_partialpca = PartialPCA(n_components=None,standardize=True,partial=0,parallelize=False)
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
        
        # check if standardize is a boolean
        if not isinstance(self.standardize,bool):
            raise TypeError("'standardize' must be a boolean.")
        
        # check if parallelize is a boolean
        if not isinstance(self.parallelize,bool):
            raise TypeError("'parallelize' must be a boolean.")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Set index name as None
        X.index.name = None
        
        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        # Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns:
                X[col] = pd.Categorical(X[col],categories=sorted(X[col].dropna().unique().tolist()),ordered=True)
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Set partial label and index
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.partial is None:
            raise ValueError("'partial' must be assigned.")  
        elif isinstance(self.partial,str):
            partial_label =  [self.partial] 
        elif isinstance(self.partial,(int,float)):
            partial_label = [X.columns[int(self.partial)]]
        elif isinstance(self.partial,(list,tuple)):
            if all(isinstance(x,str) for x in self.partial):
                partial_label = [str(x) for x in self.partial] 
            elif all(isinstance(x,(int,float)) for x in self.partial):
                partial_label = X.columns[[int(x) for x in self.partial]].tolist()
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary qualitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            if isinstance(self.quali_sup,str):
                quali_sup_label = [self.quali_sup]
            elif isinstance(self.quali_sup,(int,float)):
                quali_sup_label = [X.columns[int(self.quali_sup)]]
            elif isinstance(self.quali_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.quali_sup):
                     quali_sup_label = [str(x) for x in self.quali_sup]
                elif all(isinstance(x,(int,float)) for x in self.quali_sup):
                    quali_sup_label = X.columns[[int(x) for x in self.quali_sup]].tolist()
        else:
            quali_sup_label = None

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary quantitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            if isinstance(self.quanti_sup,str):
                quanti_sup_label = [self.quanti_sup]
            elif isinstance(self.quanti_sup,(int,float)):
                quanti_sup_label = [X.columns[int(self.quanti_sup)]]
            elif isinstance(self.quanti_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.quanti_sup):
                    quanti_sup_label = [str(x) for x in self.quanti_sup]
                elif all(isinstance(x,(int,float)) for x in self.quanti_sup):
                    quanti_sup_label = X.columns[[int(x) for x in self.quanti_sup]].tolist()
        else:
            quanti_sup_label = None
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if individuls supplementary
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            if isinstance(self.ind_sup,str):
                ind_sup_label = [self.ind_sup]
            elif isinstance(self.ind_sup,(int,float)):
                ind_sup_label = [X.index[int(self.ind_sup)]]
            elif isinstance(self.ind_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.ind_sup):
                    ind_sup_label = [str(x) for x in self.ind_sup]
                elif all(isinstance(x,(int,float)) for x in self.ind_sup):
                    ind_sup_label = X.index[[int(x) for x in self.ind_sup]].tolist()
        else:
            ind_sup_label = None
             
        ###### Store initial data
        Xtot = X.copy()

        #--------------------------
        ## Drop supplementary elements
        #----------------------------------------------------------------------------------------------------------------
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

        #Summary quantitatives variables
        summary_quanti = X.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti

        #------------------------------------------------------------------------------------------------------------------
        ## Principal Component Analysis with partial correlation matrix (PartialPCA)
        #------------------------------------------------------------------------------------------------------------------
        # Split X into target and features
        y, x = X[partial_label], X.drop(columns = partial_label)
        
        #------------------------------------------------------------------------------------------------------------------
        ## Standardisation of 
        #------------------------------------------------------------------------------------------------------------------
        # weighted mean and standard deviation
        d1 = DescrStatsW(X,weights=self.ind_weights,ddof=0)

        # Standardization
        center = d1.mean
        if self.standardize:
            scale = d1.std
        else:
            scale = np.ones(X.shape[1])
        # Z = (X - mu)/sigma
        Z = mapply(X,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)
        
        #-------------------------------------------------------------------------------------------------------------------------
        # Normalized coefficients
        #-------------------------------------------------------------------------------------------------------------------------
        coef_n = pd.DataFrame(np.zeros((x.shape[1],len(partial_label))),columns = partial_label,index=x.columns)
        for col in x.columns:
            coef_n.loc[col,:] = sm.OLS(endog=Z[col],exog=sm.add_constant(Z[partial_label])).fit().params[1:]

        # Ordinary least squares models
        ols_results = pd.DataFrame(np.zeros((x.shape[1],len(partial_label)+4)),columns = [*["intercept"],*partial_label,*["R2","Adj. R2","RMSE"]],index=x.columns)
        resid, model = pd.DataFrame(index=x.index,columns=x.columns).astype("float"), OrderedDict()
        for col in x.columns:
            ols = sm.OLS(endog=x[col],exog=sm.add_constant(y)).fit()
            ols_results.loc[col,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(x[col],ols.fittedvalues,squared=False)]]
            resid.loc[:,col] = ols.resid
            model[col] = ols

        #----------------------------------------------------------------------------------------------------
        # Fit Principal Components Analysis with resid
        #----------------------------------------------------------------------------------------------------
        res = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights).fit(resid)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary individuals
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            # Transform to float
            X_ind_sup = X_ind_sup.astype("float")

            # Apply regression to compute Residuals
            X_ind_resid = pd.concat((pd.DataFrame({f"{col}": X_ind_sup[col].sub(model[col].predict(sm.add_constant(X_ind_sup[partial_label])))},index=ind_sup_label) for col in x.columns),axis=1)
            
            # Concatenate the two datasets
            X_ind_resid = pd.concat((resid,X_ind_resid),axis=0)

            # Update PCA with supplementary individuals
            res = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights,ind_sup=self.ind_sup).fit(X_ind_resid)

            # Extract supplementary individuals informations
            self.ind_sup_ = res.ind_sup_
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary quantitative variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            # Transform to float
            X_quanti_sup = recodecont(X_quanti_sup).Xcod

            # Summary statistics
            self.summary_quanti_.insert(0,"group","active")
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            summary_quanti_sup.insert(0,"group","sup")

            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

            # Standardize supplementary quantitative variables
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=self.ind_weights,ddof=0)

            # Standardization
            center_sup = d_quanti_sup.mean
            if self.standardize:
                scale_sup = d_quanti_sup.std
            else:
                scale_sup = np.ones(X_quanti_sup.shape[1])
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - center_sup)/scale_sup,axis=1,progressbar=False,n_workers=n_workers)

            #-------------------------------------------------------------------------------------------------------------------------
            # Normalized coefficients
            #-------------------------------------------------------------------------------------------------------------------------
            coef_n_sup = pd.DataFrame(np.zeros((X_quanti_sup.shape[1],len(partial_label))),columns = partial_label,index=quanti_sup_label)
            for col in X_quanti_sup.columns:
                coef_n_sup.loc[col,:] = sm.OLS(endog=Z_quanti_sup[col],exog=sm.add_constant(Z[partial_label])).fit().params[1:]
            
            #concatenate
            coef_n.insert(0,"group","actif")
            coef_n_sup.insert(0,"group","sup")
            coef_n = pd.concat((coef_n,coef_n_sup),axis=0)

            #----------------------------------------------------------------------------------------------------------------------
            # Compute resid
            ols2_results = pd.DataFrame(np.zeros((X_quanti_sup.shape[1],len(partial_label)+4)),columns = [*["intercept"],*partial_label,*["R2","Adj. R2","RMSE"]],index=quanti_sup_label)
            X_quanti_sup_resid = pd.DataFrame(columns=quanti_sup_label,index=x.index).astype("float")
            for col in quanti_sup_label:
                ols = sm.OLS(endog=X_quanti_sup[col],exog=sm.add_constant(y)).fit()
                ols2_results.loc[col,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(X_quanti_sup[col],ols.fittedvalues,squared=False)]]
                X_quanti_sup_resid.loc[:,col] = ols.resid
                model[col] = ols

            # Concatenate Ols results
            ols_results.insert(0,"group","actif")
            ols2_results.insert(0,"group","sup")
            ols_results = pd.concat((ols_results,ols2_results))

            # Concatenate
            X_quanti_sup_resid = pd.concat((resid,X_quanti_sup_resid),axis=1)

            # Find index
            index = [X_quanti_sup_resid.columns.tolist().index(x) for x in quanti_sup_label]

            # PCA with supplementary quantitatives variables
            res = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights,quanti_sup=index).fit(X_quanti_sup_resid)
            
            # extract statistics for supplementary quantitative variables
            self.quanti_sup_ = res.quanti_sup_
            
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary qualitative variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            
            # Concatenate
            X_quali_sup_resid = pd.concat((resid,X_quali_sup),axis=1)

            # Find new index labels
            index = [X_quali_sup_resid.columns.tolist().index(x) for x in quali_sup_label]
            
            # Update PCA with supplementary qualitatives variables
            res = PCA(standardize=self.standardize,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=self.var_weights,quali_sup=index).fit(X_quali_sup_resid)
            
            # Extract supplementary qualitatives variables informations
            self.quali_sup_, self.summary_quali_ = res.quali_sup_, res.summary_quali_

        #--------------------------------------------------------------------------------------------------
        # Add others informations
        #--------------------------------------------------------------------------------------------------
        others_ = dict(
            corr = X.corr(method="pearson"),
            pcorr = X.pcorr(),
            kmo = kmo_index(X),
            coef_n = coef_n,
            statistics = ols_results
        )
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #### Store separate model
        self.separate_model_ = namedtuple("separate_model",model.keys())(*model.values())

        # Store global PCA method
        self.global_pca_ = res
        
        # Update number of components
        n_components, var_weights, ind_weights = res.call_.n_components, res.call_.var_weights, res.call_.ind_weights

        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        call_ = {"Xtot":Xtot,
                "X" : x,
                "Z" : Z,
                "y" : y,
                "ind_sup" : ind_sup_label,
                "quanti_sup" : quanti_sup_label,
                "quali_sup" : quali_sup_label,
                "center" : pd.Series(center,index=X.columns,name="center"),
                "scale" : pd.Series(scale,index=X.columns,name="scale"),
                "ind_weights" : ind_weights,
                "var_weights" : var_weights,
                "n_components" : n_components,
                "standardize" : self.standardize,
                "partial" : partial_label}
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #extract all additionals informations
        self.svd_, self.eig_, self.ind_, self.var_ = res.svd_, res.eig_, res.ind_, res.var_

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
        return self.ind_.coord

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
        resid = pd.concat((pd.DataFrame({f"{col}" : X[col].sub(self.separate_model_[i].predict(sm.add_constant(X[self.call_.partial])))},index=X.index) for i,col in enumerate(self.call_.X.columns)),axis=1)
        
        # Apply PCA transform to resid
        return self.global_pca_.transform(resid)

def predictPartialPCA(self,X=None) -> NamedTuple:
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
    namedtuple of dataframes containing all the results for the new individuals including:
    
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
    >>> res_partialpca = PartialPCA(n_components=None,standardize=True,partial="CYL",parallelize=False)
    >>> res_partialpca.fit(D)
    >>> # Load supplementary individuals
    >>> ind_sup = load_cars2006(which="ind_sup")
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
    pca_center, pca_scale = self.global_pca_.call_.center.values, self.global_pca_.call_.scale.values
    n_components, var_weights = self.call_.n_components, self.call_.var_weights.values

    # Transform to float
    X = X.astype("float")
    # Residuals
    resid = pd.concat((pd.DataFrame({f"{x}" : X[x] - self.separate_model_[i].predict(sm.add_constant(X[self.call_.partial]))},index=X.index) for i,x in enumerate(self.call_.X.columns)),axis=1)
    
    #### Standardize residuals
    Z = mapply(resid,lambda x : (x - pca_center)/pca_scale,axis=1,progressbar=False,n_workers=n_workers)
    res = predict_ind_sup(Z,self.svd_.V[:,:n_components],var_weights,n_workers)
    
    return namedtuple("predictPartialPCAResult",res.keys())(*res.values())

def supvarPartialPCA(self,X_quanti_sup=None,X_quali_sup=None) -> NamedTuple:
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
    namedtuple of namedtuple containing the results for supplementary variables including : 

    `quanti` : namedtuple containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : namedtuple containing the results of the supplementary qualitatives/categories variables including :
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
    >>> res_partialpca = PartialPCA(n_components=None,standardize=True,partial="CYL",parallelize=False)
    >>> res_partialpca.fit(D)
    >>> # Supplementary quantitatives variables
    >>> X_quanti_sup = load_cars2006(which="quanti_sup")
    >>> # Supplementary qualitatives variables
    >>> X_quali_sup = load_cars2006(which="quali_sup")
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
    ind_weights, n_components = self.call_.ind_weights.values, self.call_.n_components

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary quantitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        # Transform to float
        X_quanti_sup = recodecont(X_quanti_sup).Xcod
        # Extract
        partial_label, y = self.call_.partial, self.call_.y

        # Extract coefficients and intercept
        ols_results = pd.DataFrame(np.zeros((X_quanti_sup.shape[1],len(partial_label)+4)),columns = [*["intercept"],*partial_label,*["R2","Adj. R2","RMSE"]],index=X_quanti_sup.columns)
        quanti_sup_resid = pd.DataFrame(columns=X_quanti_sup.columns,index=X_quanti_sup.index.tolist()).astype("float")
        for col in X_quanti_sup.columns:
            ols = sm.OLS(endog=X_quanti_sup[col],exog=sm.add_constant(y)).fit()
            ols_results.loc[col,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(X_quanti_sup[col],ols.fittedvalues,squared=False)]]
            quanti_sup_resid.loc[:,col] = ols.resid
        
        # Standardize
        d_quanti_sup = DescrStatsW(quanti_sup_resid,weights=ind_weights,ddof=0)

        # Standardization
        center = d_quanti_sup.mean
        if self.standardize:
            scale = d_quanti_sup.std
        else:
            scale = np.ones(X_quanti_sup.shape[1])
        
        # Z = (X - mu)/sigma
        Z_quanti_sup_resid = mapply(quanti_sup_resid,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)

        #compute statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(Z_quanti_sup_resid,self.svd_.U[:,:n_components],ind_weights,n_workers)
        quanti_sup_["statistics"] = ols_results
        
        # Store supplementary quantitatives informations
        quanti_sup =  namedtuple("quanti_stp",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        # check if X_quali_sup is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X_quali_sup.index.name = None
        
        # Transform to object
        for col in X_quali_sup.columns:
            X_quali_sup[col] = pd.Categorical(X_quali_sup[col],categories=sorted(X_quali_sup[col].dropna().unique().tolist()),ordered=True)
        
        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Extract elements
        var_weights, center, scale  = self.global_pca_.call_.var_weights.values, self.global_pca_.call_.center.values, self.global_pca_.call_.scale.values

        # Square correlation ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_.coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

        # conditional average of original data
        barycentre = conditional_average(X=self.global_pca_.call_.X,Y=X_quali_sup,weights=ind_weights)
        n_k = pd.concat((X_quali_sup[col].value_counts().sort_index() for col in X_quali_sup.columns),axis=0)

        # Supplementary qualitatives squared distance
        bary = mapply(barycentre,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)
        quali_sup_sqdisto  = mapply(bary, lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        quali_sup_sqdisto.name = "Sq. Dist."

        # Supplementary qualitatives coordinates
        quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_.V[:,:n_components])
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Supplementary qualiatives square cosine
        quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
        
        # Supplementary qualitatives value-test
        quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((n_rows-n_k[k])/((n_rows-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

        # Supplementary categories informations
        quali_sup = namedtuple("quali_sup",["coord","cos2","vtest","dist","eta2"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_sqdisto,quali_sup_eta2)
    else:
        quali_sup = None

    # Store all informations
    return namedtuple("supvarPartialPCAResult",["quanti","quali"])(quanti_sup,quali_sup)