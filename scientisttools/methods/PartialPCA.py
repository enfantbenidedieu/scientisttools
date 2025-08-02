# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, zeros, sqrt
from pandas import DataFrame, Series, Categorical, api, concat
import statsmodels.api as sm
from typing import NamedTuple
from collections import namedtuple, OrderedDict
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

#intern functions
from .PCA import PCA
from .functions.summarize import summarize
from .functions.pcorrcoef import pcorrcoef
from .functions.kaiser_msa import kaiser_msa
from .functions.predict_sup import predict_ind_sup, predict_quanti_sup, predict_quali_sup
from .functions.recodecont import recodecont
from .functions.revaluate_cat_variable import revaluate_cat_variable
from .functions.conditional_average import conditional_average

class PartialPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Component Analysis (PartialPCA)
    -------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Partial Principal Component Analysis (PartialPCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables. Missing values are replaced by the column mean.

    Usage
    -----
    ```python
    >>> PartialPCA(n_components = 5, partial = None, ind_weights = None, var_weights = None, ind_sup = None, quanti_sup = None, quali_sup = None, parallelize = False)
    ```

    Parameters:
    -----------
    `n_components`: number of dimensions kept in the results (by default 5)

    `partiel`: an integer or a list/tuple of string specifying the name of the partial variables

    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `quanti_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `quali_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary categorical variables

    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True`: parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False`: parallelize using pandas apply

    Attributes
    ----------
    `call_` : namedtuple with some informations

    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_`  : namedtuple of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)
    
    `ind_` : namedtuple of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    `ind_sup_` : namedtuple of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `others_` : namedtuple of others statistics

    `model_` : string specifying the model fitted = 'partialpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * A. Boudou (1982), Analyse en composantes principales partielle, Statistique et analyse des données, tome 7, n°2 (1982), p. 1-21

    * Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    `get_partialpca_ind`, `get_partialpca_var`, `get_partialpca`, `summaryPartialPCA`, `fviz_partialpca_ind`, fviz_partialpca_var, fviz_partialpca_biplot, predictPartialPCA, supvarPartialPCA

    Examples
    --------
    ```python
    >>> from scientisttools import autos2006, PartialPCA, summaryPartialPCA
    >>> #with integer
    >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> #with string
    >>> res_ppca = PartialPCA(partial="CYL",ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> #with list
    >>> res_ppca = PartialPCA(partial=["CYL"],ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> #with tuple
    >>> res_ppca = PartialPCA(partial=("CYL"),ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8).fit(autos2006)
    >>> summaryPartialPCA(res_ppca)
    ```
    """
    def __init__(self,
                 partial = None,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize = False):
        self.partial = partial
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: pandas DataFrame of float, shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself

        Examples
        --------
        ```python
        >>> from scientisttools import autos2006, PartialPCA, summaryPartialPCA
        >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8)
        >>> res_ppca.fit(autos2006)
        ```
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if X is an instance of pd.DataFrame class
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if parallelize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.parallelize,bool):
            raise TypeError("'parallelize' must be a boolean.")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## rop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## checks if categoricals variables is in X
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for q in is_quali.columns:
                X[q] = Categorical(X[q],categories=sorted(X[q].dropna().unique().tolist()),ordered=True)
                
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Set partial label and index
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary qualitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
             
        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        #drop supplementary quantitative variables
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Principal Component Analysis with partial correlation matrix (PartialPCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
    
        #descriptive statistics of quantitatives variables 
        summary_quanti = summarize(X=X)

        #split X into target (y) and features (x)
        y, x = X[partial_label], X.drop(columns = partial_label)

        #number of rows/columns
        n_rows, n_cols = x.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list/tuple/array/Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array/Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Set variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list/tuple/array/Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array/Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=x.index,name="weight"), Series(var_weights,index=x.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##standardize the active data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)
        #convert to Series
        center, scale, wcorr = Series(d1.mean,index=X.columns,name="center"), Series(d1.std,index=X.columns,name="scale"), DataFrame(d1.corrcoef,index=X.columns,columns=X.columns)

        #standardization : Z = (X - mu)/sigma
        Z = mapply(X,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##normalized coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        coef_n = DataFrame(zeros((x.shape[1],len(partial_label))),columns = partial_label,index=x.columns)
        for k in x.columns:
            coef_n.loc[k,:] = sm.OLS(endog=Z[k],exog=sm.add_constant(Z[partial_label])).fit().params[1:]

        # Ordinary least squares models
        ols_results = DataFrame(zeros((x.shape[1],len(partial_label)+4)),columns = [*["intercept"],*partial_label,*["R2","Adj. R2","RMSE"]],index=x.columns)
        resid, model = DataFrame(index=x.index,columns=x.columns).astype("float"), OrderedDict()
        for k in x.columns:
            ols = sm.OLS(endog=x[k],exog=sm.add_constant(y)).fit()
            ols_results.loc[k,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(x[k],ols.fittedvalues,squared=False)]]
            resid.loc[:,k] = ols.resid
            model[k] = ols

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##fit Principal Components Analysis with resid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = PCA(standardize=True,n_components=self.n_components,ind_weights=ind_weights,var_weights=var_weights,rotate=None).fit(resid)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #apply regression to compute Residuals
            ind_sup_resid = concat((DataFrame({f"{k}": X_ind_sup[k].sub(model[k].predict(sm.add_constant(X_ind_sup[partial_label])))},index=ind_sup_label) for k in x.columns),axis=1)
            #update PCA with supplementary individuals
            fit_ = PCA(standardize=True,n_components=self.n_components,ind_weights=ind_weights,var_weights=var_weights,ind_sup=self.ind_sup,rotate=None).fit(concat((resid,ind_sup_resid),axis=0))
            #extract supplementary individuals informations
            self.ind_sup_ = fit_.ind_sup_
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            #fill missing with mean
            X_quanti_sup = recodecont(X=X_quanti_sup).X

            #standardize supplementary quantitative variables
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=n_workers)

            #normalized coefficients
            coef_n_sup = DataFrame(zeros((X_quanti_sup.shape[1],len(partial_label))),columns = partial_label,index=quanti_sup_label)
            for k in X_quanti_sup.columns:
                coef_n_sup.loc[k,:] = sm.OLS(endog=Z_quanti_sup[k],exog=sm.add_constant(Z[partial_label])).fit().params[1:]
            #insert `group` columns
            coef_n.insert(0,"group","actif")
            coef_n_sup.insert(0,"group","sup")
            #concatenate
            coef_n = concat((coef_n,coef_n_sup),axis=0)

            # Compute resid
            ols2_results = DataFrame(zeros((X_quanti_sup.shape[1],len(partial_label)+4)),columns = [*["intercept"],*partial_label,*["R2","Adj. R2","RMSE"]],index=quanti_sup_label)
            quanti_sup_resid = DataFrame(columns=quanti_sup_label,index=x.index).astype("float")
            for k in quanti_sup_label:
                ols = sm.OLS(endog=X_quanti_sup[k],exog=sm.add_constant(y)).fit()
                ols2_results.loc[k,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(X_quanti_sup[k],ols.fittedvalues,squared=False)]]
                quanti_sup_resid.loc[:,k] = ols.resid
                model[k] = ols
            #insert `group` columns
            ols_results.insert(0,"group","actif")
            ols2_results.insert(0,"group","sup")
            #concatenate Ols results
            ols_results = concat((ols_results,ols2_results))

            #update PCA with supplementary quantitatives variables
            fit_ = PCA(standardize=True,n_components=self.n_components,ind_weights=ind_weights,var_weights=var_weights,quanti_sup=quanti_sup_label,rotate=None).fit(concat((resid,quanti_sup_resid),axis=1))
            #extract statistics for supplementary quantitative variables
            self.quanti_sup_ = fit_.quanti_sup_

            #summary statistics for supplementary quantitative variables
            summary_quanti_sup = summarize(X=X_quanti_sup)
            #insert `group` columns
            summary_quanti.insert(0,"group","active")
            summary_quanti_sup.insert(0,"group","sup")
            #concatenate
            summary_quanti = concat((summary_quanti,summary_quanti_sup),axis=0,ignore_index=True)
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            
            #update PCA with supplementary qualitatives variables
            fit_= PCA(standardize=True,n_components=self.n_components,ind_weights=ind_weights,var_weights=var_weights,quali_sup=quali_sup_label,rotate=None).fit(concat((resid,X_quali_sup),axis=1))
            #extract supplementary qualitatives variables informations
            self.quali_sup_, self.summary_quali_ = fit_.quali_sup_, fit_.summary_quali_

        #Update number of components
        n_components, var_weights, ind_weights = fit_.call_.n_components, fit_.call_.var_weights, fit_.call_.ind_weights

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,x=x,y=y,resid=resid,partial=partial_label,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,quanti_sup=quanti_sup_label,quali_sup=quali_sup_label,pca=fit_)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #extract all additionals informations
        self.svd_, self.eig_, self.ind_, self.var_ = fit_.svd_, fit_.eig_.drop(columns=[]), fit_.ind_, fit_.var_

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partial pearson correlation coefficient
        pcorr =  pcorrcoef(X=X)
        #convert to ordered dictionary
        corr_ = OrderedDict(corrcoef=wcorr,pcorrcoef=pcorr)
        #convert to namedtuple
        self.corr_ = namedtuple("correlation",corr_.keys())(*corr_.values())

        #add others informations
        others_ = OrderedDict(kaiser_msa = kaiser_msa(X),coef_n = coef_n,statistics = ols_results)
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #Store separate model
        self.separate_model_ = namedtuple("separate_model",model.keys())(*model.values())
        
        self.summary_quanti_ = summary_quanti
        self.model_ = "partialpca"

        return self

    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X.
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas Dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y`: None.
            y is ignored.

        Returns
        -------
        `X_new`: pandas Dataframe of shape (n_samples, n_components)
            Transformed values.
        
        Examples
        --------
        ```python
        >>> from scientisttools import autos2006, PartialPCA, summaryPartialPCA
        >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8)
        >>> ind_coord = res_ppca.fit_transform(autos2006)
        """
        self.fit(X)
        return self.ind_.coord

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns

        Returns
        -------
        `X_new`: pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.

        Examples
        --------
        Examples
        --------
        ```python
        >>> from scientisttools import autos2006, PartialPCA, summaryPartialPCA
        >>> res_ppca = PartialPCA(partial=0,ind_sup=(18,19),quanti_sup=(6,7),quali_sup=8)
        >>> res_ppca.fit(autos2006)
        >>> ind_coord = res_ppca.transform(res_pca.call_.X)
        """ 
        #check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None

        #check if X.shape[1] = ncols
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #find intersect
        intersect_col = [x for x in X.columns if x in self.call_.X.columns]
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the PartialPCA result")
        #reorder columns
        X = X.loc[:,self.call_.X.columns]
        
        #residuals for new observations
        resid = concat((DataFrame({f"{k}" : X[k].sub(self.separate_model_[i].predict(sm.add_constant(X[self.call_.partial])))},index=X.index) for i,k in enumerate(self.call_.x.columns)),axis=1)
        #ppply PCA transform to resid
        return self.call_.pca.transform(resid)

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
    `self`: an object of class PartialPCA

    `X`: pandas Dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    a namedtuple of pandas Dataframes containing all the results for the new individuals including:
    
    `coord`: factor coordinates of the new individuals

    `cos2`: squared cosinus of the new individuals

    `dist`: squared distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_autos2006, PartialPCA, predictPartialPCA
    >>> autos2006 = load_autos2006("actif")
    >>> res_ppca = PartialPCA(partial="CYL").fit(autos2006)
    >>> #load new individuals
    >>> ind_sup = load_autos2006("ind_sup")
    >>> predict = predictPartialPCA(res_ppca,X=ind_sup)
    >>> predict.coord.head() #coordinate of the new individuals
    >>> predict.cos2.head() #squared cosinus of the new individuals
    >>> predict.dist.head() #squared distance to origin of the new individuals
    ```
    """
    # Check if self is an object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    
    #check if X is an instance of pd.DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set index name as None
    X.index.name = None

    # Check if columns are aligned
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")
    
    #check if all variables are numerics
    all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")
    
    #find intersect
    intersect_col = [x for x in X.columns if x in self.call_.X.columns]
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the PartialPCA result")
    #reorder columns
    X = X.loc[:,self.call_.X.columns]

    #residuals for new observations
    resid = concat((DataFrame({f"{x}" : X[x] - self.separate_model_[i].predict(sm.add_constant(X[self.call_.partial]))},index=X.index) for i,x in enumerate(self.call_.x.columns)),axis=1)
    #standardize residuals
    Z = mapply(resid,lambda x : (x - self.call_.pca.call_.center)/self.call_.pca.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #square distance to origin
    sqdisto = mapply(Z, lambda x : (x**2)*self.call_.pca.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #statistics for new observations
    res = predict_ind_sup(Z=Z,V=self.svd_.V,sqdisto=sqdisto,col_weights=self.call_.pca.call_.var_weights,n_workers=self.call_.n_workers)
    return namedtuple("predictPartialPCAResult",res.keys())(*res.values())

def supvarPartialPCA(self,X_quanti_sup=None,X_quali_sup=None) -> NamedTuple:
    """
    Supplementary variables in Partial Principal Components Analysis (PartialPCA)
    -----------------------------------------------------------------------------

    Description
    -----------
    Performs the factor coordinates, squared cosinus and squared distance to origin of supplementary variables (quantitative and/or qualitative) with Partial Principal Components Analysis (PartialPCA).

    Usage
    -----
    ```python
    >>> supvarPartialPCA(self,X_quanti_sup=None,X_quali_sup=None)
    ```

    Parameters
    ----------
    `self`: an object of class PartialPCA

    `X_quanti_sup`: pandas Dataframe of supplementary quantitative variables (default = None)

    `X_quali_sup`: pandas Dataframe of supplementary qualitative variables (default = None)

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary variables including : 

    `quanti`: a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables including :
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `statistics`: statistics for linear regression between supplementary quantitative variables and partial variable
    
    `quali`: a namedtuple of pandas DataFrames containing all the results of the supplementary qualitative/categories variables including :
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `vtest`: value-test
        * `dist`: squared distance to origin
        * `eta2`: squared correlation ratio

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
     ```python
    >>> from scientisttools import load_autos2006, PartialPCA, supvarPartialPCA
    >>> autos2006 = load_autos2006("actif")
    >>> res_ppca = PartialPCA(partial="CYL").fit(autos2006)
    >>> #supplementary variables (quantitative & qualitative)
    >>> X_quanti_sup, X_quali_sup = load_autos2006("quanti_sup"), load_autos2006("quali_sup")
    >>> sup_var = supvarPartialPCA(res_ppca,X_quanti_sup=X_quanti_sup,X_quali_sup=X_quali_sup)
    ``` 
    """
    # Check if self is and object of class PartialPCA
    if self.model_ != "partialpca":
        raise TypeError("'self' must be an object of class PartialPCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        #if pandas Series, transform to pandas Dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        #check if X_quanti_sup is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_quanti_sup[k]) for k in X_quanti_sup.columns)
        if not all_num:
            raise TypeError("All columns in `X_quanti_sup` must be numeric")
        
        #fill missing with mean
        X_quanti_sup = recodecont(X_quanti_sup).X
        
        #xteract coefficients and intercept
        ols_results = DataFrame(zeros((X_quanti_sup.shape[1],len(self.call_.partial)+4)),columns = [*["intercept"],*self.call_.partial,*["R2","Adj. R2","RMSE"]],index=X_quanti_sup.columns)
        quanti_sup_resid = DataFrame(columns=X_quanti_sup.columns,index=X_quanti_sup.index.tolist()).astype("float")
        for k in X_quanti_sup.columns:
            ols = sm.OLS(endog=X_quanti_sup[k],exog=sm.add_constant(self.call_.y)).fit()
            ols_results.loc[k,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(X_quanti_sup[k],ols.fittedvalues,squared=False)]]
            quanti_sup_resid.loc[:,k] = ols.resid
        
        #compute statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(X=quanti_sup_resid,row_coord=self.ind_.coord,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        quanti_sup_["statistics"] = ols_results
        #convert to namedtuple
        quanti_sup =  namedtuple("quanti_stp",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        #if pandas Series, transform to pandas Dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        #check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if all columns are categoricals
        all_cat = all(api.types.is_string_dtype(X_quali_sup[q]) for q in X_quali_sup.columns)
        if not all_cat:
            raise TypeError("All columns in `X_quali_sup` must be categoricals") 
        
        #convert to factor
        for q in X_quali_sup.columns:
            X_quali_sup[q] = Categorical(X_quali_sup[q],categories=sorted(X_quali_sup[q].dropna().unique().tolist()),ordered=True)
        #check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        #conditional average of original data
        barycentre = conditional_average(X=self.call_.pca.call_.X,Y=X_quali_sup,weights=self.call_.ind_weights)
        #standardize the data
        Z_quali_sup = mapply(barycentre,lambda x : (x - self.call_.pca.call_.center)/self.call_.pca.call_.scale,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #categories factor coordinates (scores)
        quali_sup_coord = mapply(Z_quali_sup, lambda x : x*self.call_.pca.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        #squared distance to origin
        quali_sup_sqdisto  = mapply(Z_quali_sup, lambda x : (x**2)*self.call_.pca.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
        quali_sup_sqdisto.name = "Sq. Dist."
        #categories coefficients
        n_rows, n_k = X_quali_sup.shape[0], concat((X_quali_sup[q].value_counts().sort_index() for q in X_quali_sup.columns),axis=0)
        coef_k = sqrt(((n_rows-1)*n_k)/(n_rows-n_k))
        #statistics for supplementary categories
        quali_sup_ = predict_quali_sup(X=X_quali_sup,row_coord=self.ind_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #update value-test with squared eigenvalues
        quali_sup_["vtest"] = mapply(quali_sup_["vtest"],lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #merge dictionary
        quali_sup_ = OrderedDict(**OrderedDict(barycentre=barycentre),**quali_sup_)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None

    # Store all informations
    return namedtuple("supvarPartialPCAResult",["quanti","quali"])(quanti_sup,quali_sup)