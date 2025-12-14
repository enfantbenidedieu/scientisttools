# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones, zeros, sqrt, average, cov, linalg, number
from pandas import DataFrame, Series, concat
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
from collections import namedtuple, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.gfa import gfa
from .functions.summarize import summarize, conditional_wmean
from .functions.kaiser_msa import kaiser_msa
from .functions.predict_sup import predict_sup
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.function_eta2 import function_eta2
from .functions.wcorrcoef import wcorrcoef
from .functions.wpcorrcoef import wpcorrcoef
from .functions.association import association
from .functions.corrmatrix import corrmatrix
from .functions.utils import is_dataframe

class pPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Component Analysis (pPCA)
    -------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Principal Component Analysis on partial correlation matrix (pPCA) with supplementary individuals and/or supplementary variables. Missing values are replaced by the column mean.

    Usage
    -----
    ```python
    >>> pPCA(n_components = 5, partial = None, ind_weights = None, var_weights = None, ind_sup = None, sup_var = None)
    ```

    Parameters:
    -----------
    `n_components`: an integer indicating the number of dimensions kept in the results (by default 5).

    `partiel`: an integer or a list or a tuple indicating the indexes or the names of the partial variables

    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `sup_var`: an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `quali_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary categorical variables

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
    `get_ppca_ind`, `get_ppca_var`, `get_ppca`, `summarpPCA`, `fviz_ppca_ind`, fviz_ppca_var, fviz_ppca_biplot, predictpPCA, supvarpPCA

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2006
    >>> from scientisttools import pPCA, summarypPCA
    >>> res_ppca = pPCA(partial=0,ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_ppca.fit(autos2006)
    >>> summarypPCA(res_ppca)
    ```
    """
    def __init__(self,
                 partial = None,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 sup_var = None):
        self.partial = partial
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.sup_var = sup_var

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of float, shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set partial label and index
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.partial is None:
            raise ValueError("'partial' must be assigned.") 
       
        partial_label = get_sup_label(X=X,indexes=self.partial,axis=1)  

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary individuals labels
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)

        #get supplementary variables labels
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)

        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Principal Component Analysis with partial correlation matrix (PartialPCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all active variables are numerics
        if not all(is_numeric_dtype(X[k]) for k in X.columns):
            raise TypeError("All active variables must be numerics")
    
        #split X into z (partial variables) and x (dependent variables)
        z, x = X[partial_label], X.drop(columns = partial_label)

        #number of rows/columns
        n_rows, n_cols = x.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])
        
        #set variables weights
        if self.var_weights is None:
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list or a tuple or a 1-D array or a pandas Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=x.index,name="weight"), Series(var_weights,index=x.columns,name="weight")

        # Ordinary least squares models
        ols_results = DataFrame(zeros((n_cols,len(partial_label)+4)),columns = [*["intercept"],*partial_label,*["R2","Adj. R2","RMSE"]],index=x.columns)
        Xhat, model = DataFrame(index=x.index,columns=x.columns).astype("float"), OrderedDict()
        for k in x.columns:
            ols = sm.WLS(endog=x[k].astype(float),exog=sm.add_constant(z),weights=ind_weights).fit()
            ols_results.loc[k,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(x[k],ols.fittedvalues,squared=False)]]
            Xhat.loc[:,k] = ols.resid
            model[k] = ols

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center Xhat
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #apply non-normed principal component analysis
        center = Series(average(Xhat,axis=0,weights=ind_weights),index=Xhat.columns,name="center")
        scale = Series(array([sqrt(cov(Xhat.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)]),index=Xhat.columns,name="scale")
        #standardization: z = (x - mu)/sigma
        Zhat = Xhat.sub(center,axis=1).div(scale,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Zhat)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1, n_cols))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Xhat=Xhat,Zhat=Zhat,partial=partial_label,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,
                            n_components=n_components,max_components=max_components,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Zhat,row_weights=ind_weights,col_weights=var_weights,max_components=max_components,n_components=n_components)

        #extract elements
        self.svd_, self.eig_ = fit_.svd, fit_.eig

        #convert to namedtuple
        self.ind_, self.var_ = namedtuple("ind",fit_.row.keys())(*fit_.row.values()), namedtuple("var",fit_.col.keys())(*fit_.col.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split into x and z
            z_ind_sup, x_ind_sup = X_ind_sup[partial_label], X_ind_sup.drop(columns=partial_label)
            #apply regression to compute residuals
            Xhat_ind_sup = concat((Series(x_ind_sup[k].sub(model[k].predict(sm.add_constant(z_ind_sup))),index=ind_sup_label,name=k) for k in x.columns),axis=1)
            #standardization: Z = (X - mu)/sigma
            Zhat_ind_sup = Xhat_ind_sup.sub(center,axis=1).div(scale,axis=1)
            #statistics for supplementary individuals
            ind_sup_ = predict_sup(X=Zhat_ind_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            if self.ind_sup is not None:
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2
            
            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0:
                #compute resid
                ols2_results = DataFrame(zeros((n_quanti_sup,len(partial_label)+4)),columns = ols_results.columns,index=X_quanti_sup.columns)
                Xhat_quanti_sup = DataFrame(columns=X_quanti_sup.columns,index=x.index).astype("float")
                for k in X_quanti_sup.columns:
                    ols = sm.WLS(endog=X_quanti_sup[k].astype(float),exog=sm.add_constant(z),weights=ind_weights).fit()
                    ols2_results.loc[k,:] = [*ols.params.values.tolist(),*[ols.rsquared,ols.rsquared_adj,mean_squared_error(X_quanti_sup[k],ols.fittedvalues,squared=False)]]
                    Xhat_quanti_sup.loc[:,k] = ols.resid
                    model[k] = ols
                #concatenate Ols results
                ols_results = concat((ols_results,ols2_results),axis=0)
                #compute weighted average and weighted standard deviation for supplementary quantitative variables
                center_sup = Series(average(Xhat_quanti_sup,axis=0,weights=ind_weights),index=X_quanti_sup.columns,name="center")
                scale_sup = Series(array([sqrt(cov(Xhat_quanti_sup.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_quanti_sup)]),index=X_quanti_sup.columns,name="scale")
                #standardization: Z = (X - mu)/sigma
                Zhat_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
                #statistics for supplementary quantitative variables
                quanti_sup_ = predict_sup(X=Zhat_quanti_sup,Y=fit_.svd.U,weights=ind_weights,axis=1)
                del quanti_sup_['dist2'] #delete dist2
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

             #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #recode
                rec = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec.X, rec.dummies
                #conditional mean - Barycenter of original data
                X_levels_sup = conditional_wmean(X=Xhat,Y=X_quali_sup,weights=ind_weights)
                #standardization: Z = (X - mu)/sigma
                Zhat_levels_sup = X_levels_sup.sub(center,axis=1).div(scale,axis=1)
                #statistics for supplementary levels
                quali_sup_ = predict_sup(X=Zhat_levels_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
                #vtest for the supplementary levels
                p_k_sup = dummies_sup.mul(ind_weights,axis=0).sum(axis=0)
                levels_sup_vtest = quali_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(fit_.svd.vs[:n_components],axis=1)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=fit_.row["coord"],weights=ind_weights,excl=None)
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
                
                #descriptive descriptive of qualitative variables
                self.summary_quali_ = summarize(X=X_quali_sup)

                #degree of association - multivariate goodness
                if n_quali_sup > 1:
                    self.association_ = association(X=X_quali_sup) 

        #Store separate model
        self.separate_model_ = namedtuple("separate_model",model.keys())(*model.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #all quantitative variables in original dataframe
        all_quanti = Xtot.select_dtypes(include=number)
        #drop supplementary individuals
        if self.ind_sup is not None:
            all_quanti = all_quanti.drop(index=ind_sup_label)
        #corrcoef, reproduces and partial corrcoef correlations
        wcorr, pcorr = wcorrcoef(X=all_quanti,weights=ind_weights), wpcorrcoef(X=all_quanti,partial=partial_label,weights=ind_weights)
        #convert to ordered dictionary
        corr_ = OrderedDict(corrcoef=wcorr,pcorrcoef=pcorr)
        #convert to namedtuple
        self.corr_ = namedtuple("correlation",corr_.keys())(*corr_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #add others informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and weighted standard deviation
        x_center = Series(average(all_quanti,axis=0,weights=ind_weights),index=all_quanti.columns,name="center")
        x_scale = Series(array([sqrt(cov(all_quanti.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(all_quanti.shape[1])]),index=all_quanti.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        Z_quanti = all_quanti.sub(x_center,axis=1).div(x_scale,axis=1)
        #normalized coefficients
        index = [k for k in all_quanti.columns if k not in partial_label]
        coef_n = concat((sm.OLS(endog=Z_quanti[k],exog=sm.add_constant(Z_quanti[partial_label])).fit().params[1:].T for k in index),axis=0)
        coef_n.index, coef_n.name = index, "Coefficients"
        #convert to ordered dictionary
        others_ = OrderedDict(kaiser_msa = kaiser_msa(all_quanti),coef_n = coef_n,statistics = ols_results)
        #convert to namedtuple
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #descriptive statistics of quantitatives variables 
        self.summary_quanti_ = summarize(X=all_quanti)

        #correlation tests
        all_vars = Xtot.copy()
        if self.ind_sup is not None:
            all_vars = all_vars.drop(index=ind_sup_label)
        self.corrtest_ = corrmatrix(X=all_vars,weights=ind_weights)
        
        self.model_ = "ppca"
        return self

    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X.
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y`: None.
            y is ignored.

        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
            Transformed values.
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
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns

        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """ 
        is_dataframe(X=X) #check if X is a pandas DataFrame
          
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] = ncols
            raise ValueError("'columns' aren't aligned")
        
        if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
            raise TypeError("All columns must be numeric") 
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the pPCA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns

        #split X into z (partial variables) and x (dependent variables)
        z, x = X[self.call_.partial], X.drop(columns=self.call_.partial)

        #compute residuals for new individuals
        Xhat = concat((Series(x[k].sub(self.separate_model_[i].predict(sm.add_constant(z))),index=X.index,name=k) for i,k in enumerate(x.columns)),axis=1)
        #standardization (Z = (X - mu)/sigma) and apply transition relation
        coord = Xhat.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord