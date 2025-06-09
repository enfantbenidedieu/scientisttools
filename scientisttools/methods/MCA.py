# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
import itertools
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from scientisttools.others.revaluate_cat_variable import revaluate_cat_variable
from scientisttools.others.fitfa import fitfa
from scientisttools.others.function_eta2 import function_eta2
from scientisttools.others.predict_sup import predict_ind_sup, predict_quanti_sup
from scientisttools.others.recodecont import recodecont

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)
    --------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Correspondence Analysis (MCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    >>> MCA(n_components = 5, ind_weights = None,var_weights = None,benzecri=True,greenacre=True,ind_sup = None,quali_sup = None,quanti_sup = None,parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); the weights are given only for the active individuals
    
    `var_weights` : an optional variables weights (by default, a list/tuple of 1/(number of active variables) for uniform row weights); the weights are given only for the active variables
    
    `benzecri` : boolean, if True benzecri correction is applied

    `greenacre` : boolean, if True greenacre correction is applied

    `ind_sup` : an integer:string:list/tuple indicating the indexes/names of the supplementary individuals

    `quali_sup` : an integer/string/list/tuple indicating the indexes/names of the supplementary categorical variables

    `quanti_sup` : an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Atttributes
    -----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `svd_` : namedtuple of matrices containing all the results of the singular value decomposition

    `var_` : namedtuple of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    `ind_` : namedtuple of pandas dataframe containing all the results for the active individuals (coordinates, square cosine, contributions)

    `ind_sup_` : namedtuple of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `quanti_sup_` : namedtuple of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes)

    `quali_sup_` : namedtuple of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a criterion with a Normal distribution, and eta2 which is the square correlation corefficient between a qualitative variable and a dimension)
    
    `summary_quali_` : summary statistics for supplementary qualitative variables

    `chi2_test_` : chi-squared test.

    `summary_quanti_` : summary statistics for quantitative variables if quanti_sup is not None
    
    `call_` : namedtuple with some statistics

    `others_` : namedtuple of others statistics (Kaiser threshold, ...)

    `model_` : string specifying the model fitted = 'mca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    `get_mca_ind`, `get_mca_var`, `get_mca`, `summaryMCA`, `dimdesc`, `predictMCA`, `supvarMCA`, `fviz_mca_ind`, `fviz_mca_mod`, `fviz_mca_var`, `fviz_mca`

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 benzecri = True,
                 greenacre = True,
                 ind_sup = None,
                 quali_sup = None,
                 quanti_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.benzecri = benzecri
        self.greenacre = greenacre
        self.ind_sup = ind_sup
        self.quali_sup = quali_sup
        self.quanti_sup = quanti_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas/polars DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

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
        
        if not isinstance(X,pd.DataFrame):
           raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                           "pd.DataFrame. For more information see: "
                           "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Set index name as None
        X.index.name = None

        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##checks if quantitatives variables are in X
        #----------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=np.number)
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##convert categorical variables to factor
        #----------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns:
            X[col] = pd.Categorical(X[col],categories=sorted(X[col].dropna().unique().tolist()),ordered=True)

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

        #make a copy of original data
        Xtot = X.copy()

        ##Drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        #Drop supplementary quantitatives columns
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        #drop supplementary individuls  
        if self.ind_sup is not None:
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Multiple Correspondence Analysis (MCA)
        #----------------------------------------------------------------------------------------------------------------------------------------
        # Number of rows/columns
        n_rows, n_cols = X.shape

        #check if two categorical variables have same categories
        X = revaluate_cat_variable(X)

        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X.columns:
            eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")

        #Chi2 statistic test
        chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
        idx = 0
        for i in range(X.shape[1]-1):
            for j in range(i+1,X.shape[1]):
                chi = sp.stats.chi2_contingency(pd.crosstab(X.iloc[:,i],X.iloc[:,j]),correction=False)
                row_chi2 = pd.DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                chi2_test = pd.concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                idx = idx + 1
        # Transform to int
        chi2_test["dof"] = chi2_test["dof"].astype("int")

        #dummies tables
        dummies = pd.concat((pd.get_dummies(X[col],dtype=int) for col in X.columns),axis=1)

        #number of categories, count and proportion
        n_cat, I_k, p_k = dummies.shape[1], dummies.sum(axis=0), dummies.mean(axis=0)
        I_k.name , p_k.name = "count","proportion"

        #standardize the data
        Z = pd.concat((dummies.loc[:,k]*(1/p_k[k])-1 for k  in dummies.columns),axis=1)
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set number of components
        #----------------------------------------------------------------------------------------------------------------------------------------
        max_components = n_cat - n_cols
        if self.n_components is None:
            n_components =  max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set individuals weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = np.ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,np.ndarray)):
            raise TypeError("'ind_weights' must be a list/tuple/array of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array with length {n_rows}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set variables weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = np.ones(n_cols)/n_cols
        elif not isinstance(self.var_weights,(list,tuple,np.ndarray)):
            raise ValueError("'var_weights' must be a list/tuple/array of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array with length {n_cols}.")
        else:
            var_weights = np.array([x/np.sum(self.var_weights) for x in self.var_weights])
            
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set categories weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        nb_mod = np.array([X[col].nunique() for col in X.columns])
        var_weights2 = np.array(list(itertools.chain(*[itertools.repeat(i,k) for i, k in zip(var_weights,nb_mod)])))
        mod_weights = np.array([x*y for x,y in zip(p_k,var_weights2)])

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Z=Z, 
                            ind_weights=pd.Series(ind_weights,index=X.index,name="weight"),
                            var_weights=pd.Series(var_weights,index=X.columns,name="weight"),
                            mod_weights=pd.Series(mod_weights,index=dummies.columns,name="weight"),
                            n_components=n_components,ind_sup=ind_sup_label,quali_sup=quali_sup_label,quanti_sup=quanti_sup_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## fit factor analysis model and extract all elements
        #----------------------------------------------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z,ind_weights,mod_weights,max_components,n_components,n_workers)
        
        # Extract elements
        self.svd_, self.eig_, ind, var = fit_.svd, fit_.eig, fit_.row, fit_.col

        # Convert to NamedTuple
        self.ind_ = namedtuple("ind",ind.keys())(*ind.values())

        # save eigen value greater than threshold
        lambd = self.eig_.iloc[:,0][self.eig_.iloc[:,0]>(1/n_cols)]
        
        # Benzecri correction
        if self.benzecri:
            if len(lambd) > 0:
                lambd_tilde = ((n_cols/(n_cols-1))*(lambd - 1/n_cols))**2
                # Cumulative percentage
                s_tilde = 100*(lambd_tilde/np.sum(lambd_tilde))
                # Benzecri correction
                self.benzecri_correction_ = pd.DataFrame(np.c_[lambd_tilde,s_tilde,np.cumsum(s_tilde)],columns=["eigenvalue","proportion","cumulative"],index = ["Dim."+str(x+1) for x in np.arange(0,len(lambd))])

        # Greenacre correction
        if self.greenacre:
            if len(lambd) > 0:
                lambd_tilde = ((n_cols/(n_cols-1))*(lambd - 1/n_cols))**2
                s_tilde_tilde = n_cols/(n_cols-1)*(np.sum(self.eig_.iloc[:,0]**2)-(n_cat-n_cols)/(n_cols**2))
                tau = 100*(lambd_tilde/s_tilde_tilde)
                self.greenacre_correction_ = pd.DataFrame(np.c_[lambd_tilde,tau,np.cumsum(tau)],columns=["eigenvalue","proportion","cumulative"],index = ["Dim."+str(x+1) for x in np.arange(0,len(lambd))])
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## variables additionals informations
        #----------------------------------------------------------------------------------------------------------------------------------------
        # Normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        var_coord_n = mapply(var["coord"],lambda x: x*self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        ##categoris variables value - test
        var_vtest = pd.concat(((var["coord"].loc[k,:]*np.sqrt(((n_rows-1)*I_k[k])/(n_rows-I_k[k]))).to_frame().T for k in I_k.index),axis=0)

        # Variables squared correlation ratio
        quali_var_eta2 = pd.concat((function_eta2(X=X,lab=col,x=self.ind_.coord.values,weights=ind_weights,n_workers=n_workers) for col in X.columns),axis=0)
        
        # Contribution des variables
        quali_var_contrib = pd.DataFrame().astype("float")
        for col in X.columns:
            modalite = np.unique(X[col]).tolist()
            contrib = var["contrib"].loc[modalite,:].sum(axis=0).to_frame(col).T
            quali_var_contrib = pd.concat((quali_var_contrib,contrib),axis=0)

        # Inertia for the variables
        quali_var_inertia = pd.Series((nb_mod - 1)/n_rows,index=X.columns.tolist(),name="inertia")

        #append dictionary
        var = OrderedDict(**var,**OrderedDict(coord_n=var_coord_n,vtest=var_vtest,eta2=quali_var_eta2,var_inertia=quali_var_inertia,var_contrib=quali_var_contrib))

        #convert to namedtuple
        self.var_ = namedtuple("var",var.keys())(*var.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##multiple correspondence analysis additionals informations
        #----------------------------------------------------------------------------------------------------------------------------------------
        #inertia
        inertia = (n_cat/n_cols) - 1

        # Eigenvalue threshold
        kaiser_threshold, kaiser_proportion_threshold = 1/n_cols, 100/inertia

        #convert to namedtuple
        self.others_ = namedtuple("others",["inertia","kaiser"])(inertia,namedtuple("kaiser",["threshold","proportion_threshold"])(kaiser_threshold,kaiser_proportion_threshold))
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary individuals
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            # Revaluate if at least two columns have same levels
            X_ind_sup = revaluate_cat_variable(X_ind_sup)

            # Create dummies table for supplementary individuals
            Y = pd.DataFrame(np.zeros((X_ind_sup.shape[0],n_cat)),columns=dummies.columns,index=ind_sup_label)
            for i in range(X_ind_sup.shape[0]):
                values = [X_ind_sup.iloc[i,k] for k in range(n_cols)]
                for j in range(n_cat):
                    if dummies.columns[j] in values:
                        Y.iloc[i,j] = 1
            #
            Z_ind_sup = pd.concat((Y.loc[:,k]*(1/p_k[k])-1 for k in Y.columns),axis=1)
            ind_sup_ = predict_ind_sup(Z_ind_sup,self.svd_.V[:,:n_components],mod_weights,n_workers)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary qualitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)

            # Reevaluate if two variables have the same level
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            # Compute dummies tables
            X_quali_dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in quali_sup_label),axis=1)
            quali_sup_n_k, quali_sup_p_k = X_quali_dummies.sum(axis=0), X_quali_dummies.mean(axis=0)

            #standardiz data
            Z_quali_sup = pd.concat(((X_quali_dummies.loc[:,k]/quali_sup_p_k[k])-1 for k  in X_quali_dummies.columns.tolist()),axis=1)

            # Correlation Ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_.coord.values,weights=ind_weights,n_workers=n_workers) for col in quali_sup_label),axis=0)
            
            #supplementary categories factor coordinates
            quali_sup_coord = mapply(mapply(X_quali_dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(self.ind_.coord),lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)

            #supplementary categories squared distance to origin
            quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            quali_sup_sqdisto.name = "Sq. Dist."

            #supplementary categories square cosinus
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
            
            #supplementary categories value-test
            quali_sup_vtest = pd.concat(((quali_sup_coord.loc[k,:]*np.sqrt(((n_rows-1)*quali_sup_n_k[k])/(n_rows-quali_sup_n_k[k]))).to_frame(name=k).T for k in quali_sup_n_k.index.tolist()),axis=0)

            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",["coord","cos2","vtest","dist","eta2"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_sqdisto,quali_sup_eta2)

            #Summary supplementary qualitatives variables
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            summary_quali_sup.insert(0,"group","sup")

            # Concatenate with activate summary
            summary_quali.insert(0,"group","active")
            summary_quali = pd.concat((summary_quali,summary_quali_sup),axis=0,ignore_index=True)

            #Chi2 statistic test
            chi2_test2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in quali_sup_label:
                for j in X.columns:
                    chi = sp.stats.chi2_contingency(pd.crosstab(X_quali_sup[i],X[j]),correction=False)
                    row_chi2 = pd.DataFrame(OrderedDict(variable1=i,variable2=j,statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                    chi2_test2 = pd.concat((chi2_test2,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            # Transform to int
            chi2_test2["dof"] = chi2_test2["dof"].astype("int")

            #concatenate
            chi2_test2.insert(0,"group","sup")
            chi2_test.insert(0,"group","active")
            chi2_test = pd.concat((chi2_test,chi2_test2),axis=0,ignore_index=True)
            
            #Chi2 statistics between each supplementary qualitatives columns
            if X_quali_sup.shape[1]>1:
                chi2_test3 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for i in np.arange(X_quali_sup.shape[1]-1):
                    for j in np.arange(i+1,X_quali_sup.shape[1]):
                        chi = sp.stats.chi2_contingency( pd.crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j]),correction=False)
                        row_chi2 = pd.DataFrame(OrderedDict(variable1=X_quali_sup.columns[i],variable2=X_quali_sup.columns[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                        chi2_test3 = pd.concat((chi2_test3,row_chi2),axis=0,ignore_index=True)
                        idx = idx + 1
                # Transform to int
                chi2_test3["dof"] = chi2_test3["dof"].astype("int")
                chi2_test3.insert(0,"group","sup")
                #concatenate
                chi2_test = pd.concat((chi2_test,chi2_test3),axis=0,ignore_index=True)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary quantitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)

            # Recode continuous variables : Fill NA if missing
            X_quanti_sup = recodecont(X_quanti_sup.astype("float")).Xcod
            
            # Compute weighted average and and weighted standard deviation
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)

            # Standardization
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=n_workers)
            quanti_sup_ = predict_quanti_sup(Z_quanti_sup,self.svd_.U[:,:n_components],ind_weights,n_workers)

            # Store supplementary quantitatives informations
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            # Compute statistics
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            self.summary_quanti_ = summary_quanti_sup

        self.summary_quali_ = summary_quali
        self.chi2_test_ = chi2_test

        self.model_ = "mca"

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y` : None
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
        X : pandas/polars dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

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
        
        # Set parallelize option
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Set index name as None
        X.index.name = None
        
        # Add revaluate
        X = revaluate_cat_variable(X)

        # Extract some elements
        n_components, dummies, mod_weights = self.call_.n_components, self.call_.dummies, self.call_.mod_weights.values
        p_k = dummies.mean(axis=0)
        
        #create dummies table
        Y = pd.DataFrame(np.zeros((X.shape[0],dummies.shape[1])),columns=dummies.columns,index=X.index)
        for i in np.arange(X.shape[0]):
            values = [X.iloc[i,k] for k in np.arange(0,self.call_.X.shape[1])]
            for j in np.arange(dummies.shape[1]):
                if dummies.columns[j] in values:
                    Y.iloc[i,j] = 1

        # Standardization
        Z = pd.concat(((Y.loc[:,k]/p_k[k])-1 for k in dummies.columns),axis=1)

        # Supplementary individuals Coordinates
        coord = mapply(Z,lambda x : x*mod_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_.V[:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)] 
        return coord

def predictMCA(self,X) -> NamedTuple:
    """
    Predict projection for new individuals with Multiple Correspondence Analysis (MCA)
    ----------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Multiple Correspondence Analysis (MCA)

    Usage
    -----
    ```python
    >>> predictMCA(self,X)
    ```

    Parameters
    ----------
    `self` : an object of class MCA

    `X` : a pandas/polars dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Returns
    -------
    dictionary of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : squared cosines of the new individuals

    `dist` : distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    
    # Check if columns are aligned
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")

    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()
    
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Set index name as None
    X.index.name = None

    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract some elements
    dummies  = self.call_.dummies 
    p_k = dummies.mean(axis=0)

    # Revaluate if at least two columns have same levels
    X = revaluate_cat_variable(X)

    # Create dummies table for supplementary individuals
    Y = pd.DataFrame(np.zeros((X.shape[0],dummies.shape[1])),columns=dummies.columns,index=X.index)
    for i in np.arange(X.shape[0]):
        values = [X.iloc[i,k] for k in range(X.shape[1])]
        for j in range(dummies.shape[1]):
            if dummies.columns[j] in values:
                Y.iloc[i,j] = 1

    # Standardization
    Z = pd.concat((Y.loc[:,k]*(1/p_k[k])-1 for k in dummies.columns),axis=1)
    ind_sup_ = predict_ind_sup(Z,self.svd_.V[:,:self.call_.n_components],self.call_.mod_weights.values,n_workers)

    # convert to NamedTuple
    return namedtuple("predictMCAResult",ind_sup_.keys())(*ind_sup_.values())

def supvarMCA(self,X_quanti_sup=None,X_quali_sup=None) -> NamedTuple:
    """
    Supplementary variables in Multiple Correspondence Analysis (MCA)
    -----------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Multiple Correspondence Analysis (MCA)

    Usage
    -----
    ```python
    >>> supvarMCA(self,X_quanti_sup=None,X_quali_sup=None)
    ```

    Parameters
    ----------
    `self` : an object of class MCA

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives variables

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives variables

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
    >>>
    ```
    """
    # Check if self is and object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1

    ind_weights, n_components = self.call_.ind_weights.values, self.call_.n_components
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary quantitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_quanti_sup,pl.DataFrame):
            X_quanti_sup = X_quanti_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,pd.Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,pd.DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Recode continuous variables : Fill NA if missing
        X_quanti_sup = recodecont(X_quanti_sup.astype("float")).Xcod
        
        # Compute weighted average and and weighted standard deviation
        d_quanti_sup = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)

        # Standardization
        Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=n_workers)
        quanti_sup_ = predict_quanti_sup(Z_quanti_sup,self.svd_.U[:,:n_components],ind_weights,n_workers)

        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X_quali_sup.index.name = None
        
        #convert to factor
        for col in X_quali_sup.columns:
            X_quali_sup[col] = pd.Categorical(X_quali_sup[col],categories=sorted(X_quali_sup[col].dropna().unique().tolist()),ordered=True)
        
        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Compute dummies tables
        dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in X_quali_sup.columns),axis=1)
        n_k, p_k = dummies.sum(axis=0), dummies.mean(axis=0)

        #standardization
        Z_quali_sup = pd.concat(((dummies.loc[:,k]/p_k[k])-1 for k  in dummies.columns),axis=1)

        #supplementary qualitative variables square correlation ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_.coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)
        
        #supplementary categories factor coordinates
        quali_sup_coord = mapply(mapply(dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(self.ind_.coord),lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        #supplementary categories squared distance to origin
        quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        quali_sup_sqdisto.name = "Sq. Dist."

        #supplementary categories square cosinus
        quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
        
        #supplementary categories value-test
        quali_sup_vtest = pd.concat(((quali_sup_coord.loc[k,:]*np.sqrt(((n_rows-1)*n_k[k])/(n_rows-n_k[k]))).to_frame(name=k).T for k in n_k.index),axis=0)

        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",["coord","cos2","vtest","dist","eta2"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_sqdisto,quali_sup_eta2)
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarMCAResult",["quanti","quali"])(quanti_sup,quali_sup)