# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from statsmodels.stats.weightstats import DescrStatsW
from collections import namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from scientisttools.others.fitfa import fitfa
from scientisttools.others.function_eta2 import function_eta2
from scientisttools.others.recodecont import recodecont
from scientisttools.others.revaluate_cat_variable import revaluate_cat_variable

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    ----------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Correspondence Analysis (CA) including supplementary row and/or column points, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    CA(n_components = 5, row_weights = None, row_sup = None, col_sup = None, quanti_sup = None, quali_sup = None, parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `row_weights` : an optional row weights (by default, a list/tuple of 1 and each row has a weight equals to its margin); the weights are given only for the active rows

    `row_sup` : list/tuple indicating the indexes/names of the supplementary rows

    `col_sup` : list/tuple indicating the indexes.names of the supplementary columns

    `quanti_sup` : list/tuple indicating the indexes/names of the supplementary continuous variables

    `quali_sup` : list/tuple indicating the indexes/names of the categorical supplementary variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `svd_` : dictionary of matrices containing all the results of the singular value decomposition

    `col_` : dictionary of pandas dataframe with all the results for the column variable (coordinates, square cosine, contributions, inertia)

    `row_` : dictionary of pandas dataframe with all the results for the row variable (coordinates, square cosine, contributions, inertia)

    `col_sup_` : dictionary of pandas dataframe containing all the results for the supplementary column points (coordinates, square cosine)

    `row_sup_` : dictionary of pandas dataframe containing all the results for the supplementary row points (coordinates, square cosine)

    `quanti_sup_` : if quanti_sup is not None, a dictionary of pandas dataframe containing the results for the supplementary continuous variables (coordinates, square cosine)

    `quali_sup_` : if quali.sup is not None, a dictionary of pandas dataframe with all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a criterion with a Normal distribution, square correlation ratio)

    `summary_quanti_` : summary statistics for quantitative variables (actives and supplementary)

    `summary_quali_` : summary statistics for supplementary qualitative variables if quali_sup is not None

    `chi2_test_` : chi-squared test. If supplementary qualitative variables are greater than 2. 
    
    `call_` : dictionary with some statistics

    `model_` : string specifying the model fitted = 'ca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2009). Analyse de donnees avec R, Presses Universitaires de Rennes.

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_ca_row, get_ca_col, get_ca, summaryCA, dimdesc, predictCA, supvarCA, fviz_ca_row, fviz_ca_col, fviz_ca_biplot

    Examples
    --------
    ```python
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> from scientisttools import CA, summaryCA
    >>> res_ca = CA(row_sup=list(range(14,18)),col_sup=list(range(5,8)),parallelize=True)
    >>> res_ca.fit(children)
    >>> summaryCA(res_ca)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 row_weights = None,
                 row_sup = None,
                 col_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.row_weights = row_weights
        self.row_sup = row_sup
        self.col_sup = col_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_rows, n_columns)
            Training data, where `n_rows` in the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y` : None
            y is ignored.
        
        Returns
        -------
        `self` : object
            Returns the instance itself.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Drop level if ndim greater than 1 and reset columns names
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #----------------------------------------------------------------------------------------------------------------
        ## Checks if categoricals variables is in X and transform to factor (category)
        #----------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns:
                X[col] = pd.Categorical(X[col],categories=sorted(X[col].dropna().unique().tolist()),ordered=True)
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary qualitatives variables in X
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
        ## Check if supplementary columns
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.col_sup is not None:
            if isinstance(self.col_sup,str):
                col_sup_label = [self.col_sup]
            elif isinstance(self.col_sup,(int,float)):
                col_sup_label = [X.columns[int(self.col_sup)]]
            elif isinstance(self.col_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.col_sup):
                    col_sup_label = [str(x) for x in self.col_sup]
                elif all(isinstance(x,(int,float)) for x in self.col_sup):
                    col_sup_label = X.columns[[int(x) for x in self.col_sup]].tolist()
        else:
            col_sup_label = None
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary rows
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            if isinstance(self.row_sup,str):
                row_sup_label = [self.row_sup]
            elif isinstance(self.row_sup,(int,float)):
                row_sup_label = [X.index[int(self.row_sup)]]
            elif isinstance(self.row_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.row_sup):
                    row_sup_label = [str(x) for x in self.row_sup]
                elif all(isinstance(x,(int,float)) for x in self.row_sup):
                    row_sup_label = X.index[[int(x) for x in self.row_sup]].tolist()
        else:
            row_sup_label = None
        
        
        # Store data - Save the base in a variables
        Xtot = X.copy()

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Correspondence Analysis (CA)
        #----------------------------------------------------------------------------------------------------------------------------------------

        #Drop supplementary qualitatives variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)

        #Drop supplementary quantitatives variables
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        #Drop supplementary columns
        if self.col_sup is not None:
            X = X.drop(columns=col_sup_label)
        
        # Drop supplementary rows
        if self.row_sup is not None:
            # Extract supplementary rows
            X_row_sup = X.loc[row_sup_label,:]
            X = X.drop(index=row_sup_label)

        # Active data
        X = X.astype("int")

        # Number of rows/columns
        n_rows, n_cols = X.shape

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Set rows weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.row_weights is None:
            row_weights = np.ones(n_rows)
        elif not isinstance(self.row_weights,(list,tuple,np.ndarray)):
            raise TypeError("'row_weights' must be a list/tuple/array of individuals weights.")
        elif len(self.row_weights) != n_rows:
            raise ValueError(f"'row_weights' must be a list/tuple/array with length {n_rows}.")

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Set number of components
        #----------------------------------------------------------------------------------------------------------------------------------------
        max_components = int(min(n_rows-1,n_cols-1))
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise TypeError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
 
        #total
        total = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0).sum()

        #frequencie table
        freq = mapply(X,lambda x : x*(row_weights/total),axis=0,progressbar=False,n_workers=n_workers)
        
        #Calcul des marges lignes et colones
        col_marge, row_marge = freq.sum(axis=0), freq.sum(axis=1)
        col_marge.name, row_marge.name = "col_marge", "row_marge"

        #Compute Matrix used in SVD
        Z = mapply(mapply(freq,lambda x : x/row_marge,axis=0,progressbar=False,n_workers=n_workers),lambda x : (x/col_marge)-1,axis=1,progressbar=False,n_workers=n_workers)

         ###### Store call informations
        call_ = {"Xtot" : Xtot,
                 "X" : X,
                 "Z" : Z ,
                 "row_weights" : pd.Series(row_weights,index=X.index,name="weight"),
                 "col_marge" : col_marge,
                 "row_marge" : row_marge,
                 "n_components":n_components,
                 "row_sup" : row_sup_label,
                 "col_sup" : col_sup_label,
                 "quanti_sup" : quanti_sup_label,
                 "quali_sup" : quali_sup_label}
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #-------------------------------------------------------------------------------------------------
        ## fit factor analysis model and extract all elements
        #-------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z,row_marge.values,col_marge.values,max_components,n_components,n_workers)

        # Extract elements
        self.svd_, self.eig_, row, col = fit_.svd, fit_.eig, fit_.row, fit_.col
        row_infos, col_infos = row['infos'].rename(columns={"Weight" : "Margin"}), col['infos'].rename(columns={"Weight" : "Margin"})
        row_infos.insert(0,"Weight",row_weights)
        #update dictionary
        row.update({"infos" : row_infos})
        col.update({"infos" : col_infos})

        # store row and columns
        self.row_, self.col_ = namedtuple("row",row.keys())(*row.values()), namedtuple("col",col.keys())(*col.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##Compute others indicators 
        #----------------------------------------------------------------------------------------------------------------------------------------
        # Compute chi - squared test
        statistic, pvalue, dof, expected_freq = sp.stats.chi2_contingency(X, lambda_=None,correction=False)
        chi2_qt = sp.stats.chi2.ppf(0.95,dof)
        # Return indicators
        chi2_test = pd.DataFrame([[statistic,dof,chi2_qt,pvalue]],columns=["statistic","dof","quantile","pvalue"],index=["chi2-test"])

        # log - likelihood - tes (G - test)
        g_stat, g_pvalue = sp.stats.chi2_contingency(X, lambda_="log-likelihood")[:2]
        g_test = pd.DataFrame([[g_stat,dof,chi2_qt,g_pvalue]],columns=["statistic","dof","quantile","pvalue"],index=["g-test"])

        # Absolute residuals
        resid = X.sub(expected_freq)

        # Standardized resid
        std_resid = resid.div(np.sqrt(expected_freq))

        # Adjusted residuals
        adj_resid = mapply(mapply(std_resid,lambda x : x/np.sqrt(1-row_marge),axis=0,progressbar=False,n_workers=n_workers),lambda x : x/np.sqrt(1-col_marge),axis=1,progressbar=False,n_workers=n_workers)

        ##### Chi2 contribution
        chi2_contrib = mapply(std_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False,n_workers=n_workers)

        # Attraction repeulsion index
        ari = X.div(expected_freq)
        
        # Association test
        association = pd.DataFrame([[sp.stats.contingency.association(X, method=name,correction=False) for name in ["cramer","tschuprow","pearson"]]],index=["statistic"],columns=["cramer","tschuprow","pearson"])
    
        # Inertia
        inertia = np.sum(row_infos.iloc[:,3])
        kaiser_threshold, kaiser_proportion_threshold = np.mean(self.eig_.iloc[:,0]), 100/max_components
       
       # Others informations
        others_ = {"resid" : resid,
                    "chi2" : chi2_test,
                    "g_test" : g_test,
                    "adj_resid" : adj_resid,
                    "chi2_contrib" : chi2_contrib,
                    "std_resid" : std_resid,
                    "ari" : ari,
                    "association" : association,
                    "inertia" : inertia,
                    "kaiser_threshold" : kaiser_threshold,
                    "kaiser_proportion_threshold" : kaiser_proportion_threshold}
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            X_row_sup = X_row_sup.astype("int")
            
            # Standardize with the row sum
            Z_row_sup = mapply(X_row_sup,lambda x : x/X_row_sup.sum(axis=1),axis=0,progressbar=False,n_workers=n_workers)

            # Supplementary rows factor coordinates
            row_sup_coord = pd.DataFrame(Z_row_sup.dot(self.svd_.V[:,:n_components]).values,columns=["Dim."+str(x+1) for x in range(n_components)],index=row_sup_label)
            
            # Supplementary rows square distance to origin
            row_sup_sqdisto = mapply(Z_row_sup,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            row_sup_sqdisto.name = "Sq. Dist."

            # Supplementary rows square cosine
            row_sup_cos2 = mapply(row_sup_coord,lambda x : (x**2)/row_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)

            # convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",["coord","cos2","dist"])(row_sup_coord,row_sup_cos2,row_sup_sqdisto)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.col_sup is not None:
            X_col_sup = Xtot.loc[:,col_sup_label]
            if self.row_sup is not None:
                X_col_sup = X_col_sup.drop(index=row_sup_label)
            
            # Transform to int
            X_col_sup = X_col_sup.astype("int")

            # weighted with row weight
            X_col_sup = mapply(X_col_sup,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers)

            # Standardize supplementary columns
            Z_col_sup = mapply(X_col_sup,lambda x : x/X_col_sup.sum(axis=0).values,axis=1,progressbar=False,n_workers=n_workers)

            # Supplementary columns factor coordinates
            col_sup_coord = pd.DataFrame(Z_col_sup.T.dot(self.svd_.U[:,:n_components]).values,columns=["Dim."+str(x+1) for x in range(n_components)],index=col_sup_label)

            # Supplementary columns square distance to origin
            col_sup_sqdisto = mapply(Z_col_sup,lambda x : ((x - row_marge.values)**2)/row_marge.values,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            col_sup_sqdisto.name = "Sq. Dist."

            # Supplementary columns square cosine
            col_sup_cos2 = mapply(col_sup_coord,lambda x : (x**2)/col_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)

            # convert to namedtuple
            self.col_sup_ = namedtuple("col_sup",["coord","cos2","dist"])(col_sup_coord,col_sup_cos2,col_sup_sqdisto)
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.row_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=row_sup_label)
            
            ##### From frame to DataFrame
            if isinstance(X_quanti_sup,pd.Series):
                X_quanti_sup = X_quanti_sup.to_frame()
            
            # transform and recode
            X_quanti_sup = recodecont(X_quanti_sup.astype("float")).Xcod

            #descriptive statistifs
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            self.summary_quanti_ = summary_quanti_sup
                
            # supplementary quantitative variables factor coordinates - factor correlation
            wcorr = DescrStatsW(pd.concat((X_quanti_sup,self.row_.coord),axis=1),weights=row_marge,ddof=0).corrcoef[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
            quanti_sup_coord = pd.DataFrame(wcorr,index=quanti_sup_label,columns=["Dim."+str(x+1) for x in range(n_components)])
            
            # supplementary quantitative variable ssquare cosinus
            quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers)

            # convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",["coord","cos2"])(quanti_sup_coord,quanti_sup_cos2)
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary categorical variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.row_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=row_sup_label)
            
            #convert to DataFrame if Series
            if isinstance(X_quali_sup,pd.Series):
                X_quali_sup = X_quali_sup.to_frame()
        
            # revaluate if two variables have same categories
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            # Compute statistiques summary_quali
            summary_quali = pd.DataFrame()
            for col in X.columns.tolist():
                eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
            summary_quali["count"] = summary_quali["count"].astype("int")
            self.summary_quali_ = summary_quali

            # Sum of columns by group
            quali_sup = pd.DataFrame().astype("float")
            for col in X_quali_sup.columns.tolist():
                data = pd.concat((X,X_quali_sup[col]),axis=1).groupby(by=col,as_index=True).sum()
                data.index.name = None
                quali_sup = pd.concat((quali_sup,data),axis=0)

            # Standardize the data
            Z_quali_sup = mapply(quali_sup,lambda x : x/quali_sup.sum(axis=1),axis=0,progressbar=False,n_workers=n_workers)

            # Supplementary categories factor coordinates
            quali_sup_coord = pd.DataFrame(Z_quali_sup.dot(self.svd_.V[:,:n_components]).values,columns=["Dim."+str(x+1) for x in range(n_components)],index=Z_quali_sup.index.tolist())

            # Supplementary categories square distance to origin
            quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            quali_sup_sqdisto.name="Sq. Dist."

            # Supplementary categories square cosine
            quali_sup_cos2 = mapply(quali_sup_coord,lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)

            # Disjonctif table
            dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in quali_sup_label),axis=1)
            # Compute : weighted count by categories
            n_k = mapply(dummies,lambda x : x*row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)*total

            # Supplementary categories value-test 
            if total > 1:
                coef = np.array([np.sqrt(n_k[i]*((total - 1)/(total - n_k[i]))) for i in range(len(n_k))])
            else:
                coef = np.sqrt(n_k)
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*coef,axis=0,progressbar=False,n_workers=n_workers)

            # Supplementary categories correlation ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.row_.coord.values,weights=row_marge,n_workers=n_workers) for col in quali_sup_label),axis=0)
            
            # convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",["coord","cos2","vtest","eta2","dist"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_eta2,quali_sup_sqdisto)
        
        self.model_ = "ca"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_rows, n_columns)
            Training data, where `n_rows` is the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y` : None.
            y is ignored.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.row_.coord

    def transform(self,X):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_rows, n_columns)
            New data, where `n_rows` is the number of row points and `n_columns` is the number of columns

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_rows, n_components)
            Projection of X in the principal components where `n_rows` is the number of rows and `n_components` is the number of the components.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # Checif if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Extract number of components
        n_components = self.call_.n_components

        # Set type to int
        X = X.astype("int")
        coord = mapply(X,lambda x : x/X.sum(axis=1).values,axis=0,progressbar=False,n_workers=n_workers).dot(self.svd_.V[:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        coord.index = X.index.tolist()
        return coord

def predictCA(self,X) -> NamedTuple:
    """
    Predict projection for new rows with Correspondence Analysis (CA)
    -----------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, square cosinus and square distance to origin of new rows with Correspondence Analysis

    Usage
    -----
    ```python
    >>> predictCA(self,X)
    ```

    Parameters
    ----------
    `self` : an object of class CA

    `X` : pandas/polars dataframe in which to look for columns with which to predict. X must contain columns with the same names as the original data

    Return
    ------
    namedtuple of dataframes including :

    `coord` : factor coordinates (scores) for supplementary rows

    `cos2` : square cosinus for supplementary rows

    `dist` : square distance to origin for supplementary rows

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> # Actifs elements
    >>> actif = children.iloc[:14,:5]
    >>> # Supplementary rows
    >>> row_sup = children.iloc[14:,:5]
    >>> # Correspondence Analysis (CA)
    >>> from scientisttools import CA
    >>> res_ca = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)))
    >>> res_ca.fit(children)
    >>> # Prediction on supplementary rows
    >>> from scientisttools import predictCA
    >>> predict = predictCA(res_ca, X=row_sup)
    ```
    """
    # Check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()

    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # Set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    n_components, col_marge = self.call_.n_components, self.call_.col_marge.values

    # Transform to int
    X = X.astype("int")
    
    # Standardize the data
    Z = mapply(X,lambda x : x/X.sum(axis=1).values,axis=0,progressbar=False,n_workers=n_workers)

    # Supplementary factor coordinates
    coord = pd.DataFrame(Z.dot(self.svd_.V[:,:n_components]).values,columns=["Dim."+str(x+1) for x in range(n_components)],index=X.index.tolist())
    
    # Supplementary square distance to origin
    sqdisto = mapply(Z,lambda x : ((x - col_marge)**2)/col_marge,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    
    # Supplementary square cosinus
    cos2 = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=n_workers)

    # convert to namedtuple
    return namedtuple("predictCAResult",["coord","cos2","dist"])(coord,cos2,sqdisto)

def supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None) -> NamedTuple:
    """
    Supplementary columns/variables with Correspondence Analysis (CA)
    -----------------------------------------------------------------

    Description
    -----------
    Performns the coordinates, square cosinus and square distance to origin of supplementary columns/variables with Correspondence Analysis (CA)

    Usage
    -----
    ```python
    >>> supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None)   
    ```

    Parameters
    ----------
    `self` : an object of class CA

    `X_col_sup` : pandas/polars dataframe of supplementary columns

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives columns

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives columns

    Returns
    -------
    namedtuple including : 

    `col` : namedtuple containing the results of the supplementary columns variables:
        * coord : factor coordinates (scores) of the supplementary columns
        * cos2 : square cosinus of the supplementary columns
        * dist : distance to origin of the supplementary columns

    `quanti` : namedtuple containing the results of the supplementary quantitatives variables:
        * coord : factor coordinates (scores) of the supplementary quantitativaes variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : namedtuple containing the results of the supplementary qualitatives/categories variables :
        * coord : factor coordinates (scores) of the supplementary categories
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
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> # Add qualitatives variables
    >>> children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +["D"]*4
    >>> # Supplementary columns
    >>> X_col_sup = children.iloc[:14,5:8]
    >>> # Supplementary qualitatives variables
    >>> X_quali_sup = children.iloc[:14,8]
    >>> from scientisttools import CA
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Supplementary columns/variables projections
    >>> from scientisttools import supvarCA
    >>> sup_var = supvarCA(res_ca,X_col_sup=X_col_sup,X_quanti_sup=X_col_sup,X_quali_sup=X_quali_sup)
    >>> # Extract supplementary columns results
    >>> col_sup_ = sup_var.col
    >>> # Extract supplementary quantitatives variables results
    >>> quanti_sup_ = sup_var.quanti
    >>> # Extract supplementary qualitatives variables results
    >>> quali_sup_ = sup_var.quali
    ```
    """
    # Check if self is and object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    row_weights, n_components, row_marge = self.call_.row_weights.values, self.call_.n_components, self.call_.row_marge.values
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary columns
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_col_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_col_sup,pl.DataFrame):
            X_col_sup = X_col_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_col_sup,pd.Series):
            X_col_sup = X_col_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_col_sup,pd.DataFrame):
            raise TypeError(f"{type(X_col_sup)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to int
        X_col_sup = X_col_sup.astype("int")

        # weighted with row weight
        X_col_sup = mapply(X_col_sup,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers)
        # Standardize the data
        X_col_sup = mapply(X_col_sup,lambda x : x/X_col_sup.sum(axis=0),axis=1,progressbar=False,n_workers=n_workers)

        # Supplementary columns factor coordinates
        col_sup_coord = pd.DataFrame(X_col_sup.T.dot(self.svd_.U[:,:n_components]).values,index=X_col_sup.columns.tolist(),columns = ["Dim."+str(x+1) for x in range(n_components)])

        # Supplementary columns square distance to origin
        col_sup_sqdisto = mapply(X_col_sup,lambda x : ((x - row_marge)**2)/row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        col_sup_sqdisto.name = "Sq. Dist."

        # Supplementary columns square cosinus
        col_sup_cos2 = mapply(col_sup_coord,lambda x : (x**2)/col_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
        
        #convert to namedtuple
        col_sup = namedtuple("col_sup",["coord","cos2","dist"])(col_sup_coord,col_sup_cos2,col_sup_sqdisto)
    else:
        col_sup = None

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
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Recode variables
        X_quanti_sup = recodecont(X_quanti_sup.astype("float")).Xcod

        # supplementary quantitative variables factor coordinates - factor correlation
        wcorr = DescrStatsW(pd.concat((X_quanti_sup,self.row_.coord),axis=1),weights=row_marge,ddof=0).corrcoef[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
        quanti_sup_coord = pd.DataFrame(wcorr,index=X_quanti_sup.columns.tolist(),columns=["Dim."+str(x+1) for x in range(n_components)])
        
        # supplementary quantitative variable square cosinus
        quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers)
     
        # Set all informations
        quanti_sup = namedtuple("quanti_sup",["coord","cos2"])(quanti_sup_coord,quanti_sup_cos2)
    else:
        quanti_sup = None
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitatives variables
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
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")

        # Extract active data
        X, col_marge = self.call_.X, self.call_.col_marge.values
        total = X.sum().sum()

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)

        # Sum of columns by group
        quali = pd.DataFrame().astype("float")
        for col in X_quali_sup.columns.tolist():
            data = pd.concat((X,X_quali_sup[col]),axis=1).groupby(by=col,as_index=True).sum()
            data.index.name = None
            quali = pd.concat((quali,data),axis=0)
        
        #standardize 
        Z_quali_sup = mapply(quali,lambda x : x/quali.sum(axis=1),axis=0,progressbar=False,n_workers=n_workers)

        # Categories factor coordinates
        quali_sup_coord = pd.DataFrame(Z_quali_sup.dot(self.svd_.V[:,:n_components]).values,index=quali.index.tolist(),columns = ["Dim."+str(x+1) for x in range(n_components)])

        # Categories dist2
        quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : ((x - col_marge)**2)/col_marge,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        quali_sup_sqdisto.name="Sq. Dist."

        # Sup Cos2
        quali_sup_cos2 = mapply(quali_sup_coord,lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)

        # Disjonctif table
        dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in X_quali_sup.columns),axis=1)
        # Compute : weighted count by categories
        n_k = mapply(dummies,lambda x : x*row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)*total

        ######## Weighted of coordinates to have 
        if total > 1:
            coef = np.array([np.sqrt(n_k[i]*((total - 1)/(total - n_k[i]))) for i in range(len(n_k))])
        else:
            coef = np.sqrt(n_k)
        # Value - test
        quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*coef,axis=0,progressbar=False,n_workers=n_workers)

        # Correlation ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.row_.coord.values,weights=row_marge,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)
        
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",["coord","cos2","vtest","eta2","dist"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_eta2,quali_sup_sqdisto)
    else:
        quali_sup = None
    
    # Store all informations
    return namedtuple("supvarCAResult",["col","quanti","quali"])(col_sup,quanti_sup,quali_sup)