# -*- coding: utf-8 -*-
from numpy import ones,ndarray,sqrt, mean, outer
from scipy.stats import chi2, chi2_contingency, contingency
from pandas import DataFrame, Categorical, Series, concat, get_dummies, api
from statsmodels.stats.weightstats import DescrStatsW
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.fitfa import fitfa
from .functions.recodecont import recodecont
from .functions.revaluate_cat_variable import revaluate_cat_variable
from .functions.summarize import sum_col_by
from .functions.predict_sup import predict_ind_sup, predict_quanti_sup, predict_quali_sup

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    ----------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Correspondence Analysis (CA) including supplementary rows and/or columns points, supplementary quantitative variables and supplementary qualitative variables.

    Usage
    -----
    ```python
    >>> CA(n_components = 5, row_weights = None, row_sup = None, col_sup = None, quanti_sup = None, quali_sup = None, parallelize = False)
    ```

    Parameters
    ----------
    `n_components`: number of dimensions kept in the results (by default 5)

    `row_weights`: an optional row weights (by default, a list/tuple/ndarray/Series of 1 and each row has a weight equals to its margin); the weights are given only for the active rows

    `row_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary rows points

    `col_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary columns points

    `quanti_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `quali_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary qualitative variables

    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True`: parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False`: parallelize using pandas apply

    Attributes
    ----------
    `call_`: a namedtuple with some informations:
        * `Xtot`: pandas DataFrame with all data (active and supplementary)
        * `X`: pandas DataFrame with active data
        * `Z`: pandas DataFrame with standardized data
        * `total`: numeric specifying the sum of elements in X
        * `row_weights`: pandas Series containing rows weights
        * `row_marge`: pandas Series containing rows marging
        * `col_marge`: pandas Series containing columns marging
        * `n_components`: an integer indicating the number of components kept
        * `n_workers`: an integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `row_sup`: None or a list of string indicating names of the supplementary rows
        * `col_sup`: None or a list of string indicating names of the supplementary columns
        * `quanti_sup`: None or a list of string indicating names of the supplementary quantitative variables
        * `quali_sup`: None or a list of string indicating names of the supplementary qualitative variables

    `svd_`: a namedtuple of numpy array containing all the results of the generalized singular value decomposition (GSVD)
        * `vs`: 1D numpy array containing the singular values
        * `U`: 2D numpy array whose columns contain the left singular vectors
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: a pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `col_`: a namedtuple of pandas DataFrame with all the results for the column points.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `contrib`: relative contributions
        * `infos`: additionals informations (margin, squared distance to origin and inertia)

    `row_`: a namedtuple of pandas DataFrame with all the results for the row points.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `contrib`: relative contributions
        * `infos`: additionals informations (weight, margin, squared distance to origin and inertia)

    `goodness_`: a namedtuple of pandas DataFrame for multivariate goodness of fit test
        * `chi2`: Pearson's chi-squared test
        * `gtest`: log-likelihood ratio (i.e the "G-test")
        * `association`: degree of association between two nominal variables ("cramer", "tschuprow", "pearson")
    
    `residual_` : a namedtuple of pandas DataFrame for residuals
        * `resid`: model residuals
        * `rstandard`: standardized residuals
        * `radjusted`: adjusted residuals
        * `contrib`: contribution to chi-squared
        * `att_rep_ind`: attraction repulsion index

    `others_`: a namedtuple with others CA statistics
        * `kaiser`: a namedtuple of numeric containing the kaiser threshold and proportion
            * `threshold`: a numeric value specifying the kaiser threshold
            * `porportion`: a numeric value specifying the kaiser proportion threshold

    `row_sup_`: if row_sup is not None, a namedtuple of pandas DataFrame/Series containing all the results for the supplementary row points.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `dist`: squared distance to origin

    `col_sup_`: if col_sup is not None, a namedtuple of pandas DataFrame/Series containing all the results for the supplementary column points.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `dist`: squared distance to origin

    `quanti_sup_`: if quanti_sup is not None, a namedtuple of pandas DataFrame containing the results for the supplementary continuous variables.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus

    `quali_sup_`: if quali_sup is not None, a namedtuple of pandas DataFrame with all the results for the supplementary categorical variables.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `vtest`: value-test
        * `dist`: squared distance
        * `eta2`: squared correlation ratio

    `model_`: a string specifying the model fitted = 'ca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod
    
    * Husson, F., Le, S. and Pages, J. (2009). Analyse de donnees avec R, Presses Universitaires de Rennes.

    * Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    * Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    `predictCA`, `supvarCA`, `get_ca_row`, `get_ca_col`, `get_ca`, `summaryCA`, `dimdesc`, `fviz_ca_row`, `fviz_ca_col`, `fviz_ca_biplot`

    Examples
    --------
    ```python
    >>> from scientisttools import children, CA, summaryCA
    >>> #with supplementary rows, supplementary columns and supplementary qualitative variables
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8).fit(children)
    >>> summaryCA(res_ca)
    >>> #with supplementary rows, supplementary quantitative variables and supplementary qualitative variables
    >>> res_ca2 = CA(row_sup=(14,15,16,17),quanti_sup=(5,6,7),quali_sup=8).fit(children)
    >>> summaryCA(res_ca2)
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

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_rows, n_columns),
            Training data, where `n_rows` in the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y`: None,
            y is ignored.
        
        Returns
        -------
        `self`: object
            Returns the instance itself.

        Examples
        --------
        ```python
        >>> from scientisttools import children, CA
        >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8).fit(children)
        ```
        """
        #check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if parallelize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.parallelize,bool):
            raise TypeError("'parallelize' must be a boolean.")

        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop level if ndim greater than 1 and reset columns names
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #checks if categoricals variables is in X and transform to factor (category)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for q in is_quali.columns:
                X[q] = Categorical(X[q],categories=sorted(X[q].dropna().unique().tolist()),ordered=True)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary qualitative variables
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
        #check if supplementary quantitative variables
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
        #check if supplementary columns
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
        #check if supplementary rows
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
        
        
        #Store data - Save the base in a variables
        Xtot = X.copy()

        #drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)

        #drop supplementary quantitative variables
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        #drop supplementary columns
        if self.col_sup is not None:
            X = X.drop(columns=col_sup_label)
        
        #drop supplementary rows
        if self.row_sup is not None:
            X_row_sup = X.loc[row_sup_label,:]
            X = X.drop(index=row_sup_label)

        #----------------------------------------------------------------------------------------------------------------------------------------
        #correspondence analysis (CA)
        #----------------------------------------------------------------------------------------------------------------------------------------
        # convert to integer
        X = X.astype(int)

        #number of rows/columns
        n_rows, n_cols = X.shape

        #----------------------------------------------------------------------------------------------------------------------------------------
        #set rows weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.row_weights is None:
            row_weights = ones(n_rows)
        elif not isinstance(self.row_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'row_weights' must be a list/tuple/ndarray/Series of individuals weights.")
        elif len(self.row_weights) != n_rows:
            raise ValueError(f"'row_weights' must be a list/tuple/ndarray/Series with length {n_rows}.")
        
        #convert weights to Series
        row_weights =  Series(row_weights,index=X.index,name="weight")

        #----------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
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

        #----------------------------------------------------------------------------------------------------------------------------------------
        #standardize the data
        #----------------------------------------------------------------------------------------------------------------------------------------
        #total
        total = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers).sum().sum()

        #frequencie table
        freq = mapply(X,lambda x : x*(row_weights/total),axis=0,progressbar=False,n_workers=n_workers)
        
        #calcul des marges lignes et colones
        col_marge, row_marge = freq.sum(axis=0), freq.sum(axis=1)
        col_marge.name, row_marge.name = "Margin", "Margin"

        #compute Matrix used in SVD
        Z = mapply(mapply(freq,lambda x : x/row_marge,axis=0,progressbar=False,n_workers=n_workers),lambda x : (x/col_marge)-1,axis=1,progressbar=False,n_workers=n_workers)

        # Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,total=total,row_weights=row_weights,row_marge=row_marge,col_marge=col_marge,n_components=n_components,n_workers=n_workers,
                            row_sup=row_sup_label,col_sup=col_sup_label,quanti_sup=quanti_sup_label,quali_sup=quali_sup_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        #fit factor analysis model and extract all elements
        #----------------------------------------------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z=Z,row_weights=row_marge,col_weights=col_marge,max_components=max_components,n_components=n_components,n_workers=n_workers)

        #extract elements
        self.svd_, self.eig_, row, col = fit_.svd, fit_.eig, fit_.row, fit_.col
        row_infos, col_infos = row['infos'].rename(columns={"Weight" : "Margin"}), col['infos'].rename(columns={"Weight" : "Margin"})
        row_infos.insert(0,"Weight",row_weights)
        #update dictionary
        row.update({"infos" : row_infos})
        col.update({"infos" : col_infos})

        #store row and columns
        self.row_, self.col_ = namedtuple("row",row.keys())(*row.values()), namedtuple("col",col.keys())(*col.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        #diagnostics tests - multivariate goodness of fit tests
        #----------------------------------------------------------------------------------------------------------------------------------------
        #conclusion on test
        def test_conclusion(pvalue):
            if pvalue <= 0.05:
                return "Dependent (reject H0)"
            else:
                return "Independent (H0 holds true)"
        #compute chi - squared test
        statistic, pvalue, dof, expected_freq = chi2_contingency(X, lambda_=None,correction=False)
        chi2_qt = chi2.ppf(0.95,dof)
        
        #convert to DataFrame
        chi2_test = DataFrame([[statistic,dof,chi2_qt,pvalue]],columns=["statistic","dof","critical value","pvalue"],index=["Pearson's Chi-Square Test"])
        chi2_test["conclusion"] = test_conclusion(pvalue=pvalue)
        #log - likelihood test (G - test)
        g_stat, g_pvalue = chi2_contingency(X, lambda_="log-likelihood")[:2]
        #convert to DataFrame
        g_test = DataFrame([[g_stat,dof,chi2_qt,g_pvalue]],columns=["statistic","dof","critical value","pvalue"],index=["g-test"])
        g_test["conclusion"] = test_conclusion(pvalue=g_pvalue)
        #association test
        association = DataFrame([[contingency.association(X,method=i,correction=False) for i in ["cramer","tschuprow","pearson"]]],index=["statistic"],columns=["cramer","tschuprow","pearson"])
        #convert to ordered dictionary
        goodness_ = OrderedDict(chi2=chi2_test,gtest=g_test,association=association)
        #convert to namedtuple
        self.goodness_ = namedtuple("test",goodness_.keys())(*goodness_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        #residuals
        #----------------------------------------------------------------------------------------------------------------------------------------
        #absolute residuals
        resid = X.sub(expected_freq)
        #standardized residuals
        std_resid = resid.div(sqrt(expected_freq))
        #adjusted residuals
        adj_resid = mapply(mapply(std_resid,lambda x : x/sqrt(1-row_marge),axis=0,progressbar=False,n_workers=n_workers),lambda x : x/sqrt(1-col_marge),axis=1,progressbar=False,n_workers=n_workers)
        #chi2 contribution
        chi2_contrib = mapply(std_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False,n_workers=n_workers)
        #attraction repulsion index
        att_rep_ind = X.div(expected_freq)
        #convert to ordered dictionary
        residuals_ = OrderedDict(resid=resid,rstandard=std_resid,radjusted=adj_resid,contrib=chi2_contrib,att_rep_ind=att_rep_ind)
        #convert to namedtuple
        self.residuals_ = namedtuple("residuals",residuals_.keys())(*residuals_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        #compute others indicators 
        #----------------------------------------------------------------------------------------------------------------------------------------
        #kaiser threshold
        kaiser = namedtuple("kaiser",["threshold","proportion"])(mean(self.eig_.iloc[:,0]),100/max_components)
        #convert to namedtuple
        self.others_ = namedtuple("others",["kaiser"])(kaiser)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary rows points
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            #standardize with the row sum
            Z_row_sup = mapply(X_row_sup,lambda x : x/X_row_sup.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers)
            #supplementary rows square distance to origin
            row_sup_sqdisto = mapply(Z_row_sup,lambda x : ((x - self.call_.col_marge)**2)/self.call_.col_marge,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
            row_sup_sqdisto.name = "Sq. Dist."
            #statistics for supplementary rows
            row_sup_ = predict_ind_sup(Z=Z_row_sup,V=self.svd_.V,sqdisto=row_sup_sqdisto,col_weights=None,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",row_sup_.keys())(*row_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary columns points
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.col_sup is not None:
            X_col_sup = Xtot.loc[:,col_sup_label]
            if self.row_sup is not None:
                X_col_sup = X_col_sup.drop(index=row_sup_label)
            #weighted with row weight
            X_col_sup = mapply(X_col_sup,lambda x : x*self.call_.row_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers)
            #standardize supplementary columns
            Z_col_sup = mapply(X_col_sup,lambda x : x/X_col_sup.sum(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #supplementary columns square distance to origin
            col_sup_sqdisto = mapply(Z_col_sup,lambda x : ((x - self.call_.row_marge)**2)/self.call_.row_marge,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
            col_sup_sqdisto.name = "Sq. Dist."
            #statistics for supplementary columns
            col_sup_ = predict_ind_sup(Z=Z_col_sup.T,V=self.svd_.U,sqdisto=col_sup_sqdisto,col_weights=None,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.col_sup_ = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())
         
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.row_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=row_sup_label)
            #convert to float and fill missing with mean
            X_quanti_sup = recodecont(X=X_quanti_sup).X
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_quanti_sup(X=X_quanti_sup,row_coord=self.row_.coord,row_weights=self.call_.row_marge,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary categorical variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.row_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=row_sup_label)
            #check if two columns have the same categories
            X_quali_sup = revaluate_cat_variable(X_quali_sup)
            #create contingency table with categories as rows 
            quali = sum_col_by(X=self.call_.X,X_quali=X_quali_sup)
            #standardize the data
            Z_quali_sup = mapply(quali,lambda x : x/quali.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers)
            #factor coordinates of categories
            quali_sup_coord = Z_quali_sup.dot(self.svd_.V)
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
            #supplementary categories square distance to origin
            quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : ((x - self.call_.col_marge)**2)/self.call_.col_marge,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
            quali_sup_sqdisto.name="Sq. Dist."
            #disjunctive table
            dummies = concat((get_dummies(X_quali_sup[q],dtype=int) for q in quali_sup_label),axis=1)
            #compute : weighted count by categories
            n_k = mapply(dummies,lambda x : x*self.call_.row_marge,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)*self.call_.total
            #supplementary categories value-test 
            if self.call_.total > 1:
                coef_k = sqrt(((self.call_.total - 1)*n_k)/(self.call_.total - n_k))
            else:
                coef_k = sqrt(n_k)
            #statistics for supplementary categories
            quali_sup_ = predict_quali_sup(X=X_quali_sup,row_coord=self.row_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.row_marge,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
            
        self.model_ = "ca"

        return self

    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_rows, n_columns)
            Training data, where `n_rows` is the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y`: None.
            y is ignored.

        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_rows, n_components)
            Transformed values.
        
        Examples
        --------
        ```python
        >>> from scientisttools import children, CA, summaryCA
        >>> row_coord = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8).fit_transform(children)
        ```
        """
        self.fit(X)
        return self.row_.coord
    
    def inverse_transform(self,X:DataFrame) -> DataFrame:
        """
        Transform data back to its original space
        -----------------------------------------

        Description
        -----------
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas DataFrame of shape (n_samples, n_columns)
            Original data, where `n_samples` is the number of samples and `n_columns` is the number of columns

        Examples
        --------
        ```python
        >>> from scientisttools import children, CA, summaryCA
        >>> res_ca = CA(n_components=None,row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8).fit(children)
        >>> X_original = res_ca.inverse_transform(res_ca.row_.coord)
        ```
        """
        #check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set number of components
        n_components = min(X.shape[1],self.call_.n_components)
        #extract elements
        F, G, m = X.iloc[:,:n_components], self.col_.coord.iloc[:,:n_components], self.call_.col_marge
        hatX = F.dot(mapply(G,lambda x : x/sqrt(G.pow(2).T.dot(m)),axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
        hatX = mapply(mapply(hatX.add(1),lambda x : x*self.call_.row_marge,axis=0,progressbar=False,n_workers=self.call_.n_workers),lambda x : (x*self.call_.col_marge),axis=1,progressbar=False,n_workers=self.call_.n_workers).mul(self.call_.total)
        return hatX

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_rows, n_columns)
            New data, where `n_rows` is the number of row points and `n_columns` is the number of columns

        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_rows, n_components)
            Projection of X in the principal components where `n_rows` is the number of rows and `n_components` is the number of the components.
        
        Examples
        --------
        ```python
        >>> from scientisttools import load_children, CA, predictCA
        >>> children = load_children("all")
        >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
        >>> res_ca.fit(children)
        >>> #projection on supplementary rows
        >>> row_sup = load_children("row_sup")
        >>> row_sup_coord = res_ca.transform(row_sup)
        ```
        """
        # check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # set index name to None
        X.index.name = None

        #check if X.shape[1] = ncols
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")

        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[j]) for j in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #find intersect
        intersect_col = [x for x in X.columns if x in self.call_.X.columns]
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the CA result")
        #reorder columns
        X = X.loc[:,self.call_.X.columns]

        #factor coordinates for new rows
        coord = mapply(X,lambda x : x/X.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord

def predictCA(self,X:DataFrame) -> NamedTuple:
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
    `self`: an object of class CA

    `X`: pandas DataFrame in which to look for columns with which to predict. X must contain columns with the same names as the original data

    Return
    ------
    a namedtuple of pandas DataFrame/Series containing all the results for the new row points.

    `coord`: factor coordinates

    `cos2`: squared cosinus

    `dist`: squared distance to origin

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import load_children, CA, predictCA
    >>> children = load_children("all")
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children)
    >>> #prediction on supplementary rows
    >>> row_sup = load_children("row_sup")
    >>> predict = predictCA(res_ca,X=row_sup)
    >>> predict.coord.head() # new row coordinates
    >>> predict.cos2.head() # new row cos2
    >>> predict.dist.head() # new row squared distance to origin
    ```
    """
    #check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    #check if X is an instance of pandas DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set index name as None
    X.index.name = None

    #check if X.shape[1] = ncols
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")

    #check if all variables are numerics
    all_num = all(api.types.is_numeric_dtype(X[j]) for j in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")
    
    #find intersect
    intersect_col = [x for x in X.columns if x in self.call_.X.columns]
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the CA result")
    #reorder columns
    X = X.loc[:,self.call_.X.columns]

    #standardize the data
    Z = mapply(X,lambda x : x/X.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers)
    #supplementary square distance to origin
    sqdisto = mapply(Z,lambda x : ((x - self.call_.col_marge)**2)/self.call_.col_marge,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #statistics for supplementary individuals
    predict_ = predict_ind_sup(Z=Z,V=self.svd_.V,sqdisto=sqdisto,col_weights=None,n_workers=self.call_.n_workers)
    return namedtuple("predictCAResult",predict_.keys())(*predict_.values())

def supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None) -> NamedTuple:
    """
    Supplementary columns/variables with Correspondence Analysis (CA)
    -----------------------------------------------------------------

    Description
    -----------
    Performns the coordinates, squared cosinus and squared distance to origin of supplementary columns/variables with Correspondence Analysis (CA)

    Usage
    -----
    ```python
    >>> supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None)   
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `X_col_sup`: pandas DataFrame of supplementary columns

    `X_quanti_sup`: pandas DataFrame of supplementary quantitative columns

    `X_quali_sup`: pandas DataFrame of supplementary qualitative columns

    Returns
    -------
    a namedtuple of namedtuple containing the results for supplementary columns/variables including : 

    `col`: a namedtuple of pandas DataFrame/Series containing all the results for the suppleementary columns.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `dist`: squared distance to origin

    `quanti`: a namedtuple of pandas DataFrame containing all the results for the supplementary quantitative variables.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
    
    `quali`: a namedtuple of pandas DataFrame/Series containing all the results for the supplementary qualitative variables.
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
    >>> from scientisttools import load_children, CA, supvarCA
    >>> children = load_children("all")
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children)
    >>> #supplementary columns/variables projections
    >>> X_col_sup, X_quali_sup = load_children("col_sup"), load_children("quali_sup")
    >>> sup_var = supvarCA(res_ca,X_col_sup=X_col_sup,X_quanti_sup=X_col_sup,X_quali_sup=X_quali_sup)
    >>> #extract supplementary columns informations
    >>> col_sup = sup_var.col
    >>> col_sup.coord.head() # coordinates
    >>> col_sup.cos2.head() # cos2
    >>> col_sup.dist.head() # squared distance to origin
    >>> #extract supplementary quantitatives variables informations
    >>> quanti_sup = sup_var.quanti
    >>> quanti_sup.coord.head() # coordinates
    >>> quanti_sup.cos2.head() # cos2
    >>> #extract supplementary qualitatives variables informations
    >>> quali_sup = sup_var.quali
    >>> quali_sup.coord.head() # coordinates
    >>> quali_sup.cos2.head() # cos2
    >>> quali_sup.vtest.head() # value-test
    >>> quali_sup.dist.head() # squared distance  to origin
    >>> quali_sup.eta2.head() # eta2
    ```
    """
    # Check if self is and object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary columns
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_col_sup is not None:
        #if pandas Series, convert to pandas DataFrame
        if isinstance(X_col_sup,Series):
            X_col_sup = X_col_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_col_sup,DataFrame):
            raise TypeError(f"{type(X_col_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #check if X_col_sup.shape[0] = nrows
        if X_col_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")

        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_col_sup[j]) for j in X_col_sup.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #weighted with row weight
        X_col_sup = mapply(X_col_sup,lambda x : x*self.call_.row_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers)
        #standardize the data
        Z_col_sup = mapply(X_col_sup,lambda x : x/X_col_sup.sum(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #supplementary columns square distance to origin
        col_sup_sqdisto = mapply(Z_col_sup,lambda x : ((x - self.call_.row_marge)**2)/self.call_.row_marge,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
        col_sup_sqdisto.name = "Sq. Dist."
        #statistics for supplementary columns
        col_sup_ = predict_ind_sup(Z=Z_col_sup.T,V=self.svd_.U,sqdisto=col_sup_sqdisto,col_weights=None,n_workers=self.call_.n_workers)
        #convert to namedtuple
        col_sup = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())
    else:
        col_sup = None

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        #if pandas Series, convert to pandas dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if X_quanti_sup.shape[0] = nrows
        if X_quanti_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")

        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_quanti_sup[k]) for k in X_quanti_sup.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #fill missing with mean
        X_quanti_sup = recodecont(X=X_quanti_sup).X
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(X=X_quanti_sup,row_coord=self.row_.coord,row_weights=self.call_.row_marge,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary qualitative variables
    ##---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        #if pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        #check if X is an instance of pandas DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if X_quali_sup.shape[0] = nrows
        if X_quali_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")

        #check if all columns are categoricals
        all_cat = all(api.types.is_string_dtype(X_quali_sup[q]) for q in X_quali_sup.columns)
        if not all_cat:
            raise TypeError("All columns in `X_quali_sup` must be categoricals")
        
        #convert to factor
        for q in X_quali_sup.columns:
            X_quali_sup[q] = Categorical(X_quali_sup[q],categories=sorted(X_quali_sup[q].dropna().unique().tolist()),ordered=True)

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        #sum of columns by group
        quali = sum_col_by(X=self.call_.X,X_quali=X_quali_sup)
        #standardize the data
        Z_quali_sup = mapply(quali,lambda x : x/quali.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers)
        #supplementary categories factor coordinates
        quali_sup_coord = Z_quali_sup.dot(self.svd_.V)
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        #supplementary categories square distance to origin 
        quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : ((x - self.call_.col_marge)**2)/self.call_.col_marge,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
        quali_sup_sqdisto.name="Sq. Dist."
        #disjunctive table
        dummies = concat((get_dummies(X_quali_sup[q],dtype=int) for q in X_quali_sup.columns),axis=1)
        #compute : weighted count by categories
        n_k = mapply(dummies,lambda x : x*self.call_.row_marge,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)*self.call_.total
        #supplementary categories value-test 
        if self.call_.total > 1:
            coef_k = sqrt(((self.call_.total - 1)*n_k)/(self.call_.total - n_k))
        else:
            coef_k = sqrt(n_k)
        #statistics for supplementary categories
        quali_sup_ = predict_quali_sup(X=X_quali_sup,row_coord=self.row_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.row_marge,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    # Store all informations
    return namedtuple("supvarCAResult",["col","quanti","quali"])(col_sup,quanti_sup,quali_sup)