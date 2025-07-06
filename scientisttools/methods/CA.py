# -*- coding: utf-8 -*-
from numpy import ones,ndarray,sqrt, mean
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
from .functions.sum_table import sum_table
from .functions.predict_sup import predict_ind_sup, predict_quali_sup


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
    `call_`: namedtuple with some informations:
        * `Xtot`: pandas dataframe with all data (active and supplementary)
        * `X`: pandas dataframe with active data
        * `Z`: pandas dataframe with standardized data
        * `row_weights`: pandas series containing rows weights
        * `row_marge`: pandas series containing rows marging
        * `col_marge`: pandas series containing columns marging
        * `n_components`: an integer indicating the number of components kept
        * `n_workers`: an integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `row_sup`: None or a list of string indicating names of the supplementary rows
        * `col_sup`: None or a list of string indicating names of the supplementary columns
        * `quanti_sup`: None or a list of string indicating names of the supplementary quantitative variables
        * `quali_sup`: None or a list of string indicating names of the supplementary qualitative variables

    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `svd_` : namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD)

    `col_` : namedtuple of pandas dataframe with all the results for the column variable (coordinates, square cosine, contributions, inertia)

    `row_` : dictionary of pandas dataframe with all the results for the row variable (coordinates, square cosine, contributions, inertia)

    `col_sup_` : dictionary of pandas dataframe containing all the results for the supplementary column points (coordinates, square cosine)

    `row_sup_` : dictionary of pandas dataframe containing all the results for the supplementary row points (coordinates, square cosine)

    `quanti_sup_` : if quanti_sup is not None, a dictionary of pandas dataframe containing the results for the supplementary continuous variables (coordinates, square cosine)

    `quali_sup_` : if quali.sup is not None, a dictionary of pandas dataframe with all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a criterion with a Normal distribution, square correlation ratio)

    `model_` : string specifying the model fitted = 'ca'

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
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2 = load_children2()
    >>> from scientisttools import CA, summaryCA
    >>> #with supplementary rows, supplementary columns and supplementary qualitative variables
    >>> res_ca = CA(row_sup=[14,15,16,17],col_sup=[5,6,7],quali_sup=8,parallelize=False)
    >>> res_ca.fit(children2)
    >>> #with supplementary rows, supplementary quantitative variables and supplementary qualitative variables
    >>> res_ca2 = CA(row_sup=[14,15,16,17],quanti_sup=[5,6,7],quali_sup=8,parallelize=False)
    >>> res_ca2.fit(children2)
    >>> #summary
    >>> summaryCA(res_ca)
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
        `X`: pandas dataframe of shape (n_rows, n_columns),
            Training data, where `n_rows` in the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y`: None,
            y is ignored.
        
        Returns
        -------
        `self`: object
            Returns the instance itself.
        """

        # check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Set index name as None
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Drop level if ndim greater than 1 and reset columns names
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Checks if categoricals variables is in X and transform to factor (category)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for q in is_quali.columns:
                X[q] = Categorical(X[q],categories=sorted(X[q].dropna().unique().tolist()),ordered=True)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary qualitative variables
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
        ## Check if supplementary quantitative variables
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

        # Drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)

        # Drop supplementary quantitative variables
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        # Drop supplementary columns
        if self.col_sup is not None:
            X = X.drop(columns=col_sup_label)
        
        # Drop supplementary rows
        if self.row_sup is not None:
            X_row_sup = X.loc[row_sup_label,:]
            X = X.drop(index=row_sup_label)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Correspondence Analysis (CA)
        #----------------------------------------------------------------------------------------------------------------------------------------
        # convert to integer
        X = X.astype(int)

        # Number of rows/columns
        n_rows, n_cols = X.shape

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Set rows weights
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

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Standardize the data
        #----------------------------------------------------------------------------------------------------------------------------------------
        #total
        total = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0).sum()

        #frequencie table
        freq = mapply(X,lambda x : x*(row_weights/total),axis=0,progressbar=False,n_workers=n_workers)
        
        #Calcul des marges lignes et colones
        col_marge, row_marge = freq.sum(axis=0), freq.sum(axis=1)
        col_marge.name, row_marge.name = "col_marge", "row_marge"

        #Compute Matrix used in SVD
        Z = mapply(mapply(freq,lambda x : x/row_marge,axis=0,progressbar=False,n_workers=n_workers),lambda x : (x/col_marge)-1,axis=1,progressbar=False,n_workers=n_workers)

        # Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,total=total,row_weights=row_weights,row_marge=row_marge,col_marge=col_marge,n_components=n_components,n_workers=n_workers,
                            row_sup=row_sup_label,col_sup=col_sup_label,quanti_sup=quanti_sup_label,quali_sup=quali_sup_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## fit factor analysis model and extract all elements
        #----------------------------------------------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z=Z,row_weights=row_marge,col_weights=col_marge,max_components=max_components,n_components=n_components,n_workers=n_workers)

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
        statistic, pvalue, dof, expected_freq = chi2_contingency(X, lambda_=None,correction=False)
        chi2_qt = chi2.ppf(0.95,dof)

        # Return indicators
        chi2_test = DataFrame([[statistic,dof,chi2_qt,pvalue]],columns=["statistic","dof","quantile","pvalue"],index=["chi2-test"])

        # log - likelihood - tes (G - test)
        g_stat, g_pvalue = chi2_contingency(X, lambda_="log-likelihood")[:2]
        g_test = DataFrame([[g_stat,dof,chi2_qt,g_pvalue]],columns=["statistic","dof","quantile","pvalue"],index=["g-test"])

        # Absolute residuals
        resid = X.sub(expected_freq)

        # Standardized resid
        std_resid = resid.div(sqrt(expected_freq))

        # Adjusted residuals
        adj_resid = mapply(mapply(std_resid,lambda x : x/sqrt(1-row_marge),axis=0,progressbar=False,n_workers=n_workers),lambda x : x/sqrt(1-col_marge),axis=1,progressbar=False,n_workers=n_workers)

        # Chi2 contribution
        chi2_contrib = mapply(std_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False,n_workers=n_workers)

        # Attraction repulsion index
        att_rep_ind = X.div(expected_freq)
        
        # Association test
        association = DataFrame([[contingency.association(X,method=i,correction=False) for i in ["cramer","tschuprow","pearson"]]],index=["statistic"],columns=["cramer","tschuprow","pearson"])
    
        # Kaiser threshold
        kaiser = namedtuple("kaiser",["threshold","proportion"])(mean(self.eig_.iloc[:,0]),100/max_components)
       
       # Store others informations
        others_ = OrderedDict(resid=resid,chi2_test=chi2_test,g_test=g_test,adj_resid=adj_resid,chi2_contrib=chi2_contrib,std_resid=std_resid,
                              att_rep_ind=att_rep_ind,association=association,kaiser=kaiser)
        
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            #convert to integer
            X_row_sup = X_row_sup.astype(int)
            #standardize with the row sum
            Z_row_sup = mapply(X_row_sup,lambda x : x/X_row_sup.sum(axis=1),axis=0,progressbar=False,n_workers=n_workers)
            #supplementary rows square distance to origin
            row_sup_sqdisto = mapply(Z_row_sup,lambda x : ((x - self.call_.col_marge)**2)/self.call_.col_marge,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            row_sup_sqdisto.name = "Sq. Dist."
            #statistics for supplementary rows
            row_sup_ = predict_ind_sup(Z=Z_row_sup,V=self.svd_.V,sqdisto=row_sup_sqdisto,col_weights=None,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",row_sup_.keys())(*row_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.col_sup is not None:
            X_col_sup = Xtot.loc[:,col_sup_label]
            if self.row_sup is not None:
                X_col_sup = X_col_sup.drop(index=row_sup_label)
            
            #convert to integer
            X_col_sup = X_col_sup.astype(int)
            #weighted with row weight
            X_col_sup = mapply(X_col_sup,lambda x : x*self.call_.row_weights,axis=0,progressbar=False,n_workers=n_workers)
            #standardize supplementary columns
            Z_col_sup = mapply(X_col_sup,lambda x : x/X_col_sup.sum(axis=0),axis=1,progressbar=False,n_workers=n_workers)
            #supplementary columns square distance to origin
            col_sup_sqdisto = mapply(Z_col_sup,lambda x : ((x - self.call_.row_marge)**2)/self.call_.row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            col_sup_sqdisto.name = "Sq. Dist."
            #statistics for supplementary columns
            col_sup_ = predict_ind_sup(Z=Z_col_sup.T,V=self.svd_.U,sqdisto=col_sup_sqdisto,col_weights=None,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.col_sup_ = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())
         
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.row_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=row_sup_label)
            
            #convert to float and fill missing with mean
            X_quanti_sup = recodecont(X_quanti_sup.astype("float")).X
            #supplementary quantitative variables factor coordinates - factor correlation
            wcorr = DescrStatsW(concat((X_quanti_sup,self.row_.coord),axis=1),weights=self.call_.row_marge,ddof=0).corrcoef[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
            quanti_sup_coord = DataFrame(wcorr,index=quanti_sup_label,columns=["Dim."+str(x+1) for x in range(n_components)])
            #supplementary quantitative variable ssquare cosinus
            quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers)
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",["coord","cos2"])(quanti_sup_coord,quanti_sup_cos2)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary categorical variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.row_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=row_sup_label)
            
            #check if two columns have the same categories
            X_quali_sup = revaluate_cat_variable(X_quali_sup)
            #create contingency table with categories as rows 
            quali = concat((map(lambda q : sum_table(X=self.call_.X,X_quali=X_quali_sup,q=q),X_quali_sup.columns)),axis=0)
            #standardize the data
            Z_quali_sup = mapply(quali,lambda x : x/quali.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers)
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
            quali_sup_ = predict_quali_sup(X=X_quali_sup,Z=Z_quali_sup,Y=self.row_.coord,V=self.svd_.V,col_coef=coef_k,sqdisto=quali_sup_sqdisto,
                                           row_weights=self.call_.row_marge,col_weights=None,n_workers=self.call_.n_workers)
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
        `X`: pandas dataframe of shape (n_rows, n_columns)
            Training data, where `n_rows` is the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y`: None.
            y is ignored.

        Returns
        -------
        `X_new`: pandas dataframe of shape (n_rows, n_components)
            Transformed values.
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
        `X`: pandas dataframe of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas dataframe of shape (n_samples, n_columns)
            Original data, where ``n_samples` is the number of samples and `n_columns` is the number of columns
        
        """
        # check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        return NotImplementedError("Not yet implemented")

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: pandas dataframe of shape (n_rows, n_columns)
            New data, where `n_rows` is the number of row points and `n_columns` is the number of columns

        Returns
        -------
        `X_new`: pandas dataframe of shape (n_rows, n_components)
            Projection of X in the principal components where `n_rows` is the number of rows and `n_components` is the number of the components.
        """
        
        # check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # set index name to None
        X.index.name = None

        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[j]) for j in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")

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

    `X`: pandas dataframe in which to look for columns with which to predict. X must contain columns with the same names as the original data

    Return
    ------
    namedtuple of dataframes including :

    `coord`: factor coordinates (scores) for supplementary rows

    `cos2`: square cosinus for supplementary rows

    `dist`: square distance to origin for supplementary rows

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> #Load children2 dataset
    >>> from scientisttools import load_children2
    >>> children = load_children()
    >>> #Actifs elements
    >>> actif = children.iloc[:14,:5]
    >>> #Supplementary rows
    >>> row_sup = children.iloc[14:,:5]
    >>> #Correspondence Analysis (CA)
    >>> from scientisttools import CA
    >>> res_ca = CA(n_components=None,row_sup=[14,15,16,17],col_sup=[5,6,7])
    >>> res_ca.fit(children)
    >>> #Prediction on supplementary rows
    >>> from scientisttools import predictCA
    >>> predict = predictCA(res_ca, X=row_sup)
    ```
    """
    # Check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    # check if X is an instance of pandas DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Set index name as None
    X.index.name = None

    #check if all variables are numerics
    all_num = all(api.types.is_numeric_dtype(X[j]) for j in X.columns)
    if not all_num:
        raise TypeError("All columns must be numeric")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Statistics for new rows
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    Performns the coordinates, square cosinus and square distance to origin of supplementary columns/variables with Correspondence Analysis (CA)

    Usage
    -----
    ```python
    >>> supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None)   
    ```

    Parameters
    ----------
    `self`: an object of class CA

    `X_col_sup`: pandas dataframe of supplementary columns

    `X_quanti_sup`: pandas dataframe of supplementary quantitatives columns

    `X_quali_sup`: pandas dataframe of supplementary qualitatives columns

    Returns
    -------
    namedtuple including : 

    `col`: namedtuple containing the results of the supplementary columns variables:
        * `coord`: factor coordinates (scores) of the supplementary columns
        * `cos2`: square cosinus of the supplementary columns
        * `dist`: distance to origin of the supplementary columns

    `quanti`: namedtuple containing the results of the supplementary quantitatives variables:
        * `coord`: factor coordinates (scores) of the supplementary quantitativaes variables
        * `cos2`: square cosinus of the supplementary quantitatives variables
    
    `quali`: namedtuple containing the results of the supplementary qualitatives/categories variables :
        * `coord`: factor coordinates (scores) of the supplementary categories
        * `cos2`: square cosinus of the supplementary categories
        * `vtest`: value-test of the supplementary categories
        * `dist`: square distance to origin of the supplementary categories
        * `eta2`: square correlation ratio of the supplementary qualitatives variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> #Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> #Add qualitatives variables
    >>> children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +["D"]*4
    >>> #Supplementary columns
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
    ##statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        #if pandas Series, convert to pandas dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_quanti_sup[k]) for k in X_quanti_sup.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #fill missing with mean
        X_quanti_sup = recodecont(X_quanti_sup.astype("float")).X
        #supplementary quantitative variables factor coordinates - factor correlation
        wcorr = DescrStatsW(concat((X_quanti_sup,self.row_.coord),axis=1),weights=self.call_.row_marge,ddof=0).corrcoef[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
        quanti_sup_coord = DataFrame(wcorr,index=X_quanti_sup.columns,columns=["Dim."+str(x+1) for x in range(self.call_.n_components)])
        #supplementary quantitative variable square cosinus
        quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",["coord","cos2"])(quanti_sup_coord,quanti_sup_cos2)
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitative variables
    ##---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        #if pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        #check if X is an instance of pandas DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
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
        quali = concat((map(lambda q : sum_table(X=self.call_.X,X_quali=X_quali_sup,q=q),X_quali_sup.columns)),axis=0)
        #standardize the data
        Z_quali_sup = mapply(quali,lambda x : x/quali.sum(axis=1),axis=0,progressbar=False,n_workers=self.call_.n_workers)
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
        quali_sup_ = predict_quali_sup(X=X_quali_sup,Z=Z_quali_sup,Y=self.row_.coord,V=self.svd_.V,col_coef=coef_k,sqdisto=quali_sup_sqdisto,
                                        row_weights=self.call_.row_marge,col_weights=None,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    # Store all informations
    return namedtuple("supvarCAResult",["col","quanti","quali"])(col_sup,quanti_sup,quali_sup)