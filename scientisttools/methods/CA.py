# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, sqrt, linalg, average, cov
from scipy.stats import chi2, chi2_contingency, contingency
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.gfa import gfa
from .functions.predict_sup import predict_sup
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.summarize import conditional_sum
from .functions.function_eta2 import function_eta2

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    ----------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Correspondence Analysis (CA) including supplementary points (rows and/or columns), supplementary variables (quantitative and/or qualitative).

    Usage
    -----
    ```python
    >>> CA(n_components = 5, row_weights = None, row_sup = None, col_sup = None, sup_var = None)
    ```

    Parameters
    ----------
    `n_components`: a numeric indicating the number of dimensions kept in the results (by default 5)

    `row_weights`: an optional row weights (by default, a list/tuple/ndarray/Series of 1 and each row has a weight equals to its margin); the weights are given only for the active rows

    `row_sup`: None or an integer/string/list/tuple indicating the indexes/names of the supplementary rows points

    `col_sup`: None or an integer/string/list/tuple indicating the indexes/names of the supplementary columns points

    `sup_var`: None or an integer/string/list/tuple indicating the indexes/names of the supplementary variables (quantitative and/or qualitative)

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
        * `row_sup`: None or a list of string indicating names of the supplementary rows
        * `col_sup`: None or a list of string indicating names of the supplementary columns
        * `sup_var`: None or a list of string indicating names of the supplementary variables (quantitative and/or qualitative)

    `svd_`: a namedtuple of numpy array containing all the results for the generalized singular value decomposition (GSVD)
        * `vs`: 1D numpy array containing the singular values,
        * `U`: 2D numpy array whose columns contain the left singular vectors,
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: a pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `col_`: a namedtuple of pandas DataFrames containing all the results for the active columns.
        * `coord`: coordinates for the active columns,
        * `cos2`: squared cosinus for the active columns,
        * `contrib`: relative contributions for the active columns,
        * `infos`: additionals informations (margin, squared distance to origin, inertia and percentage of inertia) for the active columns.

    `row_`: a namedtuple of pandas DataFrames containing all the results for the active rows.
        * `coord`: coordinates for the active rows,
        * `cos2`: squared cosinus for the active rows,
        * `contrib`: relative contributions for the active rows,
        * `infos`: additionals informations (weight, margin, squared distance to origin, inertia and percentage of inertia) for the active rows.

    `goodness_`: a namedtuple of pandas DataFrames for multivariate goodness of fit test.
        * `chi2`: Pearson's chi-squared test
        * `gtest`: log-likelihood ratio (i.e the "G-test")
        * `association`: degree of association between two nominal variables ("cramer", "tschuprow", "pearson")
    
    `residual_` : a namedtuple of pandas DataFrames for residuals.
        * `resid`: model residuals,
        * `rstandard`: standardized residuals,
        * `radjusted`: adjusted residuals,
        * `contrib`: contribution to chi-squared,
        * `att_rep_ind`: attraction repulsion index.

    `others_`: a namedtuple with others correspondence analysis statistics.
        * `kaiser`: a namedtuple of numeric containing the kaiser threshold and proportion
            * `threshold`: a numeric value specifying the kaiser threshold
            * `porportion`: a numeric value specifying the kaiser proportion threshold

    `row_sup_`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary rows.
        * `coord`: coordinates for the supplementary rows,
        * `cos2`: squared cosinus for the supplementary rows,
        * `dist2`: squared distance to origin for the supplementary rows.

    `col_sup_`: a namedtuple of pandas DataFrame/Series containing all the results for the supplementary columns.
        * `coord`: coordinates for the supplementary columns,
        * `cos2`: squared cosinus for the supplementary columns,
        * `dist2`: squared distance to origin for the supplementary columns.

    `quanti_sup_`: a namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables.
        * `coord`: coordinates for the supplementary quantitative variables,
        * `cos2`: squared cosinus for the supplementary quantitative variables.

    `quali_sup_`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary qualitative variables/levels.
        * `coord`: coordinates for supplementary levels,
        * `cos2`: squared cosinus for supplementary levels,
        * `vtest`: value-test for supplementary levels,
        * `dist2`: squared distance to origin for supplementary levels,
        * `eta2`: squared correlation ratio for supplementary qualitative variables.

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
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, summaryCA
    >>> #with supplementary rows, supplementary columns and supplementary qualitative variables
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),sup_var=8)
    >>> res_ca.fit(children)
    >>> summaryCA(res_ca)
    >>> #with supplementary rows, supplementary variables (quantitative and qualitative)
    >>> res_ca2 = CA(row_sup=(14,15,16,17),sup_var=(5,6,7,8))
    >>> res_ca2.fit(children)
    >>> summaryCA(res_ca2)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 row_weights = None,
                 row_sup = None,
                 col_sup = None,
                 sup_var = None):
        self.n_components = n_components
        self.row_weights = row_weights
        self.row_sup = row_sup
        self.col_sup = col_sup
        self.sup_var = sup_var

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns),
            Training data, where `n_samples` in the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y`: None,
            y is ignored.
        
        Returns
        -------
        `self`: object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary rows labels
        row_sup_label = get_sup_label(X=X, indexes=self.row_sup, axis=0)

        #get supplementary columns labels
        col_sup_label = get_sup_label(X=X, indexes=self.col_sup, axis=1)
        
        #check if supplementary variables (quantitative and/or qualitative)
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)
        
        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)

        #drop supplementary columns
        if self.col_sup is not None:
            X_col_sup, X = X.loc[:,col_sup_label],  X.drop(columns=col_sup_label)
        
        #drop supplementary rows
        if self.row_sup is not None:
            X_row_sup, X = X.loc[row_sup_label,:], X.drop(index=row_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correspondence analysis (CA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all columns are numerics
        if not all(is_numeric_dtype(X[j]) for j in X.columns):
            raise TypeError("All columns must be numeric")
        
        #number of rows/columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set rows weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_weights is None:
            row_weights = ones(n_rows)
        elif not isinstance(self.row_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'row_weights' must be a list or a tuple or 1-D array or a pandas Series of rows weights.")
        elif len(self.row_weights) != n_rows:
            raise ValueError(f"'row_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_rows}.")
        else:
            row_weights = array(self.row_weights)
        
        #convert weights to Series
        row_weights =  Series(row_weights,index=X.index,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize the data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total
        total = X.mul(row_weights,axis=0).sum().sum()

        #frequencies table
        freq = X.mul(row_weights,axis=0).div(total,axis=0)
        
        #columns and rows margins calcul
        col_marge, row_marge = freq.sum(axis=0), freq.sum(axis=1)
        col_marge.name, row_marge.name = "Margin", "Margin"

        #standardization: z_ij = (fij/(fi.*f.j)) - 1
        Z = freq.div(row_marge,axis=0).div(col_marge,axis=1).sub(1)
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #----------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1, n_cols - 1))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise TypeError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,total=total,row_weights=row_weights,row_marge=row_marge,col_marge=col_marge,n_components=n_components,
                            row_sup=row_sup_label,col_sup=col_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Z,row_weights=row_marge,col_weights=col_marge,max_components=max_components,n_components=n_components)

        #extract elements
        self.svd_, self.eig_, row, col = fit_.svd, fit_.eig, fit_.row, fit_.col
        row_infos, col_infos = row['infos'].rename(columns={"Weight" : "Margin"}), col['infos'].rename(columns={"Weight" : "Margin"})
        row_infos.insert(0,"Weight",row_weights)
        #update dictionary
        row.update({"infos" : row_infos})
        col.update({"infos" : col_infos})

        #convert to namedtuple
        self.row_, self.col_ = namedtuple("row",row.keys())(*row.values()), namedtuple("col",col.keys())(*col.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #diagnostics tests - multivariate goodness of fit tests
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        association = DataFrame([[contingency.association(X.astype("int"),method=i,correction=False) for i in ["cramer","tschuprow","pearson"]]],index=["statistic"],columns=["cramer","tschuprow","pearson"])
        #convert to ordered dictionary
        goodness_ = OrderedDict(chi2=chi2_test,gtest=g_test,association=association)
        #convert to namedtuple
        self.goodness_ = namedtuple("test",goodness_.keys())(*goodness_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #residuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #absolute residuals
        resid = X.sub(expected_freq)
        #standardized residuals
        std_resid = resid.div(sqrt(expected_freq))
        #adjusted residuals
        adj_resid = std_resid.div(sqrt(1-row_marge),axis=0).div(sqrt(1-col_marge),axis=1)
        #chi2 contribution
        chi2_ctr = std_resid.pow(2).div(statistic,axis=0)
        #attraction repulsion index
        att_rep_ind = X.div(expected_freq)
        #convert to ordered dictionary
        residuals_ = OrderedDict(resid=resid,rstandard=std_resid,radjusted=adj_resid,contrib=chi2_ctr,att_rep_ind=att_rep_ind)
        #convert to namedtuple
        self.residuals_ = namedtuple("residuals",residuals_.keys())(*residuals_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute others indicators 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #kaiser threshold
        kaiser = namedtuple("kaiser",["threshold","proportion"])(self.eig_.iloc[:,0].mean(),100/max_components)
        #convert to namedtuple
        self.others_ = namedtuple("others",["kaiser"])(kaiser)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            #frequencies of supplementary rows
            freq_row_sup = X_row_sup.div(total)
            #margins for supplementary rows
            row_sup_marge = freq_row_sup.sum(axis=1)
            #standardization: z_ij = (fij/(fi.*f.j)) - 1
            Z_row_sup = freq_row_sup.div(row_sup_marge,axis=0).div(col_marge,axis=1).sub(1)
            #statistics for supplementary rows
            row_sup_ = predict_sup(X=Z_row_sup,Y=fit_.svd.V,weights=col_marge,axis=0)
            #convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",row_sup_.keys())(*row_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.col_sup is not None:
            if self.row_sup is not None: #drop supplementary rows
                X_col_sup = X_col_sup.drop(index=row_sup_label)
            #frequencies of supplementary columns
            freq_col_sup = X_col_sup.mul(row_weights,axis=0).div(total)
            #margins for supplementary columns
            col_sup_marge = freq_col_sup.sum(axis=0)
            #standardization: z_ij = (fij/(fi.*f.j)) - 1
            Z_col_sup = freq_col_sup.div(row_marge,axis=0).div(col_sup_marge,axis=1).sub(1)
            #statistics for supplementary columns
            col_sup_ = predict_sup(X=Z_col_sup,Y=fit_.svd.U,weights=row_marge,axis=1)
            #convert to namedtuple
            self.col_sup_ = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            if self.row_sup is not None: #drop supplementary rows
                X_sup_var = X_sup_var.drop(index=row_sup_label)
            
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0:
                #compute weighted average and weighted standard deviation
                center_sup = Series(average(X_quanti_sup,axis=0,weights=row_marge),index=X_quanti_sup.columns,name="center")
                scale_sup = Series([sqrt(cov(X_quanti_sup.iloc[:,k],aweights=row_marge,ddof=0)) for k in range(n_quanti_sup)],index=X_quanti_sup.columns,name="scale")
                #standardization: Z = (X - mu)/sigma
                Z_quanti_sup = X_quanti_sup.mul(row_weights,axis=0).sub(center_sup,axis=1).div(scale_sup,axis=1)
                #statistics for supplementary quantitative variables
                quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=fit_.svd.U,weights=row_marge,axis=1)
                del quanti_sup_['dist2'] #delete dist2
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #recode
                rec = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec.X, rec.dummies
                #create contingency table with categories as rows 
                X_levels_sup = conditional_sum(X=X,Y=X_quali_sup)
                #frequencies of supplementary levels
                freq_levels_sup = X_levels_sup.div(total)
                #margins for supplementary levels
                levels_sup_marge = freq_levels_sup.sum(axis=1)
                #standardization: z_ij = (fij/(fi.*f.j)) - 1
                Z_levels_sup = freq_levels_sup.div(levels_sup_marge,axis=0).div(col_marge,axis=1).sub(1)
                #statistics for supplementary levels
                quali_sup_ = predict_sup(X=Z_levels_sup,Y=fit_.svd.V,weights=col_marge,axis=0)
                #proportion of supplementary levels
                p_k_sup = dummies_sup.mul(row_marge,axis=0).sum(axis=0)
                levels_sup_vtest = quali_sup_["coord"].mul(sqrt((total-1)/(1/p_k_sup).sub(1)),axis=0)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=fit_.row["coord"],weights=row_marge,excl=None)
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
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
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y`: None.
            y is ignored.

        Returns
        -------
        `X_new`: a pandas DataFrame of shape (n_samples, n_components)
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
        `X`: a pandas DataFrame of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas DataFrame of shape (n_samples, n_columns)
            Original data, where `n_samples` is the number of samples and `n_columns` is the number of columns
        """
        #check if X is an instance of pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set number of components
        n_components = min(X.shape[1],self.call_.n_components)
        #extract elements
        F, G = X.iloc[:,:n_components], self.col_.coord.iloc[:,:n_components]
        X_original = F.dot(G.div(sqrt(G.pow(2).T.dot(self.call_.col_marge)),axis=1).T).add(1).mul(self.call_.row_marge,axis=0).mul(self.call_.col_marge,axis=1).mul(self.call_.total)
        return X_original

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of row points and `n_columns` is the number of columns

        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of rows and `n_components` is the number of the components.
        """
        if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        X.index.name = None #set index name to None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] = ncols
            raise ValueError("'columns' aren't aligned")

        if not all(is_numeric_dtype(X[j]) for j in X.columns): #check if all variables are numerics
            raise TypeError("All columns in X must be numerics")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the CA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns

        #frequencies of supplementary rows
        freq = X.div(self.call_.total)
        #margins for supplementary rows
        marge = freq.sum(axis=1)
        #standardization: z_ij = (fij/(fi.*f.j)) - 1
        Z = freq.div(marge,axis=0).div(self.call_.col_marge,axis=1).sub(1)
        #coordinates for the supplementary rows
        coord = Z.mul(self.call_.col_marge,axis=1).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord