# -*- coding: utf-8 -*-

# https://kiwidamien.github.io/making-a-python-package.html
##################################### Chargement des librairies
import functools
from functools import reduce
import numpy as np
import pandas as pd
from mapply.mapply import mapply
import pingouin as pg
import statsmodels.formula.api as smf
from scipy.spatial.distance import pdist,squareform
from scipy import linalg
from scipy.sparse import issparse
import scipy.stats as st
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error
from scientistmetrics import scientistmetrics
from scientisttools.utils import (
    orthonormalize,
    random_orthonormal, 
    weighted_mean,
    solve_weighted, 
    check_array_with_weights,
    global_kmo_index,
    per_item_kmo_index,
    from_dummies)

####################################################################################################
#                       PRINCIPAL COMPONENTS ANALYSIS (PCA)
#####################################################################################################

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)
    ----------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    This is a standard Principal Component Analysis implementation
    bases on the Singular Value Decomposition

    Performs Principal Component Analysis (PCA) with supplementary 
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Missing values are replaced by the column mean.

    Usage
    -----
    PCA(normalize=True,
        n_components=None,
        row_labels=None,
        col_labels=None,
        row_sup_labels =None,
        quanti_sup_labels = None,
        quali_sup_labels = None,
        parallelize=False).fit(X)
    
    where X a data frame with n_rows (individuals) and p columns (numeric variables).

    Parameters
    ----------
    normalize : bool, default = True
        - If True : the data are scaled to unit variance.
        - If False : the data are not scaled to unit variance.
    
    n_components : int, float or None, default = None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
            that the amount of variance that needs to be explained is
            greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
                equal to n_components
            - If n_components is float, select the higher number of 
                components lower than n_components
    
    row_labels : array of strings or None, default = None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
     
    col_labels : array of strings or None, default = None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.
    
    row_sup_labels : array of strings or None, defulat = None
        This array provides the supplementary individuals labels
    
    quanti_sup_labels : arrays of strings or None, default = None
        This array provides the quantitative supplementary variables labels
    
    quali_sup_labels : array of strings or None, default = None
        This array provides the categorical supplementary variables labels
    
    parallelize : bool, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.
    
    row_labels_ : array of strings
        Labels for the rows.
    
    col_labels_ : array of strings
        Labels for the columns.
    
    row_sup_labels_ : array of strings or None
        Labels of supplementary individuals labels
    
    quanti_sup_labels_ : arrays of strings or None
        Labels of quantitative supplementary variables
    
    quali_sup_labels_ : arrays of strings or None

    mod_sup_labels_ : list of strings
                        labels for the categories supplementary

    short_sup_labels_ : list of strings
                        Short labels for the categories supplementary 
    
    eig_ : array of float
        A 4 x n_components_ matrix containing all the eigenvalues
        (1st row), difference (2nd row) the percentage of variance (3rd row) and the
        cumulative percentage of variance (4th row).
    
    eigen_vectors_ : array of float
        Eigen vectors extracted from the Principal Components Analysis.
    
    row_coord_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row coordinates.
    
     row_contrib_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    row_cos2_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_coord_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the column
        coordinates.

    col_contrib_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the column
        contributions.
    
    col_cos2_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the column
        cosines.
    
    col_cor_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the correlations
        between variables (= columns) and axes.
    
    col_ftest_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the fisher test
        between variables (= columns) and axes.
    
    col_corr_ : ndarray of shape (n_columns,n_columns)
        A n_columns x n_components matrix containing the pearson correlations
        between variables (= columns)
    
    col_pcorr_ : ndarray of shape (n_columns,n_columns)
        A n_columns x n_components matrix containing the partial correlations
        between variables (= columns)
    
    means_ : ndarray of shape (n_columns,)
        The mean for each variable (= for each column).
    
    std_ : ndarray of shape (n_columns,)
        The standard deviation for each variable (= for each column).
    
    ss_col_coord_ : ndarray of shape (n_columns,)
        The sum of squared of columns coordinates.
    
    row_weight_ : ndarray of shape (n_rows,)
        weights for the individuals
    
    col_weight_ : ndarray of shape (n_columns,)
        weights for the variables
    
    row_sup_coord_ : ndarray of shape (n_sup_rows,n_components_)
        A n_sup_rows x n_components_ matrix containing the 
        coordinates for the supplementary individuals
    
    row_sup_cos2_ : ndarray of shape (n_sup_rows,n_components_)
        A n_sup_rows x n_components_ matrix containing the 
        Cos2 for the supplementary individuals
    
    row_sup_disto_ : ndarray of shape (n_sup_rows,)
        Distance to origin for the supplementary individuals
    
    col_sup_corr_ : ndarray of shape (n_sup_columns,n_columns)
        A n_sup_columns x n_components matrix containing the pearson correlations
        between actives variables and supplementary quantitatives variables
    
    col_sup_coord_ : ndarray of shape (n_sup_columns,n_components_)
        A n_sup_columns x n_components_ matrix containing the 
        coordinates for the supplementary quantitatives variables
    
    col_sup_cos2_ : ndarray of shape (n_sup_columns,n_components_)
        A n_sup_columns x n_components_ matrix containing the cos2 
        for the supplementary quantitatives variables
    
    col_sup_ftest_ : ndarray of shape (n_sup_columns,n_components_)
        A n_sup_columns x n_components_ matrix containing the fisher test
        between variables (= columns) and axes.
    
    mod_sup_stats_ : pd.DataFrame of shape (n_sup_mod,2)
        Statistical informations about variables/categories
    
    mod_sup_coord_ : ndarray of shape (n_sup_mod,n_components_)
        A n_sup_mod x n_components_ matrix containing the 
        coordinates for the supplementary variables/categories
    
    mod_sup_cos2_ : ndarray of shape (n_sup_mod,n_components_)
        A n_sup_mod x n_components_ matrix containing the 
        cos2 for the supplementary variables/categories
    
    mod_sup_vtest_ : ndarray of shape (n_sup_mod,n_components_)
        A n_sup_mod x n_components_ matrix containing the 
        value-test for the supplementary variables/categories
    
    mod_sup_disto_ : ndarray of shape (n_sup_mod)
       Distance to origin for the supplementary variables/categories
    
    quali_sup_eta2_ : ndarray of shape (n_quali_sup_labels_,n_components_)
        A n_quali_sup_labels_ x n_components_ matrix containing the 
        correlation ratio for the supplementary qualitatives variables.
    
    model_ : string
        The model fitted = 'pca'
    """

    def __init__(self,
                 normalize=True,
                 n_components=None,
                 row_labels=None,
                 col_labels=None,
                 row_sup_labels =None,
                 quanti_sup_labels = None,
                 quali_sup_labels = None,
                 parallelize=False):
        self.normalize = normalize
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """Fit the model to X
        
        Parameters
        ----------
        X : DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored
        
        Returns:
        --------
        self : object
                Returns the instance itself
        """

        # Return data
        

        # Check if sparse matrix
        if issparse(X):
            raise TypeError("PCA does not support sparse input.")
        # Check if X is an instance of pd.DataFrame class
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1
        
        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary numeric or categorical columns
        self.quanti_sup_labels_ = self.quanti_sup_labels
        self.quali_sup_labels_ = self.quali_sup_labels
        if ((self.quali_sup_labels_ is not None) and (self.quanti_sup_labels_ is not None)):
            X_ = _X.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)        
        elif self.quali_sup_labels_ is not None:
            X_= _X.drop(columns = self.quali_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_)
        elif self.quanti_sup_labels_ is not None:
            X_ = _X.drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup  = row_sup.drop(columns = self.quanti_sup_labels_)
        else:
            X_ = _X
        
        # Store Data set
        self.data_ = X

        # Check if NA in DataFrame
        if X_.isnull().values.any():
            X_ = X_.fillna(X_.mean(), inplace=True)

        # Active Dataset
        self.active_data_ = X_
        
        # Supplementary initialization
        self.row_sup_coord_ = None
        self.row_sup_cos2_ = None

        # Additionnal continuous variables
        self.col_sup_coord_ = None
        self.col_sup_cos2_ = None
        self.col_sup_ftest_ = None

        # Additionnal categories
        self.mod_sup_coord_ = None
        self.mod_sup_cos2_ = None
        self.mod_sup_disto_ = None
        self.mod_sup_vtest_ = None

        # Additionnal categorical variables
        self.quali_sup_eta2_ = None

        # Pearson correlation
        self.col_corr_ = np.array(X_.corr(method="pearson"))

        # Partial correlation variables
        self.col_pcorr_ = np.array(X_.pcorr())

        # Compute SVD
        self._computed_svd(X_.values)

        # Compute supplementary quantitatives variables statistics
        if self.quanti_sup_labels_ is not None:
            self._compute_quanti_sup_stats(_X[self.quanti_sup_labels_])
        
        # Compute supllementary qualitatives variables statistics
        if self.quali_sup_labels_ is not None:
            self._compute_quali_sup_stats(X=_X[self.quali_sup_labels_])
        
        # Compute supplementrary rows statistics
        if self.row_sup_labels_ is not None:
            self._compute_row_sup_stats(X=row_sup)

        return self
    
    def _computed_svd(self,X):
        """Compute a Singular Value Decomposition
        
        Then, this function computes :
            n_components_ : number of computer.
            eig_ : eigen values.
            eigen_vectors_ : eigen vectors.
            row_coord_ : row coordinates.
            col_coord_ : columns coordinates.
            _compute_stats : 
            row_labels_ : row labels.
            col_labels_ : columns labels.
            row_infos : row informations (distance, weight, inertia).
            inertia_ : inertia.
            data_ : X
            normalized_data_ : Z
            bartlett_sphericity_test_ : Bartlett sphericity test
            kaiser_threshold_ : Kaiser threshold.
            kaiser_proportion_threshold_ : Kaiser proportional threshold
            kss_threshold_ : Kaiser - S - S threshold.
            broken_stick_threshold_ : Broken stick threshold
        
        Parameters
        ----------
        X : DataFrame of float, shape (n_row,n_columns)
            Training data, where n_rows is the number of rows and 
            n_columns is the number of columns.
            X is a table of numeric values.
        
        Returns
        -------
        None
        """

        # Shape of active data
        self.n_rows_, self.n_cols_ = X.shape

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]
        
        # Set col labels
        self.col_labels_ = self.col_labels
        if ((self.col_labels_ is None) or (len(self.col_labels_) != self.n_cols_)):
            self.col_labels_ = ["col_" + str(k+1) for k in np.arange(0,self.n_cols_)]
        
        # Initializations - scale data
        self.means_ = np.mean(X, axis=0).reshape(1,-1)
        if self.normalize:
            self.std_ = np.std(X,axis=0,ddof=0).reshape(1,-1)
            Z = (X - self.means_)/self.std_
        else:
            Z = X - self.means_
        
        # Row information
        row_disto = np.apply_along_axis(func1d=lambda  x : np.sum(x**2),arr=Z,axis=1)
        row_weight = np.ones(self.n_rows_)/self.n_rows_
        row_inertia = row_disto*row_weight
        row_infos = np.c_[np.sqrt(row_disto),row_weight,row_inertia]

        # total inertia
        inertia = np.sum(row_inertia)
        
        # Singular Value Decomposition
        U, delta, V_T = np.linalg.svd(Z,full_matrices=False)

        # Eigen - values
        eigen_values = delta**2/self.n_rows_
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = len(eigen_values)
        elif (self.n_components_ >= 0) and (self.n_components_ < 1):
            i = 0
            threshold = 100 * self.n_components_
            while cumulative[i] < threshold:
                i = i + 1
            self.n_components_ = i
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values))
                and (isinstance(self.n_components_, int))):
            self.n_components_ = int(np.trunc(self.n_components_))
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values))
                and (isinstance(self.n_components_, float))):
            self.n_components_ = int(np.floor(self.n_components_))
        else:
            self.n_components_ = len(eigen_values)
        
        # Row coordinates
        row_coord = U * delta.reshape(1,-1)

        # Columns coordinates
        col_coord = V_T.T.dot(np.diag(np.sqrt(eigen_values)))
        # Test de significativité de Fisher
        col_ftest = np.apply_along_axis(func1d=lambda x : (1/2)*np.sqrt(self.n_rows_-3)*np.log((1+x)/(1-x)),axis=0,arr=col_coord)
        self.ss_col_coord_ = (np.sum(col_coord ** 2, axis=1)).reshape(-1, 1)

        # Correlation between variables and axes
        col_cor = np.transpose(np.corrcoef(x=row_coord,y=Z,rowvar=False)[:self.n_cols_,self.n_cols_:])

        # Store all informations
        self.eig_ = np.array([eigen_values[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])

        # Bartlett - statistics
        bartlett_stats = -(self.n_rows_-1-(2*self.n_cols_+5)/6)*np.sum(np.log(eigen_values))

        # Broken stick threshold
        broken_stick_threshold = np.flip(np.cumsum(1/np.arange(self.n_cols_,0,-1)))

        # Karlis - Saporta - Spinaki threshold
        kss = 1 + 2*np.sqrt((self.n_rows_-1)/(self.n_rows_-1))

        # Store all informations
        self.eigen_vectors_= V_T.T[:,:self.n_components_]

        # Factor coordinates for rows
        self.row_coord_ = row_coord[:,:self.n_components_]

        # Factor coordinates for columns
        self.col_coord_ = col_coord[:,:self.n_components_]
        self.col_cor_ = col_cor[:,:self.n_components_]
        self.col_ftest_ = col_ftest[:,:self.n_components_]
        self.col_weight_ = np.repeat(a=1,repeats=self.n_cols_)

        self.row_infos_ = row_infos
        self.row_weight_ = np.repeat(a=1/self.n_rows_,repeats=self.n_rows_)
        self.inertia_ = inertia
        self.normalized_data_ = Z
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        # Add eigenvalue threshold informations
        self.bartlett_sphericity_test_ = dict({
            "statistic" : bartlett_stats, 
            "p-value" : 1-st.chi2.cdf(bartlett_stats,df=(self.n_cols_*(self.n_cols_-1)/2)),
            "dof" : self.n_cols_*(self.n_cols_-1)/2
        })
        self.kaiser_threshold_ = np.mean(eigen_values)
        self.kaiser_proportion_threshold_ = 100/inertia
        self.kss_threshold_ = kss
        self.broken_stick_threshold_ = broken_stick_threshold[:self.n_components_]

        # Compute stats : contribution and cos2
        self._compute_stats()

        # store model name
        self.model_ = "pca"
    
    def _compute_stats(self):
        """Computed statistics
            row_contrib_ : row contributions.
            col_contrib_ : columns contributions.
            row_cos2_ : row cosines
            col_cos2_ : columns cosines

        Parameters
        ----------
        X : DataFrame of float, shape (n_row, n_columns)
            Training data, where n_rows is the number of rows and 
            n_columns is the number of columns
            X is a table containing numeric values
        """
        # Row and col contributions
        row_contrib = 100 * ((1/self.n_rows_)*(self.row_coord_**2)*(1/self.eig_[0].T))
        col_contrib = 100 * (self.col_coord_ ** 2) * (1/self.eig_[0].T)

        # Row and col cos2
        row_cos2 = ((self.row_coord_ ** 2)/ (np.linalg.norm(self.normalized_data_, axis=1).reshape(-1, 1) ** 2))
        col_cos2 = (self.col_coord_ ** 2) / self.ss_col_coord_
        self.ss_col_coord_ = None
        
        # Store row and col contrib and cos2 with additional informations
        self.row_contrib_   =   row_contrib[:, :self.n_components_]
        self.col_contrib_   =   col_contrib[:, :self.n_components_]
        self.row_cos2_      =   row_cos2[:, :self.n_components_]
        self.col_cos2_      =   col_cos2[:, :self.n_components_]

    def _compute_quanti_sup_stats(self,X,y=None):
        """Comupute supplementary continuous variables statistics

        Parameters
        ----------
        self    :   An instance of class PCA
        X       :   DataFrame (n_rows,n_columns)
        y : None
            y is ignored

        Return
        ------
        col_sup_corr_   :   Pearson correlation between new and old continuous variables
        col_sup_coord_  :   Supplementary continuous coordinates
        col_sup_cos2_   :   Supplementary continuous cosines
        col_sup_ftest_  :   Supplementary continuous Fisher - test
        """

        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Correlation between New and old continuous variables
        col_sup_corr = np.transpose(np.corrcoef(x=self.active_data_,y=X.values,rowvar=False))[self.n_cols_:,:self.n_cols_]

        # Supplementary quantitatives coordinates
        col_sup_coord = np.transpose(np.corrcoef(x=self.row_coord_,y=X.values,rowvar=False)[:self.n_components_,self.n_components_:])

        # Test de significativité de Fisher
        col_sup_ftest = np.apply_along_axis(func1d=lambda x : (1/2)*np.sqrt(self.n_rows_-3)*np.log((1+x)/(1-x)),axis=0,arr=col_sup_coord)

        # Supplementary quantitatives cos2
        col_sup_cos2 = np.apply_along_axis(func1d=lambda x : x**2,arr = col_sup_coord,axis=0)

        # Store supplementary quantitatives informations
        self.col_sup_corr_      =   col_sup_corr
        self.col_sup_coord_     =   col_sup_coord[:,:self.n_components_]
        self.col_sup_cos2_      =   col_sup_cos2[:,:self.n_components_]
        self.col_sup_ftest_     =   col_sup_ftest[:,:self.n_components_]
        
        # Supplementray continuous labels
        self.col_sup_labels_ = X.columns

        return dict({"corr"     :  pd.DataFrame(col_sup_corr,index=X.columns,columns=self.col_labels_), 
                    "coord"     :  pd.DataFrame(col_sup_coord[:,:self.n_components_],index=X.columns,columns=self.dim_index_),
                     "cos2"     :  pd.DataFrame(col_sup_cos2[:,:self.n_components_],index=X.columns,columns=self.dim_index_),
                     "ftest"    :  pd.DataFrame(col_sup_ftest[:,:self.n_components_],index=X.columns,columns=self.dim_index_)
        })
    
    def _compute_quali_sup_stats(self,X,y=None):
        """Compute statistics supplementary categorical variables

        Parameters
        ----------
        self    :   An instance of class PCA
        X       :   DataFrame (n_rows,n_columns)
        y : None
            y is ignored

        Return
        ------
        mod_sup_coord_  : Supplementary categories coordinates
        mod_sup_cos2_   : Supplementary categories cosines
        mod_sup_disto_  : Supplementary categories distance
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Dummies variables
        dummies = pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols in X.columns),axis=1)
        mod_sup_stats = dummies.agg(func=[np.sum,np.mean]).T
        n_k = dummies.sum(axis=0)
        p_k = dummies.mean(axis=0)
        mod_sup_labels = dummies.columns
        short_sup_labels = list([x.split("_",1)[-1] for x in mod_sup_labels])
        
        # Supplementary categories coordinates
        mod_sup_coord = pd.concat((pd.concat((pd.DataFrame(self.row_coord_,index=self.row_labels_,
                                                   columns=self.dim_index_),dummies[cols]),axis=1)
                                      .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)
        
        # Rapport de corrélation
        quali_sup_eta2 = pd.concat(((mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_).mul(p_k,axis="index")
                                                  .loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:]
                                                  .sum(axis=0).to_frame(name=cols).T.div(self.eig_[0])) for cols in X.columns),axis=0)
        
        # Supplementary categories v-test
        mod_sup_vtest = mapply(mapply(mod_sup_coord,lambda x : x/np.sqrt((self.n_rows_-n_k)/((self.n_rows_-1)*n_k)),
                                        axis=0,progressbar=False,n_workers=self.n_workers_),
                                 lambda x : x/np.sqrt(self.eig_[0]),axis=1,progressbar=False,n_workers=self.n_workers_)
        
        # Moyennes conditionnelles sur la variable Z
        mz_g = pd.concat((pd.concat((pd.DataFrame(self.normalized_data_,index=self.row_labels_,
                                                   columns=self.col_labels_),dummies[cols]),axis=1)
                                    .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)

        # Distance des modalités à  l'origine
        mod_sup_disto = mapply(mz_g,lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary categories cos2
        mod_sup_cos2 = mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_).div(mod_sup_disto,axis="index")

        # Supplementary categories eta2 - correlation
        quali_sup_eta2 = pd.concat((mapply(mod_sup_coord.loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:],
                                           lambda x : x**2,axis=1,progressbar=False,n_workers=self.n_workers_)
                                           .mul(p_k.loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index)],axis="index")
                                           .div(self.eig_[0],axis="columns")
                                           .sum(axis=0).to_frame(name=cols).T for cols in X.columns),axis=0)

        # Supplementary categories informations
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_disto_     =   np.sqrt(np.array(mod_sup_disto))
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)

        self.mod_sup_labels_    =   mod_sup_labels
        self.short_sup_labels_  =   short_sup_labels

        # Supplementary qualitatives variables
        self.quali_sup_eta2_    =   quali_sup_eta2

        return dict({"stats"    :   pd.DataFrame(np.array(mod_sup_stats),columns=["n(k)","p(k)"],index=mod_sup_labels),
                    "coord"     :  pd.DataFrame(np.array(mod_sup_coord),index=mod_sup_labels,columns=self.dim_index_),
                     "cos2"     :   pd.DataFrame(np.array(mod_sup_cos2),index=mod_sup_labels,columns=self.dim_index_), 
                     "dist"     :   pd.DataFrame(np.array(mod_sup_disto),index=mod_sup_labels,columns=["dist"]),
                     "eta2"     :   pd.DataFrame(quali_sup_eta2,index=X.columns,columns=self.dim_index_),
                     "vtest"    :   pd.DataFrame( np.array(mod_sup_vtest),index=mod_sup_labels,columns=self.dim_index_)
                     })
    
    def _compute_row_sup_stats(self,X,y=None):
        """Compute supplementary individuals coordinates

        Parameters
        ----------
        X : DataFrame, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        row_sup_coord_ : DataFrame of float, shape (n_rows_sup, n_components_)
                row_sup_coord_ : coordinates of the projections of the supplementary
                row points on the axes.
        
        row_sup_cos2_ : DataFrame of float, shape (n_rows_sup,n_compoents_)
                row_sup_cos2_ : Cosines of the projection of the supplementary 
                row points
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if self.normalize:
            Z = (X - self.means_)/self.std_
        else:
            Z = X - self.means_

        row_sup_coord = np.array(Z.dot(self.eigen_vectors_))
        row_sup_cos2 = ((row_sup_coord ** 2)/ (np.linalg.norm(Z, axis=1).reshape(-1, 1) ** 2))

        # Distance à l'origine
        row_sup_disto = np.apply_along_axis(func1d=lambda  x : np.sum(x**2),arr=Z,axis=1)

        # Store all informations
        self.row_sup_coord_     =   row_sup_coord[:,:self.n_components_]
        self.row_sup_cos2_      =   row_sup_cos2[:,:self.n_components_]
        self.row_sup_disto_     =   np.sqrt(row_sup_disto)

        return dict({"coord"    :   row_sup_coord[:,:self.n_components_],
                     "cos2"     :   row_sup_cos2[:,:self.n_components_],
                     "dist"     :   np.sqrt(row_sup_disto)})


    def transform(self,X,y=None):
        """Apply the dimensionality reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if self.normalize:
            Z = (X - self.means_)/self.std_
        else:
            Z = X - self.means_
        return np.array(Z.dot(self.eigen_vectors_))[:,:self.n_components_]
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        self.fit(X)

        return self.row_coord_
    


##########################################################################################
#           Partial PRINCIPAL COMPONENTS ANALYSIS (PPCA)
##########################################################################################

class PartialPCA(BaseEstimator,TransformerMixin):
    """
    Partial Principal Components Analysis (PartialPCA)
    --------------------------------------------------
    """
    def __init__(self,
                 n_components=None,
                 normalize=True,
                 row_labels=None,
                 col_labels=None,
                 partial_labels=None,
                 parallelize = False):
        self.n_components = n_components
        self.normalize = normalize
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.partial_labels = partial_labels
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize option
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1


        self.n_rows_, self.n_cols_ = X.shape
        self.data_ = X

        self._compute_stats(X)
        self._compute_svds(X)

        return self


    def _compute_stats(self,X,y=None):
        """
        
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        global_kmo = global_kmo_index(X)
        per_var_kmo = per_item_kmo_index(X)
        corr = X.corr(method="pearson")
        pcorr = X.pcorr()

        self.global_kmo_index_ = global_kmo
        self.partial_kmo_index_ = per_var_kmo
        self.pearson_correlation_ = corr
        self.partial_correlation_ = pcorr
    
    def _compute_svds(self,X,y=None):
        """
        
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

            
        self.partial_labels_ = self.partial_labels
        X = X.drop(columns = self.partial_labels_)
        
        # Extract coefficients and intercept
        coef = pd.DataFrame(np.zeros((len(self.partial_labels_)+1,X.shape[1])),
                            index = ["intercept"]+self.partial_labels_,columns=X.columns)
        rsquared = pd.DataFrame(np.zeros((1,X.shape[1])),index = ["R carré"],columns=X.columns)
        rmse = pd.DataFrame(np.zeros((1,X.shape[1])),index = ["RMSE"],columns=X.columns)
        E = pd.DataFrame(np.zeros((self.n_rows_,X.shape[1])),index=X.index,columns=X.columns) # Résidu de régression

        for lab in X.columns:
            res = smf.ols(formula="{}~{}".format(lab,"+".join(self.partial_labels_)), data=self.data_).fit()
            coef.loc[:,lab] = res.params.values
            rsquared.loc[:,lab] = res.rsquared
            rmse.loc[:,lab] = mean_squared_error(self.data_[lab],res.fittedvalues,squared=False)
            E.loc[:,lab] = res.resid
        
        # Coefficients normalisés
        normalized_data = mapply(self.data_,lambda x : (x - x.mean())/x.std(),axis=0,progressbar=False,n_workers=self.n_workers_)
        normalized_coef = pd.DataFrame(np.zeros((len(self.partial_labels_),X.shape[1])),
                                       index = self.partial_labels_,columns=X.columns)
        for lab in X.columns:
            normalized_coef.loc[:,lab] = smf.ols(formula="{}~{}".format(lab,"+".join(self.partial_labels_)),data=normalized_data).fit().params[1:]

        # Matrice des corrélations partielles vers y
        resid_corr = E.corr(method="pearson")
        
        # Matrice des corrélations brutes
        R = X.corr(method="pearson")

        # ACP sur les résidus
        self.row_labels_ = self.row_labels
        my_pca = PCA(normalize=self.normalize,n_components=self.n_components,row_labels=self.row_labels_,col_labels=E.columns).fit(E)
    
        self.resid_corr_ = resid_corr

        self.n_components_ = my_pca.n_components_

        self.eig_ = my_pca.eig_
        self.eigen_vectors_ = my_pca.eigen_vectors_
        self.inertia_ = my_pca.inertia_
        self.dim_index_ =  my_pca.dim_index_
        
        self.row_coord_ = my_pca.row_coord_
        self.row_contrib_ = my_pca.row_contrib_
        self.row_cos2_ = my_pca.row_cos2_
        self.row_infos_ = my_pca.row_infos_

        self.col_coord_ = my_pca.col_coord_
        self.col_cor_ = my_pca.col_cor_
        self.col_ftest = my_pca.col_ftest_
        self.col_cos2_ = my_pca.col_cos2_
        self.col_contrib_ = my_pca.col_contrib_

        self.bartlett_sphericity_test_ = my_pca.bartlett_sphericity_test_
        self.kaiser_proportion_threshold_ = my_pca.kaiser_proportion_threshold_
        self.kaiser_threshold_ = my_pca.kaiser_threshold_
        self.broken_stick_threshold_ = my_pca.broken_stick_threshold_
        self.kss_threshold_ = my_pca.kss_threshold_
        self.col_labels_ = my_pca.col_labels_

        self.rsquared_ = rsquared
        self.rmse_ = rmse
        self.coef_ = coef
        self.normalized_coef_ = normalized_coef
        self.normalized_data_ = normalized_data
        self.resid_ = E
        self.R_ = R

        self.model_ = "ppca"
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.row_coord_
    
    def transform(self,X,y=None):
        """Apply the Partial Principal Components Analysis reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        raise NotImplementedError("Error : This method is not implemented yet.")

#############################################################################################
#       Weighted PRINCIPAL COMPONENTS ANALYSIS (WPCA)
############################################################################################

class WPCA(BaseEstimator, TransformerMixin):
    """Weighted Principal Component Analysis

    This is a direct implementation of weighted PCA based on the eigenvalue
    decomposition of the weighted covariance matrix following
    Delchambre (2014) [1]_.

    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept
    xi : float (optional)
        Degree of weight enhancement.
    regularization : float (optional)
        Control the strength of ridge regularization used to compute the
        transform.
    copy_data : boolean, optional, default True
        If True, X and weights will be copied; else, they may be overwritten.
    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.
    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.
    See Also
    --------
    - PCA
    - sklearn.decomposition.PCA
    References
    ----------
    .. [1] Delchambre, L. MNRAS 2014 446 (2): 3545-3555 (2014)
           http://arxiv.org/abs/1412.4533
    """
    def __init__(self, n_components=None, xi=0, regularization=None,
                 copy_data=True):
        self.n_components = n_components
        self.xi = xi
        self.regularization = regularization
        self.copy_data = copy_data

    def _center_and_weight(self, X, weights, fit_mean=False):
        """Compute centered and weighted version of X.
        If fit_mean is True, then also save the mean to self.mean_
        """
        X, weights = check_array_with_weights(X, weights, dtype=float,
                                              copy=self.copy_data)

        if fit_mean:
            self.mean_ = weighted_mean(X, weights, axis=0)

        # now let X <- (X - mean) * weights
        X -= self.mean_

        if weights is not None:
            X *= weights
        else:
            weights = np.ones_like(X)

        return X, weights

    def fit(self, X, y=None, weights=None):
        """Compute principal components for X
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # let X <- (X - mean) * weights
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_precentered(X, weights)
        return self

    def _fit_precentered(self, X, weights):
        """fit pre-centered data"""
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # TODO: filter NaN warnings
        covar = np.dot(X.T, X)
        covar /= np.dot(weights.T, weights)
        covar[np.isnan(covar)] = 0

        # enhance weights if desired
        if self.xi != 0:
            Ws = weights.sum(0)
            covar *= np.outer(Ws, Ws) ** self.xi

        eigvals = (X.shape[1] - n_components, X.shape[1] - 1)
        evals, evecs = linalg.eigh(covar, eigvals=eigvals)
        self.components_ = evecs[:, ::-1].T
        self.explained_variance_ = evals[::-1]
        self.explained_variance_ratio_ = evals[::-1] / covar.trace()

    def transform(self, X, weights=None):
        """Apply dimensionality reduction on X.
        X is projected on the first principal components previous extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=False)
        return self._transform_precentered(X, weights)

    def _transform_precentered(self, X, weights):
        """transform pre-centered data"""
        # TODO: parallelize this?
        Y = np.zeros((X.shape[0], self.components_.shape[0]))
        for i in range(X.shape[0]):
            cW = self.components_ * weights[i]
            cWX = np.dot(cW, X[i])
            cWc = np.dot(cW, cW.T)
            if self.regularization is not None:
                cWc += np.diag(self.regularization / self.explained_variance_)
            Y[i] = np.linalg.solve(cWc, cWX)
        return Y

    def fit_transform(self, X, y=None, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_precentered(X, weights)
        return self._transform_precentered(X, weights)

    def inverse_transform(self, X):
        """Transform data back to its original space.
        Returns an array X_original whose transform would be X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.
        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        X = check_array(X)
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X, weights=None):
        """Reconstruct the data using the PCA model
        This is equivalent to calling transform followed by inverse_transform.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.transform(X, weights=weights))

    def fit_reconstruct(self, X, weights=None):
        """Fit the model and reconstruct the data using the PCA model
        This is equivalent to calling fit_transform()
        followed by inverse_transform().
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights=weights))

#####################################################################################
#           Expected - Maximization PRINCIPAL COMPONENTS ANALYSIS ( EMPCA)
#######################################################################################

class EMPCA(BaseEstimator, TransformerMixin):
    """Expectation-Maximization PCA

    This is an iterative implementation of weighted PCA based on an
    Expectation-Maximization approach, following Bailey (2012) [1]_.
    
    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept
    max_iter : int (default=100)
        Maximum number of Expectation-Maximization iterations
    random_state : int or None
        Seed for the random initialization of eigenvectors
    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.
    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.
    See Also
    --------
    - PCA
    - WPCA
    - sklearn.decomposition.PCA
    References
    ----------
    .. [1] Bailey, S. PASP 124:919 (2012)
           http://arxiv.org/abs/1208.4122
    """
    def __init__(self, n_components=None, max_iter=100, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def _Estep(self, data, weights, eigvec):
        """E-step: solve for coeff given eigvec"""
        if weights is None:
            return np.dot(data, eigvec.T)
        else:
            return np.array([solve_weighted(eigvec.T, data[i], weights[i])
                             for i in range(data.shape[0])])

    def _Mstep(self, data, weights, eigvec, coeff):
        """M-step: solve for eigvec given coeff"""
        w2 = 1 if weights is None else weights ** 2

        for i in range(eigvec.shape[0]):
            # remove contribution of previous eigenvectors from data
            d = data - np.dot(coeff[:, :i], eigvec[:i])
            c = coeff[:, i:i + 1]
            eigvec[i] = np.dot(c.T, w2 * d) / np.dot(c.T, w2 * c)
            # orthonormalize computed vectors: in theory not necessary,
            # but numerically it's a good idea
            # TODO: perhaps do this more efficiently?
            eigvec[:i + 1] = orthonormalize(eigvec[:i + 1])
        return eigvec

    def fit_transform(self, X, y=None, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = check_array_with_weights(X, weights)

        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        self.mean_ = weighted_mean(X, weights, axis=0)
        X_c = X - self.mean_

        eigvec = random_orthonormal(n_components, X.shape[1],
                                    random_state=self.random_state)

        # TODO: add a convergence check
        for k in range(self.max_iter):
            coeff = self._Estep(X_c, weights, eigvec)
            eigvec = self._Mstep(X_c, weights, eigvec, coeff)
        coeff = self._Estep(X_c, weights, eigvec)

        self.components_ = eigvec
        self.explained_variance_ = (coeff ** 2).sum(0) / X.shape[0]

        if weights is None:
            total_var = X_c.var(0).sum()
        else:
            XW = X_c * weights
            total_var = np.sum((XW ** 2).sum(0) / (weights ** 2).sum(0))
        self.explained_variance_ratio_ = (self.explained_variance_ / total_var)
        return coeff

    def fit(self, X, y=None, weights=None):
        """Compute principal components for X
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X, weights=weights)
        return self

    def transform(self, X, weights=None):
        """Apply dimensionality reduction on X.
        X is projected on the first principal components previous extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = check_array_with_weights(X, weights)

        X_c = X - self.mean_
        if weights is not None:
            assert X.shape == weights.shape
            X_c[weights == 0] = 0
        return self._Estep(X_c, weights, self.components_)

    def inverse_transform(self, X):
        """Transform data back to its original space.
        Returns an array X_original whose transform would be X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.
        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        X = check_array(X)
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X, weights=None):
        """Reconstruct the data using the PCA model
        This is equivalent to calling transform followed by inverse_transform.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.transform(X, weights=weights))

    def fit_reconstruct(self, X, weights=None):
        """Fit the model and reconstruct the data using the PCA model
        This is equivalent to calling fit_transform()
        followed by inverse_transform().
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.
        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.
        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights=weights))
    
##############################################################################################
#       EXPLORATORY FACTOR ANALYSIS (EFA)
###############################################################################################

class EFA(BaseEstimator,TransformerMixin):
    """Exploratory Factor Analysis

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    EFA performs a Exploratory Factor Analysis, given a table of
    numeric variables; shape = n_rows x n_columns

    Parameters
    ----------
    normalize : bool
        - If true : the data are scaled to unit variance
        - If False : the data are not scaled to unit variance
    
    n_components: int or None
        number of components to keep
    
    row_labels : list of string or None
        The list provides the row labels
    
    col_labels : list of strings or None
        The list provides the columns labels
    
    method : {"principal","harris"}
        - If method = "principal" : performs Exploratory Factor Analyis using principal approach
        - If method = "harris" : performs Exploratory Factor Analysis using Harris approach
    
    row_sup_labels : list of strings or None
        The list provides the supplementary row labels
    
    quanti_sup_labels : list of strings or None
        The list provides the supplementary continuous columns
    
    quali_sup_labels : list of strings or None
        The list provides the supplementary categorical variables
    
    graph : bool or None
        - If True : return graph
    
    figsize = tuple of int or None

    Returns:
    --------
    
    """
    def __init__(self,
                normalize=True,
                n_components = None,
                row_labels = None,
                col_labels = None,
                method = "principal",
                row_sup_labels = None,
                quanti_sup_labels = None,
                quali_sup_labels = None):
        self.normalize = normalize
        self.n_components =n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.method = method
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels

    def fit(self,X,y=None):
        """Fit the model to X
        
        Parameters
        ----------
        X : DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored
        
        Returns:
        --------
        self : object
                Returns the instance itself
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary numeric or categorical columns
        self.quanti_sup_labels_ = self.quanti_sup_labels
        self.quali_sup_labels_ = self.quali_sup_labels
        if ((self.quali_sup_labels_ is not None) and (self.quanti_sup_labels_ is not None)):
            X_ = _X.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)        
        elif self.quali_sup_labels_ is not None:
            X_= _X.drop(columns = self.quali_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_)
        elif self.quanti_sup_labels_ is not None:
            X_ = _X.drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup  = row_sup.drop(columns = self.quanti_sup_labels_)
        else:
            X_ = _X
        
        # Save dataframe
        self.data_ = X
        self.active_data_ = X_
        
        # Dimension
        self.n_rows_, self.n_cols_ = X_.shape

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]
        
        # Set col labels
        self.col_labels_ = self.col_labels
        if ((self.col_labels_ is None) or (len(self.col_labels_) != self.n_cols_)):
            self.col_labels_ = ["col_" + str(k+1) for k in np.arange(0,self.n_cols_)]

        # Initialisation
        self.uniqueness_    = None
        self.row_sup_coord_ = None
        self.col_sup_coord_ = None

        #
        self.estimated_communality_ = None
        self.col_coord_             = None
        self.col_contrib_           = None
        self.explained_variance_    = None
        self.percentage_variance_   = None
        self.factor_score_          = None
        self.factor_fidelity_       = None
        self.row_coord_             = None

        # Correlation Matrix
        self.correlation_matrix_ = X_.corr(method= "pearson")

        # Rsquared
        self.initial_communality_ =  np.array([1 - (1/x) for x in np.diag(np.linalg.inv(self.correlation_matrix_))])
        # Total inertia
        self.inertia_ = np.sum(self.initial_communality_)

        # Scale - data
        self.means_ = np.mean(X_.values, axis=0).reshape(1,-1)
        if self.normalize:
            self.std_ = np.std(X_.values,axis=0,ddof=0).reshape(1,-1)
            Z = (X_ - self.means_)/self.std_
        else:
            Z = X_ - self.means_
        
        self.normalized_data_ = Z
        
        if self.method == "principal":
            self._compute_principal(X_)
        elif self.method == "harris":
            self._compute_harris(X_)
        
        # Compute supplementrary rows statistics
        if self.row_sup_labels_ is not None:
            self._compute_row_sup_stats(X=row_sup)
        
        self.model_ = "efa"
        
        return self
    
    def _compute_eig(self,X):
        """Compute eigen decomposition
        
        """

        # Eigen decomposition
        eigenvalue, eigenvector = np.linalg.eigh(X)

        # Sort eigenvalue
        eigen_values = np.flip(eigenvalue)
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = (eigenvalue > 0).sum()

        self.eig_ = np.array([eigen_values[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])

        self.eigen_vectors_ = eigenvector
        return eigenvalue, eigenvector

    def _compute_principal(self,X):
        """Compute EFA using principal approach
        
        
        """
        # Compute Pearson correlation matrix 
        corr_prim = X.corr(method="pearson")

        # Fill diagonal with nitial communality
        np.fill_diagonal(corr_prim.values,self.initial_communality_)
        
        # eigen decomposition
        eigen_value,eigen_vector = self._compute_eig(corr_prim)
        eigen_value = np.flip(eigen_value)
        eigen_vector = np.fliplr(eigen_vector)

        # Compute columns coordinates
        col_coord = eigen_vector*np.sqrt(eigen_value)
        self.col_coord_ = col_coord[:,:self.n_components_]
        
        # Variance restituées
        explained_variance = np.sum(np.square(self.col_coord_),axis=0)

        # Communalité estimée
        estimated_communality = np.sum(np.square(self.col_coord_),axis=1)

        # Pourcentage expliquée par variables
        percentage_variance = estimated_communality/self.initial_communality_

        # F - scores
        factor_score = np.dot(np.linalg.inv(X.corr(method="pearson")),self.col_coord_)

        # Contribution des variances
        col_contrib = np.square(factor_score)/np.sum(np.square(factor_score),axis=0)

        # Fidélité des facteurs
        factor_fidelity = np.sum(factor_score*self.col_coord_,axis=0)
        
        # Row coordinates
        row_coord = np.dot(self.normalized_data_,factor_score)

        # Broken stick threshold
        broken_stick_threshold = np.flip(np.cumsum(1/np.arange(self.n_cols_,0,-1)))

        # Karlis - Saporta - Spinaki threshold
        kss = 1 + 2*np.sqrt((self.n_rows_-1)/(self.n_rows_-1))
        
        # Store all result
        self.estimated_communality_ = estimated_communality
       
        self.col_contrib_ = col_contrib[:,:self.n_components_]
        self.explained_variance_ = explained_variance
        self.percentage_variance_ = percentage_variance
        self.factor_score_ = factor_score
        self.factor_fidelity_ = factor_fidelity
        self.row_coord_ = row_coord[:,:self.n_components_]
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        # Add eigenvalue threshold informations
        self.kaiser_threshold_ = 1.0
        self.kaiser_proportion_threshold_ = 100/self.inertia_
        self.kss_threshold_ = kss
        self.broken_stick_threshold_ = broken_stick_threshold[:self.n_components_]

    
    def _compute_harris(self,X):
        """Compute EFA using harris method
        
        """

        self.uniqueness_ = 1 - self.initial_communality_

        # Save 
        corr_prim = X.corr(method="pearson")
        np.fill_diagonal(corr_prim.values,self.initial_communality_)

        #  New correlation matrix
        corr_snd = np.zeros((self.n_cols_,self.n_cols_))
        for k in np.arange(0,self.n_cols_,1):
            for l in np.arange(0,self.n_cols_,1):
                corr_snd[k,l] = corr_prim.iloc[k,l]/np.sqrt(self.uniqueness_[k]*self.uniqueness_[l])
        
        eigen_value,eigen_vector = self._compute_eig(corr_snd)
        
    def _compute_row_sup_stats(self,X,y=None):
        """Compute statistics supplementary row

        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.method == "principal":
            if self.normalize:
                Z = (X - self.means_)/self.std_
            else:
                Z = X - self.means_

            self.row_sup_coord_ = np.dot(Z,self.factor_score_)[:,:self.n_components_]
        else:
            raise NotImplementedError("Error : This method is not implemented yet.")
    
    def _compute_quanti_sup_stats(self,X,y=None):
        """Compute quantitative supplementary variables
        
        """
        raise NotImplementedError("Error : This method is not implemented yet.")
    
    def _compute_quali_sup_stats(self,X,y=None):
        """Compute qualitative supplementary variables
        
        """
        raise NotImplementedError("Error : This method is not implemented yet.")
    
    def transform(self,X,y=None):
        """Apply the dimensionality reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.method == "principal":
            if self.normalize:
                Z = (X - self.means_)/self.std_
            else:
                Z = X - self.means_
            return np.dot(Z,self.factor_score_)[:,:self.n_components_]
        else:
            raise NotImplementedError("Error : This method is not implemented yet.")
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        self.fit(X)
        return self.row_coord_

################################################################################################
#                                   CORRESPONDENCE ANALYSIS (CA)
################################################################################################

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    ----------------------------
    
    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class
    
    CA performs a Correspondence Analysis, given a contingency table
    containing absolute frequencies ; shape= n_rows x n_columns.
    This implementation only works for dense dataframe.

    It Performs Correspondence Analysis (CA) including supplementary row and/or column points.

    Usage
    -----
    CA(n_components=None,
       row_labels=None,
       col_labels=None,
       row_sup_labels=None,
       col_sup_labels=None,
       parallelize = False).fit(X)

    where X a data frame or a table with n rows and p columns, i.e. a contingency table.

    Parameters
    ----------
    n_components : int, float or None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
          that the amount of variance that needs to be explained is
          greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
              equal to n_components.
            - If n_components is float, select the higher number of
              components lower than n_components.
        
    row_labels : list of strings or None
        - If row_labels is a list of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
    
    col_labels : list of strings or None
        - If col_labels is a list of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.

    row_sup_labels : list of strings or None
        - If row_sup_labels is a list of strings : this array provides the
          supplementary row labels.
    
    col_sup_labels :  list of strings or None
        - If col_sup_labels is a list of strings : this array provides the
          supplementary columns labels.
    
    Attributes
    ----------
    n_components_ : int
        The estimated number of components.
    
    row_labels_ : array of strings
        Labels for the rows.
    
    col_labels_ : array of strings
        Labels for the columns.
    
    eig_ : array of float
        A 4 x n_components_ matrix containing all the eigenvalues
        (1st row), difference (2nd row), the percentage of variance (3th row) and the
        cumulative percentage of variance (4th row).
    
    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.
    
    col_coord_ : array of float
        A n_columns x n_components_ matrix containing the column
        coordinates.
        
    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    col_contrib_ : array of float
        A n_columns x n_components_ matrix containing the column
        contributions.
    
    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_cos2_ : array of float
        A n_columns x n_components_ matrix containing the column
        cosines.

    total_ : float
        The sum of the absolute frequencies in the X array.
    
    model_ : string
        The model fitted = 'ca'
    """
    
    def __init__(self,
                 n_components=None,
                 row_labels=None,
                 col_labels=None,
                 row_sup_labels=None,
                 col_sup_labels=None,
                 parallelize = False):
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.row_sup_labels = row_sup_labels
        self.col_sup_labels = col_sup_labels
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """ Fit the model to X
        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.
        
        y : None
            y is ignored.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary columns
        self.col_sup_labels_ = self.col_sup_labels 
        if self.col_sup_labels is not None:
            X_= _X.drop(columns = self.col_sup_labels_)
            col_sup = _X[self.col_sup_labels_]
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.col_sup_labels_)
        else:
            X_ = _X
        
        # Save data
        self.data_ = X
        self.active_data_ = X_
        
        # Supplementary initialization
        self.row_sup_coord_ = None
        self.row_sup_cos2_ = None

        self.col_sup_coord_ = None
        self.col_sup_cos2_ = None

        # Save shape
        self.n_rows_, self.n_cols_ = X_.shape
        self.total_ = X_.sum().sum()

        # Computes Singular Values Decomposition
        self._compute_svd(X=X_)
        
        # Computes Dependance indicators
        self._compute_indicators(X_)

        if self.row_sup_labels is not None:
            self._compute_sup(X=row_sup,row=True)
        
        if self.col_sup_labels is not None:
            self._compute_sup(X=col_sup,row=False)

        return self
    
    def _compute_stats(self,rowprob,colprob,rowdisto,coldisto):

        row_contrib = np.apply_along_axis(func1d=lambda x : x/self.eig_[0], axis=1,
                        arr=np.apply_along_axis(func1d=lambda x: 100*x**2*rowprob,axis=0,arr=self.row_coord_))
        col_contrib = np.apply_along_axis(func1d=lambda x : x/self.eig_[0], axis=1,
                        arr=np.apply_along_axis(func1d=lambda x: 100*x**2*colprob,axis=0,arr=self.col_coord_))

        # 
        row_cos2 = np.apply_along_axis(func1d=lambda x: x**2/rowdisto, axis = 0, arr=self.row_coord_)
        col_cos2 = np.apply_along_axis(func1d=lambda x: x**2/coldisto, axis = 0, arr=self.col_coord_)
    
        self.row_contrib_ = row_contrib[:,:self.n_components_]
        self.col_contrib_ = col_contrib[:,:self.n_components_]
        self.row_cos2_ = row_cos2[:,:self.n_components_]
        self.col_cos2_ = col_cos2[:,:self.n_components_]

    def _compute_indicators(self,X):
        """
        """
        # 
        prob_conj = mapply(X,lambda x : x/self.total_,axis=0,progressbar=False,n_workers=self.n_workers_)

        # probabilité marginale de V1 - marge colonne
        row_prob = prob_conj.sum(axis = 1)

        # Marge ligne (probabilité marginale)
        col_prob = prob_conj.sum(axis = 0)

        # Totaux lignes
        row_sum = X.sum(axis=1)

        # Totaux colonnes
        col_sum = X.sum(axis=0)

        # Compute chi - squared test
        statistic,pvalue,dof, _ = st.chi2_contingency(X, lambda_=None)

        # log - likelihood - tes (G - test)
        g_test_res = st.chi2_contingency(X, lambda_="log-likelihood")

        # Residuaal
        resid = X - self.expected_freq_

        standardized_resid = pd.DataFrame(self.standardized_resid_,index=self.row_labels_,columns=self.col_labels_)

        adjusted_resid = mapply(mapply(standardized_resid,lambda x : x/np.sqrt(1 - col_prob),axis=1,progressbar=False,n_workers=self.n_workers_),
                                lambda x : x/np.sqrt(1-row_prob),axis=0,progressbar=False,n_workers=self.n_workers_)
        
        chi2_contribution = mapply(standardized_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False,n_workers=self.n_workers_)
        # 
        attraction_repulsion_index = X/self.expected_freq_

        # Profils lignes
        row_prof = mapply(prob_conj,lambda x : x/np.sum(x), axis=1,progressbar=False,n_workers=self.n_workers_)
        
        ## Profils colonnes
        col_prof = mapply(prob_conj,lambda x : x/np.sum(x), axis=0,progressbar=False,n_workers=self.n_workers_)

        # Row distance
        row_dist = squareform(pdist(row_prof,metric= "seuclidean",V=col_prob)**2)
        
        # Distance entre individus et l'origine
        row_disto = mapply(row_prof,lambda x :np.sum((x-col_prob)**2/col_prob),axis = 1,progressbar=False,n_workers=self.n_workers_)

        # Poids des observations
        row_weight = row_sum/np.sum(row_sum)
        # Inertie des lignes
        row_inertie = row_disto*row_weight
        # Affichage
        row_infos = np.c_[row_disto, row_weight, row_inertie]
        
        ###################################################################################
        #               Informations sur les profils colonnes
        ###################################################################################

        col_dist = squareform(pdist(col_prof.T,metric= "seuclidean",V=row_prob)**2)

        # Distance à l'origine
        col_disto = mapply(col_prof.T,lambda x : np.sum((x-row_prob)**2/row_prob),axis = 1,progressbar=False,n_workers=self.n_workers_)

        # Poids des colonnes
        col_weight = col_sum/np.sum(col_sum)

        # Inertie des colonnes
        col_inertie = col_disto*col_weight
        # Affichage
        col_infos = np.c_[col_disto, col_weight, col_inertie]

        inertia = np.sum(row_inertie)

        # 
        self._compute_stats(row_prob,col_prob,row_disto,col_disto)

        # Return indicators
        self.chi2_test_ = pd.DataFrame(dict({"statistic" : statistic,"pvalue":pvalue,"dof":dof}),index=["chi-squared test"])
        self.log_likelihood_test_ = pd.DataFrame(dict({"statistic" : g_test_res[0],"pvalue":g_test_res[1]}),index=["G - test"])
        self.contingency_association_ = pd.DataFrame(dict({"cramer" : st.contingency.association(X, method="cramer"),
                                                           "tschuprow" : st.contingency.association(X, method="tschuprow"),
                                                           "pearson" : st.contingency.association(X, method="pearson")}),
                                                           index=["association"])
        self.resid_ = resid
        self.row_infos_ = row_infos
        self.col_infos_ = col_infos
        self.adjusted_resid_ = adjusted_resid
        self.chi2_contribution_ = chi2_contribution
        self.attraction_repulsion_index_ = attraction_repulsion_index
        self.inertia_ = inertia
        self.row_dist_ = row_dist
        self.col_dist_ = col_dist
        self.col_weight_ = col_weight
        self.row_weight_ = row_weight
    
    def _compute_svd(self,X):
        """"Compute a Singular Value Decomposition

        Then, this function computes :
            n_components_ : 
        """
        # Set row labels
        self.row_labels_ = self.row_labels
        if (self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]
        
        # Set col labels
        self.col_labels_ = self.col_labels
        if (self.col_labels_ is None) or (len(self.col_labels_) !=self.n_cols_):
            self.col_labels_ = ["col_" + str(k+1) for k in np.arange(0,self.n_cols_)]
        
        # Expected frequency
        self.expected_freq_ = st.contingency.expected_freq(X)
        
        # Standardized resid
        self.standardized_resid_ = (X - self.expected_freq_)/np.sqrt(self.expected_freq_)

        # Singular Values Decomposition
        U, delta, V_T = np.linalg.svd(self.standardized_resid_/np.sqrt(self.total_),full_matrices=False)

        # Eigenvalues
        lamb = delta**2

        f_max = min(self.n_rows_ -1,self.n_cols_ - 1)
        eigen_values = lamb[:f_max]
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # 
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = (delta > 1e-16).sum()
        
        self.eig_ = np.array([eigen_values[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])
        row_weight = X.sum(axis=1)/self.total_
        col_weight = X.sum(axis=0)/self.total_

        row_coord = np.apply_along_axis(func1d=lambda x : x/np.sqrt(row_weight),axis=0,arr=U[:,:f_max]*delta[:f_max])
        
        col_coord = np.apply_along_axis(func1d=lambda x : x/np.sqrt(col_weight),axis=0,arr=V_T[:f_max,:].T*delta[:f_max])
        #self.data_ = np.array(X)
        self.row_coord_ = row_coord[:,:self.n_components_]
        self.col_coord_ = col_coord[:,:self.n_components_]
        self.dim_index_ = ["Dim."+str(i+1) for i in np.arange(0,self.n_components_)]
        self.kaiser_threshold_ = np.mean(eigen_values)
        self.kaiser_proportion_threshold_ = 100/f_max
        self.res_row_dist_ = squareform(pdist(self.row_coord_,metric="sqeuclidean"))
        self.res_col_dist_ = squareform(pdist(self.col_coord_,metric="sqeuclidean"))

        self.model_ = "ca"
    
    def _compute_sup(self,X,row=True):
        """Compute row/columns supplementary coordinates
        
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if row:
            row_sup_prof = np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=1,arr=X).dot(self.col_coord_)/np.sqrt(self.eig_[0])
            self.row_sup_coord_ = row_sup_prof[:,:self.n_components_]
        else:
            col_sup_prof = np.transpose(np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=0,arr=X)).dot(self.row_coord_)/np.sqrt(self.eig_[0])
            self.col_sup_coord_ = col_sup_prof[:,:self.n_components_]

    
    def transform(self,X,y=None,row=True):
        """Apply the dimensionality reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame of float, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if row: 
            row_sup_prof = np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=1,arr=X)
            return row_sup_prof.dot(self.col_coord_) / np.sqrt(self.eig_[0])
        else:
            col_sup_prof = np.apply_along_axis(func1d=lambda x : x/np.sum(x),axis=0,arr=X)
            return col_sup_prof.T.dot(self.row_coord_)/np.sqrt(self.eig_[0])
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        y : None
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)

        return self.row_coord_
    

####################################################################################
#       MULTIPLE CORRESPONDENCE ANALYSIS (MCA)
####################################################################################

def _mul(*args):
    """An internal method to multiply matrices."""
    return functools.reduce(np.dot,args)

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)
    ---------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    This class performs Multiple Correspondence Analysis (MCA) with supplementary 
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Usage
    ----

    Parameters
    ----------
    
    """
    

    def __init__(self,n_components=None,
                 row_labels=None,
                 var_labels=None,
                 mod_labels= None,
                 matrix_type="completed",
                 benzecri=True,
                 greenacre=True,
                 tol = 1e-4,
                 approximate=False,
                 row_sup_labels = None,
                 quali_sup_labels = None,
                 quanti_sup_labels=None,
                 parallelize = False):
        self.n_components = n_components
        self.row_labels = row_labels
        self.var_labels = var_labels
        self.mod_labels = mod_labels
        self.matrix_type = matrix_type
        self.benzecri = benzecri
        self.greenacre = greenacre
        self.tol = tol
        self.approximate = approximate
        self.row_sup_labels = row_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        
        """
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary numeric or categorical columns
        self.quanti_sup_labels_ = self.quanti_sup_labels
        self.quali_sup_labels_ = self.quali_sup_labels
        if ((self.quali_sup_labels_ is not None) and (self.quanti_sup_labels_ is not None)):
            X_ = _X.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)        
        elif self.quali_sup_labels_ is not None:
            X_= _X.drop(columns = self.quali_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_)
        elif self.quanti_sup_labels_ is not None:
            X_ = _X.drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup  = row_sup.drop(columns = self.quanti_sup_labels_)
        else:
            X_ = _X
        
        self.data_ = X
        self.original_data_ = None
        
        # Supplementary initialization
        self.row_sup_coord_ = None
        self.row_sup_cos2_ = None

        self.col_sup_coord_ = None
        self.col_sup_cos2_ = None

        # Supplementary variables/categories informations
        self.mod_sup_coord_ = None
        self.mod_sup_cos2_ = None
        self.mod_sup_disto_ = None
        self.mod_sup_vtest_ = None

        # Supplementary categorical variables correlation ratio
        self.quali_sup_eta2_ = None

        # Benzerci and Greenacre coorection
        self.benzecri_correction_ = None
        self.greenacre_correction_ = None

        self.var_labels_ = self.var_labels
        if ((self.var_labels_ is not None) and (len(X_.columns) < len(self.var_labels_))):
            raise ValueError("length of 'var_labels' must be less or equal to number of X columns.")
        
        if self.n_components == 1:
            raise ValueError("n_components must be grather than 1.")
        
        self._compute_svds(X_)
        
        # Compute supplementary quantitatives variables statistics
        if self.quanti_sup_labels_ is not None:
            self._compute_quanti_sup_stats(_X[self.quanti_sup_labels_])
        
        # Compute supllementary qualitatives variables statistics
        if self.quali_sup_labels_ is not None:
            self._compute_quali_sup_stats(X=_X[self.quali_sup_labels_])
        
        # Compute supplementrary rows statistics
        if self.row_sup_labels_ is not None:
            self._compute_row_sup_stats(X=row_sup)

        return self
        
    def _get_dummies(self,X):
        """Convert categorical variable into dummy/indicator variables.
        Each variable is converted in as many 0/1 variables as there are different values. Columns in the
        output are each named after a value; if the input is a DataFrame, the name of the original variable
        is prepended to the value.

        Parameters
        ----------
        X : Series, or DataFrame
            Data of which to get dummy indicators.
        
        Return
        ------
        DataFrame
            Dummy-coded data. If data contains other columns than the dummy-coded
            one(s), these will be prepended, unaltered, to the result.
        """
        dummies = (pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols in (X.columns if self.var_labels_ is None else self.var_labels_))
        return pd.concat(dummies,axis=1)

    def _compute_disjonctif_table(self,X):
        """Compute dummies tables
        
        """
        self.mod_labels_ = self.mod_labels
        if ((self.var_labels_ is None) and (self.mod_labels_ is None)):
            raise ValueError("Error : You must pass either 'var_labels' or 'mod_labels'.")
        
        self.n_rows_ = X.shape[0]
        if self.matrix_type == "completed":
            self.original_data_ = X
            self.disjonctif_ = self._get_dummies(X)
            if self.var_labels_ is None:
                self.var_labels_ = list(X.columns)
        elif self.matrix_type == "disjonctif":
            # Chack if duplicate columns
            duplicate = {x for x in list(X.columns) if list(X.columns).count(x) > 1}
            if len(duplicate)>1:
                raise ValueError("Error : 'X' must have unique columns.")
            
            # Check if underscore <<"_">> in columns
            if False in [x.__contains__('_') for x in list(X.columns)]:
                raise ValueError("Error : 'X' columns must have '_' to separate 'variable name' with 'modality'.",
                                 "\n see 'https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html'.")
            
            self.original_data_ = from_dummies(X,sep="_")
            self.disjonctif_ = X
            if self.var_labels_ is None:
                var = list([x.split("_",1)[0] for x in self.disjonctif_.columns])
                self.var_labels_ = reduce(lambda re, x: re+[x] if x not in re else re, var, [])
        else:
            raise ValueError("Error : Allowed values for the argument matrix_type are : 'completed' or 'disjonctif'.")
        
        self.mod_labels_ = self.mod_labels
        if self.mod_labels_ is None:
            self.mod_labels_ = self.disjonctif_.columns
        
        self.n_mods_ = len(self.mod_labels_)
        self.n_vars_ = len(self.var_labels_)
        self.short_labels_ = list([x.split("_",1)[-1] for x in self.mod_labels_])

    def _compute_svds(self,X):

        """
        
        """

        self._compute_disjonctif_table(X)
        self._compute_stats()

         # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]

        
        S = self.disjonctif_.sum().sum()
        Z = self.disjonctif_/S  # Correspondence matrix
        self.r_ = Z.sum(axis=1)
        self.c_ = Z.sum(axis=0)

        eps = np.finfo(float).eps
        self.D_r = np.diag(1/(eps + np.sqrt(self.r_)))
        self.D_c = np.diag(1/(eps + np.sqrt(self.c_)))
        Z_c = Z -  np.outer(self.r_, self.c_)  # standardized residuals matrix

        product = self.D_r.dot(Z_c).dot(self.D_c)
        self._numitems = len(X)
        U, delta, V_T = np.linalg.svd(product)

        eigen_value = delta ** 2
        difference = np.insert(-np.diff(eigen_value),len(eigen_value)-1,np.nan)
        proportion = 100*eigen_value/np.sum(eigen_value)
        cumulative = np.cumsum(proportion)


        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = self.n_mods_ - self.n_vars_
        
        self.dim_index_ = ["Dim."+str(i+1) for i in np.arange(0,self.n_components_)]

        if self.benzecri:
            self._benzecri(eigen_value)
        
        if self.greenacre:
            self._greenacre(eigen_value)

        # Row and columns coordinates
        row_coord = (self.D_r.dot(U).dot(sp.linalg.diagsvd(delta[:self.n_components_],self._numitems,self.n_components_)))
        mod_coord = _mul(self.D_c, V_T.T, sp.linalg.diagsvd(delta[:self.n_components_],len(V_T),self.n_components_))

        # Store information
        self.eig_ = np.array([eigen_value[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])
        
        # Normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        corrected_mod_coord = np.apply_along_axis(func1d=lambda x: x*np.sqrt(self.eig_[0]),axis=1,arr=mod_coord)

        # Row and columns cos2
        row_cos2 = np.apply_along_axis(lambda x : x**2/np.linalg.norm(row_coord,axis=1)**2,axis=0,arr=row_coord)
        mod_cos2 = np.apply_along_axis(lambda x : x**2/np.linalg.norm(mod_coord,axis=1)**2,axis=0,arr=mod_coord)

        # Row and columns contribution
        row_contrib = np.apply_along_axis(lambda x : 100*x**2/(self.n_rows_*eigen_value[:self.n_components_]),axis=1,arr=row_coord)
        mod_contrib = np.apply_along_axis(lambda x : x/eigen_value[:self.n_components_],axis=1,
                                          arr=np.apply_along_axis(lambda x : 100*x**2*self.c_,axis=0,arr=mod_coord))
        
        # Valeur test des modalités
        dummies_sum = self.disjonctif_.sum(axis=0)
        mod_vtest = np.apply_along_axis(func1d=lambda x : x*np.sqrt(((self.n_rows_ - 1)*dummies_sum)/(self.n_rows_ - dummies_sum)),
                                        axis=0,arr=mod_coord)
        # Qualitative informations
        mod_coord_df = pd.DataFrame(mod_coord,index=self.mod_labels_,columns=self.dim_index_)
        dummies_mean = self.disjonctif_.mean(axis=0)
        var_eta2 = mapply(mod_coord_df,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_).mul(dummies_mean, axis='index')
        var_eta2 = pd.concat((mapply(var_eta2.loc[filter(lambda x: x.startswith(cols),var_eta2.index),:],lambda x : np.sum(x),
                             axis=0,progressbar=False,n_workers=self.n_workers_).to_frame(name=cols).T for cols in  self.var_labels_),axis=0)
    
        ## Cosinus carrés des variables qualitatives
        # Nombre de modalités par variables
        nb_mod = pd.Series(np.array([len(np.unique(self.original_data_[[col]])) for col in self.var_labels_]),index=self.var_labels_,name="count")

        # Cosinus des variables
        var_cos2 = pd.DataFrame(mod_cos2,index=self.mod_labels_,columns=self.dim_index_)
        var_cos2 = pd.concat((var_cos2.loc[var_cos2.index.str.startswith(cols)].sum(axis=0).to_frame(name=cols).T/(nb_mod[cols]-1) for cols in self.var_labels_),axis=0)
        
        # Contribution des variables
        var_contrib = pd.DataFrame(mod_contrib,index=self.mod_labels_,columns=self.dim_index_)
        var_contrib = pd.concat((var_contrib.loc[var_contrib.index.str.startswith(cols)].sum(axis=0).to_frame(name=cols).T for cols in self.var_labels_),axis=0)

        # Store all informations
        self.row_coord_ = row_coord
        self.row_cos2_ = row_cos2
        self.row_contrib_ = row_contrib

        self.mod_coord_ = mod_coord
        self.corrected_mod_coord_ = corrected_mod_coord
        self.mod_cos2_ = mod_cos2
        self.mod_contrib_ = mod_contrib
        self.mod_vtest_ = mod_vtest

        # Inertia
        self.inertia_ = self.n_mods_/self.n_vars_ - 1

        # Eigenvalue threshold
        self.kaiser_threshold_ = 1/self.n_vars_
        self.kaiser_proportion_threshold_ = 100/self.inertia_

        ## Ajout des informations sur les variables
        self.var_eta2_ =  np.array(var_eta2)
        self.var_cos2_ = np.array(var_cos2)
        self.var_contrib_ = np.array(var_contrib)

        self.model_ = "mca"
    
    def _compute_stats(self):

        """
        
        
        """

        chi2_stats = np.zeros(shape=(self.n_vars_,self.n_vars_))
        chi2_pvalue = np.zeros(shape=(self.n_vars_,self.n_vars_))
        for i in np.arange(0,self.n_vars_):
            for j in np.arange(0,self.n_vars_):
                tab = pd.crosstab(self.original_data_.iloc[:,i],self.original_data_.iloc[:,j])
                chi = st.chi2_contingency(tab)
                chi2_stats[i,j],chi2_pvalue[i,j]= chi[0],chi[1]
            
        self.chi2_test_ = dict({"statistic": pd.DataFrame(chi2_stats,index=self.var_labels_,columns=self.var_labels_),
                                "pvalue" : pd.DataFrame(chi2_pvalue,index=self.var_labels_,columns=self.var_labels_)
                        })

        # Marke ligne
        row_marge = self.disjonctif_.sum(axis=0)

        # Profil individu moyen
        ind_moyen = row_marge/(self.n_rows_*self.n_vars_)

        ########################################################################################################################
        #                       Informations sur les individus
        #######################################################################################################################

        # Distance du chi2 entre les individus
        row_dist = squareform(pdist(self.disjonctif_/self.n_vars_,metric="seuclidean",V=ind_moyen)**2)

        # Distance des observations à l'origine
        row_disto = mapply(self.disjonctif_,lambda x : np.sum((1/ind_moyen)*(x/self.n_vars_ - ind_moyen)**2),axis=1,progressbar=False,n_workers=self.n_workers_)
        
        # Poids des observations
        row_weight = np.ones(self.n_rows_)/self.n_rows_

        # Inertie des observations
        row_inertia = row_disto*row_weight

        # Stockage - concatenation
        row_infos = np.c_[np.sqrt(row_disto), row_weight, row_inertia]

        #########################################################################################################
        #                   Informations sur les modalités
        #########################################################################################################

        # Distance chi2 entre les modalités
        dummies_weight = self.disjonctif_.div(row_marge,axis="columns")
        mod_dist = self.n_rows_*squareform(pdist(dummies_weight.T,metric="sqeuclidean"))

        # Distance des modalités à l'origine
        mod_disto = mapply(dummies_weight,lambda x : np.sum(self.n_rows_*(x-row_weight)**2),axis = 0,progressbar=False)

        # Poids des modalités
        mod_weight = ind_moyen

        # Inertie des modalités
        mod_inertia = mod_disto * mod_weight

        # Stockage des informations
        mod_infos = np.c_[np.sqrt(mod_disto),mod_weight,mod_inertia]

        #########################################################################################################
        #                   Informations sur les variables
        #########################################################################################################
        
        # Inertia for the variables
        var_inertia = np.c_[np.array([(len(np.unique(self.original_data_[col]))-1)/self.n_vars_ for col in self.original_data_.columns])]

        #########################################################################################################
        #                                         Store all informations
        #########################################################################################################

        # Store informations
        self.row_dist_ = row_dist
        self.row_infos_ = row_infos
        self.mod_dist_ = mod_dist
        self.mod_infos_ = mod_infos
        self.var_inertia_ = var_inertia
    
    def _benzecri(self,X):
        """Compute Benzécri correction

        Prameters:
        ----------
        X : array

        Return
        -------
        None
        
        """
        # save eigen value grather than threshold
        lambd = X[X>(1/self.n_vars_)]

        if len(lambd) > 0:
            # Apply benzecri correction
            lambd_tilde = ((self.n_vars_/(self.n_vars_-1))*(lambd - 1/self.n_vars_))**2

            # Cumulative percentage
            s_tilde = 100*(lambd_tilde/np.sum(lambd_tilde))

            # Benzecri correction
            self.benzecri_correction_ = pd.DataFrame(np.c_[lambd_tilde,s_tilde,np.cumsum(s_tilde)],
                                                     columns=["eigenvalue","proportion","cumulative"],
                                                     index = list(["Dim."+str(x+1) for x in np.arange(0,len(lambd))]))

    def _greenacre(self,X):
        """Compute Greenacre correction

        Parameters
        ----------
        X : array

        Return
        ------
        None
        
        """
        # save eigen value grather than threshold
        lambd = X[X>(1/self.n_vars_)]

        if len(lambd) > 0:
            lambd_tilde = ((self.n_vars_/(self.n_vars_-1))*(lambd - 1/self.n_vars_))**2

            s_tilde_tilde = self.n_vars_/(self.n_vars_-1)*(np.sum(X**2)-(self.n_mods_-self.n_vars_)/(self.n_vars_**2))

            tau = 100*(lambd_tilde/s_tilde_tilde)

            self.greenacre_correction_ = pd.DataFrame(np.c_[lambd_tilde,tau,np.cumsum(tau)],
                                                      columns=["eigenvalue","proportion","cumulative"],
                                                      index = list(["Dim."+str(x+1) for x in np.arange(0,len(lambd))]))
    
    def _compute_row_sup_stats(self,X, y=None):
        """ Apply the dimensionality reduction on X. X is projected on
        the first axes previous extracted from a training set.
        Parameters
        ----------
        X : array of string, int or float, shape (n_rows_sup, n_vars)
            New data, where n_rows_sup is the number of supplementary
            row points and n_vars is the number of variables.
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.
            X rows correspond to supplementary row points that are
            projected onto the axes.
        
        y : None
            y is ignored.
        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points onto the axes.
        """
        if self.matrix_type == "completed":
            n_rows = X.shape[0]
            n_cols = len(self.mod_labels_)
            Y = np.zeros((n_rows,n_cols))
            for i in np.arange(0,n_rows,1):
                values = [self.var_labels_[k] +"_"+str(X.iloc[i,k]) for k in np.arange(0,self.n_vars_)]
                for j in np.arange(0,n_cols,1):
                    if self.mod_labels_[j] in values:
                        Y[i,j] = 1
            row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)
        else:
            row_sup_dummies = X
        row_sup_profil = (mapply(row_sup_dummies,lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=self.n_workers_)
                                .dot(self.mod_coord_)/np.sqrt(self.eig_[0]))
        
        self.row_sup_coord_ = np.array(row_sup_profil)
        self.row_sup_cos2_ = np.apply_along_axis(lambda x : x**2/np.linalg.norm(self.row_sup_coord_,axis=1)**2,
                                                    axis=0,arr=self.row_sup_coord_)
        
        return {"coord"    :   self.row_sup_coord_,
                "cos2"     :   self.row_sup_cos2_}
    
    def _compute_quali_sup_stats(self,X,y=None):
        """Find the supplementary categorical columns factor

        Parameters
        ----------

        
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        dummies = pd.concat((pd.get_dummies(X,prefix=cols,prefix_sep = "_") for cols in X.columns),axis=1)
        mod_sup_stats = dummies.agg(func=[np.sum,np.mean]).T

        n_k = dummies.sum(axis=0)
        p_k = dummies.mean(axis=0)

        mod_sup_labels = dummies.columns
        short_sup_labels = list([x.split("_",1)[-1] for x in mod_sup_labels])

        # Distance à l'origine des modalités supplémentaires
        mod_sup_disto = (1/p_k)-1
        
        # Coordinates of supplementary categories - corrected
        mod_sup_coord = mapply(dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=self.n_workers_).T.dot(self.row_coord_)/np.sqrt(self.eig_[0])

        # Square of coordinates
        sq_mod_sup_coord = mapply(mod_sup_coord,lambda x: x**2,axis=0,progressbar=False,n_workers=self.n_workers_)

        # Cosinus carré des modalités supplémentaires
        mod_sup_cos2 = pd.concat(((sq_mod_sup_coord.loc[k,:]/mod_sup_disto[k]).to_frame().T for k in mod_sup_coord.index),axis=0)

        # Valeur test des modalités supplémentaires
        mod_sup_vtest = pd.concat(((mod_sup_coord.loc[k,:]*np.sqrt(((self.n_rows_-1)*n_k[k])/(self.n_rows_ - n_k[k]))).to_frame(name=k).T for k in mod_sup_coord.index),axis=0)
        mod_sup_vtest.columns = self.dim_index_

        # Correlation Ratio
        quali_sup_eta2 = pd.concat(((sq_mod_sup_coord.loc[k,:]*p_k[k]).to_frame().T for k in mod_sup_coord.index),axis=0)
        quali_sup_eta2 = pd.concat((quali_sup_eta2.loc[quali_sup_eta2.index.str.startswith(cols)].sum(axis=0).to_frame(name=cols).T for cols in X.columns),axis=0)
        quali_sup_eta2.columns = self.dim_index_

        # Nombre de modalités par variables
        nb_sup_mod = pd.Series(np.array([len(np.unique(X[[col]])) for col in X.columns]),index=X.columns,name="count")
       
        # Cosinus des variables qualitatives supplémentaires
        quali_sup_cos2 = pd.concat((mod_sup_cos2.loc[mod_sup_cos2.index.str.startswith(cols)].sum(axis=0).to_frame(name=cols).T/(nb_sup_mod[cols]-1) for cols in X.columns),axis=0)
        quali_sup_cos2.columns = self.dim_index_

        # Store supplementary categories informations
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_disto_     =   np.array(mod_sup_disto)
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)
       
        self.mod_sup_labels_ = mod_sup_labels
        self.short_sup_labels_ = short_sup_labels

        # Supplementative categorical variables correlation ratio and cos2
        self.quali_sup_eta2_ = quali_sup_eta2
        self.quali_sup_cos2_ = quali_sup_cos2

        return {"coord"   :   mod_sup_coord,
                "cos2"    :   mod_sup_cos2,
                "dist"    :   mod_sup_disto.to_frame("Dist"),
                "stats"   :   mod_sup_stats,
                "vtest"   :   mod_sup_vtest}
    
    def _compute_quanti_sup_stats(self,X,y=None):
        """Find the supplementary quantitative columns factor

        Parameters
        ----------
        X : DataFrame

        Returns:
        -------

        
        """

        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Supplementary quantitatives coordinates
        col_sup_coord = np.transpose(np.corrcoef(x=self.row_coord_,y=X.values,rowvar=False)[:self.n_components_,self.n_components_:])

        # Supplementary quantitatives cos2
        col_sup_cos2 = np.apply_along_axis(func1d=lambda x : x**2,arr = col_sup_coord,axis=0)        

        # Store supplementary quantitatives informations
        self.col_sup_coord_  =   col_sup_coord[:,:self.n_components_]
        self.col_sup_cos2_   =   col_sup_cos2[:,:self.n_components_]
        self.col_sup_labels_ = X.columns

        return {"coord"    :   col_sup_coord[:,:self.n_components_],
                "cos2"     :   col_sup_cos2[:,:self.n_components_]}
    
    def transform(self,X,y=None):
        """ Apply the dimensionality reduction on X. X is projected on
        the first axes previous extracted from a training set.
        Parameters
        ----------
        X : array of string, int or float, shape (n_rows_sup, n_vars)
            New data, where n_rows_sup is the number of supplementary
            row points and n_vars is the number of variables.
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.
            X rows correspond to supplementary row points that are
            projected onto the axes.
        
        y : None
            y is ignored.
        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points onto the axes.
        """
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #self._compute_row_sup_stats(X)
        if self.matrix_type == "completed":
            n_rows = X.shape[0]
            n_cols = len(self.mod_labels_)
            Y = np.zeros((n_rows,n_cols))
            for i in np.arange(0,n_rows,1):
                values = [self.var_labels_[k] +"_"+str(X.iloc[i,k]) for k in np.arange(0,self.n_vars_)]
                for j in np.arange(0,n_cols,1):
                    if self.mod_labels_[j] in values:
                        Y[i,j] = 1
            row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)
        else:
            row_sup_dummies = X
        row_sup_coord = (mapply(row_sup_dummies,lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=self.n_workers_)
                                .dot(self.mod_coord_)/np.sqrt(self.eig_[0]))
        
        row_sup_coord = np.array(row_sup_coord)
        return row_sup_coord
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        self.fit(X)
        return self.row_coord_

#############################################################################################
#               FACTOR ANALYSIS OF MIXED DATA (FAMD)
#############################################################################################

class FAMD(BaseEstimator,TransformerMixin):
    """Factor Analysis of Mixed Data

    Performs Factor Analysis of Mixed Data (FAMD) with supplementary 
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Parameters:
    -----------
    see scientisttools.decomposition.PCA and scientisttools.decomposition.MCA
    
    """
    def __init__(self,
                 normalize=True,
                 n_components=None,
                 row_labels=None,
                 quanti_labels=None,
                 quali_labels=None,
                 row_sup_labels=None,
                 quanti_sup_labels=None,
                 quali_sup_labels=None,
                 parallelize = False):
        self.normalize =normalize
        self.n_components = n_components
        self.row_labels = row_labels
        self.quanti_labels = quanti_labels
        self.quali_labels = quali_labels
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.parallelize = parallelize
    
    def fit(self,X):
        """
        
        
        """

        # Chack if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize option
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Extract supplementary rows
        self.row_sup_labels_ = self.row_sup_labels
        if self.row_sup_labels_ is not None:
            _X = X.drop(index = self.row_sup_labels_)
            row_sup = X.loc[self.row_sup_labels_,:]
        else:
            _X = X

        # Extract supplementary numeric or categorical columns
        self.quali_sup_labels_ = self.quali_sup_labels
        self.quanti_sup_labels_ = self.quanti_sup_labels
        if ((self.quali_sup_labels_ is not None) and (self.quanti_sup_labels_ is not None)):
            X_ = _X.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_).drop(columns = self.quanti_sup_labels_)        
        elif self.quali_sup_labels_ is not None:
            X_= _X.drop(columns = self.quali_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup = row_sup.drop(columns = self.quali_sup_labels_)
        elif self.quanti_sup_labels_ is not None:
            X_ = _X.drop(columns = self.quanti_sup_labels_)
            if self.row_sup_labels_ is not None:
                row_sup  = row_sup.drop(columns = self.quanti_sup_labels_)
        else:
            X_ = _X
        
        # Save initial data
        self.data_ = X
        self.active_data_ = X_
        
        # Supplementary initialization
        self.row_sup_coord_ = None
        self.row_sup_cos2_ = None

        # Additional information for supplementary continuous variables
        self.col_sup_coord_ = None
        self.col_sup_cos2_ = None
        self.col_sup_ftest_ = None

        # Additionnal informations supplementary categories
        self.mod_sup_stats_ = None
        self.mod_sup_coord_ = None
        self.mod_sup_cos2_ = None
        self.mod_sup_disto_ = None
        self.mod_sup_vtest_ = None

        #Additionnal informations for supplementary categorical informations
        self.quali_sup_eta2_ = None
        self.quali_sup_cos2_ = None

        
        # Compute statistics
        self.n_rows_ = X_.shape[0]
        X_quant = X_.select_dtypes(include=np.number)
        X_qual = X_.select_dtypes(include=["object","category"])

        # Check if NULL
        if X_quant.empty and not X_qual.empty:
            raise ValueError("Error : There is no continuous variables in X. Please use MCA function.")
        elif X_qual.empty and not X_quant.empty:
            raise ValueError("Error : There is no categoricals variables in X. Please use PCA function.")

        #Initialize lables
        self.row_labels_ = self.row_labels
        if self.row_labels_ is None:
            self.row_labels_ = X.index

        self.quali_labels_ = self.quali_labels
        if self.quali_labels_ is None:
            self.quali_labels_ = X_qual.columns
        
        self.quanti_labels_ = self.quanti_labels
        if self.quanti_labels_ is None:
            self.quanti_labels_ = X_quant.columns

        self.quanti_data_ = X_quant
        self.quali_data_ = X_qual

        # Pearson correlation between continuous variables
        self.col_corr_ = np.array(X_quant.corr(method="pearson"))

        # Partial correlation between continuous variables
        self.col_pcorr_ = np.array(X_quant.pcorr())
    
        # Measure of association
        self.chi2_test_ = scientistmetrics(X=X_qual,method="chi2")
        self.cramerv_ = scientistmetrics(X=X_qual,method="cramer")
        self.tschuprowt_ = scientistmetrics(X=X_qual,method="tschuprow")
        
        # Normalisation des variables qualitatives
        dummies = pd.concat((pd.get_dummies(X_qual[cols],prefix=cols,prefix_sep='_') for cols in self.quali_labels_),axis=1)

        n_k = dummies.sum(axis=0)
        self.dummies_means_ = dummies.mean(axis=0)
        self.dummies_std_ = np.sqrt(self.dummies_means_)
        mod_stats = dummies.agg(func=[np.sum,np.mean]).T

        # Centrage et réduction
        self.means_ = np.mean(X_quant.values, axis=0).reshape(1,-1)
        if self.normalize:
            self.std_ = np.std(X_quant.values,axis=0,ddof=0).reshape(1,-1)
            Z1 = (X_quant - self.means_)/self.std_
        else:
            Z1 = X_quant - self.means_

        # Normalize Z2
        Z2 = mapply(dummies,lambda x: x/np.sqrt(self.dummies_means_.values),axis = 1,progressbar=False,n_workers=self.n_workers_)

        # Concatenate the 2 dataframe
        Z = pd.concat([Z1,Z2],axis=1)

        # Distance between individuals
        row_dist = squareform(pdist(Z,metric='sqeuclidean'))

        # Distance between individuals and inertia center
        row_disto = (mapply(Z1,lambda x:np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_) + 
                     mapply(dummies,lambda x:np.sum(1/self.dummies_means_.values*(x-self.dummies_means_.values)**2),
                            axis=1,progressbar=False,n_workers=self.n_workers_))
        # Individuals weight
        row_weight = np.ones(self.n_rows_)/self.n_rows_

        # Individuals inertia
        row_inertie = row_disto*row_weight

        # Save all informations
        row_infos = np.c_[np.sqrt(row_disto),row_weight,row_inertie]

        ################################
        dummies_weight = dummies.div(n_k,axis="columns")

        mod_dist = self.n_rows_*squareform(pdist(dummies_weight.T,metric="sqeuclidean"))

        # Distance à l'origine
        mod_disto = mapply(dummies_weight,lambda x : np.sum(self.n_rows_*(x-row_weight)**2),axis=0,progressbar=False,n_workers=self.n_workers_)

        # Poids des modalités
        mod_weight = n_k/(self.n_rows_*dummies.shape[1])

        # Inertie des lignes
        mod_inertie = mod_disto*mod_weight

        mod_infos = np.c_[np.sqrt(mod_disto), mod_weight, mod_inertie]

        self.row_infos_ = row_infos
        self.mod_infos_ = mod_infos
        self.row_dist_ = row_dist
        self.mod_dist_ = mod_dist
        self.mod_stats_ = np.array(mod_stats)
        self.normalized_data_ = Z
        self.mod_labels_ = dummies.columns
        self.short_labels_ = list([x.split("_",1)[-1] for x in dummies.columns])

        self._compute_svd(X=Z,Xq=X_qual,Iq=n_k)

        if self.row_sup_labels_ is not None:
            self._compute_row_sup_stats(X=row_sup)
        
        if self.quanti_sup_labels_ is not None:
            self._compute_quanti_sup_stats(X=_X[self.quanti_sup_labels_])
        
        if self.quali_sup_labels_ is not None:
            self._compute_quali_sup_stats(X=_X[self.quali_sup_labels_])

        return self

    def _compute_svd(self,X,Xq,Iq):
        """Compute Singular Value Decomposition
        
        
        
        """

        f_max = X.shape[1] - len(self.quali_labels_)

        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = f_max
        elif not isinstance(self.n_components_,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components_ <= 0:
            raise ValueError("Error : 'n_components' must be positive integers")
        elif self.n_components_ > f_max:
            raise ValueError(f"Error : 'n_components' must be less or equal to {f_max}")
        
        self.dim_index_ = ["Dim."+str(x+1) for x in np.arange(0,self.n_components_)]

        res = PCA(normalize=False,n_components=self.n_components_,row_labels=X.index,col_labels=X.columns).fit(X)

        ########### Store all informations
        self.eig_ = res.eig_
        self.eigen_vectors_ = res.eigen_vectors_

        ####### Row - Cos2 & contrib
        row_cos2 = np.apply_along_axis(func1d=lambda x : x**2/(self.row_infos_[:,0]**2),axis=0,arr=res.row_coord_)
        row_contrib = np.apply_along_axis(func1d=lambda x : 100*x**2/(self.n_rows_*res.eig_[0]),axis=1,arr=res.row_coord_)

        # Row informations
        self.row_coord_     = res.row_coord_
        self.row_contrib_   = row_contrib
        self.row_cos2_      = row_cos2
        self.res_row_dist_  = squareform(pdist(self.row_coord_,metric="sqeuclidean"))

        # Coordinates for quantitatives columns
        var_mod_coord = pd.DataFrame(res.col_coord_,index=X.columns,columns=self.dim_index_)
        col_coord = var_mod_coord.loc[self.quanti_labels_,:]

        ####### Quantitative columns - Cos2 & Contrib
        col_cos2 = mapply(col_coord,lambda x : x**2, axis=1,progressbar=False,n_workers=self.n_workers_)
        col_contrib = mapply(col_coord,lambda x : 100*x**2/res.eig_[0],axis=1,progressbar=False,n_workers=self.n_workers_)

        # Test de significativité de Fisher
        col_ftest = mapply(col_coord,lambda x : (1/2)*np.sqrt(self.n_rows_-3)*np.log((1+x)/(1-x)),axis=0,progressbar=False,n_workers=self.n_workers_)

        # Quantitatives informations
        self.col_coord_     = np.array(col_coord)
        self.col_cos2_      = np.array(col_cos2)
        self.col_contrib_   = np.array(col_contrib)
        self.col_ftest_     = np.array(col_ftest)

        # Continuous labels
        self.col_labels_    = self.quanti_labels_

        # Modality informations
        mod_coord = self._correct_modality(X=Xq)

        coord_mod = var_mod_coord.loc[self.mod_labels_,:]

        mod_cos2 = mapply(mod_coord,lambda x : x**2/(self.mod_infos_[:,0]**2), axis=0,progressbar=False,n_workers=self.n_workers_)
        mod_contrib = mapply(coord_mod,lambda x : 100*x**2/res.eig_[0],axis = 1,progressbar=False,n_workers=self.n_workers_)
        mod_vtest = mapply(mapply(mod_coord,lambda x : x*np.sqrt(((self.n_rows_-1)*Iq.values)/(self.n_rows_-Iq.values)),
                                  axis=0,progressbar=False,n_workers=self.n_workers_),
                           lambda x : x/np.sqrt(res.eig_[0]),axis=1,progressbar=False,n_workers=self.n_workers_)
        
        # Qualitative informations
        var_eta2 = pd.concat((mapply(coord_mod.loc[filter(lambda x: x.startswith(cols),coord_mod.index),:],
                                     lambda x : x**2,axis=1,progressbar=False,n_workers=self.n_workers_).sum().to_frame(name=cols).T for cols in self.quali_labels_),axis=0)
        
        # Cosinus carrés des variables qualitatives
        nb_mod = pd.Series([len(np.unique(Xq[[col]])) for col in self.quali_labels_],index=self.quali_labels_,name="count")
        var_cos2 = pd.concat(((var_eta2.loc[cols,:]/nb_mod[cols]).to_frame(name=cols).T for cols in var_eta2.index),axis=0)
        
        # Contributions des variables qualitatives
        var_contrib = mapply(var_eta2,lambda x : 100*x/res.eig_[0],axis=1,progressbar=False,n_workers=self.n_workers_)
        
        # Modality informations
        self.coord_mod_ = np.array(coord_mod)
        self.mod_coord_ = np.array(mod_coord)
        self.mod_cos2_ = np.array(mod_cos2)
        self.mod_contrib_ = np.array(mod_contrib)
        self.mod_vtest_ = np.array(mod_vtest)

        # Information sur les variables qualitatives
        self.var_mod_coord_ = np.array(var_mod_coord)
        self.var_eta2_ = np.array(var_eta2)
        self.var_cos2_ = np.array(var_cos2)
        self.var_contrib_ = np.array(var_contrib)

        self.model_ = "famd"

    def _correct_modality(self,X):
        """
        
        
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")


        # Modified modality coordinates
        dummies = pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols in X.columns),axis=1)
        modified_mod_coord = pd.concat((pd.concat((pd.DataFrame(self.row_coord_,index=self.row_labels_,
                                                   columns=self.dim_index_),dummies[cols]),axis=1)
                                      .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)
        
        return modified_mod_coord
    
    def _compute_row_sup_stats(self,X):
        """Compute supplementary individuals coordinates

        Parameters
        ----------
       X : DataFrame, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        X_sup_quant = X[self.quanti_labels_]
        X_sup_qual = X[self.quali_labels_]

        if self.normalize:
            Z1 = (X_sup_quant - self.means_)/self.std_ 
        else:
            Z1 = X_sup_quant - self.means_

        # Standardscale Categorical Variable
        n_rows = X_sup_qual.shape[0]
        n_cols = len(self.mod_labels_)
        Y = np.zeros((n_rows,n_cols))
        for i in np.arange(0,n_rows,1):
            values = [self.quali_labels_[k] +"_"+str(X_sup_qual.iloc[i,k]) for k in np.arange(0,len(self.quali_labels_))]
            for j in np.arange(0,n_cols,1):
                if self.mod_labels_[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)

        # New normalized Data
        Z2 = mapply(row_sup_dummies,lambda x : (x - self.dummies_means_)/self.dummies_std_,axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary individuals coordinates
        row_sup_coord = np.dot(pd.concat([Z1,Z2],axis=1),self.eigen_vectors_)

        # Supplementary individuals distance to inertia
        row_sup_disto = (mapply(Z1,lambda x:np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_) + 
                            mapply(row_sup_dummies,lambda x:np.sum(1/self.dummies_means_.values*(x-self.dummies_means_.values)**2),
                            axis=1,progressbar=False,n_workers=self.n_workers_))
        
        row_sup_cos2 = np.apply_along_axis(func1d=lambda x : x**2/(row_sup_disto),axis=0,arr=row_sup_coord)

        # Save
        self.row_sup_coord_ = row_sup_coord[:,:self.n_components_]
        self.row_sup_disto_ = np.sqrt(np.array(row_sup_disto))
        self.row_sup_cos2_ = row_sup_cos2
    
    def _compute_quanti_sup_stats(self,X,y=None):
        """Comupute supplementary continuous variables statistics

        Parameters
        ----------
        self    :   An instance of class FAMD
        X       :   DataFrame (n_rows,n_columns)
        y : None
            y is ignored

        Return
        ------
        col_sup_corr_   : Pearson correlation between new continuous variables and old continuous variables
        col_sup_coord_  :   Supplementary continuous coordinates
        col_sup_cos2_   :   Supplementary continuous cosines
        col_sup_ftest_  :   Supplementary continuous Fisher - test
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Correlation between New continuous variables and old continuous variables
        col_sup_corr = np.transpose(np.corrcoef(x=self.quanti_data_.values,y=X.values,rowvar=False))[len(self.quanti_labels_):,:len(self.quanti_labels_)]

        # Supplementary continuous coordinates
        col_sup_coord = np.transpose(np.corrcoef(x=self.row_coord_,y=X.values,rowvar=False)[:self.n_components_,self.n_components_:])

        # Fisher - test for columns coordinates
        col_sup_ftest = np.apply_along_axis(func1d=lambda x : (1/2)*np.sqrt(self.n_rows_-3)*np.log((1+x)/(1-x)),axis=0,arr=col_sup_coord)

        # Supplementary continuous cos2
        col_sup_cos2 = np.apply_along_axis(func1d=lambda x : x**2,arr = col_sup_coord,axis=0)

        # Store supplementary continuous informations
        self.col_sup_corr_  =   col_sup_corr
        self.col_sup_coord_ =   col_sup_coord[:,:self.n_components_]
        self.col_sup_cos2_  =   col_sup_cos2[:,:self.n_components_]
        self.col_sup_ftest_ =   col_sup_ftest[:,:self.n_components_]

        # Self
        self.col_sup_labels_ = X.columns

        return {"corr"     :   pd.DataFrame(col_sup_corr, index=X.columns,columns=self.col_labels_),
                     "coord"    :   pd.DataFrame(col_sup_coord[:,:self.n_components_],index=X.columns,columns=self.dim_index_),
                     "cos2"     :   pd.DataFrame(col_sup_cos2[:,:self.n_components_], index = X.columns,columns=self.dim_index_),
                     "ftest"    :   pd.DataFrame(col_sup_ftest[:,:self.n_components_],index =X.columns,columns=self.dim_index_)
                     }
    
    def _compute_quali_sup_stats(self,X,y=None):
        """Compute statistics supplementary categorical variables

        Parameters
        ----------
        self    :   An instance of class FAMD
        X       :   DataFrame (n_rows,n_columns)
        y : None
            y is ignored

        Return
        ------
        chi2_sup_test_  : chi-squared test
        mod_sup_coord_  : Supplementary categories coordinates
        mod_sup_cos2_   : Supplementary categories cosines
        mod_sup_disto_  : Supplementary categories distance
        mod_sup_stats_  : Statistic for supplementary categories (count and percentage)
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Chi-squared test between old and new categorical variables
        chi2_sup_stats = np.zeros(shape=(X.shape[1],len(self.quali_labels_)))
        chi2_sup_pvalue = np.zeros(shape=(X.shape[1],len(self.quali_labels_)))
        for i in np.arange(0,X.shape[1]):
            for j,lab in enumerate(self.quali_labels_):
                tab = pd.crosstab(X.iloc[:,i],self.quali_data_[lab])
                chi = st.chi2_contingency(tab)
                chi2_sup_stats[i,j],chi2_sup_pvalue[i,j]= chi[0],chi[1]
        
        # Dummies variables
        dummies = pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols in X.columns),axis=1)
        mod_sup_stats = dummies.agg(func=[np.sum,np.mean]).T
        n_k = dummies.sum(axis=0)
        p_k = dummies.mean(axis=0)
        mod_sup_labels = dummies.columns
        short_sup_labels = list([x.split("_",1)[-1] for x in mod_sup_labels])
        
        # Supplementary categories coordinates
        mod_sup_coord = pd.concat((pd.concat((pd.DataFrame(self.row_coord_,index=self.row_labels_,
                                                   columns=self.dim_index_),dummies[cols]),axis=1)
                                      .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)
        
        # Supplementary categories v-test
        mod_sup_vtest = pd.concat(((mod_sup_coord.loc[k,:]/np.sqrt((self.n_rows_-n_k[k])/((self.n_rows_-1)*n_k[k]))).to_frame(name=k).T for k in mod_sup_coord.index),
                                  axis=0)/np.sqrt(self.eig_[0])
        
        # Moyennes conditionnelles sur la variable Z
        mz_g = pd.concat((pd.concat((self.normalized_data_,dummies[cols]),axis=1)
                                    .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)

        # Distance des modalités à  l'origine
        mod_sup_disto = mapply(mz_g,lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary categories cos2
        mod_sup_cos2 = mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_).div(mod_sup_disto,axis="index")

        # Supplementary categories eta2 - correlation
        quali_sup_eta2 = pd.concat((mapply(mod_sup_coord.loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:],
                                           lambda x : x**2,axis=1,progressbar=False,n_workers=self.n_workers_)
                                           .mul(p_k.loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index)],axis="index")
                                           .div(self.eig_[0],axis="columns")
                                           .sum(axis=0).to_frame(name=cols).T for cols in X.columns),axis=0)
        
        # Cosinus carrés des variables qualitatives supplémentaires
        nb_mod = pd.Series([len(np.unique(X[[col]])) for col in X.columns],index=X.columns,name="count")
        quali_sup_cos2 = pd.concat(((quali_sup_eta2.loc[cols,:]/nb_mod[cols]).to_frame(name=cols).T for cols in quali_sup_eta2.index),axis=0)

        # Supplementary categories informations
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_disto_     =   np.array(mod_sup_disto)
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)

        self.mod_sup_labels_    =   mod_sup_labels
        self.short_sup_labels_  =   short_sup_labels

        # Categorical variables
        self.quali_sup_eta2_    =   np.array(quali_sup_eta2)
        self.quali_sup_cos2_    =   np.array(quali_sup_cos2)
        self.chi2_sup_test_     = {"statistic" : pd.DataFrame(chi2_sup_stats,index=X.columns,columns=self.quali_labels_),
                                    "pvalue"    : pd.DataFrame(chi2_sup_pvalue,index=X.columns,columns=self.quali_labels_)
                                    }

        return {"chi2"     :   self.chi2_sup_test_,
                    "coord"     :   pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
                     "cos2"     :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_), 
                     "dist"     :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=["dist"]),
                     "eta2"     :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
                     "vtest"    :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
                     }
    
    def transform(self,X):
        """Apply the dimensionality reduction on X

        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        X : DataFrame, shape (n_rows_sup, n_columns)
            New data, where n_row_sup is the number of supplementary
            row points and n_columns is the number of columns
            X rows correspond to supplementary row points that are 
            projected on the axes
            X is a table containing numeric values
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of float, shape (n_rows_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Store continuous and categorical variables
        X_sup_quant = X[self.quanti_labels_]
        X_sup_qual = X[self.quali_labels_]

        # Standardscaler numerical variable
        if self.normalize:
            Z1 = (X_sup_quant - self.means_)/self.std_ 
        else:
            Z1 = X_sup_quant - self.means_

        # Standardscaler categorical Variable
        n_rows = X_sup_qual.shape[0]
        n_cols = len(self.mod_labels_)
        Y = np.zeros((n_rows,n_cols))
        for i in np.arange(0,n_rows,1):
            values = [self.quali_labels_[k] +"_"+str(X_sup_qual.iloc[i,k]) for k in np.arange(0,len(self.quali_labels_))]
            for j in np.arange(0,n_cols,1):
                if self.mod_labels_[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)

        # New normalized data
        Z2 = mapply(row_sup_dummies,lambda x : (x - self.dummies_means_)/self.dummies_std_,axis=1,progressbar=False,n_workers=self.n_workers_)

        # Supplementary individuals coordinates
        row_sup_coord = np.dot(np.array(pd.concat([Z1,Z2],axis=1)),self.eigen_vectors_)
        
        return  row_sup_coord[:,:self.n_components_]
    
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        
        y : None
            y is ignored
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        self.fit(X)
        return self.row_coord_

######################################################################################################
#               Multiple Factor Analysis (MFA)
#####################################################################################################

# https://husson.github.io/MOOC_AnaDo/AFM.html
# https://math.institut-agro-rennes-angers.fr/fr/ouvrages/analyse-factorielle-multiple-avec-r
# https://eudml.org/subject/MSC/62H25

class MFA(BaseEstimator,TransformerMixin):
    """Multiple Factor Analysis (MFA)

    Performs Multiple Factor Analysis

    Parameters:
    ----------
    normalize : 
    n_components :
    groups : list of string
    groups : list of string
    groups_sup : list of string

    
    
    """


    def __init__(self,
                 normalize=True,
                 n_components=int|None,
                 groups=list[str]|None,
                 groups_sup = list[str]|None,
                 row_labels = list[str]|None,
                 parallelize=False):
        self.normalize = normalize
        self.n_components =n_components
        self.groups = groups
        self.groups_sup = groups_sup
        self.row_labels = row_labels
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
         # set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1
        
        # Check if groups is None 
        if self.groups is None:
            raise ValueError("Error : 'groups' must be assigned.")
        
        # Remove supplementary group
        self.groups_sup_ = self.groups_sup
        if self.groups_sup_ is not None:
            diff = [i for i in self.groups + self.groups_sup_ if i not in self.groups or i not in self.groups_sup_]
            if len(diff)==0:
                raise ValueError("Error : ")
            else:
                Xsup = X[self.groups_sup_]
                X_ = X[self.groups]
        else:
            X_ = X
        
        # Save data
        self.data_ = X
        self.active_data_ = X_

        # Compute stats
        self._compute_stats(X_)


        return self
    
    def _compute_stats(self,X):
        """
        
        """

        # Shape of X
        self.n_rows_, self.n_cols_ = X.shape

        # Set row labels
        self.row_labels_ = self.row_labels
        if ((self.row_labels_ is None) or (len(self.row_labels_) != self.n_rows_)):
            self.row_labels_ = ["row_" + str(i+1) for i in np.arange(0,self.n_rows_)]
        
        # Checks groups are provided
        self.groups_ = self._determine_groups(X=X)

        # Set columns labels - columns group labels
        self.col_labels_ = []
        self.col_group_labels_ = []
        for _, cols in self.groups_.items():
            for i in range(len(cols)):
                group = cols[i][0]
                col_label = cols[i][1]
                self.col_labels_.append(col_label)
                self.col_group_labels_.append(group)
    
        # Chack group types are consistent
        self.all_nums_ = dict()
        for name, cols in self.groups_.items():
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError(f'Not all columns in "{name}" group are of the same type')
            self.all_nums_[name] = all_num
        
        # Run a Factor Analysis in each group
        model = dict()
        for group, cols in self.groups_.items():
            if self.all_nums_[group]:
                fa = PCA(normalize=self.normalize,
                         n_components=self.n_components,
                         row_labels=X.loc[:,cols][group].index,
                         col_labels=X.loc[:,cols][group].columns,
                         parallelize=self.parallelize)
            else:
                raise NotImplementedError("Groups of non-numerical variables are not supported yet")
            model[group] = fa.fit(X.loc[:,cols][group])
            
        self.fa_model_ = model

        # Normalize data
        self.means_ = np.mean(X.values, axis=0).reshape(1,-1)
        if self.normalize:
            self.std_ = np.std(X.values,axis=0,ddof=0).reshape(1,-1)
            Z = (X - self.means_)/self.std_
        else:
            Z = X - self.means_
        
        # Ponderation
        Zb = pd.concat((mapply(Z.loc[:,cols],lambda x : x/np.sqrt(model[group].eig_[0][0]),axis=0,progressbar=False,n_workers=self.n_workers_) for group, cols in self.groups_.items()),axis=1)
        
        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        pca_model = PCA(normalize=False,
                        n_components=None,
                        row_labels=Zb.index,
                        col_labels=Zb.columns,
                        parallelize=self.parallelize).fit(Zb)

        # Number of components
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = pca_model.n_components_
        
        dim_index = ["Dim."+str(x+1) for x in np.arange(self.n_components_)]
        
        # Store all informations
        self.eig_ = pca_model.eig_[:,:self.n_components_]
        # Eigenvectors
        self.eigen_vectors_ = pca_model.eigen_vectors_[:,:self.n_components]
        # Row coordinates
        self.row_coord_ = pca_model.row_coord_[:,:self.n_components_]
        # Row contributions
        self.row_contrib_ = pca_model.row_contrib_[:,:self.n_components_]
        # Row - Quality of representation
        self.row_cos2_ = pca_model.row_cos2_[:,:self.n_components_]
        # Partial row coordinates
        #self.row_coord_partial_ = self._row_coord_partial(X,Zb,pca_model.eig_[0])
        self.dim_index_ = dim_index
        self.normalied_data_ = Z
        self.pnormalized_data_ = Zb

        ###### Columns informations ################################################
        # Columns coordinates
        self.col_coord_ = pca_model.col_coord_[:,:self.n_components_]
        # Columns contributions
        self.col_contrib_ = pca_model.col_contrib_[:,:self.n_components_]
        # Columns Quality of representations
        self.col_cos2_ = pca_model.col_cos2_
        # Correlation with axis
        self.col_cor_ = pca_model.col_cor_

        # Model Name
        self.model_ = "mfa"
    
    def _row_coord_partial(self,X,Zb,s):
        # 
        X = (X - X.mean()) / ((X - X.mean()) ** 2).sum() ** 0.5
        print(X.shape )
        Z = pd.concat((X.loc[:,cols][group]/self.fa_model_[group].eig_[0][0]**2) for group, cols in self.groups_.items())
        # Matrice des poids
        M = np.full(len(X), 1 / len(X))
        U = np.linalg.svd(Zb)[0] #[:,:self.n_components_]
        #s = np.sqrt(self.eig_[0])
        return len(self.groups_) * pd.concat(
            [
                self.add_index(
                    df=(Zb[g] @ Zb[g].T) @ (M[:, np.newaxis] ** (-0.5) * U * s**-1),
                    group_name=g,
                )
                for g, cols in self.groups_.items()
            ],
            axis="columns",
        )

    @staticmethod
    def add_index(df,group_name):
        df.columns = pd.MultiIndex.from_tuples([(group_name,col) for col in df.columns],names=("group","component"))
        return df


    def _determine_groups(self,X):
        """
        
        
        """

        if isinstance(self.groups,list):
            if not isinstance(X.columns,pd.MultiIndex):
                raise ValueError("Error : Groups have to be provided as a dict when X is not a MultiIndex")
            groups = { g: [(g, c) for c in X.columns.get_level_values(1)[X.columns.get_level_values(0) == g]] for g in self.groups}
        else:
            groups = self.groups
        
        return groups
    
    def _compute_groups_sup_coord(self,X):
        """
        """
        if not isinstance(X,pd.DataFrame):
            raise ValueError()
            

    def fit_transform(self,X,y=None):
        """
        
        
        """

        self.fit(X)
        return self.row_coord_
    
    def transform(self,X):
        """
        
        """
        raise NotImplementedError("Error : This method is not yet implemented")
        

        


########################################################################################################
#       Hierarchical Multiple Factor Analysis (HMFA)
#######################################################################################################

class HMFA(BaseEstimator,TransformerMixin):
    """
    
    
    
    """


    def __init__(self,n_components=None):
        self.n_components = n_components
    

    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not yet implemented.")

 
        





    








    
        