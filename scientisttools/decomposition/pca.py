# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from mapply.mapply import mapply
import pingouin as pg
from scientisttools.graphics import plotPCA
from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator,TransformerMixin):
    """Principal Component Analysis

    This is a standard Principal Component Analysis implementation
    bases on the Singular Value Decomposition

    Performs Principal Component Analysis (PCA) with supplementary 
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

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
    
    graph : boolean

    figsize : tuple or None
    
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
    
    quanti_sup_labels_ : arrays f strings or None
        Labels of quantitative supplementary variables
    
    quali_sup_labels
    
    col_labels_short_ : array of strings
        Short labels for the columns.
        Useful only for MCA, which inherits from Base class. In that
        case, the short labels for the columns at not prefixed by the
        names of the variables.
    
    eig_ : array of float
        A 4 x n_components_ matrix containing all the eigenvalues
        (1st row), difference (2nd row) the percentage of variance (3rd row) and the
        cumulative percentage of variance (4th row).
    
    eigen_vectors_ : array of float
        Eigen vectors extracted from the Principal Components Analysis.
    
    row_coord_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row coordinates.
    
    col_coord_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the column
        coordinates.
        
    row_contrib_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    col_contrib_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the column
        contributions.
    
    row_cos2_ : ndarray of shape (n_rows,n_components_)
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_cos2_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the column
        cosines.
    
    col_cor_ : ndarray of shape (n_columns,n_components_)
        A n_columns x n_components_ matrix containing the correlations
        between variables (= columns) and axes.
    
    means_ : ndarray of shape (n_columns,)
        The mean for each variable (= for each column).
    
    std_ : ndarray of shape (n_columns,)
        The standard deviation for each variable (= for each column).
    
    ss_col_coord_ : ndarray of shape (n_columns,)
        The sum of squared of columns coordinates.
    
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
                 graph=False,
                 figsize=None):
        self.normalize = normalize
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.graph = graph
        self.figsize = figsize

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
        
        # Store data and active data
        self.data_ = X
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
        
        if self.graph:
            fig, axe = plt.subplots(1,2,figsize=self.figsize)
            plotPCA(self,choice="ind",repel=True,ax=axe[0])
            plotPCA(self,choice="var",repel=True,ax=axe[1],xlim=(-1.1,1.1),ylim=(-1.1,1.1))

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

        self.row_infos_ = row_infos
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
        self    :   An instance of class FAMD
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
        col_sup_corr = np.zeros((len(X.columns),len(self.col_labels_)))
        for i, lab in enumerate(X.columns):
            for j, name in enumerate(self.col_labels_):
                col_sup_corr[i,j] = st.pearsonr(X[lab],self.active_data_[name]).statistic

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

        return dict({"corr"     :  pd.DataFrame(self.col_sup_corr_,index=self.quanti_sup_labels_,columns=self.col_labels_), 
                    "coord"     :  pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
                     "cos2"     :  pd.DataFrame(self.col_sup_cos2_,index=self.col_sup_labels_,columns=self.dim_index_),
                     "ftest"    :  pd.DataFrame(self.col_sup_ftest_,index=self.col_sup_labels_,columns=self.dim_index_)
        })
    
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
        quali_sup_eta2 = pd.concat(((mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False).mul(p_k,axis="index")
                                                  .loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:]
                                                  .sum(axis=0).to_frame(name=cols).T.div(self.eig_[0])) for cols in X.columns),axis=0)
        
        # Supplementary categories v-test
        mod_sup_vtest = mapply(mapply(mod_sup_coord,lambda x : x/np.sqrt((self.n_rows_-n_k)/((self.n_rows_-1)*n_k)),
                                        axis=0,progressbar=False),
                                 lambda x : x/np.sqrt(self.eig_[0]),axis=1,progressbar=False)
        
        # Moyennes conditionnelles sur la variable Z
        mz_g = pd.concat((pd.concat((pd.DataFrame(self.normalized_data_,index=self.row_labels_,
                                                   columns=self.col_labels_),dummies[cols]),axis=1)
                                    .groupby(cols).mean().iloc[1,:].to_frame(name=cols).T for cols in dummies.columns),axis=0)

        # Distance des modalités à  l'origine
        mod_sup_disto = mapply(mz_g,lambda x : np.sum(x**2),axis=1,progressbar=False)

        # Supplementary categories cos2
        mod_sup_cos2 = mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False).div(mod_sup_disto,axis="index")

        # Supplementary categories eta2 - correlation
        quali_sup_eta2 = pd.concat((mapply(mod_sup_coord.loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:],
                                           lambda x : x**2,axis=1,progressbar=False)
                                           .mul(p_k.loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index)],axis="index")
                                           .div(self.eig_[0],axis="columns")
                                           .sum(axis=0).to_frame(name=cols).T for cols in X.columns),axis=0)

        # Supplementary categories informations
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_disto_     =   np.array(mod_sup_disto)
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)

        self.mod_sup_labels_    =   mod_sup_labels
        self.short_sup_labels_  =   short_sup_labels

        # Supplementary qualitatives variables
        self.quali_sup_eta2_    =   quali_sup_eta2

        return dict({"stats"    :   pd.DataFrame(self.mod_sup_stats_,columns=["n(k)","p(k)"],index=self.mod_sup_labels_),
                    "coord"     :  pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
                     "cos2"     :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_), 
                     "dist"     :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=["dist"]),
                     "eta2"     :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
                     "vtest"    :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
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

        # Store all informations
        self.row_sup_coord_     =   row_sup_coord[:,:self.n_components_]
        self.row_sup_cos2_      =   row_sup_cos2[:,:self.n_components_]

        return dict({"coord"    :   row_sup_coord[:,:self.n_components_],
                     "cos2"     :   row_sup_cos2[:,:self.n_components_]})


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