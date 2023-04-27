# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st
import functools
import scipy as sp
import matplotlib.pyplot as plt
from scientisttools.utils import from_dummies
from mapply.mapply import mapply
from functools import reduce
from scipy.spatial.distance import pdist,squareform
from scientisttools.graphics import plotMCA
from sklearn.base import BaseEstimator, TransformerMixin

def _mul(*args):
    """An internal method to multiply matrices."""
    return functools.reduce(np.dot,args)

class MCA(BaseEstimator,TransformerMixin):

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
                 graph=True,
                 figsize=None):
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
        self.graph = graph
        self.figsize = figsize

    def fit(self, X, y=None):
        """
        
        """
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")


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

        self.quanti_sup_coord_ = None
        self.quanti_sup_cos2_ = None

        self.quali_sup_coord_ = None
        self.quali_sup_cos2_ = None
        self.quali_sup_eta2_ = None
        self.quali_sup_disto_ = None
        self.quali_sup_vtest_ = None

        self.var_sup_eta2_ = None

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
        
        if self.graph:
            fig, (axe1,axe2,axe3) = plt.subplots(1,3,figsize=self.figsize)
            plotMCA(self,choice="ind",repel=True,ax=axe1)
            plotMCA(self,choice="mod",repel=True,ax=axe2)
            plotMCA(self,choice="var",repel=True,ax=axe3,xlim=(0,1),ylim=(0,1))

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
        dummies = (pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols 
                        in (X.columns if self.var_labels_ is None else self.var_labels_))
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
        var_eta2 = mapply(mod_coord_df,lambda x : x**2,axis=0,progressbar=False).mul(dummies_mean, axis='index')
        var_eta2 = pd.concat((mapply(var_eta2.loc[filter(lambda x: x.startswith(cols),var_eta2.index),:],lambda x : np.sum(x),
                             axis=0,progressbar=False).to_frame(name=cols).T for cols in  self.var_labels_),axis=0)
    
        # Cosinus carrés des variables qualitatives
        denom = np.array([len(np.unique(self.original_data_[[col]]))-1 for col in self.var_labels_])
        var_cos2 = var_eta2.div(denom,axis="index")
        var_contrib = mapply(var_eta2,lambda x : 100*x/self.eig_[0],axis=1,progressbar=False)

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

        # Distance du chi2 entre les individus
        row_dist = squareform(pdist(self.disjonctif_/self.n_vars_,metric="seuclidean",V=ind_moyen)**2)

        # Distance des observations à l'origine
        row_disto = mapply(self.disjonctif_,lambda x : np.sum((1/ind_moyen)*(x/self.n_vars_ - ind_moyen)**2),axis=1,progressbar=False)
        
        # Poids des observations
        row_weight = np.ones(self.n_rows_)/self.n_rows_

        # Inertie des observations
        row_inertia = row_disto*row_weight

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
        row_sup_profil = (mapply(row_sup_dummies,lambda x : x/np.sum(x),axis=1,progressbar=False)
                                .dot(self.mod_coord_)/np.sqrt(self.eig_[0]))
        
        self.row_sup_coord_ = np.array(row_sup_profil)
        self.row_sup_cos2_ = np.apply_along_axis(lambda x : x**2/np.linalg.norm(self.row_sup_coord_,axis=1)**2,
                                                    axis=0,arr=self.row_sup_coord_)
    
    def _compute_quali_sup_stats(self,X,y=None):
        """Find the supplementary categorical columns factor
        
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

        mod_sup_coord = mapply(dummies,lambda x : x/np.sum(x),axis=0,progressbar=False).T.dot(self.row_coord_)/np.sqrt(self.eig_[0])

        # Rapport de corrélation
        """
        quali_sup_eta2 = pd.concat(((mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False).mul(p_k,axis="index")
                                                  .loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:]
                                                  .sum(axis=0).to_frame(name=cols).T.div(self.eig_[0])) for cols in X.columns),axis=0)
        """

        mod_sup_cos2 = mapply(mod_sup_coord,lambda x: x**2/np.linalg.norm(mod_sup_coord,axis=1)**2,axis=0,progressbar=False)

        mod_sup_disto = (1/p_k)-1
        mod_sup_vtest = mapply(mod_sup_coord,lambda x : x*np.sqrt(((self.n_rows_-1)*n_k.values)/(self.n_rows_ - n_k.values)),axis=0,progressbar=False)

        # Store supplementary categories informations
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_disto_     =   np.array(mod_sup_disto)
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)
       
        self.mod_sup_labels_ = mod_sup_labels
        self.short_sup_labels_ = short_sup_labels

        return dict({"coord"   :   mod_sup_coord,
                     "cos2"    :   mod_sup_cos2,
                     "dist"    :   mod_sup_disto.to_frame("Dist"),
                     "stats"   :   mod_sup_stats,
                     "vtest"   :   mod_sup_vtest})  
    
    def _compute_quanti_sup_stats(self,X,y=None):
        """Find the supplementary quantitative columns factor
        
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
        quanti_sup_coord = np.transpose(np.corrcoef(x=self.row_coord_,y=X.values,rowvar=False)[:self.n_components_,self.n_components_:])

        # Supplementary quantitatives cos2
        quanti_sup_cos2 = np.apply_along_axis(func1d=lambda x : x**2,arr = quanti_sup_coord,axis=0)

        # Store supplementary quantitatives informations
        self.quanti_sup_coord_  =   quanti_sup_coord[:,:self.n_components_]
        self.quanti_sup_cos2_   =   quanti_sup_cos2[:,:self.n_components_]

        return dict({"coord"    :   quanti_sup_coord[:,:self.n_components_],
                     "cos2"     :   quanti_sup_cos2[:,:self.n_components_]})
    
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

        self._compute_row_sup_stats(X)
        return self.row_sup_coord_
    
    def fit_transform(self,X,y=None):
        """
        
        Return
        ------
        """

        self.fit(X)
        return self.row_coord_









