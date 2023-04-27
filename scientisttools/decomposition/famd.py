# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.stats as st
import pingouin as pg
from scientisttools.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

class FAMD(BaseEstimator,TransformerMixin):
    """Factor Analysis of Mixed Data

    Performs Factor Analysis of Mixed Data (FAMD) with supplementary 
    individuals, supplementary quantitative variables and supplementary
    categorical variables.
    
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
                 graph=False,
                 figsize=None):
        self.normalize =normalize
        self.n_components = n_components
        self.row_labels = row_labels
        self.quanti_labels = quanti_labels
        self.quali_labels = quali_labels
        self.row_sup_labels = row_sup_labels
        self.quanti_sup_labels = quanti_sup_labels
        self.quali_sup_labels = quali_sup_labels
        self.graph = graph
        self.figsize= figsize
    
    def fit(self,X):
        if not isinstance(X,pd.DataFrame):
            raise ValueError("Error : 'X' must be a data.frame")
        
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

        # Compute statistics
        self.n_rows_ = X_.shape[0]
        X_quant = X_.select_dtypes(include=np.number)
        X_qual = X_.select_dtypes(include=["object"])

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
        
        chi2_stats = np.zeros(shape=(len(self.quali_labels_),len(self.quali_labels_)))
        chi2_pvalue = np.zeros(shape=(len(self.quali_labels_),len(self.quali_labels_)))
        for i,lab1 in enumerate(self.quali_labels_):
            for j,lab2 in enumerate(self.quali_labels_):
                tab = pd.crosstab(X_.iloc[:,i],X_.iloc[:,j])
                chi = st.chi2_contingency(tab)
                chi2_stats[i,j],chi2_pvalue[i,j]= chi[0],chi[1]
            
        self.chi2_test_ = dict({"statistic": pd.DataFrame(chi2_stats,index=self.quali_labels_,columns=self.quali_labels_),
                                "pvalue" : pd.DataFrame(chi2_pvalue,index=self.quali_labels_,columns=self.quali_labels_)
                        })
        
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

        Z2 = mapply(dummies,lambda x: x/np.sqrt(self.dummies_means_.values),axis = 1,progressbar=False)

        Z = pd.concat([Z1,Z2],axis=1)

        # Distance between individuals
        row_dist = squareform(pdist(Z,metric='sqeuclidean'))

        # Distance between individuals and inertia center
        row_disto = (mapply(Z1,lambda x:np.sum(x**2),axis=1,progressbar=False) + 
                     mapply(dummies,lambda x:np.sum(1/self.dummies_means_.values*(x-self.dummies_means_.values)**2),
                            axis=1,progressbar=False))
        # Individuals weight
        row_weight = np.ones(self.n_rows_)/self.n_rows_

        # Individuals inertia
        row_inertie = row_disto*row_weight

        row_infos = np.c_[np.sqrt(row_disto),row_weight,row_inertie]

        ################################
        dummies_weight = dummies.div(n_k,axis="columns")

        mod_dist = self.n_rows_*squareform(pdist(dummies_weight.T,metric="sqeuclidean"))

        # Distance à l'origine
        mod_disto = mapply(dummies_weight,lambda x : np.sum(self.n_rows_*(x-row_weight)**2),axis=0,progressbar=False)

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
        col_cos2 = mapply(col_coord,lambda x : x**2, axis=1,progressbar=False)
        col_contrib = mapply(col_coord,lambda x : 100*x**2/res.eig_[0],axis=1,progressbar=False)

        # Test de significativité de Fisher
        col_ftest = mapply(col_coord,lambda x : (1/2)*np.sqrt(self.n_rows_-3)*np.log((1+x)/(1-x)),axis=0,progressbar=False)

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

        mod_cos2 = mapply(mod_coord,lambda x : x**2/(self.mod_infos_[:,0]**2), axis=0,progressbar=False)
        mod_contrib = mapply(coord_mod,lambda x : 100*x**2/res.eig_[0],axis = 1,progressbar=False)
        mod_vtest = mapply(mapply(mod_coord,lambda x : x*np.sqrt(((self.n_rows_-1)*Iq.values)/(self.n_rows_-Iq.values)),
                                  axis=0,progressbar=False),
                           lambda x : x/np.sqrt(res.eig_[0]),axis=1,progressbar=False)
        
        # Qualitative informations
        var_eta2 = pd.concat((mapply(coord_mod.loc[filter(lambda x: x.startswith(cols),coord_mod.index),:],
                                     lambda x : x**2,axis=1,progressbar=False).sum().to_frame(name=cols).T for cols in self.quali_labels_),axis=0)
        
        # Cosinus carrés des variables qualitatives
        denom = np.array([len(np.unique(Xq[[col]]))-1 for col in self.quali_labels_])
        var_cos2 = var_eta2.div(denom,axis="index")
        var_contrib = mapply(var_eta2,lambda x : 100*x/res.eig_[0],axis=1,progressbar=False)
    
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
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise ValueError("Error : 'X' must be a DataFrame.")

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
            values = [self.quali_labels_[k] +"_"+str(X.iloc[i,k]) for k in np.arange(0,len(self.quali_labels_))]
            for j in np.arange(0,n_cols,1):
                if self.mod_labels_[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)

        # New normalized Data
        Z2 = mapply(row_sup_dummies,lambda x : (x - self.dummies_means_)/self.dummies_std_,axis=1,progressbar=False)
        row_sup_coord = np.dot(pd.concat([Z1,Z2],axis=1),self.eigen_vectors_)

        # Save
        self.row_sup_coord_ = row_sup_coord[:,:self.n_components_]
    
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
        col_sup_corr = np.zeros((len(X.columns),len(self.quanti_labels_)))
        for i, lab in enumerate(X.columns):
            for j, name in enumerate(self.quanti_labels_):
                col_sup_corr[i,j] = st.pearsonr(X[lab],self.quanti_data_[name]).statistic

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

        return dict({"corr"     :   pd.DataFrame(self.col_sup_corr_, index=self.col_sup_labels_,columns=self.col_labels_),
                     "coord"    :   pd.DataFrame(self.col_sup_coord_,index=self.col_sup_labels_,columns=self.dim_index_),
                     "cos2"     :   pd.DataFrame(self.col_sup_cos2_, index = self.col_sup_labels_,columns=self.dim_index_),
                     "ftest"    :   pd.DataFrame(self.col_sup_ftest_,index = self.col_sup_labels_,columns=self.dim_index_)
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
        
        # Rapport de corrélation
        quali_sup_eta2 = pd.concat(((mapply(mod_sup_coord,lambda x : x**2,axis=0,progressbar=False).mul(p_k,axis="index")
                                                  .loc[filter(lambda x: x.startswith(cols),mod_sup_coord.index),:]
                                                  .sum(axis=0).to_frame(name=cols).T.div(self.eig_[0])) for cols in X.columns),axis=0)
        
        # Supplementary categories v-test
        mod_sup_vtest = mapply(mapply(mod_sup_coord,lambda x : x/np.sqrt((self.n_rows_-n_k)/((self.n_rows_-1)*n_k)),
                                        axis=0,progressbar=False),
                                 lambda x : x/np.sqrt(self.eig_[0]),axis=1,progressbar=False)
        
        # Moyennes conditionnelles sur la variable Z
        mz_g = pd.concat((pd.concat((self.normalized_data_,dummies[cols]),axis=1)
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
        self.mod_sup_coord_     =   np.array(mod_sup_coord)
        self.mod_sup_cos2_      =   np.array(mod_sup_cos2)
        self.mod_sup_disto_     =   np.array(mod_sup_disto)
        self.mod_sup_stats_     =   np.array(mod_sup_stats)
        self.mod_sup_vtest_     =   np.array(mod_sup_vtest)

        self.mod_sup_labels_    =   mod_sup_labels
        self.short_sup_labels_  =   short_sup_labels

        # Categorical variables
        self.quali_sup_eta2_    =   np.array(quali_sup_eta2)
        self.chi2_sup_test_ = dict({"statistic" : pd.DataFrame(chi2_sup_stats,index=X.columns,columns=self.quali_labels_),
                                    "pvalue"    : pd.DataFrame(chi2_sup_pvalue,index=X.columns,columns=self.quali_labels_)
                                    })

        return dict({"chi2"     :   self.chi2_sup_test_,
                    "coord"     :   pd.DataFrame(self.mod_sup_coord_,index=self.mod_sup_labels_,columns=self.dim_index_),
                     "cos2"     :   pd.DataFrame(self.mod_sup_cos2_,index=self.mod_sup_labels_,columns=self.dim_index_), 
                     "dist"     :   pd.DataFrame(self.mod_sup_disto_,index=self.mod_sup_labels_,columns=["dist"]),
                     "eta2"     :   pd.DataFrame(self.quali_sup_eta2_,index=self.quali_sup_labels_,columns=self.dim_index_),
                     "vtest"    :   pd.DataFrame(self.mod_sup_vtest_,index=self.mod_sup_labels_,columns=self.dim_index_)
                     })
    
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
            values = [self.quali_labels_[k] +"_"+str(X.iloc[i,k]) for k in np.arange(0,len(self.quali_labels_))]
            for j in np.arange(0,n_cols,1):
                if self.mod_labels_[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)

        # New normalized data
        Z2 = mapply(row_sup_dummies,lambda x : (x - self.dummies_means_)/self.dummies_std_,axis=1,progressbar=False)

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
    
    
 
        

        

