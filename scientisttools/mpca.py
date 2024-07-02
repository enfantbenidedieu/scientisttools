# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .pca import PCA
from .weightedcorrcoef import weightedcorrcoef
from .function_eta2 import function_eta2
from .revaluate_cat_variable import revaluate_cat_variable

from .recodevarfamd import recodevarfamd
from .recodecont import recodecont
from .recodecat import recodecat
from .splitmix import splitmix

class MPCA(BaseEstimator,TransformerMixin):
    """
    Mixed Principal Components Analysis (MPCA)
    ------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs principal component analysis of a set of individuals (observations) described by a mixture of qualitative and quantitative variables with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    >>> MPCA(n_components = 5,ind_weights = None,quanti_weights = None,quali_weights = None,ind_sup=None,quanti_sup=None,quali_sup=None,parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals
    
    `quanti_weights` : an optional quantitatives variables weights (by default, a list/tuple of 1 for uniform quantitative variables weights), the weights are given only for the active quantitative variables
    
    `quali_weights` : an optional qualitatives variables weights (by default, a list/tuple of 1 for uniform qualitative variables weights), the weights are given only for the active qualitative variables
    
    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `quanti_sup` : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    `quali_sup` : an integer or a list/tuple indicating the indexes of the categorical supplementary variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_`  : dictionary of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)

    `ind_` : dictionary of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    `ind_sup_` : dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `quali_var_` : dictionary of pandas dataframe with all the results for the categorical variables (coordinates, square cosine, contributions, v.test)

    `quali_sup_` : dictionary of pandas dataframe with all the results for the supplementary categorical variables (coordinates, square cosine, v.test)
    
    `quanti_var_` : dictionary of pandas dataframe with all the results for the quantitative variables (coordinates, correlation, square cosine, contributions)

    `quanti_sup_` : dictionary of pandas dataframe with all the results for the supplementary quantitative variables (coordinates, correlation, square cosine)

    `call_` : dictionary with some statistics

    `summary_quanti_` : descriptive statistics of quantitatives variables

    `summary_quali_` : statistics of categories variables

    `chi2_test_` : chi2 statistics test

    `model_` : string specifying the model fitted = 'mpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Abdesselam R. (2006), Analyse en Composantes Principales Mixtes, CREM UMR CNRS 6211
    
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Pages J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Rakotomalala, R (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_mpca_ind, get_mpca_var, get_mpca, summaryMPCA, dimdesc, predictMPCA, supvarMPCA, fviz_mpca_ind, fviz_mpca_col, fviz_mpca_mod, fviz_mpca_var, 

    Examples
    --------
    ```python
    >>> # Load cars dataset
    >>> from scientisttools import load_cars
    >>> cars = load_cars()
    >>> from scientisttools import MPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(cars)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 ind_weights = None,
                 quanti_weights = None,
                 quali_weights = None,
                 ind_sup=None,
                 quanti_sup=None,
                 quali_sup=None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.quanti_weights = quanti_weights
        self.quali_weights  = quali_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X, y=None):
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

        # Chack if X is a DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X.index.name = None

        # Set parallelize option
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Set supplementary qualitative variables labels
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup]
        else:
            quali_sup_label = None

        #  Set supplementary quantitative variables labels
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup]
        else:
            quanti_sup_label = None

        # Set supplementary individuals labels
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None
        
        # Store data
        Xtot = X.copy()

        # Drop supplementary qualitative columns
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        # Drop supplementary quantitative columns
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        # Drop supplementary individuals
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)
        
        # Mixed Principal Components Analysis
        rec = recodevarfamd(X)

        # Extract elements
        X = rec["X"]
        n_rows = rec["n"]
        n_cont = rec["k1"]
        n_cat = rec["k2"]
        X_quanti = rec["quanti"]
        X_quali = rec["quali"]
        dummies = rec["dummies"]
        nb_moda = rec["nb_moda"]

        # Check if no quantitatives variables
        if n_cont==0:
            raise TypeError("No quantitatives variables in X. X must be a mixed data")

        # Check if no qualitatives variables
        if n_cat == 0:
            raise TypeError("No qualitatives variables in X. X must be a mixed data")
        
        # Set individuals weights
        if self.ind_weights is None:
            ind_weights = np.ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind _weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        #########################################################################################
        #                                 For quantitatives variables
        #########################################################################################

        # Summary quantitatives variables
        summary_quanti = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti

        # Compute weighted average mean and standard deviation
        d1 = DescrStatsW(X_quanti,weights=ind_weights,ddof=0)
        means1 = pd.Series(d1.mean,index=X_quanti.columns,name="weight")

        # Center quantitatives variables
        Z1 = X_quanti - means1.values.reshape(1,-1)

        # Set quantitatives variables weights
        if self.quanti_weights is None:
            weights1 = np.ones(n_cont)
        elif not isinstance(self.quanti_weights,list):
            raise TypeError("'quanti_weights' must be a list of quantitatives weights")
        elif len(self.quanti_weights) != n_cont:
            raise TypeError(f"'quanti_weights' must be a list with length {n_cont}.")
        else:
            weights1 = np.array(self.quanti_weights)
        
        # Apply weighted correction
        weights1 = pd.Series(weights1,index=X_quanti.columns)

        ########################################################################################
        #                    For qualitatives variables
        ##########################################################################################
    
        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X_quali.columns.tolist():
            eff = X_quali[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")
        self.summary_quali_ = summary_quali
    
        # Chi2 statistic test
        if n_cat > 1:
            chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in np.arange(n_cat-1):
                for j in np.arange(i+1,n_cat):
                    tab = pd.crosstab(X_quali.iloc[:,i],X_quali.iloc[:,j])
                    chi = sp.stats.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_quali.columns[i],
                                            "variable2" : X_quali.columns[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[idx])
                    chi2_test = pd.concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            chi2_test["dof"] = chi2_test["dof"].astype("int")
            self.chi2_test_ = chi2_test
        
        # Set qualitative variables weights
        if self.quali_weights is None:
            quali_weights = pd.Series(np.ones(n_cat),index=X_quali.columns,name="weight").astype("float")
        elif not isinstance(self.quali_weights,pd.Series):
            raise ValueError("'quali_weights' must be a pandas series where index are qualitatives variables names and values are qualitatives variables weights.")
        else:
            quali_weights = self.quali_weights

        # Diagonal matrix of individuals weights
        D = np.diag(ind_weights)

        # Initialize dataframe
        dummies = pd.DataFrame().astype("int")
        Z2 = pd.DataFrame().astype("float")
        means2 = pd.Series().astype("float")
        weights2 = pd.Series(name="weight").astype("float")
        for col in X_quali.columns:
            Yl = pd.get_dummies(X_quali[col],dtype=int)
            Vx, Vylx = np.dot(np.dot(Z1.T,D),Z1), np.dot(np.dot(Yl.T,D),Z1)
            # Compute the mean
            Gl =  np.dot(np.dot(np.dot(np.dot(Vylx,np.linalg.pinv(Vx,hermitian=True)),Z1.T),D),np.ones(X.shape[0]))
            # Center the dummies table
            Zl = Yl - Gl
            # Concatenate
            Z2 = pd.concat((Z2,Zl),axis=1)
            dummies = pd.concat((dummies,Yl),axis=1)
            means2 = pd.concat((means2,pd.Series(Gl,index=Yl.columns)),axis=0)
            # Categories weights
            weights = pd.Series([quali_weights[col]]*Yl.shape[1],index=Yl.columns,name="weight")
            weights2 = pd.concat((weights2,weights),axis=0) 
        
        # Concatenate
        Z = pd.concat((Z1,Z2),axis=1)
        means = pd.concat((means1,means2),axis=0)
        var_weights = pd.concat((weights1,weights2),axis=0)

        # set maximum number of components
        max_components = min(n_rows-1,n_cont + dummies.shape[1] - n_cat)

        #################### Set number of components
        if self.n_components is None:
            n_components = min(n_rows-1,n_cont + dummies.shape[1] - n_cat)
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("'n_components' must be greater or equal than 1.")
        else:
            n_components = min(self.n_components, max_components)

        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "Z" : Z,
                      "ind" : X.index,
                      "quali" : X_quali.columns,
                      "quanti" : X_quanti.columns,
                      "ind_sup" : ind_sup_label,
                      "quanti_sup" : quanti_sup_label,
                      "quali_sup" : quali_sup_label,
                      "means" : means,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                      "var_weights" : var_weights,
                      "dummies" : dummies,
                      "rec" : rec}

        # Normalized PCA
        global_pca = PCA(standardize=True,n_components=n_components,ind_weights=ind_weights.tolist(),var_weights=var_weights.values.tolist()).fit(Z)

        ###########################################################################################################
        #                            Compute supplementary individuals informations
        ##########################################################################################################
        # Statistics for supplementary individuals
        if self.ind_sup is not None:
            ##### Prepare supplementary columns
            X_ind_sup_quant = X_ind_sup[X_quanti.columns]
            X_ind_sup_qual = X_ind_sup[X_quali.columns]

            # Revaluate categories
            X_ind_sup_qual = revaluate_cat_variable(X_ind_sup_qual)

            # Dummies with qualitatives variables in supplementary individuals
            Y = pd.DataFrame(np.zeros((X_ind_sup.shape[0],dummies.shape[1])),columns=dummies.columns,index=X_ind_sup.index)
            for i in np.arange(0,X_ind_sup.shape[0],1):
                values = [str(X_ind_sup_qual.iloc[i,k]) for k in np.arange(X_quali.shape[1])]
                for j in np.arange(dummies.shape[1]):
                    if dummies.columns[j] in values:
                        Y.iloc[i,j] = 1

            # Concatenate and centered the data
            Z_ind_sup = pd.concat((X_ind_sup_quant,Y),axis=1) - means.values.reshape(1,-1)

            # Concatenate
            Z_ind_sup = pd.concat((Z,Z_ind_sup),axis=0)

            # Update PCA using supplementary individuals
            global_pca = PCA(standardize=True,n_components=n_components,ind_weights=ind_weights.tolist(),var_weights=var_weights.values.tolist(),ind_sup=ind_sup).fit(Z_ind_sup)

            # Extract all elements
            ind_sup_coord = global_pca.ind_sup_["coord"].iloc[:,:n_components]
            ind_sup_cos2 = global_pca.ind_sup_["cos2"].iloc[:,:n_components] 
            ind_sup_dist = global_pca.ind_sup_["dist"]

            # Save all informations
            self.ind_sup_ = {"coord" : ind_sup_coord, "cos2" : ind_sup_cos2, "dist" : ind_sup_dist}
        
        ##########################################################################################################
        #                         Compute supplementary quantitatives variables statistics
        ###########################################################################################################
        # Statistics for supplementary quantitatives variables
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            # Recode to fill NA with means
            X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]
            
            # Summary statistics with supplementary quantitatives variables.
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            summary_quanti_sup.insert(0,"group","sup")
            self.summary_quanti_.insert(0,"group","active")
            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)
            
            # Centered
            d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = X_quanti_sup - d2.mean.reshape(1,-1)

            # Concatenate
            Z_quanti_sup = pd.concat((Z,Z_quanti_sup),axis=1)

            # Find supplementary quantitatives columns index
            index = [Z_quanti_sup.columns.tolist().index(x) for x in X_quanti_sup.columns.tolist()]
            
            # Update PCA with supplementary qualitatives variables
            global_pca = PCA(standardize=True,n_components=n_components,ind_weights=ind_weights.tolist(),var_weights=var_weights.values.tolist(),quanti_sup=index).fit(Z_quanti_sup)
            
            # Extract all informations
            quanti_sup_coord = global_pca.quanti_sup_["coord"].iloc[:,:n_components]
            quanti_sup_cos2 = global_pca.quanti_sup_["cos2"].iloc[:,:n_components]
            
            # Store informations
            self.quanti_sup_ = {"coord" : quanti_sup_coord, "cor": quanti_sup_coord, "cos2" : quanti_sup_cos2}

        ##########################################################################################################
        #                         Compute supplementary qualitatives variables statistics
        ###########################################################################################################
        # Statistics for supplementary qualitatives variables
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            
            # Chi-squared test between new categorie
            if X_quali_sup.shape[1] > 1:
                chi2_sup_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                cpt = 0
                for i in range(X_quali_sup.shape[1]-1):
                    for j in range(i+1,X_quali_sup.shape[1]):
                        tab = pd.crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j])
                        chi = sp.stats.chi2_contingency(tab,correction=False)
                        row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns[i],
                                    "variable2" : X_quali_sup.columns[j],
                                    "statistic" : chi.statistic,
                                    "dof"       : chi.dof,
                                    "pvalue"    : chi.pvalue},index=[cpt])
                        chi2_sup_test = pd.concat([chi2_sup_test,row_chi2],axis=0)
                        cpt = cpt + 1
                chi2_sup_test["dof"] = chi2_sup_test["dof"].astype("int")
            
            # Chi-squared between old and new qualitatives variables
            chi2_sup_test2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
            cpt = 0
            for i in range(X_quali_sup.shape[1]):
                for j in range(X_quali.shape[1]):
                    tab = pd.crosstab(X_quali_sup.iloc[:,i],X_quali.iloc[:,j])
                    chi = sp.stats.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns[i],
                                            "variable2" : X_quali.columns[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[cpt])
                    chi2_sup_test2 = pd.concat([chi2_sup_test2,row_chi2],axis=0,ignore_index=True)
                    cpt = cpt + 1
            chi2_sup_test2["dof"] = chi2_sup_test2["dof"].astype("int")
            
            ###### Add
            if n_cat > 1:
                if X_quali_sup.shape[1] > 1 :
                    chi2_sup_test = pd.concat([chi2_sup_test,chi2_sup_test2],axis=0,ignore_index=True)
                else:
                    chi2_sup_test = chi2_sup_test2
                self.chi2_test_ = pd.concat((self.chi2_test_,chi2_sup_test),axis=0,ignore_index=True)
            else:
                if X_quali_sup.shape[1] > 1 :
                    self.chi2_test_ = chi2_sup_test2
            
            #################################### Summary quali
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            summary_quali_sup.insert(0,"group","sup")

            #########
            self.summary_quali_.insert(0,"group","active")
            self.summary_quali_ = pd.concat([self.summary_quali_,summary_quali_sup],axis=0,ignore_index=True)

            ##########################################################################################################################
            #
            #########################################################################################################################
            Z_quali_sup = pd.concat((Z,X_quali_sup),axis=1)
            # Find supplementary quantitatives columns index
            index = [Z_quali_sup.columns.tolist().index(x) for x in X_quali_sup.columns]

            # Update PCA with supplementary qualitatives variables
            global_pca = PCA(standardize=True,n_components=n_components,ind_weights=ind_weights.tolist(),var_weights=var_weights.values.tolist(),quali_sup=index).fit(Z_quali_sup)
            
            # Extract all elements
            quali_sup_coord = global_pca.quali_sup_["coord"].iloc[:,:n_components]
            quali_sup_cos2 = global_pca.quali_sup_["cos2"].iloc[:,:n_components]
            quali_sup_vtest = global_pca.quali_sup_["vtest"].iloc[:,:n_components]
            quali_sup_dist = global_pca.quali_sup_["dist"]
            quali_sup_eta2 = global_pca.quali_sup_["eta2"].iloc[:,:n_components]

            # Store all elements
            self.quali_sup_ = {"coord" : quali_sup_coord, "cos2" : quali_sup_cos2, "vtest": quali_sup_vtest, "dist": quali_sup_dist, "eta2" : quali_sup_eta2}
        
        # Store singular value decomposition
        vs = global_pca.svd_["vs"][:max_components]
        U = global_pca.svd_["U"][:,:n_components]
        V = global_pca.svd_["V"][:,:n_components]
        
        self.svd_ = {"vs" : vs, "U" : U, "V" : V}

        # Eigen - values
        eigen_values = global_pca.svd_["vs"][:max_components]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        ## Individuals informations
        # Coordinates
        ind_coord = global_pca.ind_["coord"].iloc[:,:n_components]
        # Square cosinus
        ind_cos2  = global_pca.ind_["cos2"].iloc[:,:n_components]
        # Relative contributions
        ind_contrib = global_pca.ind_["contrib"].iloc[:,:n_components]
        # Additionals informations
        ind_infos = global_pca.ind_["infos"]

        # Store informations
        self.ind_ = {"coord": ind_coord, "cos2" :ind_cos2, "contrib" : ind_contrib,"infos" : ind_infos}

        ## Quantitatives variables informations
        # Coordinates
        quanti_coord =  global_pca.var_["coord"].iloc[:n_cont,:n_components]
        # Square cosinus
        quanti_cos2 = global_pca.var_["cos2"].iloc[:n_cont,:n_components]
        # Relative contributions
        quanti_contrib = global_pca.var_["contrib"].iloc[:n_cont,:n_components]

        # Store all informations
        self.quanti_var_ = {"coord" : quanti_coord, "contrib" : quanti_contrib,"cor":quanti_coord,"cos2" : quanti_cos2}
        
        ## Qualitatives variables informations
        # Coordinates
        quali_coord = global_pca.var_["coord"].iloc[n_cont:,:n_components]
        # Square cosinus
        quali_cos2 = global_pca.var_["cos2"].iloc[n_cont:,:n_components]
        # Relative contributions
        quali_contrib = global_pca.var_["contrib"].iloc[n_cont:,:n_components]
        # Value - test
        I_k = dummies.sum(axis=0)
        quali_vtest = pd.concat(((quali_coord.loc[k,:]*np.sqrt(((n_rows-1)*I_k[k])/(n_rows-I_k[k]))).to_frame(k).T for k in dummies.columns),axis=0)
        quali_vtest = mapply(quali_vtest,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
        
        # Store all informations
        self.quali_var_ = {"coord" : quali_coord, "contrib" : quali_contrib, "cos2" : quali_cos2,"vtest":quali_vtest}

        # Square correlation ratio
        quali_var_eta2 = pd.concat((function_eta2(X=X_quali,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali.columns),axis=0)
        
        # Contributions des variables qualitatives
        quali_var_contrib = mapply(quali_var_eta2,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        
        # Cosinus carrés des variables qualitatives
        quali_var_cos2 = pd.concat((((quali_var_eta2.loc[col,:]**2)/(nb_moda[col]-1)).to_frame(name=col).T for col in X_quali.columns),axis=0)
        
        ######################################## Global variables informations ######################################
        var_coord = pd.concat((quanti_cos2,quali_var_eta2),axis=0)
        var_contrib = pd.concat((quanti_contrib,quali_var_contrib),axis=0)
        var_cos2 = pd.concat((quanti_cos2**2,quali_var_cos2),axis=0)

        # Store all informations
        self.var_ = {"coord" : var_coord, "contrib" : var_contrib, "cos2" : var_cos2}

        self.model_ = "mpca"

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
        return self.ind_["coord"]

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

        # check if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X.index.name = None
        
        # Split data
        X_quanti = splitmix(X)["quanti"]
        X_quali = splitmix(X)["quali"]

        # Revaluate
        X_quali = revaluate_cat_variable(X_quali)

        # Standardscaler according to PCA
        means = self.call_["Z"].mean(axis=0)
        std = self.call_["Z"].std(axis=0,ddof=0)
        dummies = self.call_["dummies"]
        n_components = self.call_["n_components"]
        
        # Dummies encoding
        Y = pd.DataFrame(np.zeros((X.shape[0],dummies.shape[1])),index=X.index,columns=dummies.columns)
        for i in np.arange(X.shape[0]):
            values = [str(X_quali.iloc[i,k]) for k in np.arange(X_quali.shape[1])]
            for j in np.arange(dummies.shape[1]):
                if dummies.columns[j] in values:
                    Y.iloc[i,j] = 1

        # Concatenate and centered according to PCAMIX
        Z = pd.concat((X_quanti,Y),axis=1) - self.call_["means"].values.reshape(1,-1)

        # Standardscaler according to PCA
        Z = (Z - means.values.reshape(1,-1))/std.values.reshape(1,-1)

        ## Multiply by columns weight & Apply transition relation
        coord = Z.apply(lambda x : x*self.call_["var_weights"].values,axis=1).dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        return coord

def predictMPCA(self,X=None):
    """
    Predict projection for new individuals with Mixed Principal Component Analysis (MPCA)
    -------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and square distance to origin of new individuals with Mixed Principal Component Analysis (MPCA)

    Usage
    -----
    ```python
    >>> predictMPCA(self,X=None)
    ```

    Parameters
    ----------
    `self` : an object of class MPCA

    `X` : pandas/polars dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    dictionary of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : square cosinus of the new individuals

    `dist` : square distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    Examples
    --------
    ```python
    >>> # Load cars dataset
    >>> from scientisttools import load_cars
    >>> cars = load_cars()
    >>> from scientisttools import MPCA, predictMPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(cars)
    >>> predict = predictMPCA(res_mpca, X=cars)
    ```
    """
    # Check if self is an object of class MPCA
    if self.model_ != "mpca":
        raise TypeError("'self' must be an object of class MPCA")
    
    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()
    
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Set index name as None
    X.index.name = None

    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Split data
    X_quanti = splitmix(X)["quanti"]
    X_quali = splitmix(X)["quali"]

    # Revaluate
    X_quali = revaluate_cat_variable(X_quali)

    # Standardscaler according to PCA
    means = self.call_["Z"].mean(axis=0)
    std = self.call_["Z"].std(axis=0,ddof=0)
    dummies = self.call_["dummies"]
    n_components = self.call_["n_components"]
    var_weights = self.call_["var_weights"].values
    
    # Dummies encoding
    Y = pd.DataFrame(np.zeros((X.shape[0],dummies.shape[1])),index=X.index,columns=dummies.columns)
    for i in np.arange(X.shape[0]):
        values = [str(X_quali.iloc[i,k]) for k in np.arange(X_quali.shape[1])]
        for j in np.arange(dummies.shape[1]):
            if dummies.columns[j] in values:
                Y.iloc[i,j] = 1

    # Concatenate and centered according to PCAMIX
    Z = pd.concat((X_quanti,Y),axis=1) - self.call_["means"].values.reshape(1,-1)

    # Standardscaler according to PCA
    Z = (Z - means.values.reshape(1,-1))/std.values.reshape(1,-1)

    # Coordinates
    coord = Z.apply(lambda x : x*var_weights,axis=1).dot(self.svd_["V"][:,:n_components])
    coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

    #  Square distance to origin
    dist2 = mapply(Z,lambda  x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    dist2.name = "Sq. Dist."

    # Square cosinus
    cos2 = mapply(coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)

    # Store all informations
    res = {"coord" : coord, "cos2" : cos2,"dist" : dist2}
    return res

def supvarMPCA(self,X_quanti_sup=None, X_quali_sup=None):
    """
    Supplementary variables in Mixed Principal Components Analysis (MPCA)
    ---------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Mixed Principal Components Analysis (MPCA)

    Usage
    -----
    ```python
    >>> supvarMPCA(self,X_quanti_sup=None, X_quali_sup=None)
    ```

    Parameters
    ----------
    `self` : an object of class MPCA

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives variables (default = None)

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives variables (default = None)

    Returns
    -------
    dictionary of dictionary containing the results for supplementary variables including : 

    `quanti` : dictionary containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : dictionary containing the results of the supplementary qualitatives/categories variables including :
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
    >>> # Load cars dataset
    >>> from scientisttools import load_cars
    >>> cars = load_cars()
    >>> from scientisttools import MPCA, supvarMPCA, splitmix
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(cars)
    >>> X_quant = splitmix(cars)["quanti"]
    >>> X_qual = splitmix(cars)["quali"]
    >>> supvar = supvarMPCA(res_mpca, X_quanti_sup=X_quant,X_quali_sup=X_qual)
    ```
    """
    # Check if self is and object of class MPCA
    if self.model_ != "mpca":
        raise TypeError("'self' must be an object of class MPCA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    ind_weights = self.call_["ind_weights"].values
    n_components = self.call_["n_components"]

    # Extract X for PCA
    X = self.call_["Z"]

    ########################################################################################################################
    #                                          For supplementary quantitatives variables
    #########################################################################################################################
    # Supplementary quantitatives variables statistics
    if X_quanti_sup is not None:
        # check if X_quanti_sup is an instance of polars dataframe
        if isinstance(X_quanti_sup,pl.DataFrame):
            X_quanti_sup = X_quanti_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,pd.Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X_quanti_sup is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X_quanti_sup.index.name = None
        
        # Transform to float
        X_quanti_sup = X_quanti_sup.astype("float")

        # Recode variables
        X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]

        # Compute weighted average and standard deviation
        d1 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
    
        # Standardization
        Z_quanti_sup = (X_quanti_sup - d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)

        # Coordinates
        quanti_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(self.svd_["U"][:,:n_components])
        quanti_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Square distance
        quanti_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        quanti_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_cor)

        # Square cosinus
        quanti_cos2 = mapply(quanti_coord,lambda x : (x**2)/np.sqrt(quanti_dist2),axis=0,progressbar=False,n_workers=n_workers)       

        # Store all informations
        quanti_sup =  {"coord": quanti_coord, "cor" : quanti_coord, "cos2" : quanti_cos2}
    else:
        quanti_sup = None
    
    ###########################################################################################################################
    #                                                   For supplementary qualitatives variables
    ###########################################################################################################################
    # Supplementary qualitatives statistics
    if X_quali_sup is not None:
        # check if X_quali_sup is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X_quali_sup.index.name = None
        
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Extract informations
        var_weights = self.call_["var_weights"].values
        means = self.call_["Z"].mean(axis=0).values
        std = self.call_["Z"].std(axis=0,ddof=0).values

        # Square correlation ratio
        quali_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

        # Barycenter
        barycentre = pd.DataFrame().astype("float")
        n_k = pd.Series().astype("float")
        for col in X_quali_sup.columns:
            vsQual = X_quali_sup[col]
            modalite, counts = np.unique(vsQual, return_counts=True)
            n_k = pd.concat([n_k,pd.Series(counts,index=modalite)],axis=0)
            bary = pd.DataFrame(index=modalite,columns=self.call_["Z"].columns)
            for mod in modalite:
                idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
                bary.loc[mod,:] = np.average(self.call_["Z"].iloc[idx,:],axis=0,weights=ind_weights[idx])
            barycentre = pd.concat((barycentre,bary),axis=0)
        
        # Standardization
        bary = (barycentre - means.reshape(1,-1))/std.reshape(1,-1)

        # Square distance
        quali_dist2  = mapply(bary, lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        quali_dist2.name = "Sq. Dist."

        # Coordinates
        quali_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        quali_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Square cosinus
        quali_cos2 = mapply(quali_coord, lambda x : (x**2)/quali_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # Value - test
        quali_vtest = mapply(quali_coord,lambda x : x/self.svd_["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)
        quali_vtest = pd.concat(((quali_vtest.loc[k,:]/np.sqrt((n_rows-n_k[k])/((n_rows-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

        # Store all informations
        quali_sup = {"coord" : quali_coord,"cos2" : quali_cos2,"vtest" : quali_vtest,"dist" : quali_dist2,"eta2" : quali_eta2}
    else:
        quali_sup = None
    
    # Store all informations
    res = {"quanti" : quanti_sup, "quali" : quali_sup}
    return res