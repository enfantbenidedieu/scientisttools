# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .function_eta2 import function_eta2
from .svd_triplet import svd_triplet
from .revaluate_cat_variable import revaluate_cat_variable
from .recodecont import recodecont
from .recodevar import recodevar
from .splitmix import splitmix

class PCAMIX(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis of Mixed Data (PCAMIX)
    ---------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Principal Components Analysis of Mixed Data (PCAMIX) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables. 
    
    Details
    -------

    PCAMIX includes standard Principal Component Analysis (PCA) and Multiple Correspondence Analysis (MCA) as special cases. If all variables are quantitative, standard PCA is performed.
    if all variables are qualitative, then standard MCA is performed.

    Missing values are replaced by means for quantitative variables. Note that, when all the variables are qualitative, the factor coordinates of the individuals are equal to the factor scores
    of standard MCA times squares root of J (the number of qualitatives variables) and the eigenvalues are then equal to the usual eigenvalues of MCA times J.
    When all the variables are quantitative, PCAMIX gives exactly the same results as standard PCA.

    Usage
    -----

    PCAMIX(n_components = 5,ind_weights = None,quanti_weights = None,quali_weights = None,ind_sup=None,quanti_sup=None,quali_sup=None,parallelize = False).fit(X)

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals
    
    quanti_weights : an optional quantitatives variables weights (by default, a list/tuple of 1 for uniform quantitative variables weights), the weights are given only for the active quantitative variables
    
    quali_weights : an optional qualitatives variables weights (by default, a list/tuple of 1/(number of active qualitative variable) for uniform qualitative variables weights), the weights are given only for the active qualitative variables
    
    ind_sup : a list/tuple indicating the indexes of the supplementary individuals

    quanti_sup : a list/tuple indicating the indexes of the quantitative supplementary variables

    quali_sup : a list/tuple indicating the indexes of the categorical supplementary variables

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    var_  : a dictionary of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)
    
    var_sup_  : a dictionary of pandas dataframe containing all the results for the supplementary variables considered as group (coordinates, square cosine)
    
    ind_ : a dictionary of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    quali_var_ : a dictionary of pandas dataframe with all the results for the categorical variables (coordinates, square cosine, contributions, v.test)
    
    quali_sup_ : a dictionary of pandas dataframe with all the results for the supplementary categorical variables (coordinates, square cosine, v.test)

    quanti_var_ : a dictionary of pandas datafrme with all the results for the quantitative variables (coordinates, correlation, square cosine, contributions)

    quanti_sup_ : a dictionary of pandas dataframe with all the results for the supplementary quantitative variables (coordinates, correlation, square cosine)

    call_ : a dictionary with some statistics

    summary_quanti_ : descriptive statistics of quantitatives variables

    summary_quali_ : statistics of categories variables

    chi2_test_ : chi2 statistics test

    model_ : string. The model fitted = 'pcamix'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Hill M., Smith A. (1976). Principal Component Analysis of taxonomic data withmulti-state discrete characters. Taxon, 25, pp. 249-255

    Husson F., Le S. and Pagès J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Husson F., Josse L, Lê S. & Mazet J. (2009). FactoMineR : Factor Analysis and Data Mining with R. R package version 2.11

    Kiers H.A.L (1991). Simple structure in Component Analysis Techniques for mixtures of qualitative and quantitative variables. Psychometrika, 56, pp. 197-212.
    
    Lebart L., Piron M. & Morineau A. (2006). Statistique exploratoire multidimensionelle. Dunod Paris 4ed

    Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1–18. https://doi.org/10.18637/jss.v025.i01

    Pagès J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_pcamix_ind, get_pcamix_var, get_pcamix, summaryPCAMIX, dimdesc

    Examples
    --------
    > # Load gironde dataset

    > from scientisttools import load_gironde

    > gironde = load_gironde()

    > # Split data

    > from scientisttools import splitmix

    > X_quant = splitmix(gironde)["quanti"]

    > X_qual = splitmix(gironde)["quali"]

    > from scientisttools import PCAMIX

    > # PCA with PCAMIX function

    > res_pca = PCAMIX().fit(X_quant)

    > # MCA with PCAMIX function

    > res_mca = PCAMIX().fit(X_qual)

    > # FAMD with PCAMIX function

    > res_pcamix = PCAMIX().fit(gironde)

    > summaryPCAMIX(res_pcamix)
    """
    def __init__(self,
                 n_components = 5,
                 ind_weights = None,
                 quanti_weights = None,
                 quali_weights = None,
                 ind_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.quanti_weights = quanti_weights
        self.quali_weights = quali_weights
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
        X : pandas/polars DataFrame of float, shape (n_rows, n_columns)

        y : None
            y is ignored

        Returns:
        --------
        self : object
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

        # Set parallelize option
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        # Check is quali sup
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup]
        else:
            quali_sup_label = None

        #  Check if quanti sup
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup]
        else:
            quanti_sup_label = None
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None
        
        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

        ####################################### Drop supplementary qualitative columns ########################################
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        ######################################## Drop supplementary quantitatives columns #######################################
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)
        
        ############################ Apply recodevar
        rec = recodevar(X=X)

        # Extract elemnts
        n_rows = rec["n"]
        n_cont = rec["k1"]
        n_cat = rec["k2"]
        dummies = rec["dummies"]
        nb_moda = rec["nb_moda"]
        X = rec["X"]
        base = rec["W"]
        X_quanti = rec["quanti"]
        X_quali = rec["quali"]
    
        # Set individuals weights
        if self.ind_weights is None:
            ind_weights = np.ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,list):
            raise TypeError("'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != n_rows:
            raise TypeError(f"'row_weights' must be a list with length {n_rows}")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        # Set variables weights
        var_weights = pd.Series(name="weight").astype("float")

        # Set quantitatives variables weights
        if n_cont > 0:
            # Summary quantitatives variables 
            summary_quanti = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti["count"] = summary_quanti["count"].astype("int")
            self.summary_quanti_ = summary_quanti

            # Continuous weights
            quanti_weights = np.ones(n_cont)
            if self.quanti_weights is None:
                weights1 = np.ones(n_cont)
            elif not isinstance(self.quanti_weights,list):
                raise TypeError("'quanti_weights' must be a list of quantitatives weights")
            elif len(self.quanti_weights) != n_cont:
                raise TypeError(f"'quanti_weights' must be a list with length {n_cont}.")
            else:
                weights1 = np.array(self.quanti_weights)
            
            # Apply weighted correction
            quanti_weights = pd.Series(quanti_weights*weights1,index=X_quanti.columns)
            # Concatenate
            var_weights = pd.concat((var_weights,quanti_weights),axis=0)
        
        # Set categoricals variables weights
        if n_cat > 0:
            # Compute statistiques
            summary_quali = pd.DataFrame()
            for col in X_quali.columns.tolist():
                eff = X_quali[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
            summary_quali["count"] = summary_quali["count"].astype("int")
            self.summary_quali_ = summary_quali

            # Chi2 statistic test
            if n_cat>1:
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
                # Transform to int
                chi2_test["dof"] = chi2_test["dof"].astype("int")
                self.chi2_test_ = chi2_test

            # Set qualitative variables weights
            if self.quali_weights is None:
                weights2 = pd.Series(np.ones(n_cat),index=X_quali.columns,name="weight").astype("float")
            elif not isinstance(self.quali_weights,pd.Series):
                raise ValueError("'quali_weights' must be a pandas series where index are qualitatives variables names and values are qualitatives variables weights.")
            else:
                weights2 = self.quali_weights
            
            # Set categories weights
            mod_weights = pd.Series(name="weight").astype("float")
            for col in X_quali.columns:
                data = pd.get_dummies(X_quali[col],dtype=int)
                weights = (1/data.mean(axis=0))*weights2[col]
                mod_weights = pd.concat((mod_weights,weights),axis=0)
            
            # Concatenate
            var_weights = pd.concat((var_weights,mod_weights),axis=0)
        
        # Apply Generalized Singular Values decomposition (GSVD)
        svd = svd_triplet(X=base,row_weights=ind_weights,col_weights=var_weights)

        # QR decomposition
        Q, R = np.linalg.qr(base)
        max_components = min(np.linalg.matrix_rank(Q),np.linalg.matrix_rank(R))

        # Eigen - values
        eigen_values = svd["vs"][:max_components]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])

        #################### Set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise TypeError("'n_components' must be greater or equal than 1.")
        else:
            n_components = min(self.n_components, max_components)
        
        #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "Z" : rec["Z"],
                      "W" : rec["X"],
                      "ind_sup" : ind_sup_label,
                      "quanti_sup" : quanti_sup_label,
                      "quali_sup" : quali_sup_label,
                      "n_components" : n_components,
                      "rec" :  rec,
                      "var_weights" : var_weights}
    
        # Set using n_components
        vs = svd["vs"][:n_components]
        U = svd["U"][:,:n_components]
        V = svd["V"][:,:n_components]

        # Store Generalized Singular Value Decomposition (GSVD)
        self.svd_ = {"vs" : vs, "U" : U, "V" : V}

        ########################### Individuals informations #################################################################
        # Individuals coordinates
        ind_coord = U.dot(np.diag(vs))
        ind_coord = pd.DataFrame(ind_coord,index=X.index,columns=["Dim."+str(x+1) for x in range(ind_coord.shape[1])])
    
        # Individuals contributions
        ind_contrib = mapply(ind_coord,lambda x : 100*(x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : x/(vs**2),axis=1,progressbar=False,n_workers=n_workers)

        # Indviduals distances 
        ind_dist2 = np.apply_along_axis(func1d=lambda x : np.sum(x**2),axis=1,arr=svd["U"]*svd["vs"])

        # Individuals Cos2
        ind_cos2 = mapply(ind_coord,lambda x : x**2/ind_dist2,axis=0,progressbar=False,n_workers=n_workers)
    
        # Individuals inertia
        ind_inertia = ind_dist2*ind_weights
        
        # Save all informations
        ind_infos = np.c_[np.sqrt(ind_dist2),ind_weights,ind_inertia]
        ind_infos = pd.DataFrame(ind_infos,columns=["dist","weight","inertia"],index=X.index)
        
        self.ind_ = {"coord" : ind_coord, "contrib" : ind_contrib,"cos2" : ind_cos2,"infos" : ind_infos}

        ############# Quantitatives information
        if n_cont > 0:
            # continues variables coordinates
            quanti_var_coord =  V[:n_cont,:].dot(np.diag(vs))
            quanti_var_coord = pd.DataFrame(quanti_var_coord,index=X_quanti.columns,columns=["Dim."+str(x+1) for x in range(quanti_var_coord.shape[1])])

            # Continues variables contrib
            quanti_var_contrib = mapply(quanti_var_coord,lambda x : 100*(x**2)*var_weights.values[:n_cont],axis=0,progressbar=False,n_workers=n_workers)
            quanti_var_contrib = mapply(quanti_var_contrib, lambda x : x/(vs**2),axis=1,progressbar=False,n_workers=n_workers)

            # Continuous variables cos2
            quanti_var_cos2  = mapply(quanti_var_coord,lambda x : (x**2)/((svd["V"][:n_cont,:]*svd["vs"])**2).sum(axis=1),axis=0,progressbar=False,n_workers=n_workers)
            self.quanti_var_ = {"coord" : quanti_var_coord, "contrib" : quanti_var_contrib,"cor":quanti_var_coord,"cos2" : quanti_var_cos2}
        
        if n_cat > 0:
            # Categoricals variables coordinates
            moda_var_coord = V[n_cont:,:].dot(np.diag(vs))
            moda_var_coord = pd.DataFrame(moda_var_coord,index=base.columns[n_cont:],columns=["Dim."+str(x+1) for x in range(moda_var_coord.shape[1])])
            moda_var_coord = mapply(moda_var_coord,lambda x : x*var_weights.values[n_cont:],axis=0,progressbar=False,n_workers=n_workers)

            # Categoricals variables contributions
            moda_var_contrib = mapply(moda_var_coord,lambda x : 100*(x**2)*(1/var_weights.values[n_cont:]),axis=0,progressbar=False,n_workers=n_workers)
            moda_var_contrib = mapply(moda_var_contrib,lambda x : x/(vs**2),axis=1,progressbar=False,n_workers=n_workers)

            # Categoricals variables cos2
            moda_var_dist2 = (np.apply_along_axis(func1d=lambda x : x*var_weights.values[n_cont:],axis=0,arr=svd["V"][n_cont:,:]*svd["vs"])**2).sum(axis=1)
            moda_var_cos2 = mapply(moda_var_coord,lambda x : (x**2)/moda_var_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Categoricals variables Vtest
            I_k = dummies.sum(axis=0)
            moda_var_vtest = pd.concat(((moda_var_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*I_k[k])/(X.shape[0]-I_k[k]))).to_frame(k).T for k in dummies.columns),axis=0)
            moda_var_vtest = mapply(moda_var_vtest,lambda x : x/vs,axis=1,progressbar=False,n_workers=n_workers)

            # 
            moda_dist2 = pd.Series(np.sqrt(moda_var_dist2),index=base.columns[n_cont:],name="dist")

            self.quali_var_ = {"coord" : moda_var_coord, "contrib" : moda_var_contrib, "cos2" : moda_var_cos2,"vtest":moda_var_vtest,"dist" : moda_dist2}

            ## Qualitatives eta2
            quali_var_eta2 = pd.concat((function_eta2(X=X_quali,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali.columns),axis=0)

            if n_cont == 0:
                self.quali_var_["eta2"] = quali_var_eta2
            elif n_cont > 0:
                # Contributions des variables qualitatives
                quali_var_contrib = mapply(quali_var_eta2,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                # Cosinus carrés des variables qualitatives
                quali_var_cos2 = pd.concat((((quali_var_eta2.loc[col,:]**2)/(nb_moda[col]-1)).to_frame(name=col).T for col in X_quali.columns.tolist()),axis=0)
                
                var_coord = pd.concat((quanti_var_cos2,quali_var_eta2),axis=0)
                var_contrib = pd.concat((quanti_var_contrib,quali_var_contrib),axis=0)
                var_cos2 = pd.concat((quanti_var_cos2**2,quali_var_cos2),axis=0)
                # Store all informations
                self.var_ = {"coord" : var_coord,"contrib" : var_contrib,"cos2" : var_cos2}

        ###########################################################################################################
        #                            Compute supplementary individuals informations
        ##########################################################################################################
        if self.ind_sup is not None:
            ##### Prepare supplementary columns
            X_ind_sup_quant = splitmix(X=X_ind_sup)["quanti"]
            X_ind_sup_quali = splitmix(X=X_ind_sup)["quali"]

            # Prepare DataFrame
            ind_sup_coord = pd.DataFrame(np.zeros((X_ind_sup.shape[0],n_components)),index=X_ind_sup.index,columns=["Dim."+str(x+1) for x in range(n_components)])
            ind_sup_cos2 = pd.DataFrame(np.zeros((X_ind_sup.shape[0],n_components)),index=X_ind_sup.index,columns=["Dim."+str(x+1) for x in range(n_components)])
            ind_sup_dist2 = pd.Series(np.zeros(X_ind_sup.shape[0]),index=X_ind_sup.index,name="dist")
            
            if n_cont > 0:
                X_ind_sup_quant = X_ind_sup_quant.astype("float")

                # Standardize the data
                Z_ind_sup_quant = (X_ind_sup_quant - rec["means"].iloc[:n_cont].values.reshape(1,-1))/rec["std"].iloc[:n_cont].values.reshape(1,-1)

                # Supplementary individuals coordinates
                ind_sup_coord1 = mapply(Z_ind_sup_quant,lambda x : x*var_weights.values[:n_cont],axis=1,progressbar=False,n_workers=n_workers)
                ind_sup_coord1 = np.dot(ind_sup_coord1,svd["V"][:n_cont,:n_components])
                ind_sup_coord1 = pd.DataFrame(ind_sup_coord1,index=X_ind_sup.index,columns=["Dim."+str(x+1) for x in range(ind_sup_coord1.shape[1])])

                # Supplementary individuals dist2
                ind_sup_dist21 = mapply(Z_ind_sup_quant,lambda  x : (x**2)*var_weights.values[:n_cont],axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
                ind_sup_dist21.name = "dist"

                # Supplementary individuals cos2
                ind_sup_cos21 = mapply(ind_sup_coord1,lambda x : (x**2)/ind_sup_dist21,axis=0,progressbar=False,n_workers=n_workers)

                # Add 
                ind_sup_coord = ind_sup_coord + ind_sup_coord1
                ind_sup_cos2 = ind_sup_cos2 + ind_sup_cos21
                ind_sup_dist2 = ind_sup_dist2 + ind_sup_dist21

            if n_cat > 0:
                # Revaluate the categorical variable
                X_ind_sup_quali = revaluate_cat_variable(X_ind_sup_quali)

                # Recode to dummies
                Y = np.zeros((X_ind_sup_quali.shape[0],dummies.shape[1]))
                for i in np.arange(0,X_ind_sup.shape[0],1):
                    values = [str(X_ind_sup_quali.iloc[i,k]) for k in np.arange(0,X_quali.shape[1])]
                    for j in np.arange(0,dummies.shape[1],1):
                        if dummies.columns[j] in values:
                            Y[i,j] = 1
                Y = pd.DataFrame(Y,columns=dummies.columns,index=X_ind_sup_quali.index)

                # Standardize the data
                Z_ind_sup_qual = (Y - (dummies.sum(axis=0)/n_rows).values.reshape(1,-1))

                # Supplementary individuals coordinates
                ind_sup_coord2 = mapply(Z_ind_sup_qual,lambda x : x*var_weights.values[n_cont:],axis=1,progressbar=False,n_workers=n_workers)
                ind_sup_coord2 = np.dot(ind_sup_coord2,svd["V"][n_cont:,:n_components])
                ind_sup_coord2 = pd.DataFrame(ind_sup_coord2,index=X_ind_sup.index,columns=["Dim."+str(x+1) for x in range(ind_sup_coord2.shape[1])])

                # Supplementary individuals dist2
                ind_sup_dist22 = mapply(Z_ind_sup_qual,lambda x : (x**2)*var_weights.values[n_cont:],axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
                ind_sup_dist22.name = "dist"

                # SSupplementary individuals cos2
                ind_sup_cos22 = mapply(ind_sup_coord2,lambda x : (x**2)/ind_sup_dist22,axis=0,progressbar=False,n_workers=n_workers)
                
                # Add to elements
                ind_sup_coord = ind_sup_coord + ind_sup_coord2
                ind_sup_cos2 = ind_sup_cos2 + ind_sup_cos22
                ind_sup_dist2 = ind_sup_dist2 + ind_sup_dist22
                
            # Store all informations
            self.ind_sup_ = {"coord" : ind_sup_coord,"cos2" : ind_sup_cos2,"dist" : np.sqrt(ind_sup_dist2)}
        
        ##########################################################################################################
        #                         Compute supplementary quantitatives variables statistics
        ###########################################################################################################
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")

            # Store
            if n_cont > 0:
                self.summary_quanti_.insert(0,"group","active")
                summary_quanti_sup.insert(0,"group","sup")
                self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)
            elif n_cont == 0:
                self.summary_quanti_ = summary_quanti_sup

            # Standardize
            d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = (X_quanti_sup - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)

            ####### Compute Supplementary quantitatives variables coordinates
            Z_quanti_sup = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            quanti_sup_coord = Z_quanti_sup.T.dot(U)
            quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(quanti_sup_coord.shape[1])]

            ############# Supplementary quantitatives variables Cos2
            var_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            var_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),var_sup_cor)
            quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2)/np.sqrt(var_sup_dist2),axis=0,progressbar=False,n_workers=n_workers)

            # Store supplementary quantitatives informations
            self.quanti_sup_ =  {"coord":quanti_sup_coord,"cor" : quanti_sup_coord, "cos2" : quanti_sup_cos2}

        ##########################################################################################################
        #                         Compute supplementary qualitatives variables statistics
        ###########################################################################################################
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)

            # Transform all categorical to object
            X_quali_sup = X_quali_sup.astype("object")
            # Check if two columns have the same categories levels
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

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
            if n_cat > 0:
                chi2_sup_test2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
                cpt = 0
                for i in range(X_quali_sup.shape[1]):
                    for j in range(n_cat):
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
            if n_cat > 0:
                if X_quali_sup.shape[1] > 1 :
                    chi2_sup_test = pd.concat([chi2_sup_test,chi2_sup_test2],axis=0,ignore_index=True)
                else:
                    chi2_sup_test = chi2_sup_test2
                chi2_sup_test["dof"] = chi2_sup_test["dof"].astype("int")
                self.chi2_test_ = pd.concat((self.chi2_test_,chi2_sup_test),axis=0,ignore_index=True)
            else:
                if X_quali_sup.shape[1] > 1 :
                    self.chi2_test_ = chi2_sup_test2
            
            #################################### Summary quali
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns:
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")

            if n_cat == 0:
                self.summary_quali_sup_ = summary_quali_sup
            elif n_cat > 0:
                summary_quali_sup.insert(0,"group","sup")
                self.summary_quali_.insert(0,"group","active")
                self.summary_quali_ = pd.concat([self.summary_quali_,summary_quali_sup],axis=0,ignore_index=True)

            ####################################" Correlation ratio #####################################################
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

            # Recode qualitative variables
            rec_quali = recodevar(X_quali_sup)
            w_quali = rec_quali["W"]
            n_k_sup = rec_quali["dummies"].sum(axis=0)
            A = w_quali.T.dot(np.diag(ind_weights)).dot(U)

            ################################### Barycentrique coordinates #############################################
            quali_sup_coord = mapply(A, lambda x : x/(n_k_sup.values/n_rows),axis=0,progressbar=False,n_workers=n_workers)
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(quali_sup_coord.shape[1])]
            
            # Distance 
            quali_sup_dist2  = mapply(w_quali, lambda x : (x**2),axis=1,progressbar=False,n_workers=n_workers).sum(axis=0)/n_rows
            quali_sup_dist2.name = "dist"

            ################################## Cos2
            quali_sup_cos2 = mapply(A, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
            quali_sup_cos2.columns = ["Dim."+str(x+1) for x in range(quali_sup_cos2.shape[1])]
           
            ################################## v-test
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/vs,axis=1,progressbar=False,n_workers=n_workers)
            quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((n_rows-n_k_sup[k])/((n_rows-1)*n_k_sup[k]))).to_frame().T for k in n_k_sup.index),axis=0)

            # Supplementary categories informations
            self.quali_sup_ = {"coord" : quali_sup_coord,"cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"dist" : np.sqrt(quali_sup_dist2), "eta2" : quali_sup_eta2}

        self.model_ = "pcamix"

        return self

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

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
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # check if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
    
         ##### Prepare supplementary columns
        X_quant = splitmix(X=X)["quanti"]
        X_quali = splitmix(X=X)["quali"]

        ####
        rec = self.call_["rec"]
        n_components = self.call_["n_components"]
        n_rows = rec["n"]
        n_cont = rec["k1"]
        n_cat = rec["k2"]
        dummies = rec["dummies"]
        var_weights = self.call_["var_weights"]

        # Initialize standardize DataFrame
        Z = pd.DataFrame().astype("float")
        if n_cont > 0:
            X_quant = X_quant.astype("float")

            # Standardize the data
            Z1 = (X_quant - rec["means"].iloc[:n_cont].values.reshape(1,-1))/rec["std"].iloc[:n_cont].values.reshape(1,-1)
            # Concatenate
            Z = pd.concat((Z,Z1),axis=1)
            
        if n_cat > 0:
            # Revaluate the categorical variable
            X_quali = revaluate_cat_variable(X_quali)

            # Recode to dummies
            Y = np.zeros((X_quali.shape[0],dummies.shape[1]))
            for i in np.arange(0,X.shape[0],1):
                values = [str(X_quali.iloc[i,k]) for k in np.arange(0,rec["quali"].shape[1])]
                for j in np.arange(0,dummies.shape[1],1):
                    if dummies.columns[j] in values:
                        Y[i,j] = 1
            Y = pd.DataFrame(Y,columns=dummies.columns,index=X.index)

            # Standardize the data
            Z2 = (Y - (dummies.sum(axis=0)/n_rows).values.reshape(1,-1))
            # Concatenate
            Z = pd.concat((Z,Z2),axis=1)

        # Multiply by variable weights
        Z = mapply(Z,lambda x : x*var_weights.values,axis=1,progressbar=False,n_workers=n_workers)
        coord = Z.dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return  coord

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        ----------------------------------------------------------------

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
        return self.ind_["coord"]

def predictPCAMIX(self,X):
    """
    
    """
    pass

def supvarPCAMIX(self,X_quanti_sup=None,X_quali_sup=None):
    """
    
    """
    pass