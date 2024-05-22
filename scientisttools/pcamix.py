# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .pca import PCA
from .function_eta2 import function_eta2
from .svd_triplet import svd_triplet
from .revaluate_cat_variable import revaluate_cat_variable
from .recodevar import recodevar
from .splitmix import splitmix

class PCAMIX(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis of Mixed Data (PCAMIX)
    ---------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Principal Components Analysis of Mixed Data (PCAMIX) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables. It includes ordinary principal component analysis (PCA) and multiple correspondence analysis (MCA) as special cases.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); 
                    the weights are given only for the active individuals
    
    quanti_weights : an optional quantitatives variables weights (by default, a list/tuple of 1 for uniform quantitative variables weights), 
                        the weights are given only for the active quantitative variables
    
    quali_weights : an optional qualitatives variables weights (by default, a list/tuple of 1/(number of active qualitative variable) for uniform qualitative variables weights), 
                        the weights are given only for the active qualitative variables
    
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
    
    ind_ : a dictionary of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    quali_var_ : a dictionary of pandas dataframe with all the results for the categorical variables (coordinates, square cosine, contributions, v.test)
    
    quanti_var_ : a dictionary of pandas datafrme with all the results for the quantitative variables (coordinates, correlation, square cosine, contributions)

    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'mca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2008), Analyses Factorielles Simples et Multiples.4ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Pages J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_famd_ind, get_famd_var, get_famd, summaryFAMD, dimdesc

    Examples
    --------
    > X = wine # from FactoMineR R package

    > res_famd = FAMD(parallelize=True)

    > res_famd.fit(X)

    > summaryFAMD(res_famd)
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
        
        ###### Checks if categoricals variables re in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns.tolist():
                X[col] = X[col].astype("object")
        
        ##### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")

        ############################
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
        
        ####################################### Fill NA in quantitatives columns wih mean
        if is_quanti.shape[1]>0:
            if is_quanti.isnull().any().any():
                for col in is_quanti.columns:
                    if X[col].isnull().any().any():
                        X[col].fillna(X[col].mean(),inplace=True)
                print("Missing values are imputed by the mean of the variable.")

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
        
        # 
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
            if self.quanti_weights is None:
                weights = np.ones(X_quanti.shape[1])
            elif not isinstance(self.quanti_weights,list):
                raise TypeError("'quanti_weights' must be a list of quantitatives weights")
            elif len(self.quanti_weights) != X_quanti.shape[1]:
                raise TypeError(f"'quanti_weights' must be a list with length {X_quanti.shape[1]}.")
            else:
                weights = np.array(self.quanti_weights)
            # Concatenate
            quanti_weights = pd.Series(weights,index=X_quanti.columns)
            var_weights = pd.concat((var_weights,quanti_weights),axis=0)
        
        # Set categoricals variables weights
        if n_cat > 0:
            quali_weights = pd.Series(index=X_quali.columns.tolist(),name="weight").astype("float")
            if self.quali_weights is None:
                for col in X_quali.columns.tolist():
                    quali_weights[col] = 1/n_cat
            elif not isinstance(self.quali_weights,dict):
                raise TypeError("'quali_weights' must be a dictionary where keys are qualitatives variables names and values are qualitatives variables weights.")
            elif len(self.quali_weights.keys()) != n_cat:
                raise TypeError(f"'quali_weights' must be a dictionary with length {n_cat}.")
            else:
                for col in X_quali.columns.tolist():
                    quali_weights[col] = self.quali_weights[col]/sum(self.quali_weights)
        
            ###### Define mod weights
            mod_weights = 1/dummies.mean(axis=0)
            # Concatenate
            var_weights = pd.concat((var_weights,mod_weights),axis=0)
        
        # Apply Generalized Singular Values decomposition (GSVD)
        svd = svd_triplet(X=base,row_weights=ind_weights,col_weights=var_weights)
        self.svd_ = svd

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
            ind_sup_dist2 = pd.Series([0]*X_ind_sup.shape[0],index=X_ind_sup.index,name="dist")
            
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
                ind_sup_dummies = pd.DataFrame(Y,columns=dummies.columns,index=X_ind_sup_quali.index)

                # Standardize the data
                Z_ind_sup_qual = (ind_sup_dummies - (dummies.sum(axis=0)/n_rows).values.reshape(1,-1))

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
            
            d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = (X_quanti_sup - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)

            ####### Compute Supplementary quantitatives variables coordinates
            var_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            var_sup_coord = np.dot(var_sup_coord.T,U)
            var_sup_coord = pd.DataFrame(var_sup_coord,index=X_quanti_sup.columns,columns = ["Dim."+str(x+1) for x in range(var_sup_coord.shape[1])])

            ############# Supplementary quantitatives variables Cos2
            var_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            var_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),var_sup_cor)
            var_sup_cos2 = mapply(var_sup_coord,lambda x : (x**2)/np.sqrt(var_sup_dist2),axis=0,progressbar=False,n_workers=n_workers)

            # Store supplementary quantitatives informations
            self.quanti_sup_ =  {"coord":var_sup_coord,"cor" : var_sup_coord,"cos2" : var_sup_cos2}

        ##########################################################################################################
        #                         Compute supplementary qualitatives variables statistics
        ###########################################################################################################
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)

            ######################################## Barycentre of DataFrame ########################################
            X_quali_sup = X_quali_sup.astype("object")
            ############################################################################################################
            # Check if two columns have the same categories
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            ####################################" Correlation ratio #####################################################
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)

            ###################################### Coordinates ############################################################
            barycentre = pd.DataFrame().astype("float")
            n_k = pd.Series().astype("float")
            for col in X_quali_sup.columns:
                vsQual = X_quali_sup[col]
                modalite, counts = np.unique(vsQual, return_counts=True)
                n_k = pd.concat([n_k,pd.Series(counts,index=modalite)],axis=0)
                bary = pd.DataFrame(index=modalite,columns=base.columns)
                for mod in modalite:
                    idx = [elt for elt, cat in enumerate(vsQual) if  cat == mod]
                    bary.loc[mod,:] = np.average(X.iloc[idx,:],axis=0,weights=ind_weights[idx])
                barycentre = pd.concat((barycentre,bary),axis=0)
            
            ############### Standardize the barycenter
            bary = (barycentre - rec["means"].values.reshape(1,-1))/rec["std"].values.reshape(1,-1)
            quali_sup_dist2  = mapply(bary, lambda x : x**2*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            quali_sup_dist2.name = "dist"

            ################################### Barycentrique coordinates #############################################
            quali_sup_coord = mapply(bary, lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers)
            quali_sup_coord = quali_sup_coord.dot(V)
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(quali_sup_coord.shape[1])]

            ################################## Cos2
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
            
            ################################## v-test
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x/vs,axis=1,progressbar=False,n_workers=n_workers)
            quali_sup_vtest = pd.concat(((quali_sup_vtest.loc[k,:]/np.sqrt((X.shape[0]-n_k[k])/((X.shape[0]-1)*n_k[k]))).to_frame().T for k in n_k.index),axis=0)

            # Supplementary categories informations
            self.quali_sup_ = {"coord" : quali_sup_coord,
                               "cos2" : quali_sup_cos2,
                               "vtest" : quali_sup_vtest,
                               "dist" : np.sqrt(quali_sup_dist2),
                               "eta2" : quali_sup_eta2,
                               "barycentre" : barycentre}
        
        # If Mixed Data
        if n_cont > 0 and n_cat > 0:
            if self.quanti_sup is not None and self.quali_sup is not None:
                var_sup_coord = pd.concat((self.quanti_sup_["cos2"],self.quali_sup_["eta2"]),axis=0)
                var_sup_cos2 = pd.concat((self.quanti_sup_["cos2"]**2,self.quali_sup_["cos2"]),axis=0)
                self.var_sup_ = {"coord" : var_sup_coord, "cos2" : var_sup_cos2}
            elif self.quanti_sup is not None:
                var_sup_coord = self.quanti_sup_["cos2"]
                var_sup_cos2 = self.quanti_sup_["cos2"]**2
                self.var_sup_ = {"coord" : var_sup_coord, "cos2" : var_sup_cos2}
            elif self.quali_sup is not None:
                var_sup_coord = self.quali_sup_["eta2"]
                var_sup_cos2 = self.quali_sup_["cos2"]
                self.var_sup_ = {"coord" : var_sup_coord, "cos2" : var_sup_cos2}

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

        # Initialize coord
        coord = pd.DataFrame(np.zeros((X.shape[0],n_components)),index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])
        
        if n_cont > 0:
            X_quant = X_quant.astype("float")

            # Standardize the data
            Z_quant = (X_quant - rec["means"].iloc[:n_cont].values.reshape(1,-1))/rec["std"].iloc[:n_cont].values.reshape(1,-1)

            # Supplementary individuals coordinates
            coord1 = mapply(Z_quant,lambda x : x*var_weights.values[:n_cont],axis=1,progressbar=False,n_workers=n_workers)
            coord1 = np.dot(coord1,self.svd_["V"][:n_cont,:n_components])
            coord1 = pd.DataFrame(coord1,index=X.index,columns=["Dim."+str(x+1) for x in range(coord1.shape[1])])
            # Update coord
            coord = coord + coord1
            
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
            ind_sup_dummies = pd.DataFrame(Y,columns=dummies.columns,index=X.index)

            # Standardize the data
            Z_qual = (ind_sup_dummies - (dummies.sum(axis=0)/n_rows).values.reshape(1,-1))

            # Supplementary individuals coordinates
            coord2 = mapply(Z_qual,lambda x : x*var_weights.values[n_cont:],axis=1,progressbar=False,n_workers=n_workers)
            coord2 = np.dot(coord2,self.svd_["V"][n_cont:,:n_components])
            coord2 = pd.DataFrame(coord2,index=X.index,columns=["Dim."+str(x+1) for x in range(coord2.shape[1])])
            # Update Coord
            coord = coord + coord2
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