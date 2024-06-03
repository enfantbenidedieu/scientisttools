# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .pca import PCA
from .revaluate_cat_variable import revaluate_cat_variable
from .recodecont import recodecont
from .recodevarfamd import recodevarfamd

class FAMD(BaseEstimator,TransformerMixin):
    """
    Factor Analysis of Mixed Data (FAMD)
    ------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Factor Analysis of Mixed Data (FAMD) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    FAMD is a principal component method dedicated to explore data with both continuous and categorical variables. 
    It can be seen roughly as a mixed between PCA and MCA. More precisely, 
    the continuous variables are scaled to unit variance and the categorical variables are transformed 
    into a disjunctive data table (crisp coding) and then scaled using the specific scaling of MCA. 
    This ensures to balance the influence of both continous and categorical variables in the analysis. 
    It means that both variables are on a equal foot to determine the dimensions of variability. 
    This method allows one to study the similarities between individuals taking into account mixed 
    variables and to study the relationships between all the variables.

    Details
    -------

    FAMD includes standard Principal Component Analysis (PCA) and Multiple Correspondence Analysis (MCA) as special cases. If all variables are quantitative, standard PCA is performed.
    if all variables are qualitative, then standard MCA is performed.

    Missing values are replaced by means for quantitative variables. Note that, when all the variable are qualitative, the factor coordinates of the individuals are equal to the factor scores
    of standard MCA times squares root of J (the number of qualitatives variables) and the eigenvalues are then equal to the usual eigenvalues of MCA times J.
    When all the variables are quantitative, FAMD gives exactly the same results as standard PCA.

    Usage
    -----

    FAMD(n_components = 5,ind_weights = None,quanti_weights = None,quali_weights = None,ind_sup=None,quanti_sup=None,quali_sup=None,parallelize = False).fit(X)

    Parameters
    ----------

    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals
    
    quanti_weights : an optional quantitatives variables weights (by default, a list/tuple of 1 for uniform quantitative variables weights), the weights are given only for the active quantitative variables
    
    quali_weights : an optional qualitatives variables weights (by default, a list/tuple of 1 for uniform qualitative variables weights), the weights are given only for the active qualitative variables
    
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
    
    quanti_var_ : a dictionary of pandas dataframe with all the results for the quantitative variables (coordinates, correlation, square cosine, contributions)

    quanti_sup_ : a dictionary of pandas dataframe with all the results for the supplementary quantitative variables (coordinates, correlation, square cosine)

    call_ : a dictionary with some statistics

    summary_quanti_ : descriptive statistics of quantitatives variables

    summary_quali_ : statistics of categories variables

    chi2_test_ : chi2 statistics test

    model_ : string. The model fitted = 'famd'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson F., Le S. and Pagès J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Husson F., Josse L, Lê S. & Mazet J. (2009). FactoMineR : Factor Analysis and Data Mining iwith R. R package version 2.11

    Lebart L., Piron M. & Morineau A. (2006). Statistique exploratoire multidimensionelle. Dunod Paris 4ed

    Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1–18. https://doi.org/10.18637/jss.v025.i01

    Pagès J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_famd_ind, get_famd_var, get_famd, summaryFAMD, dimdesc

    Examples
    --------
    > # Load gironde dataset

    > from scientisttools import load_gironde

    > gironde = load_gironde()

    > # Split data

    > from scientisttools import splitmix

    > X_quant = splitmix(gironde)["quanti"]

    > X_qual = splitmix(gironde)["quali"]

    > from scientisttools import FAMD

    > # PCA with FAMD function

    > res_pca = FAMD().fit(X_quant)

    > # MCA with FAMD function

    > res_mca = FAMD().fit(X_qual)

    > # FAMD with FAMD function

    > res_famd = FAMD().fit(gironde)

    > summaryFAMD(res_famd)
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
        
        ##########################################################################################
        # Set supplementary qualitative variables labels
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup]
        else:
            quali_sup_label = None

        ############################################################################################
        #  Set supplementary qualitative variables labels
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup]
        else:
            quanti_sup_label = None
        
        ###################################################################################################
        # Set supplementary individuals labels
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

        ###################################### Factor Analysis of Mixed Data ######################################################
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

        # Set individuals weights
        if self.ind_weights is None:
            ind_weights = np.ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,list):
            raise TypeError("'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != n_rows:
            raise TypeError(f"'ind_weights' must be a list with length {n_rows}")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        # Set variables weights
        Z = pd.DataFrame().astype("float")
        means = pd.Series(name="weight").astype("float") # set means series
        std = pd.Series(name="weight").astype("float") # set std dev series
        var_weights = pd.Series(name="weight").astype("float")

        # Set quantitatives variables weights
        if n_cont > 0:
            # Summary quantitatives variables 
            summary_quanti = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti["count"] = summary_quanti["count"].astype("int")
            self.summary_quanti_ = summary_quanti

            # Compute means and standard deviation
            d1 = DescrStatsW(X_quanti,weights=ind_weights,ddof=0)
            means1 = pd.Series(d1.mean,index=X_quanti.columns,name="weight")
            std1 = pd.Series(d1.std,index=X_quanti.columns,name="weight")

            # Standardize the data
            Z1 = (X_quanti - means1.values.reshape(1,-1))/std1.values.reshape(1,-1)

            # Concatenate
            Z = pd.concat((Z,Z1),axis=1)
            means = pd.concat((means,means1),axis=0)
            std = pd.concat((std,std1),axis=0)

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
            # Concatenate
            var_weights = pd.concat((var_weights,weights1),axis=0)
        
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
            
            ############################ Compute weighted mean and weighted standards 
            # Normalize Z2
            prop = mapply(dummies,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            means2 = pd.Series(np.average(dummies,axis=0,weights=ind_weights),index=dummies.columns,name="weight")
            std2 = pd.Series(np.sqrt(prop),index=dummies.columns,name="weight")

            # Standardize
            Z2 = (dummies - means2.values.reshape(1,-1))/std2.values.reshape(1,-1)

            # Concatenate
            Z = pd.concat((Z,Z2),axis=1)
            means = pd.concat((means,means2),axis=0)
            std = pd.concat((std,std2),axis=0)

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

        # QR decomposition (to set number of components)
        Q, R = np.linalg.qr(Z)
        max_components = min(np.linalg.matrix_rank(Q),np.linalg.matrix_rank(R))

        #################### Set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise TypeError("'n_components' must be greater or equal than 1.")
        else:
            n_components = min(self.n_components, max_components)
        
        # Store elements
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "Z" : Z,
                      "ind_sup" : ind_sup_label,
                      "quanti_sup" : quanti_sup_label,
                      "quali_sup" : quali_sup_label,
                      "n_components" : n_components,
                      "means" : means,
                      "std" : std,
                      "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                      "var_weights" : var_weights,
                      "rec" : rec}
        
        #########################################################################################
        #Global PCA without supplementary elememnts
        global_pca = PCA(standardize=False,n_components=int(max_components)).fit(Z)

        if n_cat > 0:
            D = pd.concat((Z,X_quali),axis=1)
            index = [D.columns.tolist().index(x) for x in X_quali.columns]
            global_pca = PCA(standardize=False,n_components=int(max_components),quali_sup=index).fit(D)

            # Extract elements for categoricals variables
            quali_infos = global_pca.var_["infos"].iloc[n_cont:,:]
            # coordinates
            quali_coord = global_pca.quali_sup_["coord"].iloc[:,:n_components]
            # Contributions
            quali_contrib = global_pca.var_["contrib"].iloc[n_cont:,:n_components]
            # Cos2
            quali_cos2 = global_pca.quali_sup_["cos2"].iloc[:,:n_components]
            # Vtest
            quali_vtest = global_pca.quali_sup_["vtest"].iloc[:,:n_components]
            # eta2
            quali_var_eta2 = global_pca.quali_sup_["eta2"].iloc[:,:n_components]
            # Store informations
            self.quali_var_ = {"coord" : quali_coord, "contrib" : quali_contrib, "cos2" : quali_cos2,"vtest":quali_vtest,
                               "dist" : global_pca.quali_sup_["dist"],"infos" : quali_infos}

        ###########################################################################################################
        #                            Compute supplementary individuals informations
        ##########################################################################################################
        if self.ind_sup is not None:
            # Recode the variables
            rec2 = recodevarfamd(X_ind_sup)
            X_ind_sup = rec2["X"]
            n_rows2 = rec2["n"]
            n_cont2 = rec2["k1"]
            n_cat2 = rec2["k2"]
            X_ind_sup_quanti = rec2["quanti"]
            X_ind_sup_quali = rec2["quali"]

            # Initialize the data
            Z_ind_sup = pd.DataFrame().astype("float")

            if n_cont2 > 0:
                if n_cont != n_cont2:
                    raise TypeError("The number of continuous columns must be the same")
                
                # Standardize the data
                Z1_ind_sup = (X_ind_sup_quanti - means.values[:n_cont].reshape(1,-1))/std.values[:n_cont].reshape(1,-1)
                # Concatenate
                Z_ind_sup = pd.concat((Z_ind_sup,Z1_ind_sup),axis=1)
            
            if n_cat2 > 0:
                if n_cat != n_cat2:
                    raise TypeError("The number of qualitatives columns must be the same")

                # Create supplementary individuals dummies
                Y = np.zeros((n_rows2,dummies.shape[1]))
                for i in np.arange(n_rows2):
                    values = [str(X_ind_sup_quali.iloc[i,k]) for k in np.arange(n_cat)]
                    for j in np.arange(dummies.shape[1]):
                        if dummies.columns[j] in values:
                            Y[i,j] = 1
                Y = pd.DataFrame(Y,columns=dummies.columns,index=X_ind_sup.index)

                # Standardize the data
                Z2_ind_sup = (Y - means.values[n_cont:].reshape(1,-1))/std.values[n_cont:].reshape(1,-1)
                # Concatenate
                Z_ind_sup = pd.concat((Z_ind_sup,Z2_ind_sup),axis=1)
            
            # Concatenate
            Z_ind_sup = pd.concat((Z,Z_ind_sup),axis=0)
            # Update PCA
            global_pca = PCA(standardize=False,n_components=int(max_components),ind_sup=ind_sup).fit(Z_ind_sup)
            
            # Extract elements
            ind_sup_coord = global_pca.ind_sup_["coord"].iloc[:,:n_components]
            ind_sup_cos2 = global_pca.ind_sup_["cos2"].iloc[:,:n_components]
            ind_sup_dist = global_pca.ind_sup_["dist"]
            # Store all informations
            self.ind_sup_ = {"coord" : ind_sup_coord, "cos2" : ind_sup_cos2, "dist" : ind_sup_dist}
        
        ##########################################################################################################
        #                         Compute supplementary quantitatives variables statistics
        ###########################################################################################################
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            # Recode to fill NA
            X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]

            # Summary statistics with supplementary quantitatives variables
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
            Z_quanti_sup = pd.concat((Z,Z_quanti_sup),axis=1)

            # Find supplementary quantitatives columns index
            index = [Z_quanti_sup.columns.tolist().index(x) for x in X_quanti_sup.columns]
            # Update PCA
            global_pca = PCA(standardize=False,n_components=int(max_components),ind_sup=None,quanti_sup=index).fit(Z_quanti_sup)
            
            # Extract elements
            quanti_sup_coord = global_pca.quanti_sup_["coord"].iloc[:,:n_components]
            quanti_sup_cos2  = global_pca.quanti_sup_["cos2"].iloc[:,:n_components]
            # store informations
            self.quanti_sup_ = {"coord" : quanti_sup_coord, "cor" : quanti_sup_coord,"cos2" : quanti_sup_cos2}
        
        ##########################################################################################################
        #                         Compute supplementary qualitatives variables statistics
        ###########################################################################################################
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
                        chi2_sup_test  = pd.concat([chi2_sup_test,row_chi2],axis=0)
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
            for col in X_quali_sup.columns:
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")

            if n_cat == 0:
                self.summary_quali_ = summary_quali_sup
            elif n_cat > 0:
                summary_quali_sup.insert(0,"group","sup")
                self.summary_quali_.insert(0,"group","active")
                self.summary_quali_ = pd.concat([self.summary_quali_,summary_quali_sup],axis=0,ignore_index=True)

            # Concatenate
            Z_quali_sup = pd.concat((Z,X_quali_sup),axis=1)
            # Find supplementary quantitatives columns index
            index = [Z_quali_sup.columns.tolist().index(x) for x in X_quali_sup.columns]
            # Update PCA
            global_pca = PCA(standardize=False,n_components=int(max_components),ind_sup=None,quali_sup=index).fit(Z_quali_sup)
            
            # Extract elements
            quali_sup_coord = global_pca.quali_sup_["coord"].iloc[:,:n_components]
            quali_sup_cos2 = global_pca.quali_sup_["cos2"].iloc[:,:n_components]
            quali_sup_vtest = global_pca.quali_sup_["vtest"].iloc[:,:n_components]
            quali_sup_eta2 = global_pca.quali_sup_["eta2"].iloc[:,:n_components]
            quali_sup_dist = global_pca.quali_sup_["dist"]
            # Store all informations
            self.quali_sup_ = {"coord" : quali_sup_coord, "cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"dist" : quali_sup_dist,"eta2" : quali_sup_eta2}

        #############################
        # Store Singular Value Decomposition
        self.svd_ = global_pca.svd_
        
        # Eigen - values
        eigen_values = global_pca.svd_["vs"][:max_components]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])

        ############################################## Individuals informations ###########################################################################
        self.ind_ = {"coord" : global_pca.ind_["coord"].iloc[:,:n_components], 
                     "contrib" : global_pca.ind_["contrib"].iloc[:,:n_components],
                     "cos2" : global_pca.ind_["cos2"].iloc[:,:n_components],
                     "infos" : global_pca.ind_["infos"]}
                    
        ############################################## Quantitatives variables informations ###############################################################
        if n_cont > 0:
            # Coordinates
            quanti_coord =  global_pca.var_["coord"].iloc[:n_cont,:n_components]
            # Contributions
            quanti_contrib = global_pca.var_["contrib"].iloc[:n_cont,:n_components]
            # Cos2
            quanti_cos2 = global_pca.var_["cos2"].iloc[:n_cont,:n_components]
            # Store informations
            self.quanti_var_ = {"coord" : quanti_coord, "contrib" : quanti_contrib,"cor": quanti_coord, "cos2" : quanti_cos2}
        
        ############################################## Qualitatives/categories variables informations #######################################################
        if n_cat > 0:
            # Add to qualitatives/categoricals variables if not continuous variables
            if n_cont == 0:
                self.quali_var_["eta2"] = quali_var_eta2
            elif n_cont > 0:
                # Contributions des variables qualitatives
                quali_var_contrib = mapply(quali_var_eta2,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                # Cosinus carrés des variables qualitatives
                quali_var_cos2 = pd.concat((((quali_var_eta2.loc[col,:]**2)/(nb_moda[col]-1)).to_frame(name=col).T for col in X_quali.columns),axis=0)

                var_coord = pd.concat((quanti_cos2,quali_var_eta2),axis=0)
                var_contrib = pd.concat((quanti_contrib,quali_var_contrib),axis=0)
                var_cos2 = pd.concat((quanti_cos2**2,quali_var_cos2),axis=0)
                #Store all informations
                self.var_ = {"coord" : var_coord,"contrib" : var_contrib,"cos2" : var_cos2}        

        self.model_ = "famd"

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
        
        # Recode
        rec2 = recodevarfamd(X=X)
        X_quanti = rec2["quanti"]
        X_quali = rec2["quali"]
        n_cont2 = rec2["k1"]
        n_cat2 = rec2["k2"]


        ####
        rec = self.call_["rec"]
        n_components = self.call_["n_components"]
        n_cont = rec["k1"]
        n_cat = rec["k2"]
        dummies = rec["dummies"]

        # Initialize coord
        #coord = pd.DataFrame(np.zeros((X.shape[0],n_components)),index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])

        Z = pd.DataFrame().astype("float")
        if n_cont2 > 0:
            if n_cont != n_cont2:
                raise TypeError("The number of continuous columns must be the same")
            
            # Standardscaler numerical variable
            Z1 = (X_quanti - self.call_["means"].values[:n_cont].reshape(1,-1))/self.call_["std"].values[:n_cont].reshape(1,-1)
            # Concatenate
            Z = pd.concat((Z,Z1),axis=1)
        
        if n_cat2 > 0:
            if n_cat != n_cat2:
                raise TypeError("The number of qualitatives columns must be the same")
            X_quali = revaluate_cat_variable(X_quali)
        
            # Standardscaler categorical Variable
            Y = np.zeros((X.shape[0],dummies.shape[1]))
            for i in np.arange(0,X.shape[0],1):
                values = [str(X_quali.iloc[i,k]) for k in np.arange(0,rec["quali"].shape[1])]
                for j in np.arange(dummies.shape[1]):
                    if dummies.columns[j] in values:
                        Y[i,j] = 1
            Y = pd.DataFrame(Y,index=X.index.tolist(),columns=dummies.columns)
            # New normalized data
            Z2 = (Y - self.call_["means"].values[n_cont:].reshape(1,-1))/self.call_["std"].values[n_cont:].reshape(1,-1)
            # Concatenate
            Z = pd.concat((Z,Z2),axis=1)
        
        # Supplementary individuals coordinates
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
    

def predictFAMD(self,X):
    """
    """
    pass

def supvarFAMD(self,X_quanti_sup=None,X_quali_sup=None):
    """
    
    """
    pass