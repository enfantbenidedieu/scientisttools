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
class FAMD(BaseEstimator,TransformerMixin):
    """
    Factor Analysis of Mixed Data (FAMD)
    ------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Factor Analysis of Mixed Data (FAMD) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    FAMD is a principal component method dedicated to explore data with both continuous and categorical variables. 
    It can be seen roughly as a mixed between PCA and MCA. More precisely, the continuous variables are scaled 
    to unit variance and the categorical variables are transformed into a disjunctive data table (crisp coding) 
    and then scaled using the specific scaling of MCA. This ensures to balance the influence of both continous and 
    categorical variables in the analysis. It means that both variables are on a equal foot to determine the dimensions 
    of variability. This method allows one to study the similarities between individuals taking into account mixed 
    variables and to study the relationships between all the variables.

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
                 n_components = None,
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
        else:
            raise TypeError("Error : No qualitatives columns in data. Please use PCA function instead.")
        
        ##### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        else:
            raise TypeError("Error : No quantitatives columns in data. Please use MCA function instead.")

        ############################
        # Check is quali sup
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif ((isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple))  and len(self.quali_sup)>=1):
                quali_sup = [int(x) for x in self.quali_sup]

        #  Check if quanti sup
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif ((isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple))  and len(self.quanti_sup)>=1):
                quanti_sup = [int(x) for x in self.quanti_sup]
        
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        ####################################### Fill NA in quantitatives columns wih mean
        if is_quanti.isnull().any().any():
            col_list = is_quanti.columns.tolist()
            X[col_list] = mapply(X[col_list], lambda x : x.fillna(x.mean(),inplace=True),axis=0,progressbar=False,n_workers=n_workers)
            raise Warning("Missing values are imputed by the mean of the variable.")

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X

        ####################################### Drop supplementary qualitative columns ########################################
        if self.quali_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quali_sup])
        
        ######################################## Drop supplementary quantitatives columns #######################################
        if self.quanti_sup is not None:
            X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quanti_sup])
        
        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
        ############################ Split X in quantitatives and qualitatives
        # Compute statistics
        X_quant = X.select_dtypes(include=np.number)
        X_qual = X.select_dtypes(include=["object","category"])

        # Check if NULL
        if X_quant.empty and not X_qual.empty:
            raise ValueError("Error : There is no continuous variables in X. Please use MCA function.")
        elif X_qual.empty and not X_quant.empty:
            raise ValueError("Error : There is no categoricals variables in X. Please use PCA function.")

        ############################################## Summary
        ################## Summary quantitatives variables ####################
        summary_quanti = X_quant.describe().T.reset_index().rename(columns={"index" : "variable"})
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.summary_quanti_ = summary_quanti

        ################# Summary categoricals variables ##########################
        #########################################################################################################
        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X_qual.columns.tolist():
            eff = X_qual[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")
        self.summary_quali_ = summary_quali
        
        ################################### Chi2 statistic test ####################################
        if X_qual.shape[1]>1:
            chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in np.arange(X_qual.shape[1]-1):
                for j in np.arange(i+1,X_qual.shape[1]):
                    tab = pd.crosstab(X_qual.iloc[:,i],X_qual.iloc[:,j])
                    chi = sp.stats.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_qual.columns.tolist()[i],
                                            "variable2" : X_qual.columns.tolist()[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[idx])
                    chi2_test = pd.concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            # Transform to int
            chi2_test["dof"] = chi2_test["dof"].astype("int")
            self.chi2_test_ = chi2_test

        ###########################################################################################
        ########### Set row weight and quanti weight
        ###########################################################################################

        # Set row weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of row weight.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'row_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        ####################################################################################################
        ################################## Treatment of continues variables ################################
        ####################################################################################################
        # Set columns weight
        if self.quanti_weights is None:
            quanti_weights = np.ones(X_quant.shape[1])
        elif not isinstance(self.quanti_weights,list):
            raise ValueError("Error : 'quanti_weights' must be a list of quantitatives weights")
        elif len(self.quanti_weights) != X_quant.shape[1]:
            raise ValueError(f"Error : 'quanti_weights' must be a list with length {X_quant.shape[1]}.")
        else:
            quanti_weights = np.array(self.quanti_weights)
        
        ###########################################################################
        # Weighted Pearson correlation between continuous variables
        col_corr = weightedcorrcoef(x=X_quant,w=ind_weights)

        ############# Compute weighted average mean and standard deviation
        d1 = DescrStatsW(X_quant,weights=ind_weights,ddof=0)
        means = d1.mean.reshape(1,-1)
        std = d1.std.reshape(1,-1)
        Z1 = (X_quant - means)/std

        ###############################################################################################
        ##################################### Treatment of qualitatives variables #####################
        ###############################################################################################

        ################### Set variables weights ##################################################
        quali_weights = pd.Series(index=X_qual.columns.tolist(),name="weight").astype("float")
        if self.quali_weights is None:
            for col in X_qual.columns.tolist():
                quali_weights[col] = 1/X_qual.shape[1]
        elif not isinstance(self.quali_weights,dict):
            raise ValueError("Error : 'quali_weights' must be a dictionary where keys are qualitatives variables names and values are qualitatives variables weights.")
        elif len(self.quali_weights.keys()) != X_qual.shape[1]:
            raise ValueError(f"Error : 'quali_weights' must be a dictionary with length {X_qual.shape[1]}.")
        else:
            for col in X_qual.columns.tolist():
                quali_weights[col] = self.quali_weights[col]/sum(self.quali_weights)
        
        ###################### Set categories weights
        # Normalisation des variables qualitatives
        dummies = pd.concat((pd.get_dummies(X_qual[col]) for col in X_qual.columns.tolist()),axis=1)

        ###### Define mod weights
        mod_weights = pd.Series().astype("float")
        for col in X_qual.columns.tolist():
            data = pd.get_dummies(X_qual[col])
            weights = data.mean(axis=0)*quali_weights[col]
            mod_weights = pd.concat((mod_weights,weights),axis=0)
        
        ############################ Compute weighted mean and weighted standards 
        # Normalize Z2
        p_k = mapply(dummies,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        mean_k = np.average(dummies,axis=0,weights=ind_weights).reshape(1,-1)
        prop = p_k.values.reshape(1,-1)

        #####
        Z2 = (dummies - mean_k)/np.sqrt(prop)

        # Concatenate the 2 dataframe
        Z = pd.concat([Z1,Z2],axis=1)

        #################### Set number of components
        if self.n_components is None:
            n_components = min(X.shape[0]-1, Z.shape[1]-X_qual.shape[1])
        elif not isinstance(self.n_components,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("Error : 'n_components' must be greater or equal than 1.")
        else:
            n_components = min(self.n_components, X.shape[0]-1, Z.shape[1]-X_qual.shape[1])

         #Store call informations  : X = Z, M = diag(col_weight), D = diag(row_weight) : t(X)DXM
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "quanti" : X_quant,
                      "quali" : X_qual,
                      "dummies" : dummies,
                      "Z" : Z,
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "mod_weights" : pd.Series(1/p_k,index=dummies.columns.tolist(),name="weight"),
                      "means" : pd.Series(means[0],index=X_quant.columns.tolist(),name="average"),
                      "std" : pd.Series(std[0],index=X_quant.columns.tolist(),name="scale"),
                      "means_k" : pd.Series(mean_k[0],index=dummies.columns.tolist(),name="means"),
                      "prop" : pd.Series(prop[0],index=dummies.columns.tolist(),name="prop"),
                      "n_components" : n_components}

        ########################################################################################################################
        #################### Informations about individuals #################################################################### 
        ########################################################################################################################
        # Distance between individuals and inertia center
        ind_dist2 = (mapply(Z1,lambda x : (x**2)*quanti_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)+
                     mapply(Z2,lambda x:  (x - np.sqrt(p_k))**2,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1))
        ind_dist2.name = "dist"
        # Individuals inertia
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        # Save all informations
        ind_infos = pd.concat((np.sqrt(ind_dist2),ind_inertia),axis=1)
        ind_infos.insert(1,"weight",ind_weights)

        ########################################################################################################################
        ################################  Informations about categories ########################################################
        ########################################################################################################################
        # Distance between ctegories
        dummies_weight = (dummies/prop)-1
        # Distance à l'origine
        quali_dist2 = mapply(dummies_weight,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        quali_dist2.name = "dist"
        # Inertie des lignes
        quali_inertia = quali_dist2*mod_weights
        quali_inertia.name = "inertia"
        # Save all informations
        quali_infos = pd.concat((np.sqrt(quali_dist2),quali_inertia),axis=1)
        quali_infos.insert(1,"weight",mod_weights)

        #########################################################################################################
        global_pca = PCA(standardize=False,n_components=n_components).fit(Z)

        ###########################################################################################################
        #                            Compute supplementary individuals informations
        ##########################################################################################################
        if self.ind_sup is not None:
            ##### Prepare supplementary columns
            X_ind_sup_quant = X_ind_sup[X_quant.columns.tolist()]
            X_ind_sup_qual = X_ind_sup[X_qual.columns.tolist()]
            #######
            Z1_ind_sup = (X_ind_sup_quant - means)/std

            Y = np.zeros((X_ind_sup.shape[0],dummies.shape[1]))
            for i in np.arange(0,X_ind_sup.shape[0],1):
                values = [str(X_ind_sup_qual.iloc[i,k]) for k in np.arange(0,X_qual.shape[1])]
                for j in np.arange(0,dummies.shape[1],1):
                    if dummies.columns.tolist()[j] in values:
                        Y[i,j] = 1
            row_sup_dummies = pd.DataFrame(Y,columns=dummies.columns.tolist(),index=X_ind_sup.index.tolist())
            
            Z2_ind_sup = (row_sup_dummies - mean_k)/np.sqrt(prop)
            Z_ind_sup = pd.concat((Z1_ind_sup,Z2_ind_sup),axis=1)
            # Concatenate
            Z_ind_sup = pd.concat((Z,Z_ind_sup),axis=0)
            global_pca = PCA(standardize=False,n_components=n_components,ind_sup=self.ind_sup).fit(Z_ind_sup)
            self.ind_sup_ = global_pca.ind_sup_
        
        ##########################################################################################################
        #                         Compute supplementary quantitatives variables statistics
        ###########################################################################################################
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.iloc[:,quanti_sup]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            ##################################################################################################"
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            self.summary_quanti_.insert(0,"group","active")
            # Concatenate
            self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)
            
            # Standardize
            d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = (X_quanti_sup - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
            Z_quanti_sup = pd.concat((Z,Z_quanti_sup),axis=1)
            # Find supplementary quantitatives columns index
            index = [Z_quanti_sup.columns.tolist().index(x) for x in X_quanti_sup.columns.tolist()]
            # Update PCA
            global_pca = PCA(standardize=False,n_components=n_components,ind_sup=None,quanti_sup=index).fit(Z_quanti_sup)
            self.quanti_sup_ = global_pca.quanti_sup_
        
        ##########################################################################################################
        #                         Compute supplementary qualitatives variables statistics
        ###########################################################################################################
        if self.quali_sup is not None:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            # Chi-squared test between new categorie
            if X_quali_sup.shape[1] > 1:
                chi_sup_stats = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                cpt = 0
                for i in range(X_quali_sup.shpe[1]-1):
                    for j in range(i+1,X_quali_sup.shape[1]):
                        tab = pd.crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j])
                        chi = sp.stats.chi2_contingency(tab,correction=False)
                        row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                    "variable2" : X_quali_sup.columns.tolist()[j],
                                    "statistic" : chi.statistic,
                                    "dof"       : chi.dof,
                                    "pvalue"    : chi.pvalue},index=[cpt])
                        chi_sup_stats = pd.concat([chi_sup_stats,row_chi2],axis=0)
                        cpt = cpt + 1
            
            # Chi-squared between old and new qualitatives variables
            chi_sup_stats2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
            cpt = 0
            for i in range(X_quali_sup.shape[1]):
                for j in range(X_qual.shape[1]):
                    tab = pd.crosstab(X_quali_sup.iloc[:,i],X_qual.iloc[:,j])
                    chi = sp.stats.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                            "variable2" : X_qual.columns.tolist()[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[cpt])
                    chi_sup_stats2 = pd.concat([chi_sup_stats2,row_chi2],axis=0,ignore_index=True)
                    cpt = cpt + 1
            
            ###### Add 
            if X_quali_sup.shape[1] > 1 :
                chi_sup_stats = pd.concat([chi_sup_stats,chi_sup_stats2],axos=0,ignore_index=True)
            else:
                chi_sup_stats = chi_sup_stats2
            
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
            index = [Z_quali_sup.columns.tolist().index(x) for x in X_quali_sup.columns.tolist()]
            # Update PCA
            global_pca = PCA(standardize=False,n_components=n_components,ind_sup=None,quali_sup=index).fit(Z_quali_sup)
            self.quali_sup_ = global_pca.quali_sup_
        
        # Store Singular Value Decomposition
        self.svd_ = global_pca.svd_
        
        # Eigen - values
        eigen_values = global_pca.svd_["vs"][:min(X.shape[0]-1, Z.shape[1]-X_qual.shape[1])]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        ########################### Row informations #################################################################
        self.ind_ = global_pca.ind_

        ############################ Quantitatives columns ###########################################################
        quanti_coord =  global_pca.var_["coord"].loc[X_quant.columns.tolist(),:]
        quanti_contrib = global_pca.var_["contrib"].loc[X_quant.columns.tolist(),:]
        quanti_cos2 = global_pca.var_["cos2"].loc[X_quant.columns.tolist(),:]
        self.quanti_var_ = {"coord" : quanti_coord, "contrib" : quanti_contrib,"cor":quanti_coord,"cos2" : quanti_cos2,"corr" : col_corr}
        
        # Extract categories coordinates form PCA
        pca_coord_mod = global_pca.var_["coord"].loc[dummies.columns.tolist(),:]
        ### Apply correction to have categoricals coordinates
        quali_coord = mapply(pca_coord_mod,lambda x : x*np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
        quali_coord = (quali_coord.T/np.sqrt(prop)).T
        quali_contrib = global_pca.var_["contrib"].loc[dummies.columns.tolist(),:]
        quali_cos2 = mapply(quali_coord,lambda x : (x**2)/quali_dist2,axis=0,progressbar=False,n_workers=n_workers)
        I_k = dummies.sum(axis=0)
        quali_vtest = pd.concat(((quali_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*I_k[k])/(X.shape[0]-I_k[k]))).to_frame(k).T for k in dummies.columns.tolist()),axis=0)
        quali_vtest = mapply(quali_vtest,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)
        self.quali_var_ = {"coord" : quali_coord, "contrib" : quali_contrib, "cos2" : quali_cos2, "infos" : quali_infos,"vtest":quali_vtest}

        ####################################   Add elements ###############################################
        #### Qualitatives eta2
        quali_var_eta2 = pd.concat((function_eta2(X=X_qual,lab=col,x=global_pca.ind_["coord"].values,weights=ind_weights,
                                                  n_workers=n_workers) for col in X_qual.columns.tolist()),axis=0)
        # Contributions des variables qualitatives
        quali_var_contrib = mapply(quali_var_eta2,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        # Cosinus carrés des variables qualitatives
        quali_var_cos2 = pd.concat((((quali_var_eta2.loc[col,:]**2)/(len(np.unique(X_qual[[col]]))-1)).to_frame(name=col).T for col in X_qual.columns.tolist()),axis=0)

        var_coord = pd.concat((quanti_cos2,quali_var_eta2),axis=0)
        var_contrib = pd.concat((quanti_contrib,quali_var_contrib),axis=0)
        var_cos2 = pd.concat((quanti_cos2**2,quali_var_cos2),axis=0)
        self.var_ = {"coord" : var_coord,"contrib" : var_contrib,"cos2" : var_cos2}

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
        
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Store continuous and categorical variables
        X_sup_quant = X[self.call_["quanti"].columns.tolist()]
        X_sup_qual = X[self.call_["quali"].columns.tolist()]

        # Standardscaler numerical variable
        Z1 = (X_sup_quant - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)

        # Standardscaler categorical Variable
        Y = np.zeros((X.shape[0],self.call_["dummies"].shape[1]))
        for i in np.arange(0,X.shape[0],1):
            values = [str(X_sup_qual.iloc[i,k]) for k in np.arange(0,X_sup_qual.shape[1])]
            for j in np.arange(0,self.call_["dummies"].shape[1],1):
                if self.call_["dummies"].columns.tolist()[j] in values:
                    Y[i,j] = 1
        Y = pd.DataFrame(Y,index=X.index.tolist(),columns=self.call_["dummies"].columns.tolist())
        # New normalized data
        Z2 = mapply(Y,lambda x : (x - self.call_["means_k"].values)/np.sqrt(self.call_["prop"].values),axis=1,progressbar=False,n_workers=n_workers)
        # Supplementary individuals coordinates
        coord = pd.concat((Z1,Z2),axis=1).dot(self.svd_["V"])
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