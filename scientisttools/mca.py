# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .revaluate_cat_variable import revaluate_cat_variable
from .function_eta2 import function_eta2
from .svd_triplet import svd_triplet

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)
    ---------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Correspondence Analysis (MCA) with supplementary
    individuals, supplementary quantitative variables and supplementary
    categorical variables.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    ind_weights : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); 
                    the weights are given only for the active individuals
    
    var_weights : an optional variables weights (by default, a list/tuple of 1/(number of active variables) for uniform row weights); 
                    the weights are given only for the active variables
    
    benzecri : boolean, if True benzecri correction is applied

    greenacre : boolean, if True greenacre correction is applied

    ind_sup : a list/tuple indicating the indexes of the supplementary individuals

    quali_sup : a list/tuple indicating the indexes of the categorical supplementary variables

    quanti_sup : a list/tuple indicating the indexes of the quantitative supplementary variables

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    var_ : a dictionary of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine, contributions)

    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    quanti_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes)

    quali_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a 
                 criterion with a Normal distribution, and eta2 which is the square correlation corefficient between a qualitative variable and a dimension)
    
    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'mca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2008), Analyses Factorielles Simples et Multiples.4ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Version 1.0

    See Also
    --------
    get_mca_ind, get_mca_var, get_mca, summaryMCA, dimdesc

    Examples
    --------
    > X = poison # from FactoMineR R package

    > res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)

    > res_mca.fit(X)

    > summaryMCA(res_mca)
    """
    def __init__(self,
                 n_components = None,
                 ind_weights = None,
                 var_weights = None,
                 benzecri=True,
                 greenacre=True,
                 ind_sup = None,
                 quali_sup = None,
                 quanti_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.benzecri = benzecri
        self.greenacre = greenacre
        self.ind_sup = ind_sup
        self.quali_sup = quali_sup
        self.quanti_sup = quanti_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
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
        
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(include=np.number)
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
        
        ####################################### Multiple Correspondence Anlysis (MCA) ##################################################
        # Check if 
        X = revaluate_cat_variable(X)
        #########################################################################################################
        # Compute statistiques
        summary_quali = pd.DataFrame()
        for col in X.columns.tolist():
            eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
            eff.insert(0,"variable",col)
            summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")
        self.summary_quali_ = summary_quali

        ################################### Chi2 statistic test ####################################
        chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
        idx = 0
        for i in np.arange(X.shape[1]-1):
            for j in np.arange(i+1,X.shape[1]):
                tab = pd.crosstab(X.iloc[:,i],X.iloc[:,j])
                chi = sp.stats.chi2_contingency(tab,correction=False)
                row_chi2 = pd.DataFrame({"variable1" : X.columns.tolist()[i],
                                         "variable2" : X.columns.tolist()[j],
                                         "statistic" : chi.statistic,
                                         "dof"       : chi.dof,
                                         "pvalue"    : chi.pvalue},index=[idx])
                chi2_test = pd.concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                idx = idx + 1
        # Transform to int
        chi2_test["dof"] = chi2_test["dof"].astype("int")
        self.chi2_test_ = chi2_test

        ############################################### Dummies tables ############################################
        dummies = pd.concat((pd.get_dummies(X[col],dtype=float) for col in X.columns.tolist()),axis=1)
        
        ###################################### Set number of components ########################################## 
        if self.n_components is None:
            n_components =  dummies.shape[1] - X.shape[1]
        elif not isinstance(self.n_components,int):
            raise ValueError("Error : 'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("Error : 'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,dummies.shape[1] - X.shape[1])
        
        ################################################################################################
        # Set individuals weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        ################### Set variables weights ##################################################
        var_weights = pd.Series(name="weight").astype("float")
        if self.var_weights is None:
            for col in X.columns.tolist():
                var_weights[col] = 1/X.shape[1]
        elif not isinstance(self.var_weights,pd.Series):
            raise ValueError("Error : 'var_weights' must be a pandas series where index are variables names and values are variables weights.")
        else:
            for col in X.columns.tolist():
                var_weights[col] = self.var_weights[col]/self.var_weights.values.sum()

        #############################################################################################
        # Effectif par modalite
        I_k = dummies.sum(axis=0)
        # Prorportion par modalité
        p_k = dummies.mean(axis=0)
        Z = pd.concat((dummies.loc[:,k]*(1/p_k[k])-1 for k  in dummies.columns.tolist()),axis=1)

        ###### Define mod weights
        mod_weights = pd.Series(name="weight").astype("float")
        for col in X.columns.tolist():
            data = pd.get_dummies(X[col],dtype=float)
            weights = data.mean(axis=0)*var_weights[col]
            mod_weights = pd.concat((mod_weights,weights),axis=0)
        
        self.call_ = {"Xtot" : Xtot ,
                      "X" : X, 
                      "dummies" : dummies,
                      "Z" : Z , 
                      "row_marge" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "col_marge" : mod_weights,
                      "var_weights" : var_weights,
                      "n_components" : n_components}

        #################### Singular Value Decomposition (SVD) ########################################
        svd = svd_triplet(X=Z,row_weights=ind_weights,col_weights=mod_weights.values,n_components=n_components)
        # Store Singular Value Decomposition (SVD) information
        self.svd_ = svd

        # Eigen - values
        eigen_values = svd["vs"][:(dummies.shape[1]-X.shape[1])]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
    
        ###############################################################
        # Store all informations
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns = ["eigenvalue","difference","proportion","cumulative"],
                                 index=["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        # save eigen value grather than threshold
        lambd = eigen_values[eigen_values>(1/X.shape[1])]
        
        # Benzecri correction
        if self.benzecri:
            if len(lambd) > 0:
                # Apply Benzecri correction
                lambd_tilde = ((X.shape[1]/(X.shape[1]-1))*(lambd - 1/X.shape[1]))**2
                # Cumulative percentage
                s_tilde = 100*(lambd_tilde/np.sum(lambd_tilde))
                # Benzecri correction
                self.benzecri_correction_ = pd.DataFrame(np.c_[lambd_tilde,s_tilde,np.cumsum(s_tilde)],
                                                   columns=["eigenvalue","proportion","cumulative"],
                                                    index = ["Dim."+str(x+1) for x in np.arange(0,len(lambd))])

        # Greenacre correction
        if self.greenacre:
            if len(lambd) > 0:
                # Apply Greenacre correction
                lambd_tilde = ((X.shape[1]/(X.shape[1]-1))*(lambd - 1/X.shape[1]))**2
                s_tilde_tilde = X.shape[1]/(X.shape[1]-1)*(np.sum(eigen_values**2)-(dummies.shape[1]-X.shape[1])/(X.shape[1]**2))
                tau = 100*(lambd_tilde/s_tilde_tilde)
                self.greenacre_correction_ = pd.DataFrame(np.c_[lambd_tilde,tau,np.cumsum(tau)],
                                                    columns=["eigenvalue","proportion","cumulative"],
                                                    index = ["Dim."+str(x+1) for x in np.arange(0,len(lambd))])
        
        ######################################################################################################################
        #################################             Coordinates                             ################################
        ######################################################################################################################
        # Individuals coordinates
        ind_coord = svd["U"].dot(np.diag(np.sqrt(eigen_values[:n_components])))
        ind_coord = pd.DataFrame(ind_coord,index=X.index.tolist(),columns=["Dim."+str(x+1) for x in range(ind_coord.shape[1])])

        # Variables coordinates
        var_coord = svd["V"].dot(np.diag(np.sqrt(eigen_values[:n_components])))
        var_coord = pd.DataFrame(var_coord,index=dummies.columns.tolist(),columns=["Dim."+str(x+1) for x in range(var_coord.shape[1])])

        # Normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        corrected_var_coord = mapply(var_coord,lambda x: x*np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)

        ####################################### Contribution ########################################
        # Individuals contributions
        ind_contrib = mapply(ind_coord,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Variables contributions
        var_contrib = mapply(var_coord,lambda x : (x**2)*mod_weights.values,axis=0,progressbar=False,n_workers=n_workers)
        var_contrib = mapply(var_contrib,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        
        ######################################### Cos2 - Quality of representation ##################################################"
        # Row and columns cos2
        ind_dist2 = mapply(Z,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        ind_dist2.name = "dist"
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        ind_infos = pd.concat([np.sqrt(ind_dist2),ind_inertia],axis=1)
        ind_infos.insert(1,"weight",ind_weights)

        ################################ Columns informations ##################################################
        var_dist2 = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        var_dist2.name = "dist"
        var_inertia = var_dist2*mod_weights
        var_inertia.name = "inertia"
        var_infos = pd.concat([np.sqrt(var_dist2),var_inertia],axis=1)
        var_infos.insert(1,"weight",mod_weights)

        #######################################" Cos2 ########################################################
        # Individuals Cos2
        ind_cos2 = mapply(ind_coord,lambda x : (x**2)/ind_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # Variables Cos2
        var_cos2 = mapply(var_coord,lambda x : (x**2)/var_dist2,axis=0,progressbar=False,n_workers=n_workers)

        ####################################################################################################################
        # Valeur test des modalités
        var_vtest = pd.concat(((var_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*I_k[k])/(X.shape[0]-I_k[k]))).to_frame().T for k in I_k.index.tolist()),axis=0)

        ########################################################################################################################
        #       Qualitative informations
        ####################################" Correlation ratio #####################################################
        quali_eta2 = pd.concat((function_eta2(X=X,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X.columns.tolist()),axis=0)
        
        # Contribution des variables
        quali_contrib = pd.DataFrame().astype("float")
        for col in X.columns.tolist():
            modalite = np.unique(X[col]).tolist()
            contrib = var_contrib.loc[modalite,:].sum(axis=0).to_frame(col).T
            quali_contrib = pd.concat((quali_contrib,contrib),axis=0)

        # Inertia for the variables
        quali_inertia = pd.Series([(len(np.unique(X[col]))-1)/X.shape[0] for col in X.columns.tolist()],index=X.columns.tolist(),name="inertia")

        #####################################
        self.ind_ = {"coord" : ind_coord, "contrib" : ind_contrib, "cos2" : ind_cos2, "infos" : ind_infos}
        self.var_ = {"coord" : var_coord, "corrected_coord":corrected_var_coord,"contrib" : var_contrib, "cos2" : var_cos2, "infos" : var_infos,
                     "vtest" : var_vtest, "eta2" : quali_eta2, "inertia" : quali_inertia,"var_contrib" : quali_contrib}

        # Inertia
        inertia = (dummies.shape[1]/X.shape[1]) - 1

        # Eigenvalue threshold
        kaiser_threshold = 1/X.shape[1]
        kaiser_proportion_threshold = 100/inertia

        self.others_ = {"inertia" : inertia,
                        "threshold" : kaiser_threshold,
                        "proportion" : kaiser_proportion_threshold}
        
        #################################################################################################################
        #   Supplementary individuals informations
        #################################################################################################################
        # Compute supplementary individuals statistics
        if self.ind_sup is not None:
            # Convert to object
            X_ind_sup = X_ind_sup.astype("object")
            # Create dummies table for supplementary
            Y = np.zeros((X_ind_sup.shape[0],dummies.shape[1]))
            for i in np.arange(0,X_ind_sup.shape[0],1):
                values = [X_ind_sup.iloc[i,k] for k in np.arange(0,X.shape[1])]
                for j in np.arange(0,dummies.shape[1],1):
                    if dummies.columns.tolist()[j] in values:
                        Y[i,j] = 1
            ind_sup_dummies = pd.DataFrame(Y,columns=dummies.columns.tolist(),index=X_ind_sup.index.tolist())

            ################################ Supplementary row coordinates ##################################################
            ind_sup_coord = pd.DataFrame(np.zeros(shape=(X_ind_sup.shape[0],n_components)),index=X_ind_sup.index.tolist(),
                                     columns=["Dim."+str(x+1) for x in range(0,n_components)]).astype("float")
            for col in X.columns.tolist():
                modalite = np.unique(X[col]).tolist()
                data1 = ind_sup_dummies.loc[:,modalite]
                data2 = var_coord.loc[modalite,:]
                coord = (var_weights[col])*data1.dot(data2)
                ind_sup_coord = ind_sup_coord.add(coord)
            ind_sup_coord = mapply(ind_sup_coord,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)

            ################################ Supplementary row Cos2 ##########################################################
            Z_sup = pd.concat((ind_sup_dummies.loc[:,k]*(1/p_k[k])-1 for k  in ind_sup_dummies.columns.tolist()),axis=1)
            ind_sup_dist2 = mapply(Z_sup,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_sup_dist2.name = "dist"

            ##### Cos2
            ind_sup_cos2 = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            ##### 
            self.ind_sup_ = {"coord" : ind_sup_coord, "cos2" : ind_sup_cos2, "dist" : ind_sup_dist2}
        
        if self.quali_sup is not None:
            X_quali_sup = Xtot.iloc[:,quali_sup]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])

            #####################"
            X_quali_sup = X_quali_sup.astype("object")
            X_quali_dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=float) for col in X_quali_sup.columns.tolist()),axis=1)

            # Correlation Ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,
                                                      n_workers=n_workers) for col in X_quali_sup.columns.tolist()),axis=0)
            
            # # Coordinates of supplementary categories - corrected
            quali_sup_coord = mapply(X_quali_dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(ind_coord)
            quali_sup_coord = mapply(quali_sup_coord,lambda x : x/np.sqrt(eigen_values[:n_components]),axis=1,progressbar=False,n_workers=n_workers)

            #####################################################################################################################################
            ###### Distance à l'origine
            #####################################################################################################################################
            quali_sup_p_k = X_quali_dummies.mean(axis=0)
            Z_quali_sup = pd.concat((X_quali_dummies.loc[:,k]*(1/quali_sup_p_k[k])-1 for k  in X_quali_dummies.columns.tolist()),axis=1)
            quali_sup_dist2 = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            quali_sup_dist2.name = "dist"

            ################################## Cos2
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
            
            ################################## v-test
            quali_sup_n_k = X_quali_dummies.sum(axis=0)
            quali_sup_vtest = pd.concat(((quali_sup_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*quali_sup_n_k[k])/(X.shape[0] - quali_sup_n_k[k]))).to_frame(name=k).T for k in quali_sup_n_k.index.tolist()),axis=0)

            self.quali_sup_ = {"coord" : quali_sup_coord,
                               "cos2"  : quali_sup_cos2,
                               "dist"  : np.sqrt(quali_sup_dist2),
                               "vtest" : quali_sup_vtest,
                               "eta2"  : quali_sup_eta2}

            #################################### Summary supplementary qualitatives variables ##################################
            # Compute statistiques
            summary_quali_sup = pd.DataFrame()
            for col in X_quali_sup.columns.tolist():
                eff = X_quali_sup[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali_sup = pd.concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            summary_quali_sup.insert(0,"group","sup")
            # Concatenate with activate summary
            self.summary_quali_.insert(0,"group","active")
            self.summary_quali_ = pd.concat((self.summary_quali_,summary_quali_sup),axis=0,ignore_index=True)

            ################################### Chi2 statistic test ####################################
            chi2_test2 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in np.arange(X_quali_sup.shape[1]):
                for j in np.arange(X.shape[1]):
                    tab = pd.crosstab(X_quali_sup.iloc[:,i],X.iloc[:,j])
                    chi = sp.stats.chi2_contingency(tab,correction=False)
                    row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                            "variable2" : X.columns.tolist()[j],
                                            "statistic" : chi.statistic,
                                            "dof"       : chi.dof,
                                            "pvalue"    : chi.pvalue},index=[idx])
                    chi2_test2 = pd.concat((chi2_test2,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            # Transform to int
            chi2_test2["dof"] = chi2_test2["dof"].astype("int")
            chi2_test2.insert(0,"group","sup")
            self.chi2_test_.insert(0,"group","active")
            self.chi2_test_ = pd.concat((self.chi2_test_,chi2_test2),axis=0,ignore_index=True)
            
            ################################### Chi2 statistics between each supplementary qualitatives columns ###################
            if X_quali_sup.shape[1]>1:
                chi2_test3 = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for i in np.arange(X_quali_sup.shape[1]-1):
                    for j in np.arange(i+1,X_quali_sup.shape[1]):
                        tab = pd.crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j])
                        chi = sp.stats.chi2_contingency(tab,correction=False)
                        row_chi2 = pd.DataFrame({"variable1" : X_quali_sup.columns.tolist()[i],
                                                 "variable2" : X_quali_sup.columns.tolist()[j],
                                                 "statistic" : chi.statistic,
                                                 "dof"       : chi.dof,
                                                 "pvalue"    : chi.pvalue},index=[idx])
                        chi2_test3 = pd.concat((chi2_test3,row_chi2),axis=0,ignore_index=True)
                        idx = idx + 1
                # Transform to int
                chi2_test3["dof"] = chi2_test3["dof"].astype("int")
                chi2_test3.insert(0,"group","sup")
                self.chi2_test_ = pd.concat((self.chi2_test_,chi2_test3),axis=0,ignore_index=True)

        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.iloc[:,quanti_sup]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])

            ##############################################################################################################################
            X_quanti_sup = X_quanti_sup.astype("float")
            
            ############# Compute average mean and standard deviation
            d1 = DescrStatsW(X_quanti_sup.values,weights=ind_weights,ddof=0)

            # Z = (X - mu)/sigma
            Z_quanti_sup = (X_quanti_sup -  d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)

            ####### Compute row coord
            quanti_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            quanti_sup_coord = quanti_sup_coord.T.dot(svd["U"])
            quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(quanti_sup_coord.shape[1])]

            ############# Supplementary cos2 ###########################################
            quanti_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_sup_cor)
            quanti_sup_co2 = mapply(quanti_sup_coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)

            #############################################################################################################
            ##################### Compute statistics
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            
            self.quanti_sup_ = {"coord" : quanti_sup_coord,"cos2" : quanti_sup_co2}
            self.summary_quanti_ = summary_quanti_sup

        self.model_ = "mca"

        return self

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------
        
        X is projected on
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

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # Check if X is a pandas DataFrame
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
        
        # Add revaluate
        X = revaluate_cat_variable(X)
        
        Y = np.zeros((X.shape[0],self.call_["dummies"].shape[1]))
        for i in np.arange(0,X.shape[0],1):
            values = [X.iloc[i,k] for k in np.arange(0,self.call_["X"].shape[1])]
            for j in np.arange(0,self.call_["dummies"].shape[1],1):
                if self.call_["dummies"].columns.tolist()[j] in values:
                    Y[i,j] = 1
        ind_sup_dummies = pd.DataFrame(Y,columns=self.call_["dummies"].columns.tolist(),index=X.index.tolist())

        ################################ Supplementary row coordinates ##################################################
        ind_sup_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],self.ind_["coord"].shape[1])),index=X.index.tolist(),
                                     columns=self.ind_["coord"].columns.tolist()).astype("float")
        for col in self.call_["X"].columns.tolist():
            modalite = np.unique(self.call_["X"][col]).tolist()
            data1 = ind_sup_dummies.loc[:,modalite]
            data2 = self.var_["coord"].loc[modalite,:]
            coord = (self.call_["var_weights"][col])*data1.dot(data2)
            ind_sup_coord = ind_sup_coord.add(coord)
        # Apply correction
        ind_sup_coord = mapply(ind_sup_coord,lambda x : x/np.sqrt(self.eig_.iloc[:,0][:self.call_["n_components"]]),axis=1,progressbar=False,n_workers=n_workers)
        return ind_sup_coord

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
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