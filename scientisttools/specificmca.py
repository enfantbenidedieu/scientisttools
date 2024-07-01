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

class SpecificMCA(BaseEstimator,TransformerMixin):
    """
    Specific Multiple Correspondence Analysis (SpecificMCA)
    -------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Specific Multiple Correspondence Analysis (SpecificMCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```
    >>> SpecificMCA(n_components = 5,excl = None,ind_weights = None,var_weights = None,ind_sup = None,quali_sup = None,quanti_sup = None,parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `excl` : dictionary indicating the "junk" categories

    `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform row weights); the weights are given only for the active individuals
    
    `var_weights` : an optional variables weights (by default, a list/tuple of 1/(number of active variables) for uniform row weights); the weights are given only for the active variables
    
    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `quali_sup` : an integert or a list/tuple indicating the indexes of the categorical supplementary variables

    `quanti_sup` : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply

    Atttributes
    -----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `var_` : dictionary of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    `ind_` : dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine, contributions)

    `ind_sup_` : dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `quanti_sup_` : dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes)

    `quali_sup_` : dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a criterion with a Normal distribution, and eta2 which is the square correlation corefficient between a qualitative variable and a dimension)

    `call_` : dictionary with some statistics

    `model_` : string specifying the model fitted = 'specificmca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Le Roux B. and Rouanet H., Geometric Data Analysis: From Correspondence Analysis to Stuctured Data Analysis, Kluwer Academic Publishers, Dordrecht (June 2004).

    Le Roux B. and Rouanet H., Multiple Correspondence Analysis, SAGE, Series: Quantitative Applications in the Social Sciences, Volume 163, CA:Thousand Oaks (2010).

    Le Roux B. and Jean C. (2010), Développements récents en analyse des correspondances multiples, Revue MODULARD, Numéro 42

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_mca_ind, get_mca_var, get_mca, summaryMCA, predictSpecificMCA, supvarSpecificMCA, dimdesc, fviz_mca_ind, fviz_mca_mod, fviz_mca_var, fviz_mca_biplot

    Examples
    --------
    ```python
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import SpecificMCA
    >>> # Junk 
    >>> excl = {"Sick" : "Sick_n", "Sex" : "F"}
    >>> res_mca = SpecificMCA(n_components=5,excl=excl,quanti_sup=[0,1])
    >>> res_mca.fit(poison)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 excl = None,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 quali_sup = None,
                 quanti_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.excl = excl
        self.ind_weights = ind_weights
        self.var_weights = var_weights
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
        
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Check if excl is not None
        if self.excl is None:
            raise TypeError("'excl' must be assigned for speMCA function either use MCA function")
        elif not isinstance(self.excl,dict):
            raise TypeError("'excl' must be a dictionary")
        
        # Set index name as None
        X.index.name = None

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Drop level if ndim greater than 1 and reset columns name
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
        
        ####################################### Multiple Correspondence Anlysis (MCA) ##################################################
        # Check if 
        n_rows, n_cols = X.shape

        ################################################################################################
        # Set individuals weight
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        ################### Set variables weights ##################################################
        var_weights = pd.Series(name="weight").astype("float")
        if self.var_weights is None:
            for col in X.columns.tolist():
                var_weights[col] = 1/X.shape[1]
        elif not isinstance(self.var_weights,pd.Series):
            raise ValueError("'var_weights' must be a pandas series where index are variables names and values are variables weights.")
        else:
            for col in X.columns.tolist():
                var_weights[col] = self.var_weights[col]/self.var_weights.values.sum()

        ############################################### Dummies tables ############################################
        dummies = pd.concat((pd.get_dummies(X[col],prefix=col,prefix_sep="_",dtype=int) for col in X.columns.tolist()),axis=1)

        # Effectif par modalite
        I_k = dummies.sum(axis=0)
        # Prorportion par modalité
        p_k = dummies.mean(axis=0)
        # Center dummies data
        Z = pd.concat((dummies.loc[:,k]*(1/p_k[k])-1 for k  in dummies.columns.tolist()),axis=1)

        ###### Define mod weights
        mod_weights = pd.Series(name="weight").astype("float")
        for col in X.columns.tolist():
            data = pd.get_dummies(X[col],prefix=col,prefix_sep="_",dtype=int)
            weights = data.mean(axis=0)*var_weights[col]
            mod_weights = pd.concat((mod_weights,weights),axis=0)

        ###################################### Set number of components ########################################## 
        if self.n_components is None:
            n_components =  dummies.shape[1] - n_cols
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,dummies.shape[1] -n_cols)

        ##### Categorie to exclude
        exclusion = []
        for key in self.excl.keys():
            val = f"{key}_{self.excl[key]}"
            exclusion.append(val)
        
       ######################### Update Z and Modalities weights
        # Remove exclusion
        Z = Z.drop(columns=exclusion)

        # remove exclusion
        mod_weights = mod_weights.loc[[x for x in dummies.columns if x not in exclusion]]

        # 
        self.call_ = {"Xtot" : Xtot ,
                      "X" : X, 
                      "dummies" : dummies,
                      "Z" : Z , 
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "var_weights" : var_weights,
                      "mod_weights" : mod_weights,
                      "excl" : exclusion,
                      "var_weights" : var_weights,
                      "n_components" : n_components,
                      "ind_sup" : ind_sup_label,
                      "quali_sup" : quali_sup_label,
                      "quanti_sup" : quanti_sup_label}

        # Individuals informations : Weights, Squared distance to origin and Inertia
        # Individuals square distance
        ind_dist2 = mapply(Z,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        ind_dist2.name = "Sq. Dist."
        # Individuals inertia
        ind_inertia = ind_dist2*ind_weights
        ind_inertia.name = "inertia"
        # Store all informations
        ind_infos = np.c_[ind_weights,ind_dist2,ind_inertia]
        ind_infos = pd.DataFrame(ind_infos,columns=["Weight","Sq. Dist.","Inertia"],index=X.index)

        ## Variables/categories infomations : Weights, Squared distance to origin and Inertia
        # Variables squared distance to origin
        var_dist2 = mapply(Z,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        var_dist2.name = "Sq. Dist."
        # Inertia
        var_inertia = var_dist2*mod_weights
        var_inertia.name = "inertia"
        # Variables/categoires
        var_infos = np.c_[mod_weights, var_dist2,var_inertia]
        var_infos = pd.DataFrame(var_infos,columns=["Weight","Sq. Dist.","Inertia"],index=Z.columns)

        #################### Singular Value Decomposition (SVD) ########################################
        svd = svd_triplet(X=Z,row_weights=ind_weights,col_weights=mod_weights.values,n_components=n_components)
        # Store Generalized Singular Value Decomposition (SVD) information
        self.svd_ = svd

        # Eigen - values
        eigen_values = svd["vs"][:(dummies.shape[1]-n_cols)]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        ###############################################################
        # Store all informations
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns = ["eigenvalue","difference","proportion","cumulative"],
                                 index=["Dim."+str(x+1) for x in range(eig.shape[0])])
        
        # save eigen value grather than threshold
        lambd = eigen_values[eigen_values>(1/n_cols)]
        # Add modified rated
        self.eig_["modified rates"] = 0.0
        self.eig_["cumulative modified rates"] = 100.0
        pseudo = (n_cols/(n_cols-1)*(lambd-1/n_cols))**2
        self.eig_.iloc[:len(lambd),4] = 100*pseudo/np.sum(pseudo)
        self.eig_.iloc[:,5] = np.cumsum(self.eig_.iloc[:,4])

        ## Indviduals informations : coordinates, square cosinus & relative contributions
        # coordinates
        ind_coord = svd["U"].dot(np.diag(svd["vs"][:n_components]))
        ind_coord = pd.DataFrame(ind_coord,index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])
        
        # Individuals contributions
        ind_contrib = mapply(ind_coord,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        ind_contrib = mapply(ind_contrib,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Square cosinus
        ind_cos2 = mapply(ind_coord,lambda x : (x**2)/ind_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Store all informations
        self.ind_ = {"coord" : ind_coord, "contrib" : ind_contrib, "cos2" : ind_cos2, "infos" : ind_infos}
        
        ## Variables informations : coordinates, square cosinus & relative contributions
        # coordinates
        var_coord = svd["V"].dot(np.diag(svd["vs"][:n_components]))
        var_coord = pd.DataFrame(var_coord,index=Z.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
        
        # Normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        corrected_var_coord = mapply(var_coord,lambda x: x*svd["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Relative contributions
        var_contrib = mapply(var_coord,lambda x : (x**2)*mod_weights.values,axis=0,progressbar=False,n_workers=n_workers)
        var_contrib = mapply(var_contrib,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
            
        # Variables Cos2
        var_cos2 = mapply(var_coord,lambda x : (x**2)/var_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Valeur test des modalités
        var_vtest = pd.concat(((var_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*I_k[k])/(X.shape[0]-I_k[k]))).to_frame().T for k in mod_weights.index),axis=0)

        ########################################################################################################################
        #       Qualitative informations
        ####################################" Correlation ratio #################################################################
        def function2_eta2(X,lab,x,weights,excl,n_workers):
            """
            Correlation ratio - Eta2
            ------------------------

            Description
            -----------
            Perform correlation ratio eta square

            Parameters
            ----------
            X : pandas dataframe of shape (n_rows, n_columns), dataframe containing categoricals variables

            lab : name of a columns in X

            x : pandas dataframe of individuals coordinates of shape (n_rows, n_components)

            weights : array with the weights of each row

            excl : categories to exclude in a columns

            n_workers : Maximum amount of workers (processes) to spawn.

            Return
            ------
            value : pandas dataframe of shape (n_columns, n_components)

            Author(s)
            ---------
            Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
            """
            def fct_eta2(idx):
                # Tratement of exclusion cases
                if lab in excl.keys():
                    tt = pd.get_dummies(X[lab],dtype=int).drop(columns=excl[lab])
                else:
                    tt = pd.get_dummies(X[lab],dtype=int)
                ni  = mapply(tt, lambda k : k*weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
                num = mapply(mapply(tt,lambda k : k*x[:,idx],axis=0,progressbar=False,n_workers=n_workers),
                                lambda k : k*weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)**2
                num = np.sum([num[k]/ni[k] for k in num.index])
                denom = np.sum(x[:,idx]*x[:,idx]*weights)
                return num/denom
            res = pd.DataFrame(np.array(list(map(lambda i : fct_eta2(i),range(x.shape[1])))).reshape(1,-1),
                               index=[lab],columns=["Dim."+str(x+1) for x in range(x.shape[1])])
            return res
        
        quali_eta2 = pd.concat((function2_eta2(X=X,lab=col,x=ind_coord.values,weights=ind_weights,excl=self.excl,n_workers=n_workers) for col in X.columns.tolist()),axis=0)

        # Contribution des variables
        quali_contrib = pd.DataFrame().astype("float")
        for col in X.columns:
            contrib = var_contrib.loc[var_contrib.index.str.startswith(col, na=False),:].sum(axis=0).to_frame(col).T
            quali_contrib = pd.concat((quali_contrib,contrib),axis=0)

        # Inertia for the variables
        quali_inertia = pd.Series([(len(np.unique(X[col]))-1)/X.shape[0] for col in X.columns.tolist()],index=X.columns.tolist(),name="inertia")

        #####################################
       
        self.var_ = {"coord" : var_coord, 
                     "corrected_coord":corrected_var_coord,
                     "contrib" : var_contrib, 
                     "cos2" : var_cos2, 
                     "infos" : var_infos,
                     "vtest" : var_vtest, 
                     "eta2" : quali_eta2, "inertia" : quali_inertia,"var_contrib" : quali_contrib}
        # Inertia
        inertia = (dummies.shape[1]/n_cols) - 1

        # Eigenvalue threshold
        kaiser_threshold = 1/n_cols
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
            for i in np.arange(X_ind_sup.shape[0]):
                values = [X.columns[k]+"_"+X_ind_sup.iloc[i,k] for k in np.arange(0,X.shape[1])]
                for j in np.arange(dummies.shape[1]):
                    if dummies.columns[j] in values:
                        Y[i,j] = 1
            Y = pd.DataFrame(Y,columns=dummies.columns,index=X_ind_sup.index)

            ## Drop columns with exclusion
            Y = Y.drop(columns=exclusion)

            # Standardisation
            Z_sup = pd.concat((Y.loc[:,k]*(1/p_k[k])-1 for k in Y.columns),axis=1)

            # Coordinates
            ind_sup_coord = mapply(Z_sup,lambda x : x*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)] 

            # Square distance
            ind_sup_dist2 = mapply(Z_sup,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_sup_dist2.name = "Sq. Dist."

            # Square cosinus
            ind_sup_cos2 = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Store all informations 
            self.ind_sup_ = {"coord" : ind_sup_coord, "cos2" : ind_sup_cos2, "dist" : ind_sup_dist2}
        
        ###############################################################################################################
        #   Supplementary qualitatives variables
        ################################################################################################################
        # Supplementary qualitatives variables statistics
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)

            # Transform to object
            X_quali_sup = X_quali_sup.astype("object")

            # Revaluate X
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            # Dummies table
            X_quali_dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in X_quali_sup.columns.tolist()),axis=1)

            # Correlation Ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=ind_coord.values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)
            
            # # Coordinates of supplementary categories - corrected
            quali_sup_coord = mapply(X_quali_dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(ind_coord)
            quali_sup_coord = mapply(quali_sup_coord,lambda x : x/svd["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)

            # Standardisation
            quali_sup_p_k = X_quali_dummies.mean(axis=0)
            Z_quali_sup = pd.concat((X_quali_dummies.loc[:,k]*(1/quali_sup_p_k[k])-1 for k  in X_quali_dummies.columns.tolist()),axis=1)
            
            # Square distance to origin
            quali_sup_dist2 = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            quali_sup_dist2.name = "Sq. Dist."

            # Square cosinus
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2.values,axis=0,progressbar=False,n_workers=n_workers)

            # v-test
            quali_sup_n_k = X_quali_dummies.sum(axis=0)
            quali_sup_vtest = pd.concat(((quali_sup_coord.loc[k,:]*np.sqrt(((X.shape[0]-1)*quali_sup_n_k[k])/(X.shape[0] - quali_sup_n_k[k]))).to_frame(name=k).T for k in quali_sup_n_k.index),axis=0)

            # Store all informations
            self.quali_sup_ = {"coord" : quali_sup_coord,"cos2"  : quali_sup_cos2,"dist"  : quali_sup_dist2,"vtest" : quali_sup_vtest,"eta2"  : quali_sup_eta2}

        ##################################################################################################
        # Supplementary quantitatives variables
        ##################################################################################################
        # Supplementary quantiatives variables statistics
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)

            # Transform to float
            X_quanti_sup = X_quanti_sup.astype("float")

            # If NA fill with mean
            X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]
            
            # Compute average mean and standard deviation
            d1 = DescrStatsW(X_quanti_sup.values,weights=ind_weights,ddof=0)

            # Z = (X - mu)/sigma
            Z_quanti_sup = (X_quanti_sup -  d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)

            ####### Compute row coord
            quanti_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            quanti_sup_coord = quanti_sup_coord.T.dot(svd["U"][:,:n_components])
            quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            ############# Supplementary cos2 ###########################################
            quanti_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_sup_cor)
            quanti_sup_co2 = mapply(quanti_sup_coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)
            
            self.quanti_sup_ = {"coord" : quanti_sup_coord,"cos2" : quanti_sup_co2}

        self.model_ = "specificmca"
        
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

        # Check if X is a pandas DataFrame
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
        
        # Extract elements
        dummies = self.call_["dummies"]
        exclusion = self.call_["excl"]
        mod_weights = self.call_["mod_weights"]
        n_components = self.call_["n_components"]
        p_k = dummies.mean(axis=0)
        
        # Create a dummies table
        Y = np.zeros((X.shape[0],dummies.shape[1]))
        for i in np.arange(X.shape[0]):
            values = [X.columns[k]+"_"+X.iloc[i,k] for k in np.arange(0,self.call_["X"].shape[1])]
            for j in np.arange(dummies.shape[1]):
                if dummies.columns[j] in values:
                    Y[i,j] = 1
        Y = pd.DataFrame(Y,columns=dummies.columns,index=X.index)

        # Remove exclude categories
        Y = Y.drop(columns = exclusion)

        # Standardization
        Z = pd.concat((Y.loc[:,k]*(1/p_k[k])-1 for k in Y.columns),axis=1)

        # Coordinates
        coord = mapply(Z,lambda x : x*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)] 
        return coord

def predictSpecificMCA(self,X):
    """
    Predict projection for new individuals with Specific Multiple Correspondence Analysis (SpecificMCA)
    ---------------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Specific Multiple Correspondence Analysis (SpecificMCA)

    Usage
    -----
    ```python
    >>> predictSpecificMCA(self,X)
    ```

    Parameters
    ----------
    `self` : an object of class SpecificMCA

    `X` : a pandas/polars dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Returns
    -------
    dictionary of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : squared cosines of the new individuals

    `dist` : distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class PCA
    if self.model_ != "specificmca":
        raise TypeError("'self' must be an object of class SpecificMCA")
    
    # Check if columns are aligned
    if X.shape[1] != self.call_["X"].shape[1]:
        raise ValueError("'columns' aren't aligned")

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
    
    # Extract some elements
    dummies = self.call_["dummies"]
    n_components = self.call_["n_components"]
    mod_weights = self.call_["mod_weights"]
    exclusion = self.call_["excl"]
    p_k = dummies.mean(axis=0)
    
    # Convert to object
    X = X.astype("object")

    # Create dummies table for supplementary
    Y = np.zeros((X.shape[0],dummies.shape[1]))
    for i in np.arange(X.shape[0]):
        values = [X.columns[k]+"_"+X.iloc[i,k] for k in np.arange(0,X.shape[1])]
        for j in np.arange(dummies.shape[1]):
            if dummies.columns[j] in values:
                Y[i,j] = 1
    Y = pd.DataFrame(Y,columns=dummies.columns,index=X.index)

    ## Drop columns with exclusion
    Y = Y.drop(columns=exclusion)

    # Standardization
    Z = pd.concat((Y.loc[:,k]*(1/p_k[k])-1 for k in Y.columns),axis=1)

    # Coordinates
    coord = mapply(Z,lambda x : x*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
    coord.columns = ["Dim."+str(x+1) for x in range(n_components)] 
    
    # Square distance to origin
    dist2 = mapply(Z,lambda x : (x**2)*mod_weights.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    dist2.name = "Sq. Dist."

    # Square cosinus
    cos2 = mapply(coord,lambda x : (x**2)/dist2.values,axis=0,progressbar=False,n_workers=n_workers)

    # Store all informations 
    res = {"coord" : coord,"cos2" : cos2,"dist" : dist2}
    return res

def supvarSpecificMCA(self,X_quanti_sup=None, X_quali_sup=None):
    """
    Supplementary variables in Specific Multiple Correspondence Analysis (MCA)
    --------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Specific Multiple Correspondence Analysis (SpecificMCA)

    Usage
    -----
    ```python
    >>> supvarSpecificMCA(self,X_quanti_sup=None,X_quali_sup=None)
    ```

    Parameters
    ----------
    `self` : an object of class SpecificMCA

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives variables

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives variables

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
    >>>
    ```
    """
    # Check if self is and object of class MCA
    if self.model_ != "specificmca":
        raise TypeError("'self' must be an object of class SpecificMCA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1

    ind_weights = self.call_["ind_weights"].values
    n_components = self.call_["n_components"]
    
    ########################################################################################################################
    #                                          For supplementary quantitatives variables
    #########################################################################################################################

    if X_quanti_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_quanti_sup,pl.DataFrame):
            X_quanti_sup = X_quanti_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,pd.Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to float
        X_quanti_sup = X_quanti_sup.astype("float")

        # Recode variables
        X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]

        # Compute weighted average and and weighted standard deviation
        d1 = DescrStatsW(X_quanti_sup.values,weights=ind_weights,ddof=0)

        # Standardization
        Z_quanti_sup = (X_quanti_sup -  d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)

        # Multiply with individuals weights
        quanti_sup_coord = mapply(Z_quanti_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        # Apply transition relation
        quanti_sup_coord = quanti_sup_coord.T.dot(self.svd_["U"][:,:n_components])
        quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Supplementary quantitatives variables squared distance to origin
        quanti_sup_cor = mapply(Z_quanti_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
        quanti_sup_dist2 = np.dot(np.ones(X_quanti_sup.shape[0]),quanti_sup_cor)

        # Supplementary quantitatives variables squared cosinus (Cos2)
        quanti_sup_co2 = mapply(quanti_sup_coord,lambda x : (x**2)/quanti_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Store all informations
        quanti_sup = {"coord" : quanti_sup_coord,"cos2" : quanti_sup_co2}
    else:
        quanti_sup = None
    
    ###########################################################################################################################
    #                                                   For supplementary qualitatives variables
    ###########################################################################################################################

    if X_quali_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Compute dummies tables
        X_quali_dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in X_quali_sup.columns),axis=1)

        # Correlation Ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)
        
        # # Coordinates of supplementary categories - corrected
        quali_sup_coord = mapply(X_quali_dummies,lambda x : x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(self.ind_["coord"])
        quali_sup_coord = mapply(quali_sup_coord,lambda x : x/self.svd_["vs"][:n_components],axis=1,progressbar=False,n_workers=n_workers)

        # Supplementary qualitatives variables proportions
        quali_sup_p_k = X_quali_dummies.mean(axis=0)
        Z_quali_sup = pd.concat(((X_quali_dummies.loc[:,k]/quali_sup_p_k[k])-1 for k  in X_quali_dummies.columns.tolist()),axis=1)
        
        # Supplementary categories squared distance to origin
        quali_sup_dist2 = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        quali_sup_dist2.name = "Sq. Dist."

        # Supplementary qualitatives variables squared cosinus (Cos2)
        quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # v-test
        quali_sup_n_k = X_quali_dummies.sum(axis=0)
        quali_sup_vtest = pd.concat(((quali_sup_coord.loc[k,:]*np.sqrt(((n_rows-1)*quali_sup_n_k[k])/(n_rows - quali_sup_n_k[k]))).to_frame(name=k).T for k in quali_sup_n_k.index.tolist()),axis=0)

        # Store all informations
        quali_sup = {"coord" : quali_sup_coord,"cos2"  : quali_sup_cos2,"dist"  : quali_sup_dist2,"vtest" : quali_sup_vtest,"eta2"  : quali_sup_eta2}
    else:
        quali_sup = None
    
    # Store all informations
    res = {"quanti" : quanti_sup, "quali" :quali_sup}
    return res