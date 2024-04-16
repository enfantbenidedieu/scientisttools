# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
import pingouin as pg

from statsmodels.stats.weightstats import DescrStatsW
import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin

from .mca import MCA
from .pca import PCA
from .famd import FAMD
from .revaluate_cat_variable import revaluate_cat_variable
from .eta2 import eta2
from .weightedcorrcoef import weightedcorrcoef
from .function_eta2 import function_eta2


class MFAQUAL(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis for Qualitatives Variables (MFAQUAL)
    -------------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Factor Analysis for Qualitatives Variables in the sense of Pagès J. (2002) with supplementary individuals
    and supplementary groups of variables. Active groups of variables must be qualitatives. Supplementary groups of variables 
    can be quantitative or categorical
 
    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    group : a list or a tuple with the number of variables in each group

    name_group : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)

    group_type : the type of variables in each group; three possibilities : 
                    - "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
                    - "n" for categorical variables
                    - "m" for mixed variables (quantitative and qualitative variables)
    
    num_group_sup : the indexes of the illustrative groups (by default, None and no group re illustrative)

    ind_sup : an integer, a list or a tuple of the supplementary individuals

    ind_weights : an optional individuals weights (by default, a list of 1/(number of individuals) for uniform weights), the weights 
                    are given only for the active individuals
    
    var_weights_mfa : an optional qualitatives variables weights (by defaut, a list of 1/(number of categoricals variables in the group)),
                            the weights are given only for qualitatives variables
    
    Parallelize : bool, default = False. Adding multi-core methods to PandasObject.

    Return
    ------
    summary_quali_ : a summary of the results for the categorical variables

    summary_quanti_ : a summary of the results for the quantitative variables

    separate_analyses_ : the results for the separate analyses

    svd_ : a dictionary of matrices containing all the results of the singular value decomposition

    eig_ : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the
            cumulative percentge of variance
    
    group_ : a dictionary of pandas dataframe containing all the results for the groups (Lg and RV coefficients, coordinates, square cosine,
                contributions, distance to the origin, the correlations between each group and each factor)
    
    inertia_ratio_ : inertia ratio

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,
            contributions)
    
    ind_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates,
                square cosine)
    
    quali_var_ : a dictionary of pandas dataframe containing all the results for the categorical variables (coordinates of each categories
                    of each variables, contribution and vtest which is a criterion with a normal distribution)
    
    quanti_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates,
                        correlation between variables and axes, cos2)
    
    quali_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of 
                        each categories of each variables, cos2 and vtest which is a criterion with a normal distribution)
    
    partial_axes_ : a dictionary of pandas dataframe containing all the results for the partial axes (coordinates, correlation between variables
                        and axes, correlation between partial axes)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfaqual'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    Pagès J (2002), Analyse factorielle multiple appliquée aux variables qualitatives et aux données mixtes, Revue de statistique appliquée, tome 50, n°4(2002), p. 5-37
    """
    def __init__(self,
                 n_components=5,
                 group = None,
                 name_group = None,
                 group_type = None,
                 num_group_sup = None,
                 ind_sup = None,
                 ind_weights = None,
                 var_weights_mfa = None,
                 parallelize=False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.group_type = group_type
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.var_weights_mfa = var_weights_mfa
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
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        ###### Transform all categoricals variables to object
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns.tolist():
            X[col] = X[col].astype("object")

        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        
        ########################################################################################################################
        #   check if two categoricals variables have same categories
        ######################################################################################################################
        X = revaluate_cat_variable(X)

        #########################################################################################################################
        #   Check if group is None
        #########################################################################################################################
        if self.group is None:
            raise ValueError("Error : 'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("Error : 'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]
        
        #######################################################################################################################
        # Remove supplementary group
        if self.num_group_sup is not None:
            # Set default values to None
            self.quali_var_sup_ = None
            self.quanti_var_sup_ = None
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        ####################################################################
        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
        
        #########################################################################################################################
        #   Check if group type in not None
        #########################################################################################################################
        if self.group_type is None:
            raise ValueError("Error : 'group_type' must be assigned.")
        
        #######################################################################################################################
        if len(self.group) != len(self.group_type):
            raise TypeError("Error : Not convenient group definition")

        ############################################################################################################################
        #  Assigned group name
        ###########################################################################################################################
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("Error : 'group_name' must be a list or a tuple of group name")
        else:
            group_name = [x for x in self.name_group]
        
        ##############################################################################################################################
        # check if group name is an integer
        #############################################################################################################################
        for i in range(len(group_name)):
            if isinstance(group_name[i],int) or isinstance(group_name[i],float):
                group_name[i] = "Gr"+str(i+1)
        
        ##############################################################################################################################
        #   Assigned group name to label
        #############################################################################################################################
        group_active_dict = {}
        group_sup_dict = {}
        debut = 0
        for i in range(len(nb_elt_group)):
            X_group = X.iloc[:,(debut):(debut+nb_elt_group[i])]
            if self.num_group_sup is not None:
                if i in num_group_sup:
                    new_elt = {group_name[i]:X_group.columns.tolist()}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
            debut = debut + nb_elt_group[i]

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

        ######################################## Drop supplementary groups columns #######################################
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))

        ######################################## Drop supplementary individuls  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.iloc[self.ind_sup,:]
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in ind_sup])
        
         ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"{grp} group should have at least two columns")
        
        # Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in X.columns.tolist())
        if not all_cat:
            raise TypeError("All actives columns must be categoricals")

        ####################################### Multiple Factor Analysis for Qualitatives Variables (MFAQUAL) ##################################################

        ########################################## Summary qualitatives variables ###############################################
        # Compute statisiques
        summary_quali = pd.DataFrame()
        for grp, cols in group_active_dict.items():
            for col in cols:
                eff = X[col].value_counts().to_frame("effectif").reset_index().rename(columns={"index" : "modalite"})
                eff.insert(0,"variable",col)
                eff.insert(0,"group",group_name.index(grp))
                summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["effectif"] = summary_quali["effectif"].astype("int")
        self.summary_quali_ = summary_quali
        
        ########### Set row weight and columns weight
        # Set row weight
        if self.ind_weights is None:
            ind_weights = (np.ones(X.shape[0])/X.shape[0]).tolist()
        elif not isinstance(self.ind_weights,list):
            raise ValueError("Error : 'ind_weights' must be a list of individuals weights")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"Error : 'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = [x/np.sum(self.ind_weights) for x in self.ind_weights]
        
        ############################# Set variables weight MFA #######################################
        var_weights_mfa = {}
        if self.var_weights_mfa is None:
            for grp, cols in group_active_dict.items():
                var_weights = pd.Series(index=cols,name="weights").astype("float")
                for col in cols:
                    var_weights[col] = 1/len(cols)
                var_weights_mfa[grp] = var_weights
        elif not isinstance(self.var_weights_mfa,dict):
            raise ValueError("Error : 'var_weights_mfa' must be a dictionary where keys are groups names and values are pandas series of variables weights in group.")
        else:
            for grp, cols in group_active_dict.items():
                var_weights = pd.Series(index=cols,name="weights").astype("float")
                for col in cols:
                    var_weights[col] = self.var_weights_mfa[grp][col]/self.var_weights_mfa[grp].values.sum()
                var_weights_mfa[grp] = var_weights
            
        # Run a Factor Analysis in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if self.group_type[group_name.index(grp)]=="n":
                # Multiple Correspondence Analysis (MCA)
                fa = MCA(n_components=None,ind_weights=ind_weights,ind_sup=None,var_weights=var_weights_mfa[grp],benzecri=False,greenacre=False,parallelize=self.parallelize)
            else:
                raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
            model[grp] = fa.fit(X[cols])

            ##### Add supplementary individuals
            if self.ind_sup is not None:
                # Select 

                X_ind_sup = X_ind_sup.astype("float")
                if self.group_type[group_name.index(grp)]=="c":
                    fa = MCA(n_components=None,ind_weights=ind_weights,ind_sup=ind_sup,var_weights=var_weights_mfa[grp],benzecri=False,greenacre=False,parallelize=self.parallelize)
                else:
                    raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        ############################################### Separate  Factor Analysis for supplementary groups ######################################""
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in self.ind_sup])
            
            for grp, cols in group_sup_dict.items():
                # Instance the FA model
                if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        fa = PCA(standardize=False,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        fa = PCA(standardize=True,n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for continues group 'group_type' should be one of 'c', 's'")
                elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        fa = MCA(n_components=None,ind_weights=ind_weights,benzecri=False,greenacre=False,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for categoricals group 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        fa = FAMD(n_components=None,ind_weights=ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("Error : for mixed group 'group_type' should be 'm'")
                model[grp] = fa.fit(X_group_sup[cols])

        ##################### Compute group disto
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="dist2")

        ##### Compute group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="dist2")

        ##### Store separate analysis
        self.separate_analyses_ = model

        ################################################# Standardize Data ##########################################################
        base        = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        for grp,cols in group_active_dict.items():
            # Compute dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
            # Effectif par categories
            I_k = dummies.sum(axis=0)
            # Apply standardize
            Z = pd.concat((dummies.loc[:,[k]]*(X.shape[0]/I_k[k])-1 for k  in dummies.columns.tolist()),axis=1)
            # Concatenate
            base = pd.concat((base,Z),axis=1)
            ###### Define weights of categories
            weights = pd.Series(name="weight").astype("float")
            for col in cols:
                data = pd.get_dummies(X[col])
                m_k = (data.mean(axis=0)*var_weights_mfa[grp][col])/model[grp].eig_.iloc[0,0]
                weights = pd.concat((weights,m_k),axis=0)
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        ####### Update number of components
        if self.n_components is None:
            n_components = base.shape[1] - X.shape[1]
        else:
            n_components = min(self.n_components,base.shape[1] - X.shape[1])
        
        # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index.tolist(),name="weight"),
                      "var_weights" : var_weights,
                      "group" : group_active_dict,
                      "group_name" : group_name}
        
        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        # Add original data to full base and global PCA without supplementary element
        D = pd.concat((base,X),axis=1)
        index = [D.columns.tolist().index(x) for x in D.columns.tolist() if x not in base.columns.tolist()]
        global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(D)
        quali_var = global_pca.quali_sup_.copy()
        quali_var["contrib"] = global_pca.var_["contrib"]

        #### Add supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup = X_ind_sup.astype("float")
            # Create a copy
            Z_ind_sup = pd.concat((base,X_ind_sup),axis=0)
            # Apply PCA
            global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),ind_sup=ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
            self.ind_sup_ = global_pca.ind_sup_.copy()

        #######################################""
        ###### Add supplementary group
        ################################################################################################################################""
        if self.num_group_sup is not None:
            X_sup_quanti = X_group_sup.select_dtypes(exclude=["object","category"])
            X_sup_quali = X_group_sup.select_dtypes(include=["object","category"])
            if X_sup_quanti.shape[1]>0:
                ##################################################################################################"
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
                summary_quanti_sup.insert(0,"group",group_name.index(grp))
                
                ####### Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
                ### Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns.tolist()]
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            if X_sup_quali.shape[1]>1:
                # Concatenate
                Z_quali_sup = pd.concat((base,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_.copy()
                summary_quali_var_sup.insert(0,"group",group_name.index(grp))
                
                # Append 
                self.summary_quanti_ = pd.concat((self.summary_quali_,summary_quali_var_sup),axis=0,ignore_index=True)

        ##########################################
        self.global_pca_ = global_pca

        ############################################# Removing duplicate value in cumulative percent #######################"
        cumulative = sorted(list(set(global_pca.eig_.iloc[:,3])))

        ##################################################################################################################
        #   Eigenvalues
        ##################################################################################################################
        self.eig_ = global_pca.eig_.iloc[:len(cumulative),:]

        ####### Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:len(cumulative)],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################
        ind = global_pca.ind_.copy()

        ########################################################################################################### 
        # Partiel coordinates for individuals
        ###########################################################################################################
        ##### Add individuals partiels coordinaates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
            # Partial coordinates
            coord_partial = mapply(dummies.dot(quali_var["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),axis=0,progressbar=False,n_workers=n_workers)
            coord_partial = len(list(group_active_dict.keys()))*mapply(coord_partial,lambda x : x/self.eig_.iloc[:,0].values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
            coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord_partial.columns.tolist()])
            ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
        # Assign
        ind["coord_partiel"] = ind_coord_partiel
        
        ###########################################################################################################
        #   Partiel coordinates for qualitatives columns
        ###########################################################################################################
        quali_var_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            ############################################################################################################################
            # Compute categories coordinates
            quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X[col]),axis=1).groupby(col).mean()for col in X.columns.tolist()),axis=0)
            quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_coord_partiel.columns.tolist()])
            quali_var_coord_partiel = pd.concat([quali_var_coord_partiel,quali_coord_partiel],axis=1)
                
        quali_var["coord_partiel"] = quali_var_coord_partiel
        # Store informations
        self.quali_var_  = quali_var

        ##########################################################################################################
        #   Partiel coordinates for supplementary qualitatives columns
        ###########################################################################################################
        if self.num_group_sup is not None:
            quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp_sup, cols_sup in group_sup_dict.items():
                if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        ############################################################################################################################
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup[col]),axis=1).groupby(col).mean()for col in cols_sup),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
            self.quali_var_sup_["coord_partiel"] = quali_var_sup_coord_partiel
        
        ################################################################################################"
        #    Inertia Ratios
        ################################################################################################
        #### "Between" inertia on axis s
        between_inertia = len(list(group_active_dict.keys()))*mapply(ind["coord"],lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        between_inertia.name = "between_inertia"

        ### Total inertial on axis s
        total_inertia = pd.Series(name="total_inertia").astype("float")
        for dim in ind["coord"].columns.tolist():
            value = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum().sum()
            inertia = pd.Series([value],index=[dim],name="total_inertia")
            total_inertia = pd.concat((total_inertia,inertia),axis=0)

        ### Inertia ratio
        inertia_ratio = between_inertia/total_inertia
        inertia_ratio.name = "inertia_ratio"
        self.inertia_ratio_ = inertia_ratio

        ##############################################################################################################
        #   Individuals Within inertia
        ##############################################################################################################
        ############################### Within inertia ################################################################
        ind_within_inertia = pd.DataFrame(index=X.index.tolist(),columns=ind["coord"].columns.tolist()).astype("float")
        for dim in ind["coord"].columns.tolist():
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        ######################################## Within partial inertia ################################################
        data = pd.DataFrame().astype("float")
        for dim in ind["coord"].columns.tolist():
            data1 = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x - ind["coord"][dim].values)**2,axis=0,progressbar=False,n_workers=n_workers)
            data1 = 100*data1/data1.sum().sum()
            data = pd.concat([data,data1],axis=1)

        ######## Rorder inertia by group
        ind_within_partial_inertia = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            partial_inertia = data[grp]
            partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns.tolist()])
            ind_within_partial_inertia = pd.concat([ind_within_partial_inertia,partial_inertia],axis=1)
        ind["within_partial_inertia"] = ind_within_partial_inertia

        #################
        self.ind_ = ind

        ##################################################################################################
        #   Partial axes informations
        #################################################################################################
        ########################################### Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            ######### Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        ############################################## Partial axes cos2
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        #########" Partial correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        #### Add 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
                all_coord = pd.concat([all_coord,data],axis=1)
            # Reorder
            all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
        
        #################################### Partial axes contrib ################################################"
        axes_contrib = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
            eig = self.separate_analyses_[grp].eig_.iloc[:nbcol,0].values/self.separate_analyses_[grp].eig_.iloc[0,0]
            contrib = mapply(partial_axes_coord[grp].iloc[:,:nbcol],lambda x : (x**2)*eig,axis=1,progressbar=False,n_workers=n_workers)
            contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
            axes_contrib  = pd.concat([axes_contrib,contrib],axis=1)
        
        partial_axes_contrib = mapply(axes_contrib,lambda x : 100*x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)

        #### Add a null dataframe
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                nbcol = min(n_components,self.separate_analyses_[grp].call_["n_components"])
                contrib = pd.DataFrame(np.zeros(shape=(n_components,nbcol)),index=["Dim."+str(x+1) for x in range(n_components)],columns=["Dim."+str(x+1) for x in range(nbcol)])
                contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns.tolist()])
                partial_axes_contrib = pd.concat([partial_axes_contrib,contrib],axis=1)
            ## Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])
                
        ###############
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : all_coord.corr()}

        #################################################################################################################
        # Group informations : coord
        #################################################################################################################
        # group coordinates
        group_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = pd.concat((function_eta2(X=X[cols],lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in cols),axis=0)
            coord = data.sum(axis=0)/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0])
            coord  = pd.DataFrame(coord.values.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_coord = pd.concat((group_coord,coord),axis=0)
        
        ########################################### Group contributions ############################################
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)

        ######################################## group cos2 ################################################################
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_dist2.loc[grp]).to_frame(grp).T for grp in group_coord.index.tolist()),axis=0)

        ########################################### Group correlations ###############################################
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=self.ind_["coord"],w=None)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns.tolist())
            group_correlation = pd.concat((group_correlation,correl),axis=0)

        #################################################################################################################
        # Measuring how similar groups
        #################################################################################################################
        Lg = pd.DataFrame().astype("float")
        for grp1,cols1 in group_active_dict.items():
            for grp2,cols2 in group_active_dict .items():
                # Sum of chi-squared
                sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X[col1],X[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                Lg.loc[grp1,grp2] = weighted_chi2
        
        # Reorder using active group name
        Lg = Lg.loc[list(group_active_dict.keys()),list(group_active_dict.keys())]

        if self.num_group_sup is not None:
            Lg_sup = pd.DataFrame().astype("float")
            for grp1, cols1 in group_sup_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    if (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square coefficient of correlation
                        sum_corr2 = np.array([(weightedcorrcoef(x=X_group_sup[col1],y=X_group_sup[col2],w=None)[0,1])**2 for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using the eigenvalues of each group
                        weighted_corr2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_corr2
                        Lg_sup.loc[grp1,grp2] = weighted_corr2
                    elif ((pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X_group_sup[col1],X_group_sup[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg_sup.loc[grp1,grp2] = weighted_chi2
                    elif (all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col1],X_group_sup[col2],digits=10)["Eta2"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(len(cols1)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
                    elif (all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols1) and all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols2)):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X_group_sup[col2],X_group_sup[col1],digits=10)["Eta2"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols2)))*sum_eta2
                        Lg_sup.loc[grp1,grp2] = weighted_eta2
                        Lg_sup.loc[grp2,grp1] = weighted_eta2
            
            ####### Concatenate
            Lg = pd.concat((Lg,Lg_sup),axis=0)
            # Fill NA with 0.0
            Lg = Lg.fillna(0.0)

            ####
            for grp1,cols1 in group_active_dict.items():
                for grp2, cols2 in group_sup_dict.items():
                    X1, X2 = X[cols1], X_group_sup[cols2]
                    if all(pd.api.types.is_string_dtype(X2[col]) for col in cols2):
                        # Sum of chi-squared
                        sum_chi2 = np.array([st.chi2_contingency(pd.crosstab(X1[col1],X2[col2]),correction=False).statistic for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues, number of categoricals variables and number of rows
                        weighted_chi2 = (1/(X.shape[0]*len(cols1)*len(cols2)*self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]))*sum_chi2
                        Lg.loc[grp1,grp2] = weighted_chi2
                        Lg.loc[grp2,grp1] = weighted_chi2
                    elif all(pd.api.types.is_numeric_dtype(X2[col]) for col in cols2):
                        # Sum of square correlation ratio
                        sum_eta2 = np.array([eta2(X1[col1],X2[col2],digits=10)["Eta2"] for col1 in cols1 for col2 in cols2]).sum()
                        # Weighted the sum using eigenvalues and number of categoricals variables
                        weighted_eta2 = (1/(self.separate_analyses_[grp1].eig_.iloc[0,0]*self.separate_analyses_[grp2].eig_.iloc[0,0]*len(cols1)))*sum_eta2
                        Lg.loc[grp1,grp2] = weighted_eta2
                        Lg.loc[grp2,grp1] = weighted_eta2
            
            ##################
            Lg = Lg.loc[group_name,group_name]
        
        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        ## RV Coefficient
        RV = pd.DataFrame().astype("float")
        for grp1 in Lg.index:
            for grp2 in Lg.columns:
                RV.loc[grp1,grp2] = Lg.loc[grp1,grp2]/(np.sqrt(Lg.loc[grp1,grp1])*np.sqrt(Lg.loc[grp2,grp2]))
        
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}

        ##### Add supplementary elements
        if self.num_group_sup is not None:
            group_sup_coord = pd.DataFrame().astype("float")
            for grp, cols in group_sup_dict.items():
                Xg = X_group_sup[cols]
                if all(pd.api.types.is_numeric_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    correl = np.sum((weightedcorrcoef(self.separate_analyses_[grp].call_["Z"],self.ind_["coord"],w=None)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/self.separate_analyses_[grp].eig_.iloc[0,0]
                    coord = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns = ["Dim."+str(x+1) for x in range(len(correl))])
                    group_sup_coord = pd.concat((group_sup_coord,coord),axis=0)
                elif all(pd.api.types.is_string_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    data = self.quali_var_sup_["eta2"].loc[cols,:]
                    coord = (data.sum(axis=0)/(Xg.shape[1]*self.separate_analyses_[grp].eig_.iloc[0,0]))
                    group_sup_coord = pd.concat((group_sup_coord,coord.to_frame(grp).T),axis=0)
                else:
                    raise TypeError("Error : All columns should have the same type.")
            
            #################################### group sup cos2 ###########################################################
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index.tolist()),axis=0)
            
            # Append two dictionnaries
            self.group_ = {**self.group_,**{"coord_sup" : group_sup_coord, "dist2_sup" : group_sup_dist2,"cos2_sup" : group_sup_cos2}}
            
        # Model name
        self.model_ = "mfaqual"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters:
        ----------
        X : pandas/polars DataFrame of shape (n_rows_,n_cols_)

        y : None
            y is ignored
        
        Return
        ------
        X_new : pandas dataframe of shape (n_rows, n_components)
        """
        self.fit(X)
        return self.ind_["coord"]

    def transform(self,X,y=None):
        """
        Apply the dimensionality reduction on X
        ----------------------------------------

        X is projected on the first axes previous extracted from a
        training set

        Parameters
        ----------
        X : pandas/polars DataFrame of shape (n_rows_sup, n_cols_)
            New data, where n_rows_sup is the number of supplementary
            row points and n_cols_ is the number of columns.
            X rows correspond to supplementary row points that are projected
            on the axes.
        
        y : None
            y is ignored
        
        Return
        ------
        X_new : array of float, shape (n_row_sup, n_components_)
                X_new : coordinates of the projections of the supplementary
                row points on the axes.
        """
        # Check if X is an instance of pd.DataFrame class
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("Error : DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Revaluate categoricals variables
        X = revaluate_cat_variable(X=X)
        
        #### Apply transition relations
        row_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],self.quali_var_["coord"].shape[1])),index=X.index.tolist(),columns=self.quali_var_["coord"].columns.tolist())
        for grp, cols in self.call_["group"].items():
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col]) for col in cols),axis=1)
            # Apply
            coord = mapply(dummies.dot(self.quali_var_["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),axis=0,progressbar=False,n_workers=n_workers)
            row_coord = row_coord + coord
        # Weighted by the eigenvalue
        ind_coord = mapply(row_coord ,lambda x : x/self.eig_.iloc[:,0][:self.call_["n_components"]],axis=1,progressbar=False,n_workers=n_workers)
        return ind_coord
