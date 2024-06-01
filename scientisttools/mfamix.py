# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW

import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin

from .pca import PCA
from .mca import MCA
from .famd import FAMD
from .pcamix import PCAMIX
from .revaluate_cat_variable import revaluate_cat_variable
from .eta2 import eta2
from .function_eta2 import function_eta2
from .weightedcorrcoef import weightedcorrcoef

from .splitmix import splitmix

class MFAMIX(BaseEstimator,TransformerMixin):
    """
    
    """
    def __init__(self,
                n_components=5,
                group = None,
                name_group = None,
                num_group_sup = None,
                ind_sup = None,
                ind_weights=None,
                quanti_var_weights_mfa = None,
                quali_var_weights_mfa = None,
                parallelize = False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.quanti_var_weights_mfa = quanti_var_weights_mfa
        self.quali_var_weights_mfa = quali_var_weights_mfa
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : pandas DataFrame, shape (n_rows, n_columns)

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
        
        ###### Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")

        ###### Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns.tolist():
            X[col] = X[col].astype("object")
        
        ########################################################################################################################
        #   check if two categoricals variables have same categories
        ######################################################################################################################
        X = revaluate_cat_variable(X)
        
        #########################################################################################################################
        #   Check if group is None
        #########################################################################################################################
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("'group' must be a list or a tuple with the number of variables in each group")
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

        ##################################################################
        # Check if supplementary individuals
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None

        ############################################################################################################################
        #  Assigned group name
        ###########################################################################################################################
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("'group_name' must be a list or a tuple of group name")
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
        
        ######################################## Drop supplementary individuals  ##############################################
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            # Drop supplementary individuals
            X = X.drop(index=ind_sup_label)
        
        ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"{grp} group should have at least two columns")
        
        # Extract qualitatives and quantitatives groups
        all_nums = {}
        all_cats = {}
        for grp, cols in group_active_dict.items():
            all_nums[grp] = all(pd.api.types.is_numeric_dtype(X[col]) for col in cols)
            all_cats[grp]= all(pd.api.types.is_string_dtype(X[col]) for col in cols)
        
        ########################################## Summary qualitatives variables ###############################################
        # Compute statisiques
        group_label = pd.DataFrame(columns=["variable","group"])
        summary_quanti = pd.DataFrame()
        summary_quali = pd.DataFrame()
        for grp, cols in group_active_dict.items():
            if all_cats[grp]:
                for col in cols:
                    eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={col : "modalite"})
                    eff.insert(0,"variable",col)
                    eff.insert(0,"group",group_name.index(grp))
                    summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
            elif all_nums[grp]:
                summary = X[cols].describe().T.reset_index().rename(columns={"index" : "variable"})
                summary.insert(0,"group",group_name.index(grp))
                summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)
            else:
                # Split X into two group : quanti & quali
                X_quali = splitmix(X[cols])["quali"]
                X_quanti = splitmix(X[cols])["quanti"]

                ###### Summary for qualitatives variables
                for col in X_quali.columns:
                    eff = X_quali[col].value_counts().to_frame("count").reset_index().rename(columns={col: "categorie"})
                    eff.insert(0,"variable",col)
                    eff.insert(0,"group",group_name.index(grp))
                    summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
                
                ####### Summary of quantitatives variables
                summary = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary.insert(0,"group",group_name.index(grp))
                summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)
            
            #
            row_grp = pd.Series(cols,name='variable').to_frame()
            row_grp.insert(0,"group",grp)
            group_label = pd.concat((group_label,row_grp),axis=0,ignore_index=True)
               
        # Convert effectif and count to int
        summary_quali["count"] = summary_quali["count"].astype("int")
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.group_label_ = group_label

        ### Store summary
        self.summary_quanti_ = summary_quanti
        self.summary_quali_ = summary_quali

        ###############################################################################################################################
        #   Set weights : Individuals weights, quantitatives variables weights, qualitatives variables weights
        ###############################################################################################################################

        ########### Set indiviuduals weights
        if self.ind_weights is None:
            ind_weights = (np.ones(X.shape[0])/X.shape[0]).tolist()
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = [x/np.sum(self.ind_weights) for x in self.ind_weights]
        
        ############################ Run a Factor Analysis in each group
        model = {}
        for grp, cols in group_active_dict.items():
            fa = PCAMIX(n_components=None,ind_weights=ind_weights,ind_sup=None,parallelize=self.parallelize)
            model[grp] = fa.fit(X[cols])
            if self.ind_sup is not None:
                X_ind_sup = X_ind_sup.astype("float")
                fa = PCAMIX(n_components=None,ind_weights=ind_weights,ind_sup=ind_sup,parallelize=self.parallelize)
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        ############################################### Separate  Factor Analysis for supplementary groups ######################################""
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            ####### Find columns for supplementary group
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=ind_sup_label)
            
            # PCAMIX with supplementary group
            for grp, cols in group_sup_dict.items():
                fa = PCAMIX(n_components=None,ind_weights=ind_weights,parallelize=self.parallelize)
                # Fit the model
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

        ##### Global PCAMIX
        var_weights = pd.Series(name="weight")
        for grp, cols in group_active_dict.items():
            weights = pd.Series([1/model[grp].eig_.iloc[0,0]]*len(cols),index=cols,name="weights")
            var_weights = pd.concat((var_weights,weights),axis=0) 
        
        # Extract quantitat
        X_quant = splitmix(X)["quanti"]
        X_qual = splitmix(X)["quali"]
        # Split weights between quantitatives and qualitatives
        quanti_weights = var_weights.loc[X_quant.columns]
        quali_weights = var_weights.loc[X_qual.columns]
        
        ###########################################################################################################
        # Fit global PCA
        ###########################################################################################################
        # Global PCAMIX without supplementary element
        global_pca = PCAMIX(n_components = self.n_components,
                            ind_weights = ind_weights,
                            quanti_weights = quanti_weights.values.tolist(),
                            quali_weights = quali_weights,
                            parallelize = self.parallelize).fit(X)

        # PCAMIX with supplementary columns
        if self.num_group_sup is not None:
            # Split data
            X_sup_quanti = splitmix(X_group_sup)["quanti"]
            X_sup_quali = splitmix(X_group_sup)["quali"]

            # Apply model for supplementary quantitatives
            if X_sup_quanti.shape[1]>0:
                ##################################################################################################"
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
                summary_quanti_sup.insert(0,"group",group_name.index(grp))
                self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

                ####### Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti- d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
                ### Concatenate
                Z_quanti_sup = pd.concat((X,X_sup_quanti),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns.tolist()]
                global_pca = PCAMIX(n_components = self.n_components,
                                    ind_weights = ind_weights,
                                    quanti_weights=quanti_weights.values.tolist(),
                                    quali_weights=quali_weights,
                                    quanti_sup=index,
                                    parallelize = self.parallelize).fit(Z_quanti_sup)
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            if X_sup_quali.shape[1]>1:
                # Concatenate
                Z_quali_sup = pd.concat((X,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                global_pca = PCAMIX(n_components = self.n_components,
                                    ind_weights = ind_weights,
                                    quanti_weights = quanti_weights.values.tolist(),
                                    quali_weights = quali_weights,
                                    quali_sup = index,
                                    parallelize = self.parallelize).fit(Z_quali_sup)
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_sup_.copy()
                summary_quali_var_sup.insert(0,"group",group_name.index(grp))
                
                # Append 
                self.summary_quali_ = pd.concat((self.summary_quali_,summary_quali_var_sup),axis=0,ignore_index=True)

        ########################################## Store global PCA
        self.global_pca_ = global_pca

        # Update number of components
        n_components = global_pca.svd_["V"].shape[1]

        # Partial individuals
        V = global_pca.svd_["V"]
        var_weights = global_pca.call_["var_weights"]
        

        





        self.model_ = "pcamix"

        return self
