# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
import pingouin as pg

from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .mca import MCA
from .pca import PCA
from .famd import FAMD
from .revaluate_cat_variable import revaluate_cat_variable
from .weightedcorrcoef import weightedcorrcoef
from .function_eta2 import function_eta2
from .splitmix import splitmix
from .function_lg import function_lg
from .coeffRV import coeffRV

class MFAQUAL(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis for Qualitatives Variables (MFAQUAL)
    -------------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Factor Analysis for Qualitatives Variables in the sense of Pagès J. (2002) with supplementary individuals and supplementary groups of variables. Active groups of variables must be qualitatives. Supplementary groups of variables can be quantitative, qualitative or both.
    
    Usage
    -----
    ```python
    >>> MFAQUAL(n_components=5, group = None,name_group = None,group_type = None,num_group_sup = None,ind_sup = None,ind_weights = None,var_weights_mfa = None,parallelize=False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `group` : a list or a tuple with the number of variables in each group

    `name_group` : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)

    `group_type` : the type of variables in each group. Possible value are : 
        * "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (quantitative and qualitative variables)
    
    `num_group_sup` : the indexes of the illustrative groups (by default, None and no group re illustrative)

    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `ind_weights` : an optional individuals weights (by default, a list of 1/(number of individuals) for uniform weights), the weights are given only for the active individuals
    
    `var_weights_mfa` : an optional qualitatives variables weights (by defaut, a list of 1/(number of categoricals variables in the group)), the weights are given only for qualitatives variables
    
    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply

    Attributes
    ----------
    `summary_quali_` : a summary of the results for the categorical variables

    `summary_quanti_` : a summary of the results for the quantitative variables (if supplementary quantitative variables)

    `separate_analyses_` : the results for the separate analyses

    `svd_` : dictionary of matrices containing all the results of the singular value decomposition

    `eig_` : pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the cumulative percentge of variance
    
    `group_` : dictionary of pandas dataframe containing all the results for the groups (Lg and RV coefficients, factor coordinates, square cosine, contributions, distance to the origin, the correlations between each group and each factor)
    
    `inertia_ratio_` : inertia ratio

    `ind_` : dictionary of pandas dataframe containing all the results for the active individuals (factor coordinates, square cosinus, relative contributions)
    
    `ind_sup_` : dictionary of pandas dataframe containing all the results for the supplementary individuals (factor coordinates, square cosinus)
    
    `quali_var_` : dictionary of pandas dataframe containing all the results for the categorical variables (factor coordinates of each categories of each variables, relative contribution and vtest which is a criterion with a normal distribution)
    
    `quanti_var_sup_` : dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (factor coordinates, correlation between variables and axes, square cosinus)
    
    `quali_var_sup_` : dictionary of pandas dataframe containing all the results for the supplementary categorical variables (factor coordinates of each categories of each variables, square cosinus and vtest which is a criterion with a normal distribution)
    
    `partial_axes_` : dictionary of pandas dataframe containing all the results for the partial axes (factor coordinates, correlation between variables and axes, correlation between partial axes)
    
    `global_pca_` : the results of the analysis when it is considered as a unique weighted PCA

    `model_` : string specifying the model fitted = 'mfaqual'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed. Dunod
    
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    
    Pagès J (2002), Analyse factorielle multiple appliquée aux variables qualitatives et aux données mixtes, Revue de statistique appliquée, tome 50, n°4(2002), p. 5-37
    
    See also
    --------
    get_mfa_ind, get_mfa_var, get_mfa_partial_axes, summaryMFA, fviz_mfa_ind, fviz_mfa_col, fviz_mfa, dimdesc

    Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MFAQUAL
    >>> group_name = ["desc","desc2","symptom","eat"]
    >>> group = [2,2,5,6]
    >>> group_type = ["s"]+["n"]*3
    >>> num_group_sup = [0,1]
    >>> res_mfaqual = MFAQUAL(group=group,name_group=group_name,group_type=group_type,var_weights_mfa=None,num_group_sup=[0,1],parallelize=True)
    >>> res_mfaqual.fit(poison)
    ```
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
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X.index.name = None

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Transform all categoricals variables to object
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns.tolist():
            X[col] = X[col].astype("object")

        # Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        if is_quanti.shape[1]>0:
            for col in is_quanti.columns.tolist():
                X[col] = X[col].astype("float")
        
        #   check if two categoricals variables have same categories
        X = revaluate_cat_variable(X)

        #   Check if group is None
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise TypeError("'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]
        
        # Remove supplementary group
        if self.num_group_sup is not None:
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        # Check if individuls supplementary
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None
        
        #   Check if group type in not None
        if self.group_type is None:
            raise ValueError("'group_type' must be assigned")
        
        if len(self.group) != len(self.group_type):
            raise TypeError("Not convenient group definition")

        #  Assigned group name
        if self.name_group is None:
            group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
        elif not (isinstance(self.name_group,list) or isinstance(self.name_group,tuple)):
            raise TypeError("'group_name' must be a list or a tuple of group name")
        else:
            group_name = [x for x in self.name_group]
        
        # check if group name is an integer
        for i in range(len(group_name)):
            if isinstance(group_name[i],int) or isinstance(group_name[i],float):
                group_name[i] = "Gr"+str(i+1)
        
        #   Assigned group name to label
        group_active_dict = {}
        group_sup_dict = {}
        debut = 0
        for i in range(len(nb_elt_group)):
            X_group = X.iloc[:,(debut):(debut+nb_elt_group[i])]
            if self.num_group_sup is not None:
                if i in num_group_sup:
                    new_elt = {group_name[i]:X_group.columns}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns
            else:
                group_active_dict[group_name[i]] = X_group.columns
            debut = debut + nb_elt_group[i]
        
        # Create group label
        group_label = pd.DataFrame(columns=["variable","group"])
        for grp in group_active_dict.keys():
            row_grp = pd.Series(group_active_dict[grp],name='variable').to_frame()
            row_grp.insert(0,"group",grp)
            group_label = pd.concat((group_label,row_grp),axis=0,ignore_index=True)
        
        # Add supplementary group
        if self.num_group_sup is not None:
            for grp in group_sup_dict.keys():
                row_grp = pd.Series(group_sup_dict[grp],name='variable').to_frame()
                row_grp.insert(0,"group",grp)
                group_label = pd.concat((group_label,row_grp),axis=0,ignore_index=True)
        
        self.group_label_ = group_label

        # Store data
        Xtot = X.copy()

        # Drop supplementary groups columns
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))

        # Drop supplementary individuls
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            # Remove supplementary individuals
            X = X.drop(index=ind_sup_label)
        
        # Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"{grp} group should have at least two columns")
        
        # Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(X[col]) for col in X.columns)
        if not all_cat:
            raise TypeError("All actives columns must be categoricals")

        # Summary qualitative variables
        summary_quali = pd.DataFrame()
        for grp, cols in group_active_dict.items():
            for col in cols:
                summary = X[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                summary.insert(0,"variable",col)
                summary.insert(0,"group name",grp)
                summary.insert(0,"group",group_name.index(grp))
                summary_quali = pd.concat([summary_quali,summary],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")
        self.summary_quali_ = summary_quali

        # Set individuals weights
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        # Set variables weight MFA
        var_weights_mfa = pd.Series(name="weight").astype("float")
        if self.var_weights_mfa is None: 
            for grp, cols in group_active_dict.items():
                weights_mfa = pd.Series(np.ones(len(cols))/len(cols),index=cols,name="weights").astype("float")
                var_weights_mfa = pd.concat((var_weights_mfa,weights_mfa),axis=0)
        elif not isinstance(self.var_weights_mfa,pd.Series):
            raise ValueError("'var_weights_mfa' must be a pandas series where index are variables names and values variables weights")
        else:
            for grp, cols in group_active_dict.items():
                weights_mfa = pd.Series(index=cols,name="weights").astype("float")
                weights = self.var_weights_mfa.loc[cols]
                for col in cols:
                    weights_mfa[col] = weights[col]/weights.values.sum()
                var_weights_mfa = pd.concat((var_weights_mfa,weights_mfa),axis=0)
            
        # Run a multiple correspondence analysis (MCA) in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if self.group_type[group_name.index(grp)]=="n":
                # Multiple Correspondence Analysis (MCA)
                fa = MCA(n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols],benzecri=False,greenacre=False,parallelize=self.parallelize)
            else:
                raise TypeError("For active group 'group_type' should be 'n'")
            model[grp] = fa.fit(X[cols])

            # Add supplementary individuals
            if self.ind_sup is not None:
                # Transform to object
                X_ind_sup = X_ind_sup.astype("object")
                if self.group_type[group_name.index(grp)]=="c":
                    # Multiple Correspondence Analysis (MCA)
                    fa = MCA(n_components=self.n_components,ind_weights=self.ind_weights,ind_sup=self.ind_sup,var_weights=var_weights_mfa[cols],benzecri=False,greenacre=False,parallelize=self.parallelize)
                else:
                    raise TypeError("For categoricals group 'group_type' should be 'n'")
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        # Separate general factor analysis for supplementary groups
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=ind_sup_label)
            
            # General factor analysis
            for grp, cols in group_sup_dict.items():
                if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        # Center principal component analysis (PCA)
                        fa = PCA(standardize=False,n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        # Scale principal component analysis (PCA)
                        fa = PCA(standardize=True,n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("For continues variables 'group_type' should be one of 'c', 's'")
                elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        # Multiple Correspondence Analysis (MCA)
                        fa = MCA(n_components=self.n_components,ind_weights=self.ind_weights,benzecri=False,greenacre=False,parallelize=self.parallelize)
                    else:
                        raise TypeError("For categoricals variables 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        # Factor analysis of mixed data (FAMD)
                        fa = FAMD(n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("For mixed variables 'group_type' should be 'm'")
                # Fit the model
                model[grp] = fa.fit(X_group_sup[cols])
  
        # Square distance to origin for active group
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="Sq. Dist.")

        # Square distance to origin for supplementary group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="Sq. Dist.")

        # Store separate analysis
        self.separate_analyses_ = model

        # Standardize Data
        base = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        for grp,cols in group_active_dict.items():
            # Concatenate
            base = pd.concat((base,model[grp].call_["Z"]),axis=1)
            # Define weights of categories
            weights = pd.Series(name="weight").astype("float")
            for col in cols:
                data = pd.get_dummies(X[col],dtype=int)
                m_k = (data.mean(axis=0)*var_weights_mfa[col])/model[grp].eig_.iloc[0,0]
                weights = pd.concat((weights,m_k),axis=0)
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        # Update number of components
        if self.n_components is None:
            n_components = int(base.shape[1] - X.shape[1])
        else:
            n_components = int(min(self.n_components,base.shape[1] - X.shape[1]))
        
        # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                      "var_weights_mfa" : var_weights_mfa,
                      "var_weights" : var_weights,
                      "group" : group_active_dict,
                      "group_name" : group_name,
                      "ind_sup" : ind_sup_label}
        
        # Global PCA & add original data to full base and global PCA without supplementary element
        D = pd.concat((base,X),axis=1)
        index = [D.columns.tolist().index(x) for x in D.columns if x not in base.columns]
        global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(D)
        # Store a copy of all informations
        quali_var = global_pca.quali_sup_.copy()
        # Add contributions
        quali_var["contrib"] = global_pca.var_["contrib"]

        ###############################################################################################
        # Add supplementary individuals
        ###############################################################################################
        # Statistics for supplementary individuals
        if self.ind_sup is not None:
            # Transform to object
            X_ind_sup = X_ind_sup.astype("object")
            # Concatenate
            Z_ind_sup = pd.concat((base,X_ind_sup),axis=0)
            # Update PCA with supplementary individuals
            global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),ind_sup=self.ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
            # Store a copy of all informations
            self.ind_sup_ = global_pca.ind_sup_.copy()

        ###################################################################################################
        # Add supplementary group
        ################################################################################################################################""
        # Statistics for supplementary variables
        if self.num_group_sup is not None:
            X_sup_quanti = splitmix(X=X_group_sup)["quanti"]
            X_sup_quali = splitmix(X=X_group_sup)["quali"]

            # Statistics for supplementary quantitative variables
            if X_sup_quanti is not None:
                # Summary
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")

                # Find group name for quantitative variables
                quanti_sup_grp_name = [grp for grp, cols in group_sup_dict.items() for col in summary_quanti_sup["variable"] if col in cols]
                quanti_sup_grp_idx = [group_name.index(x) for x in quanti_sup_grp_name]
                summary_quanti_sup.insert(0,"group name",quanti_sup_grp_name)
                summary_quanti_sup.insert(0,"group",quanti_sup_grp_idx)
                self.summary_quanti_ = summary_quanti_sup
                
                # Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=self.ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)
                
                # Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns.tolist()]
                
                # Update PCA with supplementary quantitatives variables
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                # Store a copy of all informations
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            # Statistics for supplementary qualitative variables
            if X_sup_quali is not None:
                # Concatenate
                Z_quali_sup = pd.concat((base,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                # Update PCA with supplementary qualitatives variables
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                # Store a copy of all informations
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract summary
                summary_quali_var_sup = global_pca.summary_quali_.copy()

                # Find group name
                quali_sup_grp_name = [grp for grp, cols in group_sup_dict.items() for col in summary_quali_var_sup["variable"] if col in cols]
                quali_sup_grp_idx = [group_name.index(x) for x in quali_sup_grp_name]
                summary_quali_var_sup.insert(0,"group name", quali_sup_grp_name)
                summary_quali_var_sup.insert(0,"group", quali_sup_grp_idx)
        
                # Append
                self.summary_quali_ = pd.concat((self.summary_quali_,summary_quali_var_sup),axis=0,ignore_index=True)

        # Store global PCA
        self.global_pca_ = global_pca

        # Removing duplicate value in cumulative percent
        cumulative = sorted(list(set(global_pca.eig_.iloc[:,3])))

        # Eigenvalues
        self.eig_ = global_pca.eig_.iloc[:len(cumulative),:]

        # Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:len(cumulative)],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        # Individuals informations : factor coordinates, square cosinus & relative contributions
        ind = global_pca.ind_.copy()

        # Individuals partiels coordinaates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Fill columns for specific group with center data
            data_partiel = pd.DataFrame(np.tile(global_pca.call_["means"].values,(X.shape[0],1)),index=base.index,columns=base.columns)
            # Find columns to use
            columns = model[grp].call_["Z"].columns
            # Fill columns group by Standardize data
            data_partiel[columns] = base[columns]
            # Center the data
            Zbis = (data_partiel - global_pca.call_["means"].values.reshape(1,-1))/global_pca.call_["std"].values.reshape(1,-1)
            # Apply
            coord_partial = len(list(group_active_dict.keys()))*Zbis 
            # Apply weight
            coord_partial = mapply(coord_partial,lambda x : x*global_pca.call_["var_weights"].values,axis=1,progressbar=False,n_workers=n_workers)
            # Transition relation
            coord_partial = coord_partial.dot(global_pca.svd_["V"])
            # Set columns 
            coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
            ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
        
        # Add to dictionary
        ind["coord_partiel"] = ind_coord_partiel
        
        # Partiel coordinates for qualitatives columns
        quali_var_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Compute categories coordinates
            quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X[col]),axis=1).groupby(col).mean()for col in X.columns.tolist()),axis=0)
            quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_coord_partiel.columns.tolist()])
            quali_var_coord_partiel = pd.concat([quali_var_coord_partiel,quali_coord_partiel],axis=1)

        # Add to dictionary  
        quali_var["coord_partiel"] = quali_var_coord_partiel
        # Store all informations
        self.quali_var_  = quali_var

        # Partiel coordinates for supplementary qualitatives columns
        if self.num_group_sup is not None:
            quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp_sup, cols_sup in group_sup_dict.items():
                if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup[col]),axis=1).groupby(col).mean()for col in cols_sup),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns.tolist()])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
                # If at least one columns is categoricals
                elif any(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                    for grp, cols in group_active_dict.items():
                        X_group_sup_quali = X_group_sup[cols_sup].select_dtypes(include=['object'])
                        # Compute categories coordinates
                        quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup_quali[col]),axis=1).groupby(col).mean()for col in X_group_sup_quali.columns),axis=0)
                        quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns])
                        quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
            # Add to dictionary
            self.quali_var_sup_["coord_partiel"] = quali_var_sup_coord_partiel
        
        ## Inertia Ratios
        # "Between" inertia on axis s
        between_inertia = len(list(group_active_dict.keys()))*mapply(ind["coord"],lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        between_inertia.name = "between_inertia"

        # Total inertial on axis s
        total_inertia = pd.Series(name="total_inertia").astype("float")
        for dim in ind["coord"].columns:
            value = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum().sum()
            inertia = pd.Series([value],index=[dim],name="total_inertia")
            total_inertia = pd.concat((total_inertia,inertia),axis=0)

        # Inertia ratio
        inertia_ratio = between_inertia/total_inertia
        inertia_ratio.name = "inertia_ratio"
        # Store
        self.inertia_ratio_ = inertia_ratio

        # Individuals Within inertia
        ind_within_inertia = pd.DataFrame(index=X.index,columns=ind["coord"].columns).astype("float")
        for dim in ind["coord"].columns:
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : ((x - ind["coord"][dim].values)**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        # Individuals Within partial inertia
        data = pd.DataFrame().astype("float")
        for dim in ind["coord"].columns:
            data1 = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : ((x - ind["coord"][dim].values)**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers)
            data1 = 100*data1/data1.sum().sum()
            data = pd.concat([data,data1],axis=1)

        # Reorder inertia by group
        ind_within_partial_inertia = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            partial_inertia = data[grp]
            partial_inertia.columns = pd.MultiIndex.from_tuples([(grp,col) for col in partial_inertia.columns])
            ind_within_partial_inertia = pd.concat([ind_within_partial_inertia,partial_inertia],axis=1)
        ind["within_partial_inertia"] = ind_within_partial_inertia

        # Store all individuals informations
        self.ind_ = ind

        ## Partial axes informations
        # Partial axes coordinates
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=self.ind_["coord"].columns,columns=data.columns)
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=None)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=self.ind_["coord"].columns,columns=data.columns.tolist())
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            ######### Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        # Partial axes square cosinus
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        # Partial axes correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        # add supplementary coordinates 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns.tolist()])
                all_coord = pd.concat([all_coord,data],axis=1)
            # Reorder
            all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
        
        # Partial axes contributions
        axes_contrib = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            nbcol = min(n_components,model[grp].call_["n_components"])
            eig = model[grp].eig_.iloc[:nbcol,0].values/model[grp].eig_.iloc[0,0]
            contrib = mapply(partial_axes_coord[grp].iloc[:,:nbcol],lambda x : (x**2)*eig,axis=1,progressbar=False,n_workers=n_workers)
            contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns])
            axes_contrib  = pd.concat([axes_contrib,contrib],axis=1)
        
        partial_axes_contrib = mapply(axes_contrib,lambda x : 100*x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)

        # Add a null dataframe
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                nbcol = min(n_components,model[grp].call_["n_components"])
                contrib = pd.DataFrame(np.zeros(shape=(n_components,nbcol)),index=["Dim."+str(x+1) for x in range(n_components)],columns=["Dim."+str(x+1) for x in range(nbcol)])
                contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns])
                partial_axes_contrib = pd.concat([partial_axes_contrib,contrib],axis=1)
            # Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])
        
        # Correlation between
        cor_between = pd.DataFrame(weightedcorrcoef(x=all_coord,w=self.ind_weights),index=all_coord.columns,columns=all_coord.columns)
        # Store all informations
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : cor_between}

        ## Group informations : coordinates
        # group coordinates
        group_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = pd.concat((function_eta2(X=X[cols],lab=col,x=self.ind_["coord"].values,weights=ind_weights,n_workers=n_workers) for col in cols),axis=0)
            coord = data.sum(axis=0)/(len(cols)*model[grp].eig_.iloc[0,0])
            coord  = pd.DataFrame(coord.values.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns)
            group_coord = pd.concat((group_coord,coord),axis=0)
        
        # Group contributions
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)

        # Group square cosinus
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_dist2.loc[grp]).to_frame(grp).T for grp in group_coord.index),axis=0)

        # Group correlations
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=self.ind_["coord"],w=None)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=self.ind_["coord"].columns)
            group_correlation = pd.concat((group_correlation,correl),axis=0)

        # Measuring how similar groups - Lg coefficients
        Lg = function_lg2(model)   
        # Reorder using group name
        Lg = Lg.loc[group_name,group_name]
           
        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        # RV Coefficient
        RV = function_rv(X=Lg)
        
        # Store all informations
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}

        # Add supplementary elements
        if self.num_group_sup is not None:
            group_sup_coord = pd.DataFrame().astype("float")
            for grp, cols in group_sup_dict.items():
                Xg = X_group_sup[cols]
                if all(pd.api.types.is_numeric_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    correl = np.sum((weightedcorrcoef(model[grp].call_["Z"],self.ind_["coord"],w=None)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/model[grp].eig_.iloc[0,0]
                    coord = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns = ["Dim."+str(x+1) for x in range(len(correl))])
                    group_sup_coord = pd.concat((group_sup_coord,coord),axis=0)
                elif all(pd.api.types.is_string_dtype(Xg[col]) for col in cols):
                    # Calculate group sup coordinates
                    data = self.quali_var_sup_["eta2"].loc[cols,:]
                    coord = (data.sum(axis=0)/(Xg.shape[1]*model[grp].eig_.iloc[0,0]))
                    group_sup_coord = pd.concat((group_sup_coord,coord.to_frame(grp).T),axis=0)
                else:
                    raise TypeError("All columns should have the same type.")
            
            # Supplementary group square cosinus
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
        # Check if X is an instance of pd.DataFrame class
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set index name as None
        X.index.name = None

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise TypeError("DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Extract number of components
        n_components = self.call_["n_components"] 
        
        # Revaluate categoricals variables
        X = revaluate_cat_variable(X=X)
        
        # Apply transition relations
        row_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],n_components)),index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])
        for grp, cols in self.call_["group"].items():
            # Compute Dummies table : 0/1
            dummies = pd.concat((pd.get_dummies(X[col],dtype=int) for col in cols),axis=1)
            # Apply transition relation
            coord = mapply(dummies.dot(self.quali_var_["coord"].loc[dummies.columns,:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),axis=0,progressbar=False,n_workers=n_workers)
            row_coord = row_coord + coord
        # Weighted by the eigenvalue
        ind_coord = mapply(row_coord ,lambda x : x/self.eig_.iloc[:,0][:n_components],axis=1,progressbar=False,n_workers=n_workers)
        return ind_coord