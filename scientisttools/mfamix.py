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
from .function_eta2 import function_eta2
from .weightedcorrcoef import weightedcorrcoef
from .splitmix import splitmix
from .function_lg import function_lg
from .coeffRV import coeffRV

class MFAMIX(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis for Mixed Data (MFAMIX)
    -----------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Factor Analysis for Mixed Data in the sense of Pagès J. (2002) with supplementary individuals
    and supplementary groups of variables. Groups of variables can be quantitative, categorical.

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
    
    quanti_var_weights_mfa : an optional quantitatives variables weights (by default, a list of 1 for uniform weights), the weights
                                are given only for active quantitatives variables
    
    quali_var_weights_mfa : an optional qualitatives variables weights (by defaut, a list of 1/(number of categoricals variables in the group)),
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
    
    quanti_var_ : a dictionary of pandas dataframe containing all the results for the quantitatives variables (coordinates,
                    correlation between variables and axes, contribution, cos2)
    
    quali_var_ : a dictionary of pandas dataframe containing all the results for the categorical variables (coordinates of each categories
                    of each variables, contribution and vtest which is a criterion with a normal distribution)
    
    quanti_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates,
                        correlation between variables and axes, cos2)
    
    quali_var_sup_ : a dictionary of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of 
                        each categories of each variables, cos2 and vtest which is a criterion with a normal distribution)
    
    partial_axes_ : a dictionary of pandas dataframe containing all the results for the partial axes (coordinates, correlation between variables
                        and axes, correlation between partial axes)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfamd'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    """
    def __init__(self,
                n_components=5,
                group = None,
                name_group = None,
                group_type = None,
                num_group_sup = None,
                ind_sup = None,
                ind_weights=None,
                var_weights_mfa = None,
                mixed_factor_method = "famd",
                parallelize = False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.group_type = group_type
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.ind_weights = ind_weights
        self.var_weights_mfa = var_weights_mfa
        self.mixed_factor_method = mixed_factor_method
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

        # Set index name to None
        X.index.name = None

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Mixed factor method
        if self.mixed_factor_method not in ["famd","pcamix"]:
            raise ValueError("'mixed_factor_method' should be one of 'famd', 'pcamix'")
        
        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")

        # Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        for col in is_quali.columns.tolist():
            X[col] = X[col].astype("object")
        
        #   check if two categoricals variables have same categories
        X = revaluate_cat_variable(X)
    
        #   Check if group is None
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif not (isinstance(self.group, list) or isinstance(self.group,tuple)):
            raise ValueError("'group' must be a list or a tuple with the number of variables in each group")
        else:
            nb_elt_group = [int(x) for x in self.group]
        
        # Remove supplementary group
        if self.num_group_sup is not None:
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        # Check if supplementary individuals
        if self.ind_sup is not None:
            if (isinstance(self.ind_sup,int) or isinstance(self.ind_sup,float)):
                ind_sup = [int(self.ind_sup)]
            elif ((isinstance(self.ind_sup,list) or isinstance(self.ind_sup,tuple)) and len(self.ind_sup)>=1):
                ind_sup = [int(x) for x in self.ind_sup]
            ind_sup_label = X.index[ind_sup]
        else:
            ind_sup_label = None
        
        # Check if group type in not None
        if self.group_type is None:
            raise ValueError("'group_type' must be assigned.")
        
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
        
        # Assigned group name to label
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
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
            debut = debut + nb_elt_group[i]

        # Store data
        Xtot = X.copy()

        # Drop supplementary groups columns
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))
        
        # Drop supplementary individuals
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            # Drop supplementary individuals
            X = X.drop(index=ind_sup_label)
        
        # Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"{grp} group should have at least two columns")

        # Extract qualitatives and quantitatives groups
        all_nums = {}
        all_cats = {}
        for grp, cols in group_active_dict.items():
            all_nums[grp] = all(pd.api.types.is_numeric_dtype(X[col]) for col in cols)
            all_cats[grp]= all(pd.api.types.is_string_dtype(X[col]) for col in cols)
        
        # Compute statisiques
        group_label = pd.DataFrame(columns=["variable","group"])
        summary_quanti = pd.DataFrame()
        summary_quali = pd.DataFrame()
        for grp, cols in group_active_dict.items():
            if all_cats[grp]:
                for col in cols:
                    summary = X[col].value_counts().to_frame("count").reset_index().rename(columns={col : "categorie"})
                    summary.insert(0,"variable",col)
                    summary.insert(0,"group name",grp)
                    summary.insert(0,"group",group_name.index(grp))
                    summary_quali = pd.concat([summary_quali,summary],axis=0,ignore_index=True)
            elif all_nums[grp]:
                summary = X[cols].describe().T.reset_index().rename(columns={"index" : "variable"})
                summary.insert(0,"group name", grp)
                summary.insert(0,"group",group_name.index(grp))
                summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)

                # insert to group label
                row_grp = pd.Series(group_active_dict[grp],name='variable').to_frame()
                row_grp.insert(0,"group",grp)
                group_label = pd.concat((group_label,row_grp),axis=0,ignore_index=True)
            else:
                # Split X into two group
                X_quali = splitmix(X[cols])["quali"]
                X_quanti = splitmix(X[cols])["quanti"]

                # Summary for qualitatives variables
                for col in X_quali.columns.tolist():
                    summary1 = X_quali[col].value_counts().to_frame("count").reset_index().rename(columns={col: "categorie"})
                    summary1.insert(0,"variable",col)
                    summary1.insert(0,"group name",grp)
                    summary1.insert(0,"group",group_name.index(grp))
                    summary_quali = pd.concat([summary_quali,summary1],axis=0,ignore_index=True)
                
                # Summary of quantitatives variables
                summary2 = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary2.insert(0,"group name", grp)
                summary2.insert(0,"group",group_name.index(grp))
                summary_quanti = pd.concat((summary_quanti,summary2),axis=0,ignore_index=True)
                
        # Convert effectif and count to int
        summary_quali["count"] = summary_quali["count"].astype("int")
        summary_quanti["count"] = summary_quanti["count"].astype("int")
        self.group_label_ = group_label

        ### Store summary
        self.summary_quanti_ = summary_quanti
        self.summary_quali_ = summary_quali
        
        # Set indiviuduals weights
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])
        
        # Set variables weights for general factor analysis in each grou^p
        var_weights_mfa = pd.Series(name="weight").astype("float")
        if self.var_weights_mfa is None:
            for grp,cols in group_active_dict.items():
                if all(pd.api.types.is_numeric_dtype(X[col]) for col in cols):
                    weights = pd.Series(np.ones(len(cols)),index=cols,name="weight")
                elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
                    weights = pd.Series(np.ones(len(cols))/len(cols),index=cols,name="weight")
                else:
                    # Split X into 2 : 
                    X_quanti = splitmix(X[cols])["quanti"]
                    X_quali = splitmix(X[cols])["quali"]
                    n_cont, n_cat = X_quanti.shape[1], X_quali.shape[1]
                    cols1, cols2 = X_quanti.columns, X_quali.columns
                    weights1 = pd.Series(np.ones(n_cont),index=cols1,name="weight")
                    weights2 = pd.Series(np.ones(n_cat)/len(cols2),index=cols2,name="weight")
                    weights = pd.concat((weights1,weights2),axis=0)
                # Concatenate
                var_weights_mfa = pd.concat((var_weights_mfa,weights),axis=0)
        elif not isinstance(self.var_weights_mfa,pd.Series):
            raise TypeError("'var_weights_mfa' must be a pandas series where series are columns names and values are variables weights.")
        else:
            if len(self.var_weights_mfa)!= X.shape[1]:
                raise TypeError("Not aligned")
            var_weights_mfa = pd.concat((var_weights_mfa,self.var_weights_mfa),axis=0)

        # Run general factor analysis in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if all(pd.api.types.is_numeric_dtype(X[col]) for col in cols):
                if self.group_type[group_name.index(grp)]=="c":
                    # Center Principal Components Anlysis (PCA) 
                    fa = PCA(standardize=False,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),parallelize=self.parallelize)
                elif self.group_type[group_name.index(grp)]=="s":
                    # Scale Principal Components Anlysis (PCA)
                    fa = PCA(standardize=True,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),parallelize=self.parallelize)
                else:
                    raise TypeError("For quantitative groups 'group_type' should be one of 'c', 's'")
            elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
                if self.group_type[group_name.index(grp)]=="n":
                    # Multiple Correspondence Analysis (MCA)
                    fa = MCA(n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols],benzecri=False,greenacre=False,parallelize=self.parallelize)
                else: 
                    raise TypeError("For qualitative groups 'group_type' should be 'n'")
            else:
                if self.group_type[group_name.index(grp)]=="m":
                    if self.mixed_factor_method == "famd":
                        # Factor Analysis of Mixed Data (FAMD)
                        fa = FAMD(n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                    elif self.mixed_factor_method == "pcamix":
                        # Principal Component Analysis of Mixed Data (PCAMIX)
                        fa = PCAMIX(n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                else:
                    raise TypeError("For mixed groups 'group_type' should be 'm'")
            model[grp] = fa.fit(X[cols])

            # Add supplementary individuals
            if self.ind_sup is not None:
                if all(pd.api.types.is_numeric_dtype(X[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        # Center Principal Components Analysis (PCA)
                        fa = PCA(standardize=False,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),ind_sup=self.ind_sup,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        # Scale Principal Components Analysis (PCA)
                        fa = PCA(standardize=True,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),ind_sup=self.ind_sup,parallelize=self.parallelize)
                    else:
                        raise TypeError("For quantitative groups 'group_type' should be one of 'c', 's'")
                elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        # Multiple Correspondence Analysis (MCA)
                        fa = MCA(n_components=self.n_components,ind_weights=self.ind_weights,ind_sup=self.ind_sup,benzecri=False,greenacre=False,parallelize=self.parallelize)
                    else:
                        raise TypeError("For qualitative groups 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        if self.mixed_factor_method == "famd":
                            # Factor Analysis of Mixed Data (FAMD)
                            fa = FAMD(n_components=self.n_components,ind_weights=self.ind_weights,ind_sup=self.ind_sup,parallelize=self.parallelize)
                        elif self.mixed_factor_method == "pcamix":
                            # Principal Component Analysis of Mixed Data (PCAMIX)
                            fa = PCAMIX(n_components=self.n_components,ind_weights=self.ind_weights,ind_sup=self.ind_sup,parallelize=self.parallelize)
                    else:
                        raise TypeError("For mixed groups 'group_type' should be 'm'")
                model[grp] = fa.fit(pd.concat((X[cols],X_ind_sup[cols]),axis=0))
        
        # Separate  Factor Analysis for supplementary groups
        if self.num_group_sup is not None:
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            # Find columns for supplementary group
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=ind_sup_label)
            
            # Factor Analysis
            for grp, cols in group_sup_dict.items():
                # Instance the FA model
                if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="c":
                        # Center principal component analysis (PCA)
                        fa = PCA(standardize=False,n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                    elif self.group_type[group_name.index(grp)]=="s":
                        # Scale principal component analysis (PCA)
                        fa = PCA(standardize=True,n_components=self.n_components,ind_weights=self.ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("For continues group 'group_type' should be one of 'c', 's'")
                    
                    ##### Add to group_label
                    row_grp = pd.Series(group_sup_dict[grp],name='variable').to_frame()
                    row_grp.insert(0,"group",grp)
                    self.group_label_ = pd.concat((self.group_label_,row_grp),axis=0,ignore_index=True)

                elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    if self.group_type[group_name.index(grp)]=="n":
                        # Multiple Correspondence Analysis (MCA)
                        fa = MCA(n_components=self.n_components,ind_weights=self.ind_weights,benzecri=False,greenacre=False,parallelize=self.parallelize)
                    else:
                        raise TypeError("For categoricals group 'group_type' should be 'n'")
                else:
                    if self.group_type[group_name.index(grp)]=="m":
                        if self.mixed_factor_method == "famd":
                            # Factor Analysis of Mixed Data (FAMD)
                            fa = FAMD(n_components=self.n_components,ind_weights=ind_weights,parallelize=self.parallelize)
                        elif self.mixed_factor_method == "pcamix":
                            # Principal Component Analysis of MIXed data (PCAMIX)
                            fa = FAMD(n_components=self.n_components,ind_weights=ind_weights,parallelize=self.parallelize)
                    else:
                        raise TypeError("For mixed group 'group_type' should be 'm'")
                    
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
        means = {}
        std = {}
        base = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        for grp, cols in group_active_dict.items():
            if all_nums[grp]:
                d = DescrStatsW(X[cols],weights=ind_weights,ddof=0)
                means[grp], std[grp] = d.mean.reshape(1,-1), d.std.reshape(1,-1)
                Z = (X[cols] - means[grp])/std[grp]
                base = pd.concat([base,Z],axis=1)
                weights = np.array(var_weights_mfa[cols])*pd.Series([1/model[grp].eig_.iloc[0,0]]*len(cols),index=cols)
                var_weights = pd.concat((var_weights,weights),axis=0)
            elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
                Z = model[grp].call_["Z"]
                base = pd.concat([base,Z],axis=1)
                weights = pd.Series(name="weight").astype("float")
                for col in X[cols].columns:
                    data = pd.get_dummies(X[cols][col],dtype=int)   
                    m_k = (data.mean(axis=0)*var_weights_mfa[col])/model[grp].eig_.iloc[0,0]
                    weights = pd.concat([weights,m_k],axis=0)
                var_weights = pd.concat([var_weights,weights],axis=0)
            else:
                Z = model[grp].call_["Z"]
                base = pd.concat([base,Z],axis=1)
                means[grp] = model[grp].call_["means"].values.reshape(1,-1)
                std[grp] = model[grp].call_["std"].values.reshape(1,-1)
                weights = pd.Series([1/model[grp].eig_.iloc[0,0]]*Z.shape[1],index=Z.columns)
                var_weights = pd.concat((var_weights,weights),axis=0)
        
        # QR decomposition (to set number of components)
        Q, R = np.linalg.qr(base)
        max_components = min(np.linalg.matrix_rank(Q),np.linalg.matrix_rank(R))
        
        # Number of components
        if self.n_components is None:
            n_components = int(max_components)
        else:
            n_components = int(min(self.n_components,max_components))
        
        # Add original data to full base and global PCA without supplementary element
        D = base.copy()
        for col in X.columns.tolist():
            if X[col].dtype in ["object"]:
                D = pd.concat((D,X[col]),axis=1)
        index = [D.columns.tolist().index(x) for x in D.columns if x not in base.columns]
        # Global PCA
        global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(D)

        ## Extract actives qualitatives informations
        quali_var = global_pca.quali_sup_.copy()
        
        ###############################################################################################
        # Add supplementary individuals
        ###############################################################################################
        # Statistics for supplementary individuals
        if self.ind_sup is not None:
            # Initialiaze
            Z_ind_sup = pd.DataFrame().astype("float")
            for grp, cols in group_active_dict.items():
                if all_nums[grp]:
                    # Standardization
                    Z_sup = (X_ind_sup[cols] - means[grp])/std[grp]
                    # Concatenation
                    Z_ind_sup = pd.concat((Z_ind_sup,Z_sup),axis=1)
                elif all_cats[grp]:
                    # Dummies encoding
                    dummies = model[grp].call_["dummies"]
                    p_k = dummies.mean(axis=0)
                    # Create dummies table for supplementary individuals
                    Y = pd.DataFrame(np.zeros((X_ind_sup[cols].shape[0],dummies.shape[1])),columns=dummies.columns,index=X_ind_sup.index)
                    for i in np.arange(X_ind_sup[cols].shape[0]):
                        values = [X_ind_sup[cols].iloc[i,k] for k in np.arange(X_ind_sup[cols].shape[1])]
                        for j in np.arange(dummies.shape[1]):
                            if dummies.columns[j] in values:
                                Y.iloc[i,j] = 1

                    # Standardization
                    Z_sup = pd.concat((Y.loc[:,k]*(1/p_k[k])-1 for k in Y.columns),axis=1)
                    # Concatenate
                    Z_ind_sup = pd.concat((Z_ind_sup,Z_sup),axis=1)
                else:
                    #
                    X_ind_sup_quanti = splitmix(X=X_ind_sup[cols])["quanti"]
                    X_ind_sup_quali = splitmix(X=X_ind_sup[cols])["quali"]
                    # Dummies encoding of supplementary individuals
                    dummies = model[grp].call_["rec"]["dummies"]
                    Y = pd.DataFrame(np.zeros((X_ind_sup.shape[0],dummies.shape[1])),columns=dummies.columns,index=X_ind_sup.index)
                    for i in np.arange(X_ind_sup.shape[0]):
                        values = [X_ind_sup_quali.iloc[i,k] for k in np.arange(X_ind_sup_quali.shape[1])]
                        for j in np.arange(dummies.shape[1]):
                            if dummies.columns[j] in values:
                                Y.iloc[i,j] = 1
                    
                    # Concatenate and standardization
                    Z_sup = (pd.concat((X_ind_sup_quanti,Y),axis=1) - means[grp])/std[grp]
                    # Concatenate
                    Z_ind_sup = pd.concat((Z_ind_sup,Z_sup),axis=1)
    
            # Concatenate with active dataset
            Z_ind_sup = pd.concat((base,Z_ind_sup),axis=0)
            # Update PCA with supplementary individuals
            global_pca = PCA(standardize = False,n_components = n_components,ind_weights =self.ind_weights,var_weights = var_weights.values.tolist(),ind_sup=self.ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
            # Store all informations for supplementary individuals
            self.ind_sup_ = global_pca.ind_sup_.copy()
        
        ###############################################################################################
        # Add supplementary group
        ###############################################################################################
        # Statistics for supplementary variables
        if self.num_group_sup is not None:
            # Split into two
            X_sup_quanti = splitmix(X_group_sup)["quanti"]
            X_sup_quali = splitmix(X_group_sup)["quali"]
            
            # Statistics for supplementary quantitative variables
            if X_sup_quanti is not None:
                # Statistics
                summary_quanti_sup = X_sup_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")

                # Find group name for quantitative variables
                quanti_sup_grp_name = [grp for grp, cols in group_sup_dict.items() for col in summary_quanti_sup["variable"] if col in cols]
                quanti_sup_grp_idx = [group_name.index(x) for x in quanti_sup_grp_name]
                summary_quanti_sup.insert(0,"group name",quanti_sup_grp_name)
                summary_quanti_sup.insert(0,"group",quanti_sup_grp_idx)

                # Concatenate
                self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

                # Standardize the data
                d2 = DescrStatsW(X_sup_quanti,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_sup_quanti- d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)

                # Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)

                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_sup_quanti.columns]

                # Update global PCA with supplementary quantitative variables
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                
                # Store all informations
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            # Statistics for supplementary qualitative variables
            if X_sup_quali is not None:
                # Concatenate
                Z_quali_sup = pd.concat((base,X_sup_quali),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_sup_quali.columns.tolist()]
                # Update
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                # Store all informations
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_.copy()

                # Find group name
                quali_sup_grp_name = [grp for grp, cols in group_sup_dict.items() for col in summary_quali_var_sup["variable"] if col in cols]
                quali_sup_grp_idx = [group_name.index(x) for x in quali_sup_grp_name]
                summary_quali_var_sup.insert(0,"group name", quali_sup_grp_name)
                summary_quali_var_sup.insert(0,"group", quali_sup_grp_idx)
                
                # Concatenate
                self.summary_quali_ = pd.concat((self.summary_quali_,summary_quali_var_sup),axis=0,ignore_index=True)
        
         # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : pd.Series(ind_weights,index=X.index,name="weight"),
                      "var_weights_mfa" : var_weights_mfa,
                      "var_weights" : var_weights,
                      "means" : means,
                      "std" : std,
                      "group" : group_active_dict,
                      "group_name" : group_name}

        # Store global PCA
        self.global_pca_ = global_pca

        # Store a copy of eigenvalues
        self.eig_ = global_pca.eig_.iloc[:max_components,:]

        # Store a copy of SVD 
        self.svd_ = {"vs" : global_pca.svd_["vs"][:max_components],"U" : global_pca.svd_["U"], "V" : global_pca.svd_["V"]}

        # Individuals informations : factor coordinates, square cosinus, relative contributions
        ind = global_pca.ind_.copy()
        
        ####################################################################################################
        #   Quantitatiaves variables informations : coordinates, cos2 and contrib
        ####################################################################################################
        # Correlation between variables en axis
        quanti_var_coord = pd.DataFrame().astype("float")
        quanti_var_contrib = pd.DataFrame().astype("float")
        quanti_var_cos2 = pd.DataFrame().astype("float")
        # qualitative contrib
        quali_var_contrib =  pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            if all_nums[grp]:
                # Correlation between variables en axis
                quanti_coord = weightedcorrcoef(x=X[cols],y=ind["coord"],w=None)[:X[cols].shape[1],X[cols].shape[1]:]
                quanti_coord = pd.DataFrame(quanti_coord,index=cols,columns=["Dim."+str(x+1) for x in range(n_components)])
                quanti_var_coord = pd.concat([quanti_var_coord,quanti_coord],axis=0)
                # Relative contribution
                quanti_contrib = global_pca.var_["contrib"].loc[cols,:].iloc[:,:n_components].copy()
                quanti_var_contrib = pd.concat([quanti_var_contrib,quanti_contrib],axis=0)

                # square cosinus
                quanti_cos2 = global_pca.var_["cos2"].loc[cols,:].iloc[:,:n_components].copy()
                quanti_var_cos2 = pd.concat([quanti_var_cos2,quanti_cos2],axis=0)
            elif all_cats[grp]:
                modalite = []
                for col in cols:
                    modalite = modalite + np.unique(X[cols][col]).tolist()
                # Relative contribution
                quali_contrib = global_pca.var_["contrib"].loc[modalite,:].iloc[:,:n_components].copy()
                quali_var_contrib = pd.concat([quali_var_contrib,quali_contrib],axis=0)
            else:
                # Split data into 2
                X_quanti = splitmix(X=X[cols])["quanti"]
                X_quali = splitmix(X=X[cols])["quali"]

                # Factor coordinates : Correlation between variables en axis
                quanti_coord = weightedcorrcoef(x=X_quanti,y=ind["coord"],w=None)[:X_quanti.shape[1],X_quanti.shape[1]:]
                quanti_coord = pd.DataFrame(quanti_coord,index=X_quanti.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
                quanti_var_coord = pd.concat([quanti_var_coord,quanti_coord],axis=0)

                # Relative contributions
                quanti_contrib = global_pca.var_["contrib"].loc[X_quanti.columns,:].iloc[:,:n_components].copy()
                quanti_var_contrib = pd.concat([quanti_var_contrib,quanti_contrib],axis=0)

                # Square cosinus
                quanti_cos2 = global_pca.var_["cos2"].loc[X_quanti.columns,:].iloc[:,:n_components].copy()
                quanti_var_cos2 = pd.concat([quanti_var_cos2,quanti_cos2],axis=0)

                # Categorical variables
                modalite = []
                for col in X_quali.columns.tolist():
                    modalite = modalite + np.unique(X_quali[col]).tolist()
                # Relative contribution
                quali_contrib = global_pca.var_["contrib"].loc[modalite,:].iloc[:,:n_components].copy()
                quali_var_contrib = pd.concat([quali_var_contrib,quali_contrib],axis=0)

        # Store all informations
        self.quanti_var_ = {"coord" : quanti_var_coord,"cor" : quanti_var_coord,"contrib":quanti_var_contrib,"cos2":quanti_var_cos2}

        # Add relative contribution to dictionary
        quali_var["contrib"] = quali_var_contrib

        # Add individuals partiels coordinates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Fill columns for specific group with center data
            data_partiel = pd.DataFrame(np.tile(global_pca.call_["means"].values,(X.shape[0],1)),index=base.index,columns=base.columns)
            # Find columns
            columns = model[grp].call_["Z"].columns
            # Fill columns with standardize
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

        # Assign
        ind["coord_partiel"] = ind_coord_partiel

        # Partiel coordinates for qualitatives columns
        quali_var_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            quali_grp_coord_partiel = pd.DataFrame()
            if all_cats[grp]:
                # Compute categories coordinates
                for grp2, cols2 in group_active_dict.items():
                    quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp2],X[col]),axis=1).groupby(col).mean()for col in cols),axis=0)
                    quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp2,col) for col in quali_coord_partiel.columns])
                    # cbind.dataframe
                    quali_grp_coord_partiel = pd.concat([quali_grp_coord_partiel,quali_coord_partiel],axis=1)
            elif any(pd.api.types.is_string_dtype(X[col]) for col in cols):
                X_quali = X[cols].select_dtypes(include=["object"])
                # Compute categories coordinates
                for grp2, cols2 in group_active_dict.items():
                    quali_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp2],X_quali[col]),axis=1).groupby(col).mean()for col in X_quali.columns),axis=0)
                    quali_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp2,col) for col in quali_coord_partiel.columns])
                    # cbind.dataframe
                    quali_grp_coord_partiel = pd.concat([quali_grp_coord_partiel,quali_coord_partiel],axis=1)
            # rbind.dataframe
            quali_var_coord_partiel = pd.concat((quali_var_coord_partiel,quali_grp_coord_partiel),axis=0)
                    
        quali_var["coord_partiel"] = quali_var_coord_partiel

        # Store informations for qualitatives variables
        self.quali_var_  = quali_var

        # Partiel coordinates for supplementary qualitatives columns
        if self.num_group_sup is not None:
            if X_sup_quali is not None:
                quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
                for grp_sup, cols_sup in group_sup_dict.items():
                    if all(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                        for grp, cols in group_active_dict.items():
                            # Compute categories coordinates
                            quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup[col]),axis=1).groupby(col).mean()for col in cols_sup),axis=0)
                            quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns])
                            quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
                    elif any(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                        for grp, cols in group_active_dict.items():
                            X_group_sup_quali = X_group_sup[cols_sup].select_dtypes(include=['object'])
                            # Compute categories coordinates
                            quali_sup_coord_partiel = pd.concat((pd.concat((ind_coord_partiel[grp],X_group_sup_quali[col]),axis=1).groupby(col).mean()for col in X_group_sup_quali.columns),axis=0)
                            quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns])
                            quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)
            
                # Add to dictionary
                self.quali_var_sup_["coord_partiel"] = quali_var_sup_coord_partiel
        
        # Inertia Ratios
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
        self.inertia_ratio_ = inertia_ratio

        # Individuals Within inertia
        ind_within_inertia = pd.DataFrame(index=X.index,columns=ind["coord"].columns).astype("float")
        for dim in ind["coord"].columns:
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : ((x - ind["coord"][dim].values)**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        # Individuals within partial inertia
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

        # Store all informations for individuals
        self.ind_ = ind

        # Partial axes factor coordinates
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            correl = weightedcorrcoef(x=ind["coord"],y=data,w=None)[:ind["coord"].shape[1],ind["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=ind["coord"].columns,columns=data.columns)
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                correl = weightedcorrcoef(x=ind["coord"],y=data,w=None)[:ind["coord"].shape[1],self.ind_["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=ind["coord"].columns,columns=data.columns)
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            # Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        # Partial axes square cosinus
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        # Partial individuals coordinates
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns])
            all_coord = pd.concat([all_coord,data],axis=1)

        # Add 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns])
                all_coord = pd.concat([all_coord,data],axis=1)
            # Reorder
            all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
        
        # Partial axes relative contrib
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
            ## Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])
        
        # Correlation between
        cor_between = pd.DataFrame(weightedcorrcoef(x=all_coord,w=ind_weights),index=all_coord.columns,columns=all_coord.columns)

        # Store all informations
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : all_coord.corr()}

        # group factor coordinates
        group_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            if all_cats[grp]:
                data = pd.concat((function_eta2(X=X[cols],lab=col,x=ind["coord"].values,weights=ind_weights,n_workers=n_workers) for col in cols),axis=0)
                coord = data.sum(axis=0)/(len(cols)*model[grp].eig_.iloc[0,0])
                coord  = pd.DataFrame(coord.values.reshape(1,-1),index=[grp],columns=ind["coord"].columns)
                group_coord = pd.concat((group_coord,coord),axis=0)
            else:
                data = model[grp].call_["Z"]
                coord =  (weightedcorrcoef(data,ind["coord"],w=None)[:data.shape[1],data.shape[1]:]**2).sum(axis=0)/model[grp].eig_.iloc[0,0]
                coord  = pd.DataFrame(coord.reshape(1,-1),index=[grp],columns=ind["coord"].columns)
                group_coord = pd.concat((group_coord,coord),axis=0)
        
        # Group relative contributions
        group_contrib = mapply(group_coord,lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)

        # group square cosinus
        group_cos2 = pd.concat((((group_coord.loc[grp,:]**2)/group_dist2.loc[grp]).to_frame(grp).T for grp in group_coord.index),axis=0)

        # Group correlations
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=ind["coord"],w=None)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=ind["coord"].columns)
            group_correlation = pd.concat((group_correlation,correl),axis=0)
        
        # Measuring how similar groups - Lg coefficients
        Lg = function_lg2(separate_analysis=model)
        # Reorder using group name
        Lg = Lg.loc[group_name,group_name]

        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        # RV Coefficient
        RV = function_rv(X=Lg)
        
        # Store all informations for groups
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}

        # Add supplementary elements
        if self.num_group_sup is not None:
            # Calculate group sup coordinates
            group_sup_coord = pd.DataFrame().astype("float")
            for grp, cols in group_sup_dict.items():
                Xg = X_group_sup[cols]
                if all(pd.api.types.is_string_dtype(Xg[col]) for col in cols):
                    data = self.quali_var_sup_["eta2"].loc[cols,:]
                    coord = (data.sum(axis=0)/(Xg.shape[1]*model[grp].eig_.iloc[0,0]))
                    group_sup_coord = pd.concat((group_sup_coord,coord.to_frame(grp).T),axis=0)
                else:
                    correl = np.sum((weightedcorrcoef(model[grp].call_["Z"],self.ind_["coord"],w=None)[:Xg.shape[1],Xg.shape[1]:]**2),axis=0)/model[grp].eig_.iloc[0,0]
                    coord = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns = ["Dim."+str(x+1) for x in range(len(correl))])
                    group_sup_coord = pd.concat((group_sup_coord,coord),axis=0)
            
            # Supplementary group square cosinus
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index),axis=0)
            
            # Append two dictionnaries
            self.group_ = {**self.group_,**{"coord_sup" : group_sup_coord, "dist2_sup" : group_sup_dist2,"cos2_sup" : group_sup_cos2}}

        # Name of model
        self.model_ = "mfamix"

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
        
        # Check if X is an instance of pd.DataFrame class
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
        
        # Apply revaluate function
        X = revaluate_cat_variable(X=X)

        # Extract
        n_components = self.call_["n_components"]
        
        # Check New Data has same group
        row_coord = pd.DataFrame(np.zeros(shape=(X.shape[0],n_components)),index=X.index,columns=["Dim."+str(x+1) for x in range(n_components)])
        for grp, cols in self.call_["group"].items():
            num_row_partial = pd.DataFrame(np.zeros(shape=(X.shape[0],n_components)),index=X.index,columns=["Dim."+str(x+1) for x in range(self.call_["n_components"])])
            cat_row_partial = pd.DataFrame(np.zeros(shape=(X.shape[0],self.call_["n_components"])),index=X.index,columns=["Dim."+str(x+1) for x in range(self.call_["n_components"])])
            if all(pd.api.types.is_numeric_dtype(X[col]) for col in cols):
                # Standardize the Data
                Z = (X[cols] - self.call_["means"][grp])/self.call_["std"][grp]
                # Partiel coordinates
                coord = mapply(Z.dot(self.quanti_var_["coord"].loc[Z.columns.tolist(),:]),lambda x : x/self.separate_analyses_[grp].eig_.iloc[0,0],
                                       axis=0,progressbar=False,n_workers=n_workers)
                num_coord = len(self.call_["group"].keys())*mapply(coord,lambda x : x/np.sqrt(self.eig_.iloc[:,0][:self.call_["n_components"]]),
                                                                   axis=1,progressbar=False,n_workers=n_workers)
                num_row_partial = num_row_partial + num_coord
            # If all variables in group are categoricals
            elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
                # Compute Dummies table : 0/1
                dummies = pd.concat((pd.get_dummies(X[col],dtype=int) for col in cols),axis=1)
                # Partiel coordinates
                coord = mapply(dummies.dot(self.quali_var_["coord"].loc[dummies.columns.tolist(),:]),lambda x : x/(len(cols)*self.separate_analyses_[grp].eig_.iloc[0,0]),
                               axis=0,progressbar=False,n_workers=n_workers)
                cat_coord = len(self.call_["group"].keys())*mapply(coord,lambda x : x/self.eig_.iloc[:,0][:self.call_["n_components"]],
                                                                   axis=1,progressbar=False,n_workers=n_workers)
                cat_row_partial = cat_row_partial + cat_coord
            row_coord = row_coord + (1/len(self.call_["group"].keys()))*(num_row_partial + cat_row_partial)
        return row_coord