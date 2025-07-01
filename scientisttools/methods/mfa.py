# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import polars as pl
from mapply.mapply import mapply
import pingouin as pg
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

from .pca import PCA
from .mca import MCA
from .famd import FAMD
from .revaluate_cat_variable import revaluate_cat_variable
from .weightedcorrcoef import weightedcorrcoef
from .recodecont import recodecont
from .splitmix import splitmix
from .function_lg import function_lg
from .coeffRV import coeffRV
from .conditional_average import conditional_average
from .function_eta2 import function_eta2

class MFA(BaseEstimator,TransformerMixin):
    """
    Mutiple Factor Analysis (MFA)
    -----------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Factor Analysis in the sense of Escofier-Pages with supplementary individuals and supplementary groups of variables. Active groups of variables must be quantitative. Supplementary groups can be quantitative or categorical

    Usage
    -----
    ```python
    >>> MFA(n_components = 5,group = None,name_group = None,group_type = None,num_group_sup = None,ind_sup = None,ind_weights = None,var_weights_mfa = None,parallelize=False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `group` : a list or a tuple with the number of variables in each group

    `name_group` : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)

    `group_type` : the type of variables in each group. Possible values are : 
        * "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (quantitative and qualitative variables)
    
    `num_group_sup` : the indexes of the illustrative groups (by default, None and no group are illustrative)

    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `ind_weights` : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights_mfa` : an optional quantitatives variables weights (by default, a list of 1 for uniform weights), the weights are given only for active quantitatives variables
    
    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply

    Attributes
    ----------
    `summary_quali_` : a summary of the results for the categorical variables

    `summary_quanti_` : a summary of the results for the quantitative variables

    `separate_analyses_` : the results for the separate analyses

    `svd_` : dictionary of matrices containing all the results of the singular value decomposition

    `eig_` : pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the cumulative percentge of variance
    
    `group_` : dictionary of pandas dataframe containing all the results for the groups (Lg and RV coefficients, factor coordinates, square cosinus, relative contributions, square distance to the origin, the correlations between each group and each factor)
    
    `inertia_ratio_` : inertia ratio

    `ind_` : dictionary of pandas dataframe containing all the results for the active individuals (factor coordinates, square cosinus, relative contributions)
    
    `ind_sup_` : dictionary of pandas dataframe containing all the results for the supplementary individuals (factor coordinates, square cosinus, square distance to origin)
    
    `quanti_var_` : dictionary of pandas dataframe containing all the results for the quantitatives variables (factor coordinates, correlation between variables and axes, relative contribution, square cosinus)
    
    `quanti_var_sup_` : dictionary of pandas dataframe containing all the results for the supplementary quantitative variables (factor coordinates, correlation between variables and axes, square cosinus)
    
    `quali_var_sup_` : dictionary of pandas dataframe containing all the results for the supplementary categorical variables (factor coordinates of each categories of each variables, square cosinus and vtest which is a criterion with a normal distribution)
    
    `partial_axes_` : dictionary of pandas dataframe containing all the results for the partial axes (factor coordinates, correlation between variables and axes, correlation between partial axes)
    
    `global_pca_` : the results of the analysis when it is considered as a unique weighted PCA

    `model_` : string specifying the model fitted = 'mfa'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed. Dunod

    Escofier B, Pagès J (1984), l'Analyse factorielle multiple, Cahiers du Bureau universitaire de recherche opérationnelle. Série Recherche, tome 42 (1984), p. 3-68

    Escofier B, Pagès J (1983), Méthode pour l'analyse de plusieurs groupes de variables. Application à la caractérisation de vins rouges du Val de Loire. Revue de statistique appliquée, tome 31, n°2 (1983), p. 43-59
   
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    Notes
    -----
    https://husson.github.io/MOOC_AnaDo/AFM.html
    
    https://math.institut-agro-rennes-angers.fr/fr/ouvrages/analyse-factorielle-multiple-avec-r
    
    https://eudml.org/subject/MSC/62H25
     
    See also
    --------
    get_mfa_ind, get_mfa_var, get_mfa_partial_axes, summaryMFA, predictMFA, fviz_mfa_ind, fviz_mfa_col, fviz_mfa, dimdesc

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
    >>> group = [2,5,3,10,9,2]
    >>> num_group_sup = [0,5]
    >>> from scientisttools import MFA
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    ```
    """
    def __init__(self,
                 n_components = 5,
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
        
        # Set index name to None
        X.index.name = None
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
            
        # Checks if categoricals variables is in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns.tolist():
                X[col] = X[col].astype("object")
        
        # Transform all quantitatives columns to float
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")
        
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
            raise TypeError("'name_group' must be a list or a tuple with name of group")
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
                    group_active_dict[group_name[i]] = X_group.columns
            else:
                group_active_dict[group_name[i]] = X_group.columns
            debut = debut + nb_elt_group[i]

        # Create group label
        group_label = pd.DataFrame(columns=["group name","variable"])
        for grp in group_active_dict.keys():
            row_grp = pd.Series(group_active_dict[grp],name='variable').to_frame()
            row_grp.insert(0,"group name",grp)
            group_label = pd.concat((group_label,row_grp),axis=0,ignore_index=True)
        
        # Add supplementary group
        if self.num_group_sup is not None:
            for grp in group_sup_dict.keys():
                row_grp = pd.Series(group_sup_dict[grp],name='variable').to_frame()
                row_grp.insert(0,"group name",grp)
                group_label = pd.concat((group_label,row_grp),axis=0,ignore_index=True)
        
        self.group_label_ = group_label

        # Store data
        Xtot = X.copy()

       # Drop supplementary groups columns
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))
            # Select supplementary columns
            X_group_sup = Xtot[list(itertools.chain.from_iterable(group_sup_dict.values()))]
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=ind_sup_label)
        
        # Drop supplementary individuals
        if self.ind_sup is not None:
            # Extract supplementary individuals
            X_ind_sup = X.loc[ind_sup_label,:]
            # Drop supplementary individuals
            X = X.drop(index=ind_sup_label)
        
        # Fill NA with means
        X = recodecont(X=X)["Xcod"]
        
        # Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"{grp} group should have at least two columns")
        
        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
        if not all_num:
            raise TypeError("All actives columns must be numeric")

        # Summary quantitative variables
        summary_quanti = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            summary = X[cols].describe().T.reset_index().rename(columns={"index" : "variable"})
            summary["count"] = summary["count"].astype("int")
            summary.insert(0,"group name", grp)
            summary.insert(0,"group",group_name.index(grp))
            summary_quanti = pd.concat((summary_quanti,summary),axis=0,ignore_index=True)
        self.summary_quanti_ = summary_quanti

        # Set individuals weights
        if self.ind_weights is None:
            ind_weights = np.ones(X.shape[0])/X.shape[0]
        elif not isinstance(self.ind_weights,list):
            raise ValueError("'ind_weights' must be a list of individuals weights.")
        elif len(self.ind_weights) != X.shape[0]:
            raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
        else:
            ind_weights = np.array([x/np.sum(self.ind_weights) for x in self.ind_weights])

        # Set columns weight 
        var_weights_mfa = pd.Series(name="weight").astype("float")
        if self.var_weights_mfa is None:
            weights_mfa = pd.Series(np.ones(X.shape[1]),index=X.columns,name="weight")
            var_weights_mfa = pd.concat((var_weights_mfa,weights_mfa),axis=0)
        elif not isinstance(self.var_weights_mfa,pd.Series):
            raise TypeError("'var_weights_mfa' must be a pandas series where series are columns names and values are variables weights.")
        else:
            if len(self.var_weights_mfa)!= X.shape[1]:
                raise TypeError("Not aligned")
            var_weights_mfa = pd.concat((var_weights_mfa,self.var_weights_mfa),axis=0)
        
        # Run a principal component analysis (center or scale) in each group
        model = {}
        for grp, cols in group_active_dict.items():
            if self.group_type[group_name.index(grp)]=="c":
                # Center Principal Components Anlysis (PCA)
                fa = PCA(standardize=False,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),parallelize=self.parallelize)
            elif self.group_type[group_name.index(grp)]=="s":
                # Scale Principal Components Anlysis (PCA)
                fa = PCA(standardize=True,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),parallelize=self.parallelize)
            else:
                raise TypeError("For active group 'group_type' should be one of 'c', 's'")
            model[grp] = fa.fit(X[cols])

            # Add supplementary individuals
            if self.ind_sup is not None:
                # Transform to float
                X_ind_sup = X_ind_sup.astype("float")
                if self.group_type[group_name.index(grp)]=="c":
                    # Center Principal Components Anlysis (PCA)
                    fa = PCA(standardize=False,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),ind_sup=self.ind_sup,parallelize=self.parallelize)
                elif self.group_type[group_name.index(grp)]=="s":
                    # Scale Principal Components Anlysis (PCA)
                    fa = PCA(standardize=True,n_components=self.n_components,ind_weights=self.ind_weights,var_weights=var_weights_mfa[cols].values.tolist(),ind_sup=self.ind_sup,parallelize=self.parallelize)
                else:
                    raise TypeError("For active group 'group_type' should be one of 'c', 's'")
                # Fit the model
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
                        # Multiple correspondence analysis (MCA)
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
        group_dist2 = pd.Series([np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())],index=list(group_active_dict.keys()),name="Sq. Dist.")

        # Square distance to origin for supplementary group
        if self.num_group_sup is not None:
            group_sup_dist2 = pd.Series([np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())],index=list(group_sup_dict.keys()),name="Sq. Dist.")

        # Store separate analysis
        self.separate_analyses_ = model

        # Standardize Data for active group
        means = pd.Series().astype("float")
        std = pd.Series().astype("float")
        base = pd.DataFrame().astype("float")
        var_weights = pd.Series(name="weight").astype("float")
        columns_dict = {}
        for grp,cols in group_active_dict.items():
            Z = model[grp].call_["Z"]
            base = pd.concat([base,Z],axis=1)
            means = pd.concat((means,model[grp].call_["means"]),axis=0)
            std = pd.concat((std,model[grp].call_["std"]),axis=0)
            weights = var_weights_mfa[cols].values*pd.Series([1/model[grp].eig_.iloc[0,0]]*Z.shape[1],index=Z.columns)
            var_weights = pd.concat((var_weights,weights),axis=0)
            columns_dict[grp] = Z.columns

        # Standardize data for supplementary columns
        if self.num_group_sup is not None:
            base_sup = pd.DataFrame().astype("float")
            columns_sup_dict = {}
            var_sup_weights = pd.Series().astype("float")
            for grp, cols in group_sup_dict.items():
                Z_sup = model[grp].call_["Z"]
                base_sup = pd.concat((base_sup,Z_sup),axis=1)
                if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
                    weights = model[grp].call_["dummies"].mean(axis=0)/(len(cols)*model[grp].eig_.iloc[0,0])
                else:
                    weights = pd.Series([1/model[grp].eig_.iloc[0,0]]*Z_sup.shape[1],index=Z_sup.columns)
                var_sup_weights = pd.concat((var_sup_weights,weights),axis=0)
                columns_sup_dict[grp] = Z_sup.columns
        
        # QR decomposition (to set maximum number of components)
        Q, R = np.linalg.qr(base)
        max_components = min(np.linalg.matrix_rank(Q),np.linalg.matrix_rank(R))
        
        # Set number of components
        if self.n_components is None:
            n_components = int(max_components)
        else:
            n_components = int(min(self.n_components,max_components))

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
                      "columns_dict" : columns_dict,
                      "group_name" : group_name,
                      "ind_sup" : ind_sup_label}
        
        # Global PCA without supplementary element
        global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),parallelize = self.parallelize).fit(base)

        # Statistics for supplementary individuals
        if self.ind_sup is not None:
            # Transform to float
            X_ind_sup = X_ind_sup.astype("float")

            # Standardization
            Z_sup = (X_ind_sup  - means.values.reshape(1,-1))/std.values.reshape(1,-1)

            # Concatenate with active data
            Z_ind_sup = pd.concat((base,Z_sup),axis=0)

            # Update PCA with supplementary individuals
            global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),ind_sup=ind_sup,parallelize = self.parallelize).fit(Z_ind_sup)
            # Store a copy of all informations
            self.ind_sup_ = global_pca.ind_sup_.copy()

            # Partiels coordinates
            ind_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp, cols in columns_dict.items():
                data_partiel = pd.DataFrame(np.tile(global_pca.call_["means"].values,(X_ind_sup.shape[0],1)),index=X_ind_sup.index,columns=Z_ind_sup.columns)
                data_partiel[cols] = Z_sup[cols]
                Zbis = (data_partiel - global_pca.call_["means"].values.reshape(1,-1))/global_pca.call_["std"].values.reshape(1,-1)
                partial_coord = len(list(columns_dict.keys()))*Zbis 
                partial_coord = mapply(partial_coord,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(global_pca.svd_["V"][:,:n_components])
                partial_coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
                ind_sup_coord_partiel = pd.concat([ind_sup_coord_partiel,partial_coord],axis=1)
            # Add to dictionary
            self.ind_sup_["coord_partiel"] = ind_sup_coord_partiel
        
        # Statistics for supplementary variables
        if self.num_group_sup is not None:
            X_quanti_sup = splitmix(X=X_group_sup)["quanti"]
            X_quali_sup = splitmix(X=X_group_sup)["quali"]
            
            # Statistics for supplementary quantitative variables
            if X_quanti_sup is not None:
                # Recode to fill NA with mean
                X_quanti_sup = recodecont(X=X_quanti_sup)["Xcod"]

                # Summary
                summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
                summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")

                # Find group name for quantitative variables
                quanti_sup_grp_name = [grp for grp, cols in group_sup_dict.items() for col in summary_quanti_sup["variable"] if col in cols]
                quanti_sup_grp_idx = [group_name.index(x) for x in quanti_sup_grp_name]
                summary_quanti_sup.insert(0,"group name",quanti_sup_grp_name)
                summary_quanti_sup.insert(0,"group",quanti_sup_grp_idx)

                # Append
                self.summary_quanti_ = pd.concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)

                # Standardize the data
                d2 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
                Z_quanti_sup = (X_quanti_sup - d2.mean.reshape(1,-1))/d2.std.reshape(1,-1)

                # Concatenate
                Z_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quanti_sup.columns.tolist().index(x) for x in X_quanti_sup.columns]

                # Update PCA with supplementary quantitative variables
                global_pca = PCA(standardize = False,n_components=n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quanti_sup=index,parallelize = self.parallelize).fit(Z_quanti_sup)
                
                # Store a copy of all informations
                self.quanti_var_sup_ = global_pca.quanti_sup_.copy()
            
            # Statistics for supplementary qualitative variables
            if X_quali_sup is not None:
                # Revaluate
                X_quali_sup = revaluate_cat_variable(X=X_quali_sup)
                # Concatenate
                Z_quali_sup = pd.concat((base,X_quali_sup),axis=1)
                # Find supplementary quantitatives columns index
                index = [Z_quali_sup.columns.tolist().index(x) for x in X_quali_sup.columns]
                # Update PCA with supplementary qualitatives variables
                global_pca = PCA(standardize = False,n_components = n_components,ind_weights = self.ind_weights,var_weights = var_weights.values.tolist(),quali_sup=index,parallelize = self.parallelize).fit(Z_quali_sup)
                # Store a copy of all informations
                self.quali_var_sup_ = global_pca.quali_sup_.copy()
                # Extract
                summary_quali_var_sup = global_pca.summary_quali_.copy()
                
                # Find group name
                quali_sup_grp_name = [grp for grp, cols in group_sup_dict.items() for col in summary_quali_var_sup["variable"] if col in cols]
                quali_sup_grp_idx = [group_name.index(x) for x in quali_sup_grp_name]
                summary_quali_var_sup.insert(0,"group name", quali_sup_grp_name)
                summary_quali_var_sup.insert(0,"group", quali_sup_grp_idx)
                self.summary_quali_ = summary_quali_var_sup

        # Store global PCA
        self.global_pca_ = global_pca

        # Store a copy of eigenvalues
        self.eig_ = global_pca.eig_.copy()

        # Store a copy of singular Values Decomposition (SVD)
        self.svd_ = global_pca.svd_.copy()

        # Individuals informations : coord, cos2, contrib
        ind = global_pca.ind_.copy()

        ## Variables informations : coordinates, cos2 and contrib
        # Weighted pearson correlation between variables en axis
        quanti_var_coord = pd.DataFrame(weightedcorrcoef(x=X,y=ind["coord"],w=ind_weights)[:X.shape[1],X.shape[1]:],index=X.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
        # Contribution
        quanti_var_contrib = global_pca.var_["contrib"].copy()
        # Square cosinus
        quanti_var_cos2 = global_pca.var_["cos2"].copy()

        # Store all informations
        self.quanti_var_ = {"coord" : quanti_var_coord, "cor" : quanti_var_coord, "contrib":quanti_var_contrib,"cos2" : quanti_var_cos2}

        # Individuals partiels coordinates
        ind_coord_partiel = pd.DataFrame().astype("float")
        for grp, cols in columns_dict.items():
            # Fill columns for specific group with center data
            data_partiel = pd.DataFrame(np.tile(global_pca.call_["means"].values,(X.shape[0],1)),index=base.index,columns=base.columns)
            # Fill columns group by Standardize data
            data_partiel[cols] = base[cols]
            # Center the data
            Zbis = (data_partiel - global_pca.call_["means"].values.reshape(1,-1))/global_pca.call_["std"].values.reshape(1,-1)
            # Apply
            coord_partial = len(list(columns_dict.keys()))*Zbis 
            # Apply weight
            coord_partial = mapply(coord_partial,lambda x : x*var_weights.values,axis=1,progressbar=False,n_workers=n_workers)
            # Transition relation
            coord_partial = coord_partial.dot(global_pca.svd_["V"])
            # Set columns 
            coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
            ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
        
        # Add to dictionary
        ind["coord_partiel"] = ind_coord_partiel

        # Partiel coordinates for supplementary qualitatives columns
        if self.num_group_sup is not None:
            if hasattr(self,"quali_var_sup_"):
                quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
                for grp_sup, cols_sup in group_sup_dict.items():
                    # If all columns in group are categoricals
                    if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols_sup):
                        quali_sup_coord_partiel = pd.DataFrame().astype("float")
                        for grp, cols in group_active_dict.items():
                            quali_sup_partiel = conditional_average(X=ind_coord_partiel[grp],Y=X_group_sup[cols_sup],weights=ind_weights)
                            quali_sup_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_partiel.columns])
                            quali_sup_coord_partiel = pd.concat([quali_sup_coord_partiel,quali_sup_partiel],axis=1)
                        quali_var_sup_coord_partiel = pd.concat((quali_var_sup_coord_partiel,quali_sup_coord_partiel),axis=0)
                    # If at least one columns is categoricals
                    elif any(pd.api.types.is_string_dtype(X_group_sup[cols_sup][col]) for col in cols_sup):
                        quali_sup_coord_partiel = pd.DataFrame().astype("float")
                        for grp, cols in group_active_dict.items():
                            X_group_sup_quali = X_group_sup[cols_sup].select_dtypes(include=['object'])
                            quali_sup_partiel = conditional_average(X=ind_coord_partiel[grp],Y=X_group_sup_quali,weights=ind_weights)
                            quali_sup_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_partiel.columns])
                            quali_sup_coord_partiel = pd.concat([quali_sup_coord_partiel,quali_sup_partiel],axis=1)
                        quali_var_sup_coord_partiel = pd.concat((quali_var_sup_coord_partiel,quali_sup_coord_partiel),axis=0)
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
        # Store all
        self.inertia_ratio_ = inertia_ratio

        # Individuals Within inertia
        ind_within_inertia = pd.DataFrame(index=X.index,columns=ind["coord"].columns).astype("float")
        for dim in ind["coord"].columns:
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : ((x - ind["coord"][dim].values)**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        # Add to dictionary
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
        # Add to dictionary
        ind["within_partial_inertia"] = ind_within_partial_inertia

        # Store all individuals informations
        self.ind_ = ind

        ## Partial axes informations
        # Partial axes coordinates
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            correl = weightedcorrcoef(x=ind["coord"],y=data,w=ind_weights)[:ind["coord"].shape[1],ind["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=ind["coord"].columns,columns=data.columns)
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                correl = weightedcorrcoef(x=ind["coord"],y=data,w=ind_weights)[:ind["coord"].shape[1],ind["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=ind["coord"].columns,columns=data.columns)
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
            # Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])

        # Partial axes square cosinus
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        # Partial axes correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        # Add supplementary informations 
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns])
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
        cor_between = pd.DataFrame(weightedcorrcoef(x=all_coord,w=ind_weights),index=all_coord.columns,columns=all_coord.columns)
        
        # Store all informations
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : cor_between}
        
        ## Group informations : contributions, factor coordinates, square cosinus and correlation, Lg, RV
        # Group contributions
        group_contrib = pd.DataFrame(index=list(group_active_dict.keys()),columns=["Dim."+str(x+1) for x in range(n_components)]).astype("float")
        for grp, cols in columns_dict.items():
            group_contrib.loc[grp,:] = global_pca.var_["contrib"].loc[cols,:].iloc[:,:n_components].sum(axis=0)
        
        # group coordinates
        group_coord = mapply(group_contrib,lambda x : (1/100)*x*(global_pca.svd_["vs"][:n_components]**2),axis=1,progressbar=False,n_workers=n_workers)
        
        # Group square cosinus
        group_cos2 = mapply(group_coord, lambda x : (x**2)/group_dist2.values,axis=0,progressbar=False,n_workers=n_workers)

        # Group correlations
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=ind["coord"],w=ind_weights)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=ind["coord"].columns)
            group_correlation = pd.concat((group_correlation,correl),axis=0)

        # Measuring how similar groups - Lg coefficients
        Lg = pd.DataFrame().astype("float")
        for grp1, cols1 in columns_dict.items():
            for grp2, cols2 in columns_dict.items():
                Lg.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base[cols2],X_weights=var_weights[cols1],Y_weights=var_weights[cols2],ind_weights=ind_weights)
        
        # Calculate Lg between supplementary groups
        if self.num_group_sup is not None:
            Lg_sup = pd.DataFrame().astype("float")
            for grp1, cols1 in columns_sup_dict.items():
                for grp2, cols2 in columns_sup_dict.items():
                    Lg_sup.loc[grp1,grp2] = function_lg(X=base_sup[cols1],Y=base_sup[cols2],X_weights=var_sup_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=ind_weights)
           
            # Concatenate
            Lg = pd.concat((Lg,Lg_sup),axis=1)
            # Fill na with 0.0
            Lg = Lg.fillna(0)

            # Calculate Lg coefficients between active and supplementary groups
            for grp1, cols1 in columns_dict.items():
                for grp2, cols2 in columns_sup_dict.items(): 
                    Lg.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base_sup[cols2],X_weights=var_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=ind_weights)
                    Lg.loc[grp2,grp1] = Lg.loc[grp1,grp2] 

        # Reorder using group name
        Lg = Lg.loc[group_name,group_name]

        # Add MFA Lg coefficients
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        # RV Coefficients 
        RV = coeffRV(X=Lg)
        
        # Store all informations
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2, "RV" : RV}

        # Add supplementary elements
        if self.num_group_sup is not None:
            # Calculate group sup coordinates
            group_sup_coord = pd.DataFrame(index=list(columns_sup_dict.keys()),columns=["Dim."+str(x+1) for x in range(n_components)]).astype("float")
            for grp, cols in columns_sup_dict.items():
                for i, dim in enumerate(group_sup_coord.columns):
                    group_sup_coord.loc[grp,dim] = function_lg(X=ind["coord"][dim],Y=base_sup[cols],X_weights=1/self.eig_.iloc[i,0],Y_weights=var_sup_weights[cols],ind_weights=ind_weights)

            # Supplementary group square cosinus
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index),axis=0)
            
            # Append two dictionnaries
            self.group_ = {**self.group_,**{"coord_sup" : group_sup_coord, "cos2_sup" : group_sup_cos2, "dist2_sup" : group_sup_dist2}}

        # Model name    
        self.model_ = "mfa"

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
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
        # Set index name as None
        X.index.name = None

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
 
        # Transform to float
        X = X.astype("float")
        
        # Extract elements
        var_weights = self.call_["var_weights"].values
        n_components = self.call_["n_components"]

        # Standardize according to MFA program
        Z = (X - self.call_["means"].values.reshape(1,-1))/self.call_["std"].values.reshape(1,-1)

        # Standardizaton according to PCA program
        Z = (Z - self.global_pca_.call_["means"].values.reshape(1,-1))/self.global_pca_.call_["std"].values.reshape(1,-1)

        # Multiply by columns weight & Apply transition relation
        coord = mapply(Z,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        return coord

def predictMFA(self,X=None):
    """
    Predict projection for new individuals with Multiple Factor Analysis (MFA)
    --------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus, square distance to origin and partial coordinates of new individuals with Multiple Factor Analysis (MFA)

    Usage
    -----
    ```python
    >>> predictMFA(self,X=None)
    ```

    Parameters
    ----------
    `self` : an object of class MFA, MFAQUAL, MFAMIX

    `X` : pandas/polars dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    dictionary of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : square cosinus of the new individuals

    `dist` : square distance to origin for new individuals

    `coord_partiel` : partiel coordinates for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load wine dataset
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> from scientisttools import MFA, predictMFA
    >>> group_type = ["n"]+["s"]*5
    >>> res_mfa = MFA(n_components=5,group=group,group_type=group_type,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    >>> # Active individuals
    >>> active = wine.drop(columns=["Overall.quality","Typical","Label","Soil"])
    >>> predict = predictMFA(res_mfa, X=active)
    ```
    """
    # Check if self is an object of class MFA, MFAQUAL or MFAMIX
    if self.model_ not in ["mfa","mfaqual","mfamix"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX")
    
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
    
    # Revaluate categorical variables
    X = revaluate_cat_variable(X=X)

    # Extract elements to use
    ind_weights = self.call_["ind_weights"].values.tolist()
    var_weights = self.call_["var_weights"].values.tolist() # Variables weights
    n_components = self.call_["n_components"] # number of components
    columns_dict = self.call_["columns_dict"]
    group_active_dict = self.call_["group"]

    Z = pd.DataFrame().astype("float")
    for grp, cols in group_active_dict.items():
        if all(pd.api.types.is_numeric_dtype(X[col]) for col in cols):
            Zg = (X[cols] - self.call_["means"][cols].values.reshape(1,-1))/self.call_["std"][cols].values.reshape(1,-1)
            Z = pd.concat((Z,Zg),axis=1)
        elif all(pd.api.types.is_string_dtype(X[col]) for col in cols):
            # Dummies encoding
            dummies = self.separate_analyses_[grp].call_["dummies"]
            # Create dummies table for supplementary individuals
            Yg = pd.DataFrame(np.zeros((X.shape[0],dummies.shape[1])),columns=dummies.columns,index=X.index)
            for i in np.arange(X.shape[0]):
                values = [X[cols].iloc[i,k] for k in np.arange(len(cols))]
                for j in np.arange(dummies.shape[1]):
                    if dummies.columns[j] in values:
                        Yg.iloc[i,j] = 1
            p_k = dummies.mean(axis=0)
            # Standardization
            Zg = pd.concat((Yg.loc[:,k]*(1/p_k[k])-1 for k in Yg.columns),axis=1)
            # Concatenate
            Z = pd.concat((Z,Zg),axis=1)
        else:
            # Split into two
            X_quanti = splitmix(X=X[cols])["quanti"]
            X_quali = splitmix(X=X[cols])["quali"]
            # Dummies encoding of supplementary individuals
            dummies = self.separate_analyses_[grp].call_["rec"]["dummies"]
            Yg = pd.DataFrame(np.zeros((X.shape[0],dummies.shape[1])),columns=dummies.columns,index=X.index)
            for i in np.arange(X.shape[0]):
                values = [X_quali.iloc[i,k] for k in np.arange(X_quali.shape[1])]
                for j in np.arange(dummies.shape[1]):
                    if dummies.columns[j] in values:
                        Yg.iloc[i,j] = 1
            # Concatenate
            Xg = pd.concat((X_quanti,Yg),axis=1)
            # Standardization
            Zg = (Xg - self.call_["means"][Xg.columns].values.reshape(1,-1))/self.call_["std"][Xg.columns].values.reshape(1,-1)
            # Concatenate
            Z = pd.concat((Z,Zg),axis=1)
    
    # Concatenate
    base = pd.concat((self.call_["Z"],Z),axis=0)
    # Find index
    index = [base.index.tolist().index(x) for x in Z.index]
    # Principal Component Analysis (PCA)
    global_pca = PCA(standardize=False,n_components=n_components,ind_weights=ind_weights,var_weights=var_weights,ind_sup=index,parallelize=self.parallelize).fit(base)

    # Partiels coordinates
    coord_partiel = pd.DataFrame().astype("float")
    for grp, cols in columns_dict.items():
        data_partiel = pd.DataFrame(np.tile(self.global_pca_.call_["means"].values,(X.shape[0],1)),index=Z.index,columns=self.call_["Z"].columns)
        data_partiel[cols] = Z[cols]
        Zbis = (data_partiel - self.global_pca_.call_["means"].values.reshape(1,-1))/self.global_pca_.call_["std"].values.reshape(1,-1)
        partial_coord = len(list(columns_dict.keys()))*Zbis
        partial_coord = mapply(partial_coord,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        partial_coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
        coord_partiel = pd.concat([coord_partiel,partial_coord],axis=1)
    
    # Store all informations
    res = {"coord" : global_pca.ind_sup_["coord"], "cos2" : global_pca.ind_sup_["cos2"], "dist" : global_pca.ind_sup_["dist"], "coord_partiel" : coord_partiel}
    return res

def supvarMFA(self,X_group_sup=None,group_sup=None,name_group_sup=None,group_sup_type=None):
    """
    Supplementary variables in Multiple Factor Analysis (MFA)
    ---------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Multiple Factor Analysis (MFA)

    Usage
    -----
    ```python
    >>> supvarMFA(self,X_group_sup=None,group_sup=None,name_group_sup=None,group_sup_type=None)
    ```

    Parameters
    ----------
    `self` : an object of class MFA, MFAQUAL, MFAMIX

    `X_group_sup` : pandas/polars dataframe of supplementary groups (default=None)

    `group_sup` : a list or a tuple with the number of variables in each supplementary group

    `name_group_sup` : a list or a tuple containing the name of the supplementary groups (by default, None and the group are named Gr1, Gr2 and so on)

    `group_sup_type` : the type of variables in each supplementary group. Possible values are : 
        * "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (quantitative and qualitative variables)

    Returns
    -------
    dictionary of dictionary containing the results for supplementary variables including : 

    `group` : dictionary containing the results of the supplementary groups including :
        * coord : supplementary group factor coordinates
        * cos2 : supplementary group square cosinus
        * dist2 : supplementary group square distance to origin
        * Lg : Lg coefficients
        * RV : RV coefficients
    
    `partial_axes` : dictionary containing the results of the supplementary groups partial axes:
        * coord : factor coordinates
        * cos2 : square cosinus

    `quanti` : dictionary containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : dictionary containing the results of the supplementary qualitatives/categories variables including :
        * coord : factor coordinates of the supplementary categories
        * cos2 : square cosinus of the supplementary categories
        * vtest : value-test of the supplementary categories
        * dist : square distance to origin of the supplementary categories
        * eta2 : square correlation ratio of the supplementary qualitatives variables
        * coord_partiel : partiel coordinates of the supplementary qualitatives variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load gironde dataset
    ```
    """
    # Check if self is and object of class MFA
    if self.model_ not in ["mfa","mfaqual","mfamix"]:
        raise TypeError("'self' must be an object of class MFA, MFAQUAL, MFAMIX")

    # check if X_group_sup is an instance of polars dataframe class
    if isinstance(X_group_sup,pl.DataFrame):
        X_group_sup = X_group_sup.to_pandas()
    
    # Check if X_group_sup is an instance of pandas dataframe class
    if not isinstance(X_group_sup,pd.DataFrame):
        raise TypeError(
        f"{type(X_group_sup)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # Check if supplementary group is None
    if group_sup is None:
        raise ValueError("'group_sup' must be assigned.")
    elif not (isinstance(group_sup, list) or isinstance(group_sup,tuple)):
        raise ValueError("'group_sup' must be a list or a tuple with the number of variables in each group")
    else:
        nb_elt_group_sup = [int(x) for x in group_sup]
    
    # Check if supplementary group type in not None
    if group_sup_type is None:
        raise ValueError("'group_sup_type' must be assigned")
        
    if len(group_sup) != len(group_sup_type):
        raise TypeError("Not convenient group definition")
        
    # Assigned supplementary group name
    if name_group_sup is None:
        group_sup_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group_sup))]
    elif not (isinstance(name_group_sup,list) or isinstance(name_group_sup,tuple)):
        raise TypeError("'name_group_sup' must be a list or a tuple with name of group")
    else:
        group_sup_name = [x for x in name_group_sup]
        
    # check if supplementary group name is an integer
    for i in range(len(group_sup_name)):
        if isinstance(group_sup_name[i],int) or isinstance(group_sup_name[i],float):
            group_sup_name[i] = "Gr"+str(i+1)
    
    # Revaluate
    X_group_sup = revaluate_cat_variable(X=X_group_sup)
        
    # Assigned supplementary group name to label
    group_sup_dict = {}
    debut = 0
    for i in range(len(nb_elt_group_sup)):
        X_group = X_group_sup.iloc[:,(debut):(debut+nb_elt_group_sup[i])]
        group_sup_dict[group_sup_name[i]] = X_group.columns
        debut = debut + nb_elt_group_sup[i]
    
    # Extract elements
    n_components = self.call_["n_components"]
    ind_weights = self.call_["ind_weights"].values

    # Separate general factor analysis for supplementary groups
    model = {}
    for grp, cols in group_sup_dict.items():
        if all(pd.api.types.is_numeric_dtype(X_group_sup[col]) for col in cols):
            if group_sup_type[group_sup_name.index(grp)]=="c":
                # Center principal component analysis (PCA)
                fa = PCA(standardize=False,n_components=n_components,ind_weights=ind_weights.tolist(),parallelize=self.parallelize)
            elif group_sup_type[group_sup_name.index(grp)]=="s":
                # Scale principal component analysis (PCA)
                fa = PCA(standardize=True,n_components=n_components,ind_weights=ind_weights.tolist(),parallelize=self.parallelize)
            else:
                raise TypeError("For continues variables 'group_type' should be one of 'c', 's'")
        elif all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
            if group_sup_type[group_sup_name.index(grp)]=="n":
                # Multiple correspondence analysis (MCA)
                fa = MCA(n_components=n_components,ind_weights=ind_weights.tolist(),benzecri=False,greenacre=False,parallelize=self.parallelize)
            else:
                raise TypeError("For categoricals variables 'group_type' should be 'n'")
        else:
            if group_sup_type[group_sup_name.index(grp)]=="m":
                # Factor analysis of mixed data (FAMD)
                fa = FAMD(n_components=n_components,ind_weights=ind_weights.tolist(),parallelize=self.parallelize)
            else:
                raise TypeError("For mixed variables 'group_type' should be 'm'")
        # Fit the model
        model[grp] = fa.fit(X_group_sup[cols])
    
    # Square distance to origin for supplementary group
    group_sup_dist2 = pd.Series([np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())],index=list(group_sup_dict.keys()),name="Sq. Dist.")

    # Standardize data for supplementary columns
    base_sup = pd.DataFrame().astype("float")
    columns_sup_dict = {}
    var_sup_weights = pd.Series().astype("float")
    for grp, cols in group_sup_dict.items():
        Z_sup = model[grp].call_["Z"]
        base_sup = pd.concat((base_sup,Z_sup),axis=1)
        if all(pd.api.types.is_string_dtype(X_group_sup[col]) for col in cols):
            weights = model[grp].call_["dummies"].mean(axis=0)/(len(cols)*model[grp].eig_.iloc[0,0])
        else:
            weights = pd.Series([1/model[grp].eig_.iloc[0,0]]*Z_sup.shape[1],index=Z_sup.columns)
        var_sup_weights = pd.concat((var_sup_weights,weights),axis=0)
        columns_sup_dict[grp] = Z_sup.columns
    
    # Lg coefficients between supplementary groups
    Lg_sup = pd.DataFrame().astype("float")
    for grp1, cols1 in columns_sup_dict.items():
        for grp2, cols2 in columns_sup_dict.items():
            Lg_sup.loc[grp1,grp2] = function_lg(X=base_sup[cols1],Y=base_sup[cols2],X_weights=var_sup_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=ind_weights)
    
    # Calculate Lg coefficients between active and supplementary groups
    base = self.call_["Z"]
    var_weights = self.call_["var_weights"]
    columns_dict = self.call_["columns_dict"]
    Lg = pd.DataFrame(index=list(columns_dict.keys()),columns=list(columns_dict.keys()))
    Lg_sup = pd.concat([Lg,Lg_sup],axis=1)
    for grp1, cols1 in columns_dict.items():
        for grp2, cols2 in columns_sup_dict.items(): 
            Lg_sup.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base_sup[cols2],X_weights=var_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=ind_weights)
            Lg_sup.loc[grp2,grp1] = Lg_sup.loc[grp1,grp2]
    
    # Add Lg coefficients for active groups
    for grp1, cols1 in columns_dict.items():
        for grp2, cols2 in columns_dict.items():
            Lg_sup.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base[cols2],X_weights=var_weights[cols1],Y_weights=var_weights[cols2],ind_weights=ind_weights)
    
    # Calculate RV coefficients
    RV_sup = coeffRV(X=Lg_sup)

    # Supplementary group factor coordinates
    group_sup_coord = pd.DataFrame().astype("float")
    group_sup_coord = pd.DataFrame(index=list(columns_sup_dict.keys()),columns=["Dim."+str(x+1) for x in range(n_components)]).astype("float")
    for grp, cols in columns_sup_dict.items():
        for i, dim in enumerate(group_sup_coord.columns):
            group_sup_coord.loc[grp,dim] = function_lg(X=self.ind_["coord"][dim],Y=base_sup[cols],X_weights=1/self.eig_.iloc[i,0],Y_weights=var_sup_weights[cols],ind_weights=ind_weights)
    
    # Supplementary group square cosinus
    group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index),axis=0)
    
    # Store all informations
    group_sup_infos = {"coord" : group_sup_coord, "cos2" : group_sup_cos2, "dist2" : group_sup_dist2, "Lg" : Lg_sup, "RV" : RV_sup}

    # Partial axis coordinates
    partial_axes_coord = pd.DataFrame().astype("float")
    for grp, cols in group_sup_dict.items():
        data = model[grp].ind_["coord"]
        correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=ind_weights)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
        coord = pd.DataFrame(correl,index=self.ind_["coord"].columns,columns=data.columns)
        coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
        partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)

    # Partial axes square cosinus
    partial_axes_cos2 = partial_axes_coord**2
    # Store all informtions
    partial_axes = {"coord" : partial_axes_coord, "cos2" : partial_axes_cos2}

    # Store results
    res = {"group" : group_sup_infos, "partial_axes" : partial_axes}

    # Split X_group_sup dataframe
    X_quanti_sup = splitmix(X=X_group_sup)["quanti"]
    X_quali_sup = splitmix(X=X_group_sup)["quali"]

    # Supplementary quantitatives variables statistics
    if X_quanti_sup is not None:
        # Transform to float
        X_quanti_sup = X_quanti_sup.astype("float")
        # Fill NA with mean
        X_quanti_sup = recodecont(X=X_quanti_sup)["Xcod"]
        # Compute weighted average and standard deviation
        d1 = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
        # Standardization
        Z_quanti_sup = (X_quanti_sup - d1.mean.reshape(1,-1))/d1.std.reshape(1,-1)
        # Supplementary quantitatives variables coordinates
        base_quanti_sup = pd.concat((base,Z_quanti_sup),axis=1)
        # Find index
        index = [base_quanti_sup.columns.tolist().index(x) for x in Z_quanti_sup.columns]
        # PCA with supplementary quantitatives variables
        global_pca = PCA(standardize=False,n_components=n_components,ind_weights=ind_weights.tolist(),var_weights=var_weights.values.tolist(),quanti_sup=index,parallelize=self.parallelize).fit(base_quanti_sup)
        # Store supplementary quantitatives informations
        quanti_sup =  global_pca.quanti_sup_.copy()
        # Add to dictionary
        res = {**res, **{"quanti" : quanti_sup}}
    
    # Supplementary qualitatives statistics
    if X_quali_sup is not None:
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")
        # Concatenate with active dataset
        base_quali_sup = pd.concat((base,X_quali_sup),axis=1)
        # Find index
        index = [base_quali_sup.columns.tolist().index(x) for x in X_quali_sup.columns]
        # PCA with supplementary quantitatives variables
        global_pca = PCA(standardize=False,n_components=n_components,ind_weights=ind_weights.tolist(),var_weights=var_weights.values.tolist(),quali_sup=index,parallelize=self.parallelize).fit(base_quali_sup)
        # Store supplementary qualitatives informations
        quali_sup =  global_pca.quali_sup_.copy()
        
        # Partiel coordinates
        quali_var_sup_coord_partiel = pd.DataFrame().astype("float")
        for grp, _ in self.call_["group"].items():
            quali_sup_coord_partiel = conditional_average(X=self.ind_["coord_partiel"][grp],Y=X_quali_sup,weights=ind_weights)
            quali_sup_coord_partiel.columns = pd.MultiIndex.from_tuples([(grp,col) for col in quali_sup_coord_partiel.columns])
            quali_var_sup_coord_partiel = pd.concat([quali_var_sup_coord_partiel,quali_sup_coord_partiel],axis=1)

        # Supplementary categories informations
        quali_sup = {**quali_sup,**{"coord_partiel" : quali_var_sup_coord_partiel}}
        # Add to dictionary
        res = {**res, **{"quali" : quali_sup}}
    
    return res