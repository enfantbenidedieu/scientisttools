# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import polars as pl

from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

from .pca import PCA
from .revaluate_cat_variable import revaluate_cat_variable
from .weightedcorrcoef import weightedcorrcoef
from .function_lg import function_lg
from .coeffRV import coeffRV

class MFACT(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis for Contingency Tables (MFACT)
    -------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Factor Analysis for Contingency Tables in the sense of Pagès J. (2002) with supplementary individuals and supplementary groups of variables. Groups of variables can be quantitative, categorical.

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `group` : a list or a tuple with the number of variables in each group

    `name_group` : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)
    
    `num_group_sup` : the indexes of the illustrative groups (by default, None and no group re illustrative)

    `ind_sup` : an integer, a list or a tuple of the supplementary individuals
    
    `Parallelize` : bool, default = False. Adding multi-core methods to PandasObject.

    Attributes
    ----------
    `separate_analyses_` : the results for the separate analyses

    `svd_` : a dictionary of matrices containing all the results of the singular value decomposition

    `eig_` : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the cumulative percentge of variance

    `ind_` : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,contributions)

    `ind_sup_`: dictionary of pandas dataframe containing all the results for supplementary individuals (factor coordinates, square cosinus, partiel coordinates)
    
    `freq_` : a dictionary of pandas dataframe containing all the results for the frequencies variables (coordinates, contribution, cos2)

    `freq_sup_` : dictionary of pandas dataframe containing all the results for the supplementary columns
    
    `global_pca_` : the results of the analysis when it is considered as a unique weighted PCA

    `model_` : string. The model fitted = 'mfact'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed Dunod

    Examples
    --------
    ```
    >>> # Load children dataset
    >>> from scientisttools import load_mortality
    >>> children = load_mortality()
    >>> import pandas as pd
    >>> from scientisttools import MFACT
    >>> mortality2 = mortality.copy()
    >>> mortality2.columns = [x + "-2" for x in mortality2.columns]
    >>> dat = pd.concat((mortality,mortality2),axis=1)
    >>> res_mfact = MFACT(group=[9]*4,name_group=["1979","2006","1979-2","2006-2"],num_group_sup=[2,3],ind_sup=list(range(50,dat.shape[0])),parallelize=True)
    >>> res_mfact.fit(dat)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 group = None,
                 name_group = None,
                 num_group_sup = None,
                 ind_sup = None,
                 parallelize= False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.num_group_sup = num_group_sup
        self.ind_sup = ind_sup
        self.parallelize = parallelize
    
    def fit(self, X, y=None):
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

        # Checks if quantitatives variables are in X
        is_quanti = X.select_dtypes(exclude=["object","category"])
        for col in is_quanti.columns.tolist():
            X[col] = X[col].astype("float")
        
        # Check if qualitative variables are in X
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
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
                    new_elt = {group_name[i]:X_group.columns.tolist()}
                    group_sup_dict = {**group_sup_dict,**new_elt}
                else:
                    group_sup_dict = group_sup_dict
                    group_active_dict[group_name[i]] = X_group.columns.tolist()
            else:
                group_active_dict[group_name[i]] = X_group.columns.tolist()
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
        
        # Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"{grp} group should have at least two columns")
        
        # Compute Frequency in all table
        total = X.sum().sum()
        F = X/total

        ## group row margins and marginal columns margins
        F_jt = {}
        Fi_t = {}
        for grp, cols in group_active_dict.items():
            F_jt[grp] = F[cols].sum(axis=0)
            Fi_t[grp] = F[cols].sum(axis=1)
        
        # Set global row margin and columns margin
        row_margin = F.sum(axis=1)
        col_margin = F.sum(axis=0)

        # Sum of frequency by group
        sum_term_grp = pd.Series().astype("float")
        for grp, cols in group_active_dict.items():
            sum_term_grp.loc[grp] = F[cols].sum().sum()
        
        # Construction of table Z
        X1 = mapply(F,lambda x : x/col_margin.values,axis=1,progressbar=False,n_workers=n_workers)
        X2 = pd.DataFrame(columns=list(group_active_dict.keys()),index=X.index).astype("float")
        for grp, cols in group_active_dict.items():
            X2[grp] = F[cols].sum(axis=1)/sum_term_grp[grp]

        # Base for PCA
        base = pd.DataFrame().astype("float")
        columns_dict = {}
        for grp, cols in group_active_dict.items():
            Zb = mapply(X1[cols],lambda x : x - X2[grp].values,axis=0,progressbar=False,n_workers=n_workers)
            Zb = mapply(Zb,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
            base = pd.concat((base,Zb),axis=1)
            columns_dict[grp] = Zb.columns
        
        # Run a Principal Component Analysis (PCA) in each group
        model = {}
        for grp, cols in group_active_dict.items():
            model[grp] = PCA(standardize=False,ind_weights=row_margin.values.tolist(),var_weights=F_jt[grp].values.tolist()).fit(base[cols])
        
        # Square distance to origin for active group
        group_dist2 = pd.Series([np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())],index=list(group_active_dict.keys()),name="Sq. Dist.")

        # Variables weights
        var_weights = pd.Series(name="weight").astype("float")
        for grp, cols in group_active_dict.items():
            weights = F_jt[grp]/model[grp].eig_.iloc[0,0]
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        # Separate general factor analysis for supplementary groups
        if self.num_group_sup is not None:
            base_sup = pd.DataFrame().astype("float")
            columns_sup_dict = {}
            var_sup_weights = pd.Series(name="weight").astype("float")
            for grp, cols in group_sup_dict.items():
                # Compute frequencies
                F_sup = X_group_sup[cols]/(X_group_sup[cols].sum().sum()*total)
                # Compute margin
                F_jt[grp] = F_sup[cols].sum(axis=0)
                Fi_t[grp] = F_sup[cols].sum(axis=1)
                # Standardization data
                Z_sup = mapply(F_sup,lambda x : x/F_jt[grp].values,axis=1,progressbar=False,n_workers=n_workers)
                Z_sup = mapply(Z_sup,lambda x : x - (Fi_t[grp].values/np.sum(Fi_t[grp])),axis=0,progressbar=False,n_workers=n_workers)
                Z_sup = mapply(Z_sup,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
                # Concatenate
                base_sup = pd.concat((base_sup,Z_sup),axis=1)
                columns_sup_dict[grp] = Z_sup.columns
                # Run Principal Components Analysis (PCA)
                model[grp] = PCA(standardize=False,ind_weights=row_margin.values.tolist(),var_weights=F_jt[grp].values.tolist()).fit(Z_sup)
                # Update variables weights
                weights = F_jt[grp]/model[grp].eig_.iloc[0,0]
                var_sup_weights = pd.concat((var_sup_weights,weights),axis=0)
            
            # Square distance to origin for supplementary group
            group_sup_dist2 = pd.Series([np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())],index=list(group_sup_dict.keys()),name="Sq. Dist.")

        # Store separate analysis
        self.separate_analyses_ = model
        
        # QR decomposition (to set number of components)
        Q, R = np.linalg.qr(base)
        max_components = min(np.linalg.matrix_rank(Q),np.linalg.matrix_rank(R))

        # Set number of components
        if self.n_components is None:
            n_components = int(max_components)
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
        
        # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components, 
                      "ind_weights" : row_margin,
                      "var_weights" : var_weights,
                      "group_row_margin" : Fi_t,
                      "group_col_margin" : F_jt,
                      "group" : group_active_dict,
                      "columns_dict" : columns_dict,
                      "group_name" : group_name,
                      "ind_sup" : ind_sup_label}
        
        # Global Principal Component
        global_pca = PCA(standardize=False,n_components=n_components,ind_weights=row_margin.values.tolist(),var_weights=var_weights.values.tolist(),parallelize=self.parallelize).fit(base)

        # Standardization for supplementary individuals
        if self.ind_sup is not None:
            # Divide by total
            F_ind_sup = X_ind_sup/total
            # Supplementray rows margin
            row_sup_margin = F_ind_sup.sum(axis=1)
            Z_ind_sup = pd.DataFrame().astype("float")
            for grp, cols in group_active_dict.items():
                # Partial sum
                partial_sum = F_ind_sup[cols].sum(axis=1)
                # Standardization data
                Z_rsup = mapply(F_ind_sup[cols],lambda x : x/F_jt[grp].values,axis=1,progressbar=False,n_workers=n_workers)
                Z_rsup = mapply(Z_rsup,lambda x : x - (partial_sum.values/np.sum(partial_sum)),axis=0,progressbar=False,n_workers=n_workers)
                Z_rsup = mapply(Z_rsup,lambda x : x/row_sup_margin.values,axis=0,progressbar=False,n_workers=n_workers)
                # Concatenate
                Z_ind_sup = pd.concat((Z_ind_sup,Z_rsup),axis=1)
            
            # Concatenate active rows with supplementary rows
            base_ind_sup = pd.concat((base,Z_ind_sup),axis=0)
            # Update Principal Component Anlaysis (PCA) with supplementary rows
            global_pca = PCA(standardize=False,n_components=n_components,ind_weights=row_margin.values.tolist(),var_weights=var_weights.values.tolist(),ind_sup=ind_sup,parallelize=self.parallelize).fit(base_ind_sup)
            # Store all informations
            self.ind_sup_ = {"coord" : global_pca.ind_sup_["coord"],"cos2" : global_pca.ind_sup_["cos2"]}

            # Partiels coordinates
            ind_sup_coord_partiel = pd.DataFrame().astype("float")
            for grp, cols in columns_dict.items():
                data_partiel = pd.DataFrame(np.tile(global_pca.call_["means"].values,(X_ind_sup.shape[0],1)),index=X_ind_sup.index,columns=base.columns)
                data_partiel[cols] = Z_ind_sup[cols]
                Zbis = (data_partiel - global_pca.call_["means"].values.reshape(1,-1))/global_pca.call_["std"].values.reshape(1,-1)
                partial_coord = len(list(columns_dict.keys()))*Zbis 
                partial_coord = mapply(partial_coord,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(global_pca.svd_["V"][:,:n_components])
                partial_coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
                ind_sup_coord_partiel = pd.concat([ind_sup_coord_partiel,partial_coord],axis=1)
            # Add to dictionary
            self.ind_sup_["coord_partiel"] = ind_sup_coord_partiel

        # Statistics for supplementary 
        if self.num_group_sup is not None:
            # Concatenate active with supplementary
            base_col_sup = pd.concat((base,base_sup),axis=1)
            # Find supplementary quantitatives columns index
            index = [base_col_sup.columns.tolist().index(x) for x in base_sup.columns]
            # Global Principal Component Analysis (PCA)
            global_pca = PCA(standardize=False,n_components=n_components,ind_weights=row_margin.values.tolist(),var_weights=var_weights.values.tolist(),quanti_sup=index,parallelize=self.parallelize).fit(base_col_sup)
            # Store all informations
            self.freq_sup_ = {"coord" : global_pca.quanti_sup_["coord"],"cos2" : global_pca.quanti_sup_["cos2"]}
        
        # Store global PCA
        self.global_pca_ = global_pca

        #   Eigenvalues
        self.eig_ = global_pca.eig_.iloc[:max_components,:]

        # Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:max_components],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        # Individuals/Rows informations : coord, cos2, contrib
        ind = {"coord" : global_pca.ind_["coord"].iloc[:,:n_components],"contrib" : global_pca.ind_["contrib"].iloc[:,:n_components],"cos2" : global_pca.ind_["cos2"].iloc[:,:n_components],"infos":global_pca.ind_["infos"]}
        
        # Columns informations
        freq = {"coord" : global_pca.var_["coord"].iloc[:,:n_components],"contrib" : global_pca.var_["contrib"].iloc[:,:n_components],"cos2" : global_pca.var_["cos2"].iloc[:,:n_components]} 
        self.freq_ = freq

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
            coord_partial = mapply(coord_partial,lambda x : x*global_pca.call_["var_weights"].values,axis=1,progressbar=False,n_workers=n_workers)
            # Transition relation
            coord_partial = coord_partial.dot(global_pca.svd_["V"])
            # Set columns 
            coord_partial.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
            ind_coord_partiel = pd.concat([ind_coord_partiel,coord_partial],axis=1)
        
        # Add to dictionary
        ind["coord_partiel"] = ind_coord_partiel

        # Inertia Ratios
        # "Between" inertia on axis s
        between_inertia = len(list(group_active_dict.keys()))*mapply(ind["coord"],lambda x : (x**2)*row_margin.values,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        between_inertia.name = "between_inertia"

        # Total inertial on axis s
        total_inertia = pd.Series(name="total_inertia").astype("float")
        for dim in ind["coord"].columns:
            value = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : (x**2)*row_margin.values,axis=0,progressbar=False,n_workers=n_workers).sum().sum()
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
            data = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : ((x - ind["coord"][dim].values)**2)*row_margin.values,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_within_inertia.loc[:,dim] = mapply(data.to_frame(dim),lambda x : 100*x/np.sum(x),axis=0,progressbar=False,n_workers=n_workers)
        ind["within_inertia"] = ind_within_inertia

        # Individuals Within partial inertia
        data = pd.DataFrame().astype("float")
        for dim in ind["coord"].columns:
            data1 = mapply(ind_coord_partiel.loc[:, (slice(None),dim)],lambda x : ((x - ind["coord"][dim].values)**2)*row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
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

        # Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            correl = weightedcorrcoef(x=ind["coord"],y=data,w=row_margin.values)[:ind["coord"].shape[1],ind["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=ind["coord"].columns,columns=data.columns)
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = model[grp].ind_["coord"]
                correl = weightedcorrcoef(x=ind["coord"],y=data,w=row_margin.values)[:ind["coord"].shape[1],ind["coord"].shape[1]:]
                coord = pd.DataFrame(correl,index=ind["coord"].columns,columns=data.columns)
                coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
                partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)

            # Reorder using group position
            partial_axes_coord = partial_axes_coord.reindex(columns=partial_axes_coord.columns.reindex(group_name, level=0)[0])
        
        # Partial axes cos2
        partial_axes_cos2 = mapply(partial_axes_coord,lambda x : x**2, axis=0,progressbar=False,n_workers=n_workers)

        # Partial correlation between
        all_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = model[grp].ind_["coord"]
            data.columns = pd.MultiIndex.from_tuples([(grp,col) for col in data.columns])
            all_coord = pd.concat([all_coord,data],axis=1)
        
        # Add supplementary columns
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

        #### Add a null dataframe
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                nbcol = min(n_components,model[grp].call_["n_components"])
                contrib = pd.DataFrame(np.zeros(shape=(n_components,nbcol)),index=["Dim."+str(x+1) for x in range(n_components)],columns=["Dim."+str(x+1) for x in range(nbcol)])
                contrib.columns = pd.MultiIndex.from_tuples([(grp,col) for col in contrib.columns])
                partial_axes_contrib = pd.concat([partial_axes_contrib,contrib],axis=1)
            ## Reorder
            partial_axes_contrib = partial_axes_contrib.reindex(columns=partial_axes_contrib.columns.reindex(group_name, level=0)[0])

        # Correlation beteen
        cor_between = pd.DataFrame(weightedcorrcoef(x=all_coord,w=row_margin.values),index=all_coord.columns,columns=all_coord.columns)
        # Store all informations
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : cor_between}

        ## Group informations : coord
        # Group contributions
        group_contrib = pd.DataFrame(index=list(group_active_dict.keys()),columns=["Dim."+str(x+1) for x in range(n_components)]).astype("float")
        for grp, cols in columns_dict.items():
            columns = model[grp].call_["Z"].columns
            group_contrib.loc[grp,:] = global_pca.var_["contrib"].loc[columns,:].iloc[:,:n_components].sum(axis=0)
        
        # group coordinates
        group_coord = mapply(group_contrib,lambda x : (1/100)*x*(global_pca.svd_["vs"][:n_components]**2),axis=1,progressbar=False,n_workers=n_workers)
        
        # Group square cosinus
        group_cos2 = mapply(group_coord, lambda x : (x**2)/group_dist2.values,axis=0,progressbar=False,n_workers=n_workers)

        # Group correlations
        group_correlation = pd.DataFrame().astype("float")
        for grp in group_coord.index:
            correl = np.diag(weightedcorrcoef(x=ind_coord_partiel[grp],y=ind["coord"],w=row_margin.values)[:ind_coord_partiel[grp].shape[1],ind_coord_partiel[grp].shape[1]:])
            correl  = pd.DataFrame(correl.reshape(1,-1),index=[grp],columns=ind["coord"].columns)
            group_correlation = pd.concat((group_correlation,correl),axis=0)

        # Measuring how similar groups - Lg coefficients
        Lg = pd.DataFrame().astype("float")
        for grp1, cols1 in columns_dict.items():
            for grp2, cols2 in columns_dict.items():
                Lg.loc[grp1,grp2] = function_lg(base[cols1],base[cols2],X_weights=var_weights[cols1],Y_weights=var_weights[cols2],ind_weights=row_margin.values)
        
        # Calculate Lg between supplementary groups
        if self.num_group_sup is not None:
            Lg_sup = pd.DataFrame().astype("float")
            for grp1, cols1 in columns_sup_dict.items():
                for grp2, cols2 in columns_sup_dict.items():
                    Lg_sup.loc[grp1,grp2] = function_lg(X=base_sup[cols1],Y=base_sup[cols2],X_weights=var_sup_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=row_margin.values)
           
            # Concatenate
            Lg = pd.concat((Lg,Lg_sup),axis=1)
            # Fill na with 0.0
            Lg = Lg.fillna(0)

            # Calculate Lg coefficients between active and supplementary groups
            for grp1, cols1 in columns_dict.items():
                for grp2, cols2 in columns_sup_dict.items(): 
                    Lg.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base_sup[cols2],X_weights=var_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=row_margin.values)
                    Lg.loc[grp2,grp1] = Lg.loc[grp1,grp2] 

        # Reorder using group name
        Lg = Lg.loc[group_name,group_name]

        # Add MFA Lg
        Lg.loc["MFA",:] = Lg.loc[:,"MFA"] = Lg.loc[list(group_active_dict.keys()),:].sum(axis=0)/self.eig_.iloc[0,0]
        Lg.loc["MFA","MFA"] = Lg.loc[list(group_active_dict.keys()),"MFA"].sum()/self.eig_.iloc[0,0]

        # RV Coefficients 
        RV = coeffRV(X=Lg)
        
        # Store all informations
        self.group_ = {"coord" : group_coord, "contrib" : group_contrib, "cos2" : group_cos2,"correlation" : group_correlation,"Lg" : Lg, "dist2" : group_dist2,"RV" : RV}
        
        # Add supplementary elements
        if self.num_group_sup is not None:
            # Calculate group sup coordinates
            group_sup_coord = pd.DataFrame(index = list(columns_sup_dict.keys()),columns=["Dim."+str(x+1) for x in range(n_components)]).astype("float")
            for grp, cols in columns_sup_dict.items():
                for i, dim in enumerate(group_sup_coord.columns):
                    group_sup_coord.loc[grp,dim] = function_lg(X=ind["coord"][dim],Y=base_sup[cols],X_weights=1/self.eig_.iloc[i,0],Y_weights=var_sup_weights[cols],ind_weights=row_margin.values)
            
            # Supplementary group square cosinus
            group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index),axis=0)
   
            # Add supplementarary groups informations - append two dictionnaries
            self.group_ = {**self.group_, **{"coord_sup" : group_sup_coord, "cos2_sup" : group_sup_cos2, "dist2_sup" : group_sup_dist2}}
        
        # Name of model
        self.model_ = "mfact"

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
        
        # Extract elemnts
        total = self.call_["X"].sum().sum()
        group_active_dict = self.call_["group"]
        n_components = self.call_["n_components"]
        var_weights = self.call_["var_weights"].values
        F_jt = self.call_["group_col_margin"]
        
        # Divide by total
        F = X/total
        # Supplementray rows margin
        row_margin = F.sum(axis=1)
        Z = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            # Partial sum
            partial_sum = F[cols].sum(axis=1)
            # Standardization data
            Z_sup = mapply(F[cols],lambda x : x/F_jt[grp].values,axis=1,progressbar=False,n_workers=n_workers)
            Z_sup = mapply(Z_sup,lambda x : x - (partial_sum.values/np.sum(partial_sum)),axis=0,progressbar=False,n_workers=n_workers)
            Z_sup = mapply(Z_sup,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
            # Concatenate
            Z = pd.concat((Z,Z_sup),axis=1)
        
        # Standardize data according to PCA method
        Z = (Z  - self.global_pca_.call_["means"].values.reshape(1,-1))/self.global_pca_.call_["std"].values.reshape(1,-1)

        # Multiply by columns weight & Apply transition relation
        coord = mapply(Z,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        return coord

def predictMFACT(self,X=None):
    """
    Predict projection for new individuals with Multiple Factor Analysis for Contingency Tables (MFACT)
    ---------------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus, square distance to origin and partiel coordinates of new individuals with Multiple Factor Analysis for Contincency Tables (MFACT)

    Usage
    -----
    ```python
    >>> predictMFACT(self,X=None)
    ```

    Parameters
    ----------
    `self` : an object of class MFACT

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
    >>> # Load children dataset
    ```
    """
    # Check if self is an object of class MFACT
    if self.model_!= "mfact":
        raise TypeError("'self' must be an object of class MFACT")

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

    # Check if columns are aligned
    if X.shape[1] != self.call_["X"].shape[1]:
        raise ValueError("DataFrame aren't aligned")

    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elemnts
    total = self.call_["X"].sum().sum()
    group_active_dict = self.call_["group"]
    n_components = self.call_["n_components"]
    row_margin = self.call_["ind_weights"]
    var_weights = self.call_["var_weights"].values
    F_jt = self.call_["group_col_margin"]
    
    # Divide by total
    F = X/total
    # Supplementray rows margin
    row_sup_margin = F.sum(axis=1)
    Z = pd.DataFrame().astype("float")
    for grp, cols in group_active_dict.items():
        # Partial sum
        partial_sum = F[cols].sum(axis=1)
        # Standardization data
        Z_rsup = mapply(F[cols],lambda x : x/F_jt[grp].values,axis=1,progressbar=False,n_workers=n_workers)
        Z_rsup = mapply(Z_rsup,lambda x : x - (partial_sum.values/np.sum(partial_sum)),axis=0,progressbar=False,n_workers=n_workers)
        Z_rsup = mapply(Z_rsup,lambda x : x/row_sup_margin.values,axis=0,progressbar=False,n_workers=n_workers)
        # Concatenate
        Z = pd.concat((Z,Z_rsup),axis=1)
    
    # Concatenate active rows with supplementary rows
    base_row_sup = pd.concat((self.call_["Z"],Z),axis=0)
    # Find index of new individuals
    index = [base_row_sup.index.tolist().index(x) for x in X.index]
    # Update Principal Component Anlaysis (PCA) with supplementary rows
    global_pca = PCA(standardize=False,n_components=n_components,ind_weights=row_margin.values.tolist(),var_weights=var_weights.tolist(),ind_sup=index,parallelize=self.parallelize).fit(base_row_sup)

    # Partiels coordinates for new individuals
    ind_sup_coord_partiel = pd.DataFrame().astype("float")
    columns_dict = self.call_["columns_dict"]
    for grp, cols in columns_dict.items():
        data_partiel = pd.DataFrame(np.tile(global_pca.call_["means"].values,(X.shape[0],1)),index=X.index,columns=self.call_["Z"].columns)
        data_partiel[cols] = Z[cols]
        Zbis = (data_partiel - global_pca.call_["means"].values.reshape(1,-1))/global_pca.call_["std"].values.reshape(1,-1)
        partial_coord = len(list(columns_dict.keys()))*Zbis 
        partial_coord = mapply(partial_coord,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(global_pca.svd_["V"][:,:n_components])
        partial_coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in ["Dim."+str(x+1) for x in range(n_components)]])
        ind_sup_coord_partiel = pd.concat([ind_sup_coord_partiel,partial_coord],axis=1)
    # Store all informations
    res = {"coord" : global_pca.ind_sup_["coord"], "cos2" : global_pca.ind_sup_["cos2"], "coord_partiel" : ind_sup_coord_partiel}
    return res

def supvarMFACT(self,X_group_sup=None,group_sup=None,name_group_sup=None):
    """
    Supplementary variables in Multiple Factor Analysis for Contingence Tables (MFACT)
    ----------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus for supplementary variables with Multiple Factor Analysis for Contingency Tables (MFACT)

    Usage
    -----
    ```python
    >>> supvarMFACT(self,X_group_sup=None,group_sup=None,name_group_sup=None)
    ```

    Parameters
    ----------
    `self` : an object of class MFACT

    `X_group_sup` : pandas/polars dataframe of supplementary groups (default=None)

    `group_sup` : a list or a tuple with the number of variables in each supplementary group

    `name_group_sup` : a list or a tuple containing the name of the supplementary groups (by default, None and the group are named Gr1, Gr2 and so on)

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

    `freq` : dictionary containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load children dataset
    ```
    """
    # Check if self is and object of class MFACT
    if self.model_ != "mfact":
        raise TypeError("'self' must be an object of class MFACT")

    # check if X_group_sup is an instance of polars dataframe class
    if isinstance(X_group_sup,pl.DataFrame):
        X_group_sup = X_group_sup.to_pandas()
    
    # Check if X_group_sup is an instance of pandas dataframe class
    if not isinstance(X_group_sup,pd.DataFrame):
        raise TypeError(
        f"{type(X_group_sup)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Set length - number of rows
    if X_group_sup.shape[0] > self.call_["X"].shape[0]:
        X_group_sup = X_group_sup.iloc[:self.call_["X"].shape[1],:]

    # Check if supplementary group is None
    if group_sup is None:
        raise ValueError("'group_sup' must be assigned.")
    elif not (isinstance(group_sup, list) or isinstance(group_sup,tuple)):
        raise ValueError("'group_sup' must be a list or a tuple with the number of variables in each group")
    else:
        nb_elt_group_sup = [int(x) for x in group_sup]
        
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
        
    # Assigned supplementary group name to label
    group_sup_dict = {}
    debut = 0
    for i in range(len(nb_elt_group_sup)):
        X_group = X_group_sup.iloc[:,(debut):(debut+nb_elt_group_sup[i])]
        group_sup_dict[group_sup_name[i]] = X_group.columns
        debut = debut + nb_elt_group_sup[i]
    
    # Extract elements
    total = self.call_["X"].sum().sum()
    n_components = self.call_["n_components"]
    row_margin = self.call_["ind_weights"]

    base_sup = pd.DataFrame().astype("float")
    columns_sup_dict = {}
    var_sup_weights = pd.Series(name="weight").astype("float")
    model = {}
    for grp, cols in group_sup_dict.items():
        # Compute frequencies
        F_sup = X_group_sup[cols]/(X_group_sup[cols].sum().sum()*total)
        # Compute margin sum
        col_group_sum = F_sup[cols].sum(axis=0)
        row_group_sum = F_sup[cols].sum(axis=1)
        # Standardization data
        Z_sup = mapply(F_sup,lambda x : x/col_group_sum.values,axis=1,progressbar=False,n_workers=n_workers)
        Z_sup = mapply(Z_sup,lambda x : x - (row_group_sum.values/np.sum(row_group_sum)),axis=0,progressbar=False,n_workers=n_workers)
        Z_sup = mapply(Z_sup,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
        # Concatenate
        base_sup = pd.concat((base_sup,Z_sup),axis=1)
        columns_sup_dict[grp] = Z_sup.columns
        # Run Principal Components Analysis (PCA)
        model[grp] = PCA(standardize=False,ind_weights=row_margin.values.tolist(),var_weights=col_group_sum.values.tolist()).fit(Z_sup)
        # Update variables weights
        weights = col_group_sum/model[grp].eig_.iloc[0,0]
        var_sup_weights = pd.concat((var_sup_weights,weights),axis=0)
    
    # Square distance to origin for supplementary group
    group_sup_dist2 = pd.Series([np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())],index=list(group_sup_dict.keys()),name="Sq. Dist.")

    # Lg coefficients between supplementary groups
    Lg_sup = pd.DataFrame().astype("float")
    for grp1, cols1 in columns_sup_dict.items():
        for grp2, cols2 in columns_sup_dict.items():
            Lg_sup.loc[grp1,grp2] = function_lg(X=base_sup[cols1],Y=base_sup[cols2],X_weights=var_sup_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=row_margin.values)
    
    # Calculate Lg coefficients between active and supplementary groups
    base = self.call_["Z"]
    var_weights = self.call_["var_weights"]
    columns_dict = self.call_["columns_dict"]
    Lg = pd.DataFrame(index=list(columns_dict.keys()),columns=list(columns_dict.keys()))
    Lg_sup = pd.concat([Lg,Lg_sup],axis=1)
    for grp1, cols1 in columns_dict.items():
        for grp2, cols2 in columns_sup_dict.items(): 
            Lg_sup.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base_sup[cols2],X_weights=var_weights[cols1],Y_weights=var_sup_weights[cols2],ind_weights=row_margin.values)
            Lg_sup.loc[grp2,grp1] = Lg_sup.loc[grp1,grp2]
    
    # Add Lg coefficients for active groups
    for grp1, cols1 in columns_dict.items():
        for grp2, cols2 in columns_dict.items():
            Lg_sup.loc[grp1,grp2] = function_lg(X=base[cols1],Y=base[cols2],X_weights=var_weights[cols1],Y_weights=var_weights[cols2],ind_weights=row_margin.values)
    
    # Calculate RV coefficients
    RV_sup = coeffRV(X=Lg_sup)

    # Supplementary group factor coordinates
    group_sup_coord = pd.DataFrame().astype("float")
    group_sup_coord = pd.DataFrame(index=list(columns_sup_dict.keys()),columns=["Dim."+str(x+1) for x in range(n_components)]).astype("float")
    for grp, cols in columns_sup_dict.items():
        for i, dim in enumerate(group_sup_coord.columns):
            group_sup_coord.loc[grp,dim] = function_lg(X=self.ind_["coord"][dim],Y=base_sup[cols],X_weights=1/self.eig_.iloc[i,0],Y_weights=var_sup_weights[cols],ind_weights=row_margin.values)
    
    # Supplementary group square cosinus
    group_sup_cos2 = pd.concat((((group_sup_coord.loc[grp,:]**2)/group_sup_dist2.loc[grp]).to_frame(grp).T for grp in group_sup_coord.index),axis=0)
    
    # Store all informations
    group_sup_infos = {"coord" : group_sup_coord, "cos2" : group_sup_cos2, "dist2" : group_sup_dist2, "Lg" : Lg_sup, "RV" : RV_sup}

    # Partial axis coordinates
    partial_axes_coord = pd.DataFrame().astype("float")
    for grp, cols in group_sup_dict.items():
        data = model[grp].ind_["coord"]
        correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=row_margin.values)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
        coord = pd.DataFrame(correl,index=self.ind_["coord"].columns,columns=data.columns)
        coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns])
        partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)

    # Partial axes square cosinus
    partial_axes_cos2 = partial_axes_coord**2
    # Store all informtions
    partial_axes = {"coord" : partial_axes_coord, "cos2" : partial_axes_cos2}

    ## Supplementary columns informations
    # Concatenate active with supplementary
    base_col_sup = pd.concat((base,base_sup),axis=1)
    # Find supplementary quantitatives columns index
    index = [base_col_sup.columns.tolist().index(x) for x in base_sup.columns]
    # Global Principal Component Analysis (PCA)
    global_pca = PCA(standardize=False,n_components=n_components,ind_weights=row_margin.values.tolist(),var_weights=var_weights.values.tolist(),quanti_sup=index,parallelize=self.parallelize).fit(base_col_sup)
    # Store all informations
    freq_sup = {"coord" : global_pca.quanti_sup_["coord"],"cos2" : global_pca.quanti_sup_["cos2"]}

    # Store results
    res = {"group" : group_sup_infos, "partial_axes" : partial_axes, "freq" : freq_sup}
    return res