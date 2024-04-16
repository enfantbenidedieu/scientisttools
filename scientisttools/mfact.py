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

class MFACT(BaseEstimator,TransformerMixin):
    """
    Multiple Factor Analysis For Contingency Tables (MFACT)
    -------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Multiple Factor Analysis for Contingency Tables in the sense of Pagès J. (2002) with supplementary individuals
    and supplementary groups of variables. Groups of variables can be quantitative, categorical.

    Parameters
    ----------
    n_components : number of dimensions kept in the results (by default 5)

    group : a list or a tuple with the number of variables in each group

    name_group : a list or a tuple containing the name of the groups (by default, None and the group are named Gr1, Gr2 and so on)
    
    num_group_sup : the indexes of the illustrative groups (by default, None and no group re illustrative)

    row_sup : an integer, a list or a tuple of the supplementary rows
    
    Parallelize : bool, default = False. Adding multi-core methods to PandasObject.

    Return
    ------
    separate_analyses_ : the results for the separate analyses

    svd_ : a dictionary of matrices containing all the results of the singular value decomposition

    eig_ : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalues, the percentage of variance and the
            cumulative percentge of variance

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates, square cosine,
            contributions)
    
    freq_ : a dictionary of pandas dataframe containing all the results for the frequencies variables (coordinates, contribution, cos2)
    
    global_pca_ : the results of the analysis when it is considered as a unique weighted PCA

    model_ : string. The model fitted = 'mfact'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    Escofier B, Pagès J (1998), Analyses Factorielles Simples et Multiples. Dunod
    """
    def __init__(self,
                 n_components = 5,
                 group = None,
                 name_group = None,
                 num_group_sup = None,
                 row_sup = None,
                 parallelize= False):
        self.n_components = n_components
        self.group = group
        self.name_group = name_group
        self.num_group_sup = num_group_sup
        self.row_sup = row_sup
        self.parallelize = parallelize
    
    def fit(self, X, y=None):
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
            self.freq_sup_ = None
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif ((isinstance(self.num_group_sup,list) or isinstance(self.num_group_sup,tuple)) and len(self.num_group_sup)>=1):
                num_group_sup = [int(x) for x in self.num_group_sup]

        ##################################################################
        # Check if supplementary rows
        if self.row_sup is not None:
            if (isinstance(self.row_sup,int) or isinstance(self.row_sup,float)):
                row_sup = [int(self.row_sup)]
            elif ((isinstance(self.row_sup,list) or isinstance(self.row_sup,tuple)) and len(self.row_sup)>=1):
                row_sup = [int(x) for x in self.row_sup]

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

        ####################################### Save the base in a new variables
        # Store data
        Xtot = X.copy()

       ######################################## Drop supplementary groups columns #######################################
        if self.num_group_sup is not None:
            X = X.drop(columns=list(itertools.chain.from_iterable(group_sup_dict.values())))
        
        ######################################## Drop supplementary individuals  ##############################################
        if self.row_sup is not None:
            # Extract supplementary individuals
            X_row_sup = X.iloc[self.row_sup,:]
            # Drop supplementary individuals
            X = X.drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in row_sup])
        
        ############################# Check if an active group has only one columns
        for grp, cols in group_active_dict.items():
            if len(cols)==1:
                raise ValueError(f"Error : {grp} group should have at least two columns")
        
        ######################################## Compute Frequency in all table #######################################
        F = X/X.sum().sum()

        ########################################################################################################################"
        ##  Row margins and columns margins
        #######################################################################################################################
        F_jt = {}
        Fi_t = {}
        for grp, cols in group_active_dict.items():
            F_jt[grp] = F[cols].sum(axis=0)
            Fi_t[grp] = F[cols].sum(axis=1)
        
        #### Set row margin and columns margin
        row_margin = F.sum(axis=1)
        col_margin = F.sum(axis=0)

        ####################################### Sum of frequency by group #############################################
        sum_term_grp = pd.Series().astype("float")
        for grp, cols in group_active_dict.items():
            sum_term_grp.loc[grp] = F[cols].sum().sum()
        
        ########################################### Construction of table Z #############################################"
        X1 = mapply(F,lambda x : x/col_margin.values,axis=1,progressbar=False,n_workers=n_workers)
        X2 = pd.DataFrame(columns=list(group_active_dict.keys()),index=X.index.tolist()).astype("float")
        for grp, cols in group_active_dict.items():
            X2[grp] = F[cols].sum(axis=1)/sum_term_grp[grp]

        ##########
        base = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            Zb = mapply(X1[cols],lambda x : x - X2[grp].values,axis=0,progressbar=False,n_workers=n_workers)
            Zb = mapply(Zb,lambda x : x/row_margin.values,axis=0,progressbar=False,n_workers=n_workers)
            base = pd.concat((base,Zb),axis=1)
        
        # Run a Principal Component Analysis (PCA) in each group
        model = {}
        for grp, cols in group_active_dict.items():
            fa = PCA(standardize=False,ind_sup=None,ind_weights=row_margin.values.tolist(),var_weights=F_jt[grp].values.tolist())
            model[grp] = fa.fit(base[cols])
        
        ##################### Compute group disto
        group_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_active_dict.keys())]
        group_dist2 = pd.Series(group_dist2,index=list(group_active_dict.keys()),name="dist2")

        ##### Compute group
        if self.num_group_sup is not None:
            group_sup_dist2 = [np.sum(model[grp].eig_.iloc[:,0]**2)/model[grp].eig_.iloc[0,0]**2 for grp in list(group_sup_dict.keys())]
            group_sup_dist2 = pd.Series(group_sup_dist2,index=list(group_sup_dict.keys()),name="dist2")

        ##### Store separate analysis
        self.separate_analyses_ = model

        #################################################
        ####### columns weights
        var_weights = pd.Series(name="weight").astype("float")
        for grp, cols in group_active_dict.items():
            weights = F_jt[grp]/model[grp].eig_.iloc[0,0]
            var_weights = pd.concat((var_weights,weights),axis=0)
        
        ######################################################## Global
        global_pca = PCA(standardize=False,n_components=None,ind_weights=row_margin.values.tolist(),var_weights=var_weights.values.tolist(),parallelize=self.parallelize)
        global_pca.fit(base)

        ############################################# Removing duplicate value in cumulative percent #######################"
        cumulative = sorted(list(set(global_pca.eig_.iloc[:,3])))

        # Number of components
        if self.n_components is None:
            n_components = len(cumulative)
        else:
            n_components = min(self.n_components,len(cumulative))
        
         # Save
        self.call_ = {"Xtot" : Xtot,
                      "X" : X, 
                      "Z" : base,
                      "n_components" : n_components,
                      "ind_weights" : row_margin,
                      "var_weights" : var_weights,
                      "group" : group_active_dict,
                      "group_name" : group_name}

        ########################################## Store global PCA
        self.global_pca_ = global_pca

        ##################################################################################################################
        #   Eigenvalues
        ##################################################################################################################
        self.eig_ = global_pca.eig_.iloc[:len(cumulative),:]

        ####### Update SVD
        self.svd_ = {"vs" : global_pca.svd_["vs"][:len(cumulative)],"U" : global_pca.svd_["U"][:,:n_components],"V" : global_pca.svd_["V"][:,:n_components]}

        ####################################################################################################
        #    Individuals/Rows informations : coord, cos2, contrib
        ###################################################################################################
        ind = {"coord" : global_pca.ind_["coord"].iloc[:,:n_components],"contrib" : global_pca.ind_["contrib"].iloc[:,:n_components],"cos2" : global_pca.ind_["cos2"].iloc[:,:n_components],"infos":global_pca.ind_["infos"]}
        self.ind_ = ind

        ######################################################################################################
        #    Columns informations
        #######################################################################################################
        freq = {"coord" : global_pca.var_["coord"].iloc[:,:n_components],"contrib" : global_pca.var_["contrib"].iloc[:,:n_components],"cos2" : global_pca.var_["cos2"].iloc[:,:n_components]} 
        self.freq_ = freq

        ##################################################################################################
        #   Partial axes informations
        #################################################################################################
        ########################################### Partial axes coord
        partial_axes_coord = pd.DataFrame().astype("float")
        for grp, cols in group_active_dict.items():
            data = self.separate_analyses_[grp].ind_["coord"]
            correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=row_margin.values)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
            coord = pd.DataFrame(correl,index=self.ind_["coord"].columns.tolist(),columns=data.columns.tolist())
            coord.columns = pd.MultiIndex.from_tuples([(grp,col) for col in coord.columns.tolist()])
            partial_axes_coord = pd.concat([partial_axes_coord,coord],axis=1)
        
        if self.num_group_sup is not None:
            for grp, cols in group_sup_dict.items():
                data = self.separate_analyses_[grp].ind_["coord"]
                correl = weightedcorrcoef(x=self.ind_["coord"],y=data,w=row_margin.values)[:self.ind_["coord"].shape[1],self.ind_["coord"].shape[1]:]
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
        cor_beteween = pd.DataFrame(weightedcorrcoef(x=all_coord,w=row_margin.values),index=all_coord.columns,columns=all_coord.columns)
        self.partial_axes_ = {"coord" : partial_axes_coord,"cor" : partial_axes_coord,"contrib" : partial_axes_contrib,"cos2":partial_axes_cos2,"cor_between" : cor_beteween}

        ############################### group result
        self.group_ = {"dist2" : group_dist2}

        # Name of model
        self.model_ = "mfact"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit to data, then transform it.

        Parameters:
        ----------
        X : pandas DataFrame of shape (n_rows_,n_cols_)

        y : None
            y is ignored

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
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check new dataframe are aligned
        if X.shape[1] != self.call_["X"].shape[1]:
            raise ValueError("DataFrame aren't aligned")
        
        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        raise NotImplementedError("This method is not yet implemented")