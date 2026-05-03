# -*- coding: utf-8 -*-
from numpy import ndarray,zeros, nan, inf
from pandas import Series, DataFrame, concat
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from ..functions.gfa import gFA
from ..functions.preprocessing import preprocessing
from ..functions.func_predict import func_predict

class ICA(BaseEstimator,TransformerMixin):
    """
    Internal Correspondence Analysis (ICA)

    Performns Internal Correspondence Analysis in the sense of `Daniel Chessel and al. <https://www.numdam.org/item/RSA_1988__36_1_39_0.pdf>`_ with supplementary groups of rows and supplementary groups of columns.

    Parameters
    ----------
    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_group : list, tuple
        The number of rows in each rows groups.

    name_row_group : list, tuple, default = None
        The name of the rows groups. If ``None``, the group are named RowGr1, RowGr2 and so on.

    col_group : list, tuple
        The number of columns in each columns groups.

    name_col_group : list, tuple, default = None
        The name of the columns groups. If ``None``, the group are named ColGr1, ColGr2 and so on.
    
    num_row_group_sup : list, tuple, default = None
        The indexes of the illustrative rows groups.

    num_col_group_sup : list, tuple, default = None
        The indexes of the illustrative columns groups.

    Parameters
    ----------
    call_ : call
        An object with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        total : int
            The sum of elements in ``X``.
        row_s : Series of shape (n_rows,)
            The rows sums.
        col_s : Series of shape (n_columns,)
            The columns sums.
        row_m : Series of shape (n_rows,)
            The rows marging
        col_m : Series of shape (n_columns,)
            The columns marging
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        ncp : int
            The number of components kepted.
        row_group : list
            The number of rows in each rows groups.
        col_group : list
            The number of columns in each columns groups.
        row_sup : None, list
            The names of the supplementary rows.
        col_sup : None, list
            The names of the supplementary columns.

    col_ : col
        An object containing all the results for the active columns with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates for the active columns.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus for the active columns.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions for the active columns.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the active columns.

    col_group_ : col_group
        An object containing all the results for the columns groups with the following attributes:

        coord : DataFrame of shape (n_col_groups, ncp)
            The coordinates for the columns groups.
        contrib : DataFrame of shape (n_col_groups, ncp)
            The relative contributions for the columns groups.
        infos : DataFrame of shape (n_col_groups, 3)
            Additionals informations (weight, inertia and percentage of inertia) for the columns groups.

    col_sup_: col_sup, optional
        An object containing all the results for the supplementary columns with the following attributes:

        coord : DataFrame of shape (n_columns_sup, ncp)
            The coordinates for the supplementary columns.
        cos2 : DataFrame of shape (n_columns_sup, ncp)
            The squared cosinus for the supplementary columns.
        dist2 : Series of shape (n_columns_sup,)
            The squared distance to origin for the supplementary columns.
        coord_partiel : coord
            An object with partiel columns coordinates.

    col_sup_group_ : col_sup_group, optional
        An object containing all the results for the supplementary columns groups with the following attributes:

        coord : DataFrame of shape (n_col_sup_groups, ncp)
            The coordinates for the supplementary columns groups.
        weight : Series of shape (n_col_sup_groups,)
            Weight for the supplementary columns groups.

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    row_ : row
        An object containing all the results for the active rows with the following attributes:

        coord : DataFrame of shape (n_rows, ncp) 
            The coordinates for the active rows.
        cos2 : DataFrame of shape (n_rows, ncp)
            The squared cosinus for the active rows.
        contrib : DataFrame of shape (n_rows, ncp)
            The relative contributions for the active rows.
        dist2 : Series of shape (n_rows,)
            The squared distance to origin for the active rows.
        infos : DataFrame of shape (n_rows, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) for the active rows.
        coord_partiel : coord
            An object with partiel coordinates for the rows.

    row_group_ : row_group
        An object containing all the results for the active rows groups with the following attributes:

        coord : DataFrame of shape (n_rows_groups, ncp) 
            The coordinates for the active rows groups.
        contrib : DataFrame of shape (n_rows_groups, ncp)
            The relative contributions for the active rows.
        infos : DataFrame of shape (n_rows_groups, 3)
            Additionals informations (weight, inertia and percentage of inertia) for the active rows groups.

    row_sup_ : row_sup, optional
        An object containing all the results for the supplementary rows with the following attributes:

        coord : DataFrame of shape (n_rows_sup, ncp)
            The coordinates for the supplementary rows.
        cos2 : DataFrame of shape (n_rows_sup, ncp)
            The squared cosinus for the supplementary rows.
        dist2 : Series of shape (n_rows_sup,)
            The squared distance to origin for the supplementary rows.
        coord_partiel : coord
            An object with partiel coordinates for the supplementary rows.

    row_sup_group_ : row_sup_group, optional
        An object containing all the results for the supplementary rows groups with the following attributes:

        coord : DataFrame of shape (n_rows_sup_groups, ncp) 
            The coordinates for the supplementary rows groups.
        weight : Series of shape (n_rows_sup_groups,)
            Weight for the supplementary rows groups.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, ncp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, ncp)
            The right singular vectors.
    
    References
    ----------
    [1] Cazes, P., Chessel, D., and Dolédec, S. (1988) `L'analyse des correspondances internes d'un tableau partitionné : son usage en hydrobiologie <https://www.numdam.org/item/RSA_1988__36_1_39_0.pdf>`. Revue de Statistique Appliquée, 36, 39-54.
    
    See Also
    --------
    :class:`scientisttools.save`
        Print results for general factor analysis model in an Excel sheet.
    :class:`scientisttools.sprintf`
        Print the analysis results.
    :class:`scientisttools.summary`
        Printing summaries of general factor analysis model.

    Examples
    --------
    >>> from scientisttools.datasets import ardeche
    >>> from scientisttools import ICA
    >>> clf = ICA(row_group=ardeche.row_group,name_row_group=ardeche.name_row_group,col_group=ardeche.col_group,name_col_group=ardeche.name_col_group)
    >>> clf.fit(ardeche.data)
    """
    def __init__(
        self, ncp = 5, row_group = None, name_row_group = None, col_group = None, name_col_group = None, num_row_group_sup = None, num_col_group_sup = None
    ):
        self.ncp = ncp
        self.row_group = row_group
        self.name_row_group = name_row_group
        self.col_group = col_group
        self.name_col_group = name_col_group
        self.num_row_group_sup = num_row_group_sup
        self.num_col_group_sup = num_col_group_sup

    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns),
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.
            ``X`` is a contingency table containing absolute frequencies.

        y : None
            y is ignored.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if negative entries
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X[X<0].any().any():
            raise ValueError("negative entries in X")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if group (row_group and col_group) is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of elements in rows group
        if self.row_group is None:
            raise ValueError("'row_group' must be assigned.")
        elif (not isinstance(self.row_group, (list,tuple,Series))) or (isinstance(self.row_group,ndarray) and self.row_group.ndim != 1):
            raise ValueError("'row_group' must be a 1d array-like with the number of rows in each row group")
        else:
            row_group = [int(x) for x in self.row_group]

        #set number of elements in columns group
        if self.col_group is None:
            raise ValueError("'col_group' must be assigned.")
        elif (not isinstance(self.col_group, (list,tuple,Series))) or (isinstance(self.col_group,ndarray) and self.col_group.ndim != 1):
            raise ValueError("'col_group' must be a 1d array-like with the number of columns in each column group")
        else:
            col_group = [int(x) for x in self.col_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if any group has only one rows/columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if any(x==1 for x in row_group):
            raise ValueError("row_group should have at least two rows")
        
        if any(x==1 for x in col_group):
            raise ValueError("col_group should have at least two columns")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned name_group (name_row_group and name_col_group)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.name_row_group is None:
            name_row_group = [f"RowGr{x+1}" for x in range(len(row_group))]
        elif (not isinstance(self.name_row_group,(list,tuple,Series))) or (isinstance(self.name_row_group,ndarray) and self.name_row_group.ndim != 1):
            raise TypeError("'name_row_group' must be a 1d array-like with names of rows group")
        else:
            name_row_group = [x for x in self.name_row_group]

        if self.name_col_group is None:
            name_col_group = [f"ColGr{x+1}" for x in range(len(col_group))]
        elif (not isinstance(self.name_col_group,(list,tuple,Series))) or (isinstance(self.name_col_group,ndarray) and self.name_col_group.ndim != 1):
            raise TypeError("'name_col_group' must be a 1d array-like with names of columns group")
        else:
            name_col_group = [x for x in self.name_col_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary groups (rows and/columns)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.num_row_group_sup is not None:
            if isinstance(self.num_row_group_sup,int):
                num_row_group_sup = [int(self.num_row_group_sup)]
            elif isinstance(self.num_row_group_sup,(list,tuple)) and len(self.num_row_group_sup)>=1:
                num_row_group_sup = [int(x) for x in self.num_row_group_sup]
        else:
            num_row_group_sup = None

        if self.num_col_group_sup is not None:
            if isinstance(self.num_col_group_sup,int):
                num_col_group_sup = [int(self.num_col_group_sup)]
            elif isinstance(self.num_col_group_sup,(list,tuple)) and len(self.num_col_group_sup)>=1:
                num_col_group_sup = [int(x) for x in self.num_col_group_sup]
        else:
            num_col_group_sup = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned rows and columns to groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        row_group_dict, k = OrderedDict(), 0
        for i, g in zip(range(len(row_group)),name_row_group):
            row_group_dict[g] = list(X.index[k:(k+row_group[i])])
            k += row_group[i]

        col_group_dict, k = OrderedDict(), 0
        for i, g in zip(range(len(col_group)),name_col_group):
            col_group_dict[g] = list(X.columns[k:(k+col_group[i])])
            k += col_group[i]
        
        if self.num_row_group_sup is not None:
            row_group_sup_dict = OrderedDict({g : row_group_dict[g] for i, g in enumerate(name_row_group) if i in num_row_group_sup})
            row_group_dict = OrderedDict({g : row_group_dict[g] for i, g in enumerate(name_row_group) if not i in num_row_group_sup})
        
        if self.num_col_group_sup is not None:
            col_group_sup_dict = OrderedDict({g : col_group_dict[g] for i, g in enumerate(name_col_group) if i in num_col_group_sup})
            col_group_dict = OrderedDict({g : col_group_dict[g] for i, g in enumerate(name_col_group) if not i in num_col_group_sup})
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract supplementary labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        col_sup_label = None if self.num_col_group_sup is None else list(chain.from_iterable(col_group_sup_dict.values()))
        row_sup_label = None if self.num_row_group_sup is None else list(chain.from_iterable(row_group_sup_dict.values()))

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary groups columns
        if self.num_col_group_sup is not None:
            X_col_sup, X = X.loc[:,col_sup_label], X.drop(columns=col_sup_label)
            if self.num_row_group_sup is not None:
                X_col_sup = X_col_sup.drop(index=row_sup_label)
        
        #drop supplementary groups rows
        if self.num_row_group_sup is not None:
            X_row_sup, X = X.loc[row_sup_label,:], X.drop(index=row_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #internal correspondence analysis (ICA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #row group variables
        y_row = list(chain(*[repeat(i,len(x)) for i, x in zip(list(row_group_dict.keys()),list(row_group_dict.values()))]))
        #columns group variables
        y_col = list(chain(*[repeat(i,len(x)) for i, x in zip(list(col_group_dict.keys()),list(col_group_dict.values()))]))

        #rows sums, columns sums and sum of all elements
        row_s, col_s, total = X.sum(axis=1), X.sum(axis=0), int(X.sum().sum())
        # #rows and columns margins and frequencies
        row_m, col_m, P = row_s.div(total), col_s.div(total), X.div(total)
        #rows and columns weights
        row_w, col_w = row_m, col_m
        #set names
        row_s.name, row_m.name, row_w.name, col_s.name, col_m.name, col_w.name = "ni.", "fi.", "weight", "n.j", "f.j", "weight"

        #normalize row_w by group (sum in each row group equal to 1)
        row_m_n = concat((row_m.loc[x]/row_m.loc[x].sum() for x in list(row_group_dict.values())),axis=0)
        #normalize col_m by group (sum in each column group equal to 1)
        col_m_n = concat((col_m.loc[x]/col_m.loc[x].sum() for x in list(col_group_dict.values())),axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #columns sums of frequencies by group - sum of frequencies for all rows in each group
        A = concat((P.loc[r,:].sum(axis=0).to_frame(g) for r, g in zip(list(row_group_dict.values()),list(row_group_dict.keys()))),axis=1).T
        #weight A with normed group row weight
        Aw = A.loc[y_row,:].mul(row_m_n.values,axis=0)

        #rows sums of frequencies by group - sum of frequencies for all columns in each group
        B = concat((P.loc[:,c].sum(axis=1).to_frame(g) for c, g in zip(list(col_group_dict.values()),list(col_group_dict.keys()))),axis=1)
        #multiply col_group_freq by col_m_n
        Bw = B.loc[:,y_col].mul(col_m_n.values,axis=1)

        #sum of frequencies by rows and columns groups
        C = concat((A.loc[:,c].sum(axis=1).to_frame(g) for c, g in zip(list(col_group_dict.values()),list(col_group_dict.keys()))),axis=1)
        #multiply col_group_freq by col_m_n
        Cw = C.loc[y_row,y_col].mul(row_m_n.values,axis=0).mul(col_m_n.values,axis=1)

        #standardized (formula p45)
        Z = (P.sub(Aw.values).sub(Bw.values).add(Cw.values)).div(row_m,axis=0).div(col_m,axis=1)
        #fill NA, +/-inf if 1e-15
        Z = Z.replace([nan,inf,-inf], 1e-15)  
        #Standardized data using for GSVD
        tab = Z.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized principal components analysis and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=tab,ncp=self.ncp,row_w=row_w,col_w=col_w)
        #extract elements
        self.svd_, self.eig_, ncp = fit_.svd, fit_.eig, fit_.ncp

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,tab=tab,total=total,row_s=row_s,col_s=col_s,row_m=row_m,col_m=col_m,row_w=row_w,col_w=col_w,ncp=ncp,
                            row_group=row_group,col_group=col_group,row_sup=row_sup_label,col_sup=col_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for rows groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #weight of row group
        row_group_w = Series([row_m.loc[x].sum() for x in list(row_group_dict.values())],index=list(row_group_dict.keys()),name="weight")
        #infos for rows groups
        row_group_infos = concat((fit_.row["infos"].loc[r,["Inertia", "Inertia (%)"]].sum(axis=0).to_frame(g) for r, g in zip(list(row_group_dict.values()),list(row_group_dict.keys()))),axis=1).T
        #insert rows groups weights
        row_group_infos.insert(0,"Weight", row_group_w)
        #contributions for rows groups
        row_group_ctr = concat((fit_.row["contrib"].loc[r,:].sum(axis=0).to_frame(g) for r, g in zip(list(row_group_dict.values()),list(row_group_dict.keys()))),axis=1).T
        #coordinates for rows groups
        row_group_coord = row_group_ctr.div(row_group_w.mul(100),axis=0).mul(self.eig_.iloc[:ncp,0],axis=1)
        #convert to ordered dictionary
        row_group_ = OrderedDict(coord=row_group_coord,contrib=row_group_ctr,infos=row_group_infos)
        #convert to namedtuple
        self.row_group_ = namedtuple("row_group",row_group_.keys())(*row_group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for columns groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #weight of columns group
        col_group_w = Series([col_m.loc[x].sum() for x in list(col_group_dict.values())],index=list(col_group_dict.keys()),name="weight")
        #infos for columns group
        col_group_infos = concat((fit_.col["infos"].loc[r,["Inertia", "Inertia (%)"]].sum(axis=0).to_frame(g) for r, g in zip(list(col_group_dict.values()),list(col_group_dict.keys()))),axis=1).T
        #insert columns groups weights
        col_group_infos.insert(0,"Weight", col_group_w)
        #contributions for rows groups
        col_group_ctr = concat((fit_.col["contrib"].loc[c,:].sum(axis=0).to_frame(g) for c, g in zip(list(col_group_dict.values()),list(col_group_dict.keys()))),axis=1).T
        #coordinates for rows groups
        col_group_coord = col_group_ctr.div(col_group_w.mul(100),axis=0).mul(self.eig_.iloc[:ncp,0],axis=1)
        #convert to ordered dictionary
        col_group_ = OrderedDict(coord=col_group_coord,contrib=col_group_ctr,infos=col_group_infos)
        #convert to namedtuple
        self.col_group_ = namedtuple("col_group",col_group_.keys())(*col_group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #others rows informations : partiels coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partiels coordinates for rows
        coord_partiel = OrderedDict()
        for g, c in col_group_dict.items():
            data = DataFrame(zeros(X.shape),index=tab.index,columns=tab.columns)
            data[c] = tab[c]
            coord = data.mul(col_w,axis=1).dot(self.svd_.V[:,:ncp])
            coord.columns = self.eig_.index[:ncp]
            coord_partiel[g] = coord
        #add to dictionary
        fit_.row["coord_partiel"] = namedtuple("coord",coord_partiel.keys())(*coord_partiel.values())
        #convert to namedtuple
        self.row_ = namedtuple("row",fit_.row.keys())(*fit_.row.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #others columns informations : partiels coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partiels coordinates for columns
        coord_partiel = OrderedDict()
        for g, r in row_group_dict.items():
            data = DataFrame(zeros(X.shape),index=tab.index,columns=tab.columns)
            data.loc[r,:] = tab.loc[r,:]
            coord = data.mul(row_w,axis=0).T.dot(self.svd_.U[:,:ncp])
            coord.columns = self.eig_.index[:ncp]
            coord_partiel[g] = coord
        #add to dictionary
        fit_.col["coord_partiel"] = namedtuple("coord",coord_partiel.keys())(*coord_partiel.values())
        #convert to namedtuple
        self.col_ = namedtuple("col",fit_.col.keys())(*fit_.col.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.num_row_group_sup is not None:
            #frequencies of supplementary rows
            P_row_sup = X_row_sup.div(total)
            #margins for supplementary rows
            row_sup_m = P_row_sup.sum(axis=1)

            #normalize row margins by row_group_sup (sum in each row group equal to 1)
            row_sup_m_n = concat((row_sup_m.loc[x]/row_sup_m.loc[x].sum() for x in list(row_group_sup_dict.values())),axis=0)
            #supplementary row group variables
            y_row_sup = list(chain(*[repeat(i,len(x)) for i, x in zip(list(row_group_sup_dict.keys()),list(row_group_sup_dict.values()))]))
            
            #sum of frequencies by row group
            A = concat((P_row_sup.loc[r,:].sum(axis=0).to_frame(g) for r, g in zip(list(row_group_sup_dict.values()),list(row_group_sup_dict.keys()))),axis=1).T
            #weighted A by row_m_n
            Aw = A.loc[y_row_sup,:].mul(row_sup_m_n.values,axis=0)

            #sum of row sup frequencies by col group
            B = concat((P_row_sup.loc[:,c].sum(axis=1).to_frame(g) for c, g in zip(list(col_group_dict.values()),list(col_group_dict.keys()))),axis=1)
            #weighted B by col_m_n
            Bw = B.loc[:,y_col].mul(col_m_n.values,axis=1)

            #sum of frequencies by col group
            C = concat((A.loc[:,c].sum(axis=1).to_frame(g) for c, g in zip(list(col_group_dict.values()),list(col_group_dict.keys()))),axis=1)
            #multiply col_group_freq by col_m_n
            Cw = C.loc[y_row_sup,y_col].mul(row_sup_m_n.values,axis=0).mul(col_m_n.values,axis=1)

            #standardization (formula p45)
            Z_row_sup = (P_row_sup.sub(Aw.values).sub(Bw.values).add(Cw.values)).div(row_sup_m,axis=0).div(col_m,axis=1)
            #fill NA, +/-inf if 1e-15
            Z_row_sup = Z_row_sup.replace([nan,inf,-inf], 1e-15)  

            #statistics for supplementary rows
            row_sup_ = func_predict(X=Z_row_sup,Y=fit_.svd.V,w=col_w,axis=0)

            #partiels coordinates for supplementary rows
            coord_partiel = OrderedDict()
            for g, c in col_group_dict.items():
                data = DataFrame(zeros(Z_row_sup.shape),index=Z_row_sup.index,columns=Z_row_sup.columns)
                data[c] = Z_row_sup[c]
                coord = data.mul(col_w,axis=1).dot(self.svd_.V[:,:ncp])
                coord.columns = self.eig_.index[:ncp]
                coord_partiel[g] = coord
            #add to dictionary
            row_sup_["coord_partiel"] = namedtuple("coord",coord_partiel.keys())(*coord_partiel.values())
            #convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",row_sup_.keys())(*row_sup_.values())

            #statistics for supplementary columns groups
            #weight of supplementary columns group
            row_sup_group_w = Series([row_sup_m.loc[x].sum() for x in list(row_group_sup_dict.values())],index=list(row_group_sup_dict.keys()),name="weight")
            #coordinates for the supplementary columns groups
            A = row_sup_["coord"].pow(2).mul(row_sup_m_n,axis=0)
            coord = concat((A.loc[r,:].sum(axis=0).to_frame(g) for r, g in zip(list(row_group_sup_dict.values()),list(row_group_sup_dict.keys()))),axis=1).T
            #convert to ordered dictionary
            row_sup_group_ = OrderedDict(coord=coord,weight=row_sup_group_w)
            #convert to namedtuple
            self.row_sup_group_ = namedtuple("row_sup_group",row_sup_group_.keys())(*row_sup_group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.num_col_group_sup is not None:
            #frequencies of supplementary columns
            P_col_sup = X_col_sup.div(total)
            #margins for supplementary columns
            col_sup_m = P_col_sup.sum(axis=0)

            #normalize col_sup_m_n by group (sum in each column group equal to 1)
            col_sup_m_n = concat((col_sup_m.loc[x]/col_sup_m.loc[x].sum() for x in list(col_group_sup_dict.values())),axis=0)
            #supplementary columns variables
            y_col_sup = list(chain(*[repeat(g,len(c)) for g, c in zip(list(col_group_sup_dict.keys()),list(col_group_sup_dict.values()))]))
            
            #colums sums of supplementary columns frequencies by supplementary columns groups
            A = concat((P_col_sup.loc[r,:].sum(axis=0).to_frame(g) for r, g in zip(list(row_group_dict.values()),list(row_group_dict.keys()))),axis=1).T
            #weighted A with row_w_n
            Aw = A.loc[y_row,:].mul(row_m_n.values,axis=0)

            #rows sums of supplementary columns frequencies by supplementary columns groups
            B = concat((P_col_sup.loc[:,c].sum(axis=1).to_frame(g) for c, g in zip(list(col_group_sup_dict.values()),list(col_group_sup_dict.keys()))),axis=1)
            #multiply B with col_m_n
            Bw = B.loc[:,y_col_sup].mul(col_sup_m_n.values,axis=1)

            #sum of supplementary columns frequencies by rows groups and supplementary columns groups
            C = concat((A.loc[:,r].sum(axis=1).to_frame(g) for r, g in zip(list(col_group_sup_dict.values()),list(col_group_sup_dict.keys()))),axis=1)
            #multiply C by row_m_n and col_sup_m_n
            Cw = C.loc[y_row,y_col_sup].mul(row_m_n.values,axis=0).mul(col_sup_m_n.values,axis=1)

            #standardized (formula p45)
            Z_col_sup = (P_col_sup.sub(Aw.values).sub(Bw.values).add(Cw.values)).div(row_m,axis=0).div(col_sup_m,axis=1)
            #fill NA, +/-inf if 1e-15
            Z_col_sup = Z_col_sup.replace([nan,inf,-inf], 1e-15)  

            #statistics for supplementary columns
            col_sup_ = func_predict(X=Z_col_sup,Y=fit_.svd.U,w=row_w,axis=1)
        
            #partiels coordinates supplementary columns
            coord_partiel = OrderedDict()
            for g, c in row_group_dict.items():
                data = DataFrame(zeros(Z_col_sup.shape),index=Z_col_sup.index,columns=Z_col_sup.columns)
                data.loc[c,:] = Z.loc[c,:]
                coord = data.mul(row_w,axis=0).T.dot(self.svd_.U[:,:ncp])
                coord.columns = self.eig_.index[:ncp]
                coord_partiel[g] = coord
            #add to dictionary
            col_sup_["coord_partiel"] = namedtuple("coord",coord_partiel.keys())(*coord_partiel.values())
            #convert to namedtuple
            self.col_sup_ = namedtuple("col_sup",col_sup_.keys())(*col_sup_.values())

            #statistics for supplementary columns groups
            #weight of supplementary columns group
            col_sup_group_w = Series([col_sup_m.loc[x].sum() for x in list(col_group_sup_dict.values())],index=list(col_group_sup_dict.keys()),name="weight")
            #coordinates for the supplementary columns groupes
            A = col_sup_["coord"].pow(2).mul(col_sup_m_n,axis=0)
            coord = concat((A.loc[c,:].sum(axis=0).to_frame(g) for c, g in zip(list(col_group_sup_dict.values()),list(col_group_sup_dict.keys()))),axis=1).T
            #convert to ordered dictionary
            col_sup_group_ = OrderedDict(coord=coord,weight=col_sup_group_w)
            #convert to namedtuple
            self.col_sup_group_ = namedtuple("col_sup_group",col_sup_group_.keys())(*col_sup_group_.values())

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.
            ``X`` is a contingency table containing absolute frequencies.

        y : None
            y is ignored
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Transformed values, where ``n_rows`` is the number of rows and ``ncp`` is the number of the components.
        """
        self.fit(X)
        return self.row_.coord