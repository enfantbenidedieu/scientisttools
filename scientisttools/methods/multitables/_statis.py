# -*- coding: utf-8 -*-
from numpy import array, where, ones, ndarray, linalg, reshape, sqrt,sum,real,tile, nan, inf, insert,diff, c_, cumsum
from collections import namedtuple, OrderedDict
from itertools import chain, repeat
from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..onetable._pca import PCA
from ..onetable._mca import MCA
from ..onetable._famd import FAMD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.utils import cols_dtypes, check_is_dataframe
from ..functions.cov2corr import cov2corr
from ..functions.concat_empty import concat_empty
from ..functions.gfa import gFA
from ..functions.func_predict import func_predict
from ..functions.func_eta2 import func_eta2
from ..functions.statistics import func_groupby, wmean, wcorr
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix

class STATIS(BaseEstimator,TransformerMixin):
    """
    STATIS, a method for analysing K-tables for k groups

    Parameters
    ----------
    excl : None, list, default = None
        The "junk" categories. It can be a list or a tuple of the names of the categories or a list or a tuple of the indexes in the active disjunctive table.
    
    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : list, tuple
        The number of variables in each group.

    name_group : list, tuple, default = None
        The name of the groups. If ``None``, the group are named Gr1, Gr2 and so on.

    type_group : list, tuple
        The type of variables in each group. Possible values are: 

        * "c" or "s" for continuous variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (continuous and categorical variables)
        * "f" for frequency (from contingency tables)
    
    option : str, default = "lambda1"
        A string for the weightings of the variables.

        * 'inertia': weighting of group :math:`k` by the inverse of the total inertia of the group :math:`k`.
        * 'lambda1': weighting of group :math:`k` by the inverse of the first eigenvalue of the :math:`k`analysis.
        * 'uniform': uniform weighting of groups.
    
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional individuals weights. The weights are given only for the active individuals.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional variables weights. The weights are given only for the active variables.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    num_group_sup : list, tuple
        The indexes of the illustrative groups (by default, None and no group are illustrative)

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger than `-tol*lambda1` where `lambda1` is the largest eigenvalue.
    
    Returns
    -------
    call_ : call
        An object containing the summary called parameters, with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        Zcod : DataFrame of shape (n_rows, n_columns) 
            Standardized data rom separate analyses. 
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data for MFA.
        tab : DataFrame of shape (n_rows, n_columns)
            Data used for GSVD.
        total : int
            The sum of all freqencies table.
        center : Series of shape (n_columns,)
            The columns weighted average.
        scale : Series of shape (n_columns)
            The columns weighted standard deviation.
        z_center : Series of shape (n_columns,)
            The weighted average.
        ind_w : Series of shape (n_rows,) 
            The individuals weights.
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        alpha : Series of shape (n_groups,)
            The weighting of variables in MFA.
        ncp : int
            The number of components kepted.
        group : list
            The number of variables in each group.
        type_group : list
            The type of variables in each group.
        name_group : list
            The name of groups.
        num_group_sup : None, list
            The indexes of the illustrative groups.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).)

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    freq_ : freq, optional
        An object containing all the results for the active frequencies, with the following attributes:

        coord : DataFrame of shape (n_freq, ncp)
            The coordinates of the frequencies.
        cos2 : DataFrame of shape (n_freq, ncp)
            The squared cosinus of the frequencies.
        contrib : DataFrame of shape (n_freq, ncp)
            The relative contributions of the frequencies.
        infos : DataFrame of shape (n_freq, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the frequencies.

    freq_sup_ : freq_sup, optional
        An object containing all the results for the supplementary frequencies, with the following attributes:
        
        coord : DataFrame of shape (n_freq_sup, ncp)
            The coordinates of the supplementary frequencies.
        cos2 : DataFrame of shape (n_freq_sup, ncp)
            The squared cosinus of the supplementary frequencies.
        dist2 : Series of shape (n_freq_sup,)
            The squared distance to origin of the supplementary frequencies.

    group_ : group
        An object containing all the results for the groups, with the following attributes:

        Lg : DataFrame of shape (n_groups, n_groups)
            The trace \emph{Lg} coefficients.
        RV : DataFrame of shape (n_groups,n_groups)
            The \emph{RV} coefficients.
        coord : DataFrame of shape (n_groups, n_groups)
            The coordinates of the groups.
        contrib : DataFrame of shape (n_groups, n_groups)
            The relative contributions of the groups.
        cos2 : DataFrame of shape (n_groups, n_groups)
            The square cosinus of the groups.
        dist2 : Series of shape (n_groups,)
            The square distance to origin of the groups.
        correlation : DataFrame of shape (n_groups, n_groups)
            The correlations between each group and each factor.
        eig : DataFrame of shape (n_groups, 4)
            The eigen values of the RV matrix.
        evd : evdResult
            The eigen values decomposition of \emph{RV}, with the following attributes:

            V : 2D numpy array of shape (n_groups, n_groups)
                The eigenvectors of the \emph{RV} matrix.
            d : 1d numpy array of shape (n_groups,)
                The eigenvalues of the \emph{RV} matrix.
    
    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows,ncp)
            The coordinates of the individuals.
        cos2 : DataFrame of shape (n_rows, ncp)
            Thesquared cosinus of the individuals.
        contrib : DataFrame of shape (n_rows, ncp) 
            The relative contributions of the individuals.
        infos : DataFrame of shape (n_rows, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates of the individuals.
        within_inertia : DataFrame of shape (n_rows, ncp)
            The within inertia for the individuals.
        within_partial_inertia : inertia
            An object containing the within partial inertia for the individuals.

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.
        cos2 : DataFrame of shape (n_rows_plus, ncp)
            The squared cosinus of the supplementary individuals.
        dist2 : Series of shape (n_rows_plus,)
            The squared distance to origin of the supplementary individuals.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates for the supplementary individuals.

    levels_ : levels_sup, optional
        An object containing all the results for the active levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the levels.
        contrib : DataFrame of shape (n_levels, ncp)
            The contributions of the levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the levels.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates of the levels.

    levels_sup_ : levels_sup, optional
        An object containing all the results for the supplementary levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the supplementary levels.
        dist2 : Series of shape (n_levels,)
            The squared distance to origin of the supplementary levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the supplementary levels.
        coord_partiel: coord_partiel
            An object containing all the partiel coordinates for the supplementary levels.

    quali_var_ : quali_var, optional
        An object containing all the results for the active qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var, ncp)
            The coordinates of the qualitative variables, which is eta2, the square correlation corefficient between a qualitative variable and a dimension.
        contrib : DataFrame of shape (n_quali_var, ncp)
            The contributions of the qualitative variables.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates for the qualitatve variables.

    quali_var_sup_ : quali_var_sup, optional
        An object containing all the results for the supplementary qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var_sup, ncp)
            The coordinates of the qualitative variables, which is eta2, the square correlation corefficient between a qualitative variable and a dimension.
        coord_partiel : coord_partiel
            An object containing all the partiel coordinates for the supplementary qualitative variables.

    quanti_var_ : quanti_var, optional
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_columns, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_var_sup, ncp)
            The coordinates of the supplementary quantitative variables.
        cos2 : DataFrame of shape (n_quanti_var_sup, ncp)
            The squared cosinus of the supplementary quantitative variables.
        dist2 : Series of shape (n_quanti_var_sup,)
            The squared distance to origin of the supplementary quantitative variables.

    separate_analyses_ : OrderedDict
        The results for the separate analyses.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (maxcp,)
            The singular values.
        U : 2d numpy array of shape (n_rows, maxcp)
            The left singular vectors.
        V : 2d numpy array of shape (n_columns, maxcp)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

    References
    ----------
    Lavit, C. (1988), Analyse conjointe de tableaux quantitatifs, Masson, Paris.

    Lavit, C., Escoufier, Y., Sabatier, R. and Traissac, P. (1994) The ACT (Statis method). Computational Statistics and Data Analysis, 18, 97--119.

    See also
    --------
    :class:`scientisttools.save`
        Print results for general factor analysis model in an Excel sheet.
    :class:`scientisttools.sprintf`
        Print the analysis results.
    :class:`scientisttools.summary`
        Printing summaries of general factor analysis model.

    Examples
    --------
    >>> from scientisttools.datasets import wine, poison, gironde, mortality
    >>> from scientisttools import STATIS
    >>> # STATIS
    >>> clf = STATIS(group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    STATIS(group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name,num_group_sup=(0,5))
    >>> # Example with groups of categrical variables
    >>> clf = STATIS(group=poison.group,type_group=("s","n","n","n"),name_group=poison.name,num_group_sup=(0,1))
    >>> clf.fit(poison.data)
    STATIS(group=(2,2,5,6),name_group=("desc","desc2","symptom","eat"),num_group_sup=(0,1),type_group=("s","n","n","n"))
    >>> # Example with groups of mixed variables
    >>> clf = STATIS(group=gironde.group,type_group=("s","m","n","s"),name_group=gironde.name)
    >>> clf.fit(gironde.data.iloc[:20,:])
    STATIS(group=(9,5,9,4),type_group=("s","m","n","s"),name_group=("employment","housing","services","environment"))
    >>> # Example with groups of frequency tables
    >>> clf = STATIS(group=mortality.group,type_group=("f","f"),name_group=mortality.name)
    >>> clf.fit(mortality.data)
    STATIS(group=(9,9),name_group=("y1958","y2006"),type_group=("f","f"))
    """
    def __init__(
            self, excl=None, ncp = 5, type_group = None, group = None, name_group = None, row_w = None, col_w = None, ind_sup = None, num_group_sup = None, tol = 1e-7
    ):
        self.excl = excl
        self.ncp = ncp
        self.group = group
        self.type_group = type_group
        self.name_group = name_group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.num_group_sup = num_group_sup
        self.tol = tol

    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif not isinstance(self.group, (list,tuple,ndarray,Series)):
            raise ValueError("'group' must be a 1d array-like with the number of variables in each group")
        else:
            group = [int(x) for x in self.group]

        #check if group definition
        if sum(group) != X.shape[1]:
            raise TypeError("Not convenient group definition")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if any group has only one /columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if any(x == 1 for x in group):
            raise ValueError("groups should have at least two columns")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if type_group in not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.type_group is None:
            raise ValueError("'type_group' must be assigned")
        else:
            type_group = [str(x) for x in self.type_group]

        if any(x not in ("c","f","m","n","s") for x in type_group):
            raise ValueError("Not convenient type_group definition")
        
        if len(self.group) != len(self.type_group):
            raise TypeError("Not convenient group definition")
        
        #which type_group is f
        num_group_freq = list(where(array(type_group) == "f")[0]) if any(x == "f" for x in type_group) else None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.name_group is None:
            name_group = [f"Gr{x+1}" for x in range(len(group))]
        elif not isinstance(self.name_group,(list,tuple)):
            raise TypeError("'name_group' must be a list or a tuple with name of group")
        else:
            name_group = [x for x in self.name_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.num_group_sup is not None:
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif isinstance(self.num_group_sup,(list,tuple)) and len(self.num_group_sup)>=1:
                num_group_sup = [int(x) for x in self.num_group_sup]
        else:
            num_group_sup = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group name to label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_dict, k = OrderedDict(), 0
        for i, g in zip(range(len(group)),name_group):
            group_dict[g] = list(X.columns[k:(k+group[i])])
            k += group[i]

        if self.num_group_sup is not None:
            group_sup_dict = OrderedDict({g : group_dict[g] for i, g in enumerate(name_group) if i in num_group_sup})
            group_dict = OrderedDict({g : group_dict[g] for i, g in enumerate(name_group) if not i in num_group_sup})

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label, sup_var_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0), None if self.num_group_sup is None else list(chain.from_iterable(group_sup_dict.values()))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #type_group_var
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        type_group_var, k = None, 0
        for i in range(len(group)):
            colnames = list(X.columns[k:(k+group[i])])
            if self.num_group_sup is None:
                if type_group[i] in ("c","s"):
                    type_var = Series(repeat("quanti",group[i]),index=colnames)
                if type_group[i] == "f":
                    type_var = Series(repeat("freq",group[i]),index=colnames)
                if type_group[i] == "n":
                    type_var = Series(repeat("quali",group[i]),index=colnames)
                if type_group[i] == "m":
                    type_var = Series(cols_dtypes(X.loc[:,colnames]),index=colnames)
            else:
                if type_group[i] in ("c","s") and not i in num_group_sup:
                    type_var = Series(repeat("quanti",group[i]),index=colnames)
                if type_group[i] in ("c","s") and i in num_group_sup:
                    type_var = Series(repeat("quanti_sup",group[i]),index=colnames)
                if type_group[i] == "f" and not i in num_group_sup :
                    type_var = Series(repeat("freq",group[i]),index=colnames)
                if type_group[i] == "f" and i in num_group_sup:
                    type_var = Series(repeat("freq_sup",group[i]),index=colnames)
                if type_group[i] == "n" and not i in num_group_sup:
                    type_var = Series(repeat("quali",group[i]),index=colnames)
                if type_group[i] == "n" and i in num_group_sup :
                    type_var = Series(repeat("quali_sup",group[i]),index=colnames)
                if type_group[i] == "m" and not i in num_group_sup:
                    type_var = Series(cols_dtypes(X.loc[:,colnames]),index=colnames)
                if type_group[i] == "m" and i in num_group_sup :
                    type_var = Series([f"{x}_sup" for x in cols_dtypes(X.loc[:,colnames])],index=colnames)
            type_group_var = concat_empty(type_group_var,type_var,axis=0)
            k += group[i]

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary groups columns
        if self.num_group_sup is not None:
            X_group_sup, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
            if self.ind_sup is not None:
                X_group_sup = X_group_sup.drop(index=ind_sup_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
       
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #STATIS
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #number of rows/columns
        n_rows, n_cols = X.shape
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None:
            ind_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)):
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows:
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else:
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")

        #set variables weights
        if self.col_w is None:
            var_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)):
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_cols:
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
        else:
            var_w = Series(array(self.col_w),index=X.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Data Preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Xcod, total = X.copy(), 0
        if num_group_freq is not None:
            name_group_freq = [g for i, g in enumerate(name_group) if i in num_group_freq] if self.num_group_sup is None else [g for i, g in enumerate(name_group) if i in num_group_freq and not i in num_group_sup]
            
            if len(name_group_freq) > 0:
                group_freq_dict = OrderedDict({k : group_dict[k] for k in name_group_freq})
                freq_cols = list(chain.from_iterable(group_freq_dict.values()))
                #select frequencies data
                N = X.loc[:,freq_cols]
                #sum of all elements 
                total = N.sum().sum()
                #proportional table
                P = N.div(total)
                #set global row margin and columns margin
                row_m, col_m = P.sum(axis=1), P.sum(axis=0)
                #construction of recoded table
                for g, cols in group_freq_dict.items():
                    #row margin for group
                    row_m_g = P[cols].sum(axis=1)
                    #normalize such as sum equal to 1
                    row_w_g = row_m_g/sum(row_m_g)
                    #recoded columns and fill NA, +/-inf if 1e-15
                    Xcod[cols] = P[cols].div(col_m[cols],axis=1).sub(row_w_g,axis=0).div(row_m,axis=0).replace([nan,inf,-inf], 1e-15)
                #update weights for rows and columns
                ind_w, var_w[freq_cols] = row_m, col_m

        #rows weights
        row_w = ind_w.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis model for active group
        model = OrderedDict()
        for g, cols in group_dict.items():
            if self.type_group[name_group.index(g)] in ("c","f","s"):
                scale_unit = False if self.type_group[name_group.index(g)] in ("c","f") else True
                fa = PCA(scale_unit=scale_unit,ncp=self.ncp,row_w=row_w,col_w=var_w[cols],tol=self.tol)
            if self.type_group[name_group.index(g)] == "m":
                fa = FAMD(ncp=self.ncp,row_w=row_w,col_w=var_w[cols],tol=self.tol)
            if self.type_group[name_group.index(g)] == "n":
                fa = MCA(excl=self.excl,ncp=self.ncp,row_w=ind_w,col_w=var_w[cols],tol=self.tol)
            model[g]= fa.fit(Xcod[cols])

        #separate general factor analysis for supplementary groups
        if self.num_group_sup is not None:
            Xcod_group_sup, var_sup_w = X_group_sup.copy(), Series(ones(X_group_sup.shape[1]),index=X_group_sup.columns,name="weight")

            if num_group_freq is not None:
                name_group_sup_freq = [g for i, g in enumerate(name_group) if i in num_group_freq and i in num_group_sup]

                if len(name_group_sup_freq) > 0:
                    group_sup_freq_dict = OrderedDict({k : group_sup_dict[k] for k in name_group_sup_freq})
                    freq_sup_cols = list(chain.from_iterable(group_sup_freq_dict.values()))
                    N_col_sup = X_group_sup.loc[:,freq_sup_cols]

                if len(name_group_freq) == 0 and len(name_group_sup_freq) > 0 :
                    #sum of all elements 
                    total_col_sup = N_col_sup.sum().sum()
                    #frequencies table
                    P_col_sup = N_col_sup.div(total_col_sup)
                    #set global row margin and columns margin
                    row_m, col_m = P_col_sup.sum(axis=1), P_col_sup.sum(axis=0)
                    #construction of recoded table
                    for g, cols in group_sup_freq_dict.items():
                        #rows margins for groups
                        row_m_g = P_col_sup[cols].sum(axis=1)
                        #normalize such sum equal to 1
                        row_w_g = row_m_g/sum(row_m_g)
                        #update weights
                        Xcod_group_sup[cols], var_sup_w[cols] = P_col_sup[cols].div(col_m[cols],axis=1).sub(row_w_g,axis=0).div(row_m,axis=0).replace([nan,inf,-inf], 1e-15), col_m[cols]
                elif all(x > 0 for x in (len(name_group_freq),len(name_group_sup_freq))):
                    for g, cols in group_sup_freq_dict.items():
                        #sum of all elements in group g
                        total_col_sup_g = N_col_sup[cols].sum().sum()
                        #compute frequencies for supplementary groups
                        P_col_sup_g = N_col_sup[cols]/(total_col_sup_g*total)
                        #compute margins
                        row_m_g, col_m_g = P_col_sup_g.sum(axis=1), P_col_sup_g.sum(axis=0)
                        #normalize such sum equal to 1
                        row_w_g = row_m_g/sum(row_m_g)
                        #update
                        Xcod_group_sup[cols], var_sup_w[cols] = P_col_sup_g.div(col_m_g,axis=1).sub(row_w_g,axis=0).div(row_m,axis=0).replace([nan,inf,-inf], 1e-15), col_m_g
            
            for g, cols in group_sup_dict.items():
                if self.type_group[name_group.index(g)] in ("c","f","s"):
                    scale_unit = False if self.type_group[name_group.index(g)] in ("c","f") else True
                    fa = PCA(scale_unit=scale_unit,ncp=self.ncp,row_w=row_w,col_w=var_sup_w[cols],tol=self.tol)  
                if self.type_group[name_group.index(g)] == "m":
                    fa = FAMD(ncp=self.ncp,row_w=row_w,col_w=var_sup_w[cols],tol=self.tol)
                if self.type_group[name_group.index(g)] == "n":
                    fa = MCA(ncp=self.ncp,row_w=row_w,col_w=var_sup_w[cols],tol=self.tol) 
                model[g] = fa.fit(Xcod_group_sup[cols])

        #store separate analysis
        self.separate_analyses_ = model

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #cross-product operation: A = XMX'D
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #cross-product matrix for each table in active group
        A = OrderedDict({g : ((model[g].call_.Z * model[g].call_.col_w).dot(model[g].call_.Z.T)) * model[g].call_.row_w for g in list(group_dict.keys())})
        #cross-product matrix for supplementary group
        if self.num_group_sup is not None:
            for g in list(group_sup_dict.keys()):
                A[g] = ((model[g].call_.Z * model[g].call_.col_w).dot(model[g].call_.Z.T)) * model[g].call_.row_w
        
        #vec of each normalized cross-matrix
        Y = concat((DataFrame(reshape(A[g],shape=(-1,1),order="F"),columns=[g]) for g in name_group),axis=1)
        
        #non-normalized RV
        traceRV = Y.T.dot(Y)
        #RV coefficients
        RV = cov2corr(X=traceRV)

        #eigen decomposition of RV (singular value decomposition of hermittian)
        rv_evd = linalg.svd(RV.loc[list(group_dict.keys()),list(group_dict.keys())],hermitian=True)
        #convert to real if any complex
        rv_evdvals, rv_evdvects = real(rv_evd[1]), real(rv_evd[0])
        rv_evdvects[:,0] = abs(rv_evdvects[:,0])
        #maximum number of components
        rv_rank = sum(rv_evdvals/rv_evdvals[0] > self.tol)
        #update with rank
        rv_eigvals, rv_eigvects = rv_evdvals[:rv_rank], rv_evdvects[:,:rv_rank]

        #RV eigen values informations
        rv_eigdiff, rv_eigprop = insert(-diff(rv_eigvals),len(rv_eigvals)-1,nan), 100*rv_eigvals/sum(rv_eigvals)
        #convert to DataFrame
        rv_eig = DataFrame(c_[rv_eigvals,rv_eigdiff,rv_eigprop,cumsum(rv_eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(rv_rank)])
        
        #coordinates for the groups
        group_coord = DataFrame(rv_eigvects*sqrt(rv_eigvals),index=list(group_dict.keys()),columns=rv_eig.index)
        #squared euclidean distance
        group_sqdist = (group_coord**2).sum(axis=1)
        #cos2 of the group
        group_cos2 = ((group_coord**2).T/group_sqdist).T
        #contributions of the group
        group_ctr = (group_coord**2)/rv_eigvals
        #convert to ordered dictionary
        group_ = OrderedDict(traceRV=traceRV,RV=RV,eig=rv_eig,evd=namedtuple("evdResult",["V","d"])(rv_eigvects,rv_eigvals),coord=group_coord,contrib=group_ctr,cos2=group_cos2,dist2=group_sqdist)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Data prepartion for active groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardized data
        Zcod = concat((model[g].call_.Z for g in list(group_dict.keys())),axis=1)
        #weighted average and standard deviation
        center, scale = concat((model[g].call_.center for g in list(group_dict.keys())),axis=0), concat((model[g].call_.scale for g in list(group_dict.keys())),axis=0)
        #number of components in all models
        mncp = Series([model[g].call_.ncp for g in list(model.keys())],index=list(model.keys()))
        #active columns dictionary
        columns_dict = {g : list(model[g].call_.Z.columns) for g in list(group_dict.keys())}
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set columns weights for STATIS
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #columns weights in each groups 
        mcol_w = concat((model[g].call_.col_w for g in list(group_dict.keys())),axis=0)
        #beta
        beta = Series([1/sqrt(sum(model[g].eig_.iloc[:,0]**2)) for g in list(group_dict.keys())],index=list(group_dict.keys()),name="beta")
        #alpha
        alpha = Series(rv_eigvects[:,0],index=list(group_dict.keys()),name="alpha")
        #columns weights
        col_w = concat((mcol_w[cols]*beta[g]*alpha[g] for g, cols in columns_dict.items()),axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Data preparation for supplementary group
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.num_group_sup is not None:
            #standardized data
            Zcod_sup = concat((model[g].call_.Z for g in list(group_sup_dict.keys())),axis=1)
            #centering according to non-normed PCA
            Z_sup = Zcod_sup - wmean(X=Zcod_sup,w=row_w)

            #group coordinates
            group_sup_coord = RV.loc[list(group_sup_dict.keys()),list(group_dict.keys())].dot(rv_eigvects/sqrt(rv_eigvals))
            group_sup_coord.columns = rv_eig.index
        
            #add to OrderedDict
            group_["coord_sup"] = group_sup_coord

        #convert to namedtuple
        self.group_ = namedtuple("group",group_.keys())(*group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center according to non-normed PCA
        z_center = wmean(X=Zcod,w=row_w)
        #center
        Z = Zcod - z_center
        #fit factor analysis model
        fit_ = gFA(X=Z,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
        
        #extract elements
        self.svd_, self.eig_, ind_, ncp = fit_.svd, fit_.eig, fit_.row, fit_.ncp

        #set call_ informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Zcod=Zcod,Z=Z,total=total,row_w=row_w,var_w=var_w,col_w=col_w,alpha=alpha,beta=beta,center=center,scale=scale,z_center=z_center,ncp=ncp,
                            group=group,type_group=type_group,name_group=name_group,ind_sup=ind_sup_label,group_sup=num_group_sup)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #others individuals informations : partiels coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partiels coordinates for individuals F_j = Z_jV/alpha_j
        ind_coord_partiel = OrderedDict()
        for g, cols in columns_dict.items():
            data = DataFrame(tile(z_center.values,(n_rows,1)),index=Z.index,columns=Z.columns)
            data[cols] = Z[cols]
            coord = ((data - z_center)*col_w).dot(self.svd_.V[:,:ncp])/alpha[g]
            coord.columns = self.eig_.index[:ncp]
            ind_coord_partiel[g] = coord

        #individuals Within inertia
        ind_within_inertia = None
        for d in self.eig_.index[:ncp]:
            data = concat((ind_coord_partiel[g].loc[:,d] for g in list(group_dict.keys())),axis=1)
            within_inertia = (((data.T  - ind_["coord"][d].values)**2)*row_w).sum(axis=0).to_frame(d)
            ind_within_inertia = concat_empty(ind_within_inertia,within_inertia,axis=1)
        #normalization
        ind_within_inertia = 100*ind_within_inertia/ind_within_inertia.sum(axis=0)

        #denominator for within partial inertia
        den = Series([concat((ind_coord_partiel[g].loc[:,d] for g in list(group_dict.keys())),axis=1).sub(ind_["coord"][d].values,axis=0)
                      .pow(2).mul(row_w,axis=0).sum().sum() for d in self.eig_.index[:ncp]],index=self.eig_.index[:ncp])
        #individuals Within partial inertia
        ind_within_partial_inertia = OrderedDict({g : ind_coord_partiel[g].sub(ind_["coord"].values).pow(2).mul(row_w,axis=0).div(den,axis=1).mul(100) for g, cols in columns_dict.items()})
        
        #add elements to ind_
        ind_ = {**ind_, **OrderedDict(coord_partiel=namedtuple("coord_partiel",ind_coord_partiel.keys())(*ind_coord_partiel.values()),within_inertia=ind_within_inertia,
                                      within_partial_inertia=namedtuple("inertia",ind_within_partial_inertia.keys())(*ind_within_partial_inertia.values()))}
        #store all individuals informations
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for partial axes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the partial axes
        partial_axes_coord, partial_axes_sqcos = OrderedDict(), OrderedDict()
        for g in name_group:
            coord = ((model[g].ind_.coord/model[g].svd_.vs[:model[g].call_.ncp]).T * row_w).dot(self.svd_.U[:,:ncp])
            coord.columns = self.eig_.index[:ncp]
            for i in range(model[g].call_.ncp):
                if coord.iloc[i,i] < 0:
                    coord.iloc[i,:] = -1*coord.iloc[i,:]
            partial_axes_coord[g], partial_axes_sqcos[g] = coord, coord**2
               
        #contributions for the partial axes
        partial_axes_ctr = OrderedDict()
        for g in list(group_dict.keys()):
            nbcol = min(ncp,mncp[g])
            eig = model[g].eig_.iloc[:nbcol,0]
            contrib = beta[g]*alpha[g]*((partial_axes_coord[g].iloc[:,:nbcol]**2).T * eig).T
            partial_axes_ctr[g] = 100*contrib/contrib.sum(axis=0)

        #correlation between the partial axes
        all_coord = None
        for g in name_group:
            data = model[g].ind_.coord
            data.columns = [f"{x}{g}" for x in data.columns]
            all_coord = concat_empty(all_coord,data,axis=1)
        #correlation between
        cor_between = wcorr(all_coord,w=row_w)

        #convert to namedtuple
        partial_axes_coord, partial_axes_sqcos = namedtuple("coord",partial_axes_coord.keys())(*partial_axes_coord.values()), namedtuple("cos2",partial_axes_sqcos.keys())(*partial_axes_sqcos.values())
        partial_axes_ctr = namedtuple("contrib",partial_axes_ctr.keys())(*partial_axes_ctr.values())

        #convert to ordered dictionary 
        partial_axes_ = OrderedDict(coord=partial_axes_coord,contrib=partial_axes_ctr,cos2=partial_axes_sqcos,cor_between=cor_between)
        #convert to namedtuple
        self.partial_axes_ = namedtuple("partial_axes",partial_axes_.keys())(*partial_axes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for quantitative variables (coordinates, cos2 and contrib, infos)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quanti" in type_group_var.values:
            #quantitativate variables columns
            quanti_var_cols = list(type_group_var[type_group_var == "quanti"].index)
            #statistics for quantitative variables
            quanti_var_ = OrderedDict({k : fit_.col[k].loc[quanti_var_cols,:] for k in list(fit_.col.keys())})
            #convert to namedtuple
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitative variables (coordinates, cos2 & dist2)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quanti_sup" in type_group_var.values:
            #supplementary quantitativate variables columns
            quanti_var_sup_cols = list(type_group_var[type_group_var == "quanti_sup"].index)
            #statistics for supplementary quantitative variables
            quanti_var_sup_ = func_predict(X=Z_sup[quanti_var_sup_cols],Y=self.svd_.U[:,:ncp],w=row_w,axis=1)
            #convert to namedtuple
            self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for levels (coordinates, cos2, contributions, value-test) and qualitative variables (coordinates, contributions)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quali" in type_group_var.values:
            #qualitativate variables columns
            quali_var_cols = list(type_group_var[type_group_var == "quali"].index)
            #select all qualitative variables
            X_quali_var = X[quali_var_cols]
            #conditional average of Zcod and standardization: z_ik = x_ik - m_zk
            Z_levels = func_groupby(X=Zcod,by=X_quali_var,func="mean",w=row_w) - z_center
            #statistics for levels
            levels_ = func_predict(X=Z_levels,Y=self.svd_.V[:,:ncp],w=col_w,axis=0)
            #contributions of levels
            levels_["contrib"] = fit_.col["contrib"].loc[Z_levels.index,:]
            #proportion for the slevels
            p_k = (disjunctive(X_quali_var).T * row_w).sum(axis=1)
            #vtest for the levels
            levels_["vtest"] = levels_["coord"].mul(sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0).div(self.svd_.vs[:ncp],axis=1)
            #partiel coordinates for levels
            levels_coord_partiel = OrderedDict({g : func_groupby(X=ind_coord_partiel[g],by=X_quali_var,func="mean",w=row_w) for g in list(group_dict.keys())})
            #add to dictionary
            levels_["coord_partiel"] = namedtuple("coord_partiel",levels_coord_partiel.keys())(*levels_coord_partiel.values())
            #convert to namedtuple
            self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

            #statistics for qualitative variables
            #coordinates for the qualitative variables - Eta-squared
            quali_var_coord = func_eta2(X=ind_["coord"],by=X_quali_var,w=row_w,excl=None)
            #partiel coordinates for the qualitative variables
            quali_var_coord_partiel = OrderedDict({g : func_eta2(X=ind_coord_partiel[g],by=X_quali_var,w=row_w,excl=None) for g in list(group_dict.keys())})
            #convert to namedtuple
            quali_var_coord_partiel = namedtuple("coord_partiel",quali_var_coord_partiel.keys())(*quali_var_coord_partiel.values())
            #contributions for the qualitative variables
            quali_var_ctr = concat((levels_["contrib"].loc[levels_["contrib"].index.isin(list(X_quali_var[j].unique())),:].sum(axis=0).to_frame(j) for j in X_quali_var.columns),axis=1).T
            #convert to ordered dictionary
            quali_var_ = OrderedDict(coord=quali_var_coord,coord_partiel=quali_var_coord_partiel,contrib=quali_var_ctr)
            #convert to namedtuple
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary levels/qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quali_sup" in type_group_var.values:
            #supplementary qualitativate variables columns
            quali_var_sup_cols = list(type_group_var[type_group_var == "quali_sup"].index)
            #select all supplementary qualitative variables
            X_quali_var_sup = X_group_sup[quali_var_sup_cols]
            #conditional mean - Barycenter of original data #standardization: z_ik = (x_ik - m_k)
            Z_levels_sup = func_groupby(X=Zcod,by=X_quali_var_sup,func="mean",w=row_w) - z_center
            #statistics for supplementary levels
            levels_sup_ = func_predict(X=Z_levels_sup,Y=self.svd_.V[:,:ncp],w=col_w,axis=0)
            #proportion for the supplementary levels
            p_k_sup = (disjunctive(X_quali_var_sup).T * row_w).sum(axis=1)
            #vtest for the supplementary levels
            levels_sup_["vtest"] = levels_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(self.svd_.vs[:ncp],axis=1)
            #partiel coordinates for supplementary levels
            levels_sup_coord_partiel = OrderedDict({g : func_groupby(X=ind_coord_partiel[g],by=X_quali_var_sup,func="mean",w=row_w) for g in list(group_dict.keys())})
            #add to dictionary
            levels_sup_["coord_partiel"] = namedtuple("coord_partiel",levels_sup_coord_partiel.keys())(*levels_sup_coord_partiel.values())

            ##statistics for supplementary qualitative variables
            #coordinates for the supplementary qualitative variables
            quali_var_sup_coord = func_eta2(X=ind_["coord"],by=X_quali_var_sup,w=row_w,excl=None)
            #partiel coordinatesfor the supplementary qualitative variables
            quali_var_sup_coord_partiel = OrderedDict({g : func_eta2(X=ind_coord_partiel[g],by=X_quali_var_sup,w=row_w,excl=None) for g in list(group_dict.keys())})
            #convert to namedtuple
            quali_var_sup_coord_partiel = namedtuple("coord_partiel",quali_var_sup_coord_partiel.keys())(*quali_var_sup_coord_partiel.values())
            #convert to ordered dictionary
            quali_var_sup_ = OrderedDict(coord=quali_var_sup_coord,coord_partiel=quali_var_sup_coord_partiel)
            #convert to namedtuple
            self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for the frequencies : coordinates, cos2, contributions & infos
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "freq" in type_group_var.values:
            #frequencies columns
            freq_cols = list(type_group_var[type_group_var == "freq"].index)
            #statistics for frequencies
            freq_ = OrderedDict({k : fit_.col[k].loc[freq_cols,:] for k in list(fit_.col.keys())})
            #convert to namedtuple
            self.freq_ = namedtuple("freq",freq_.keys())(*freq_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for the supplementary frequencies : coordinates, cos2, & dist2
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "freq_sup" in type_group_var.values:
            #supplementary frequencies columns
            freq_sup_cols = list(type_group_var[type_group_var == "freq_sup"].index)
            #statistics for supplementary quantitative variables
            freq_sup_ = func_predict(X=Z_sup[freq_sup_cols],Y=self.svd_.U[:,:ncp],w=row_w,axis=1)
            #convert to namedtuple
            self.freq_sup_ = namedtuple("freq_sup",freq_sup_.keys())(*freq_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #Data preparation
            Zcod_ind_sup = DataFrame(index=X_ind_sup.index,columns=Z.columns).astype(float)
            for g, cols in group_dict.items():
                if self.type_group[name_group.index(g)] in ("c","s"):
                    Zcod_ind_sup[cols] = X_ind_sup[cols].sub(center[cols],axis=1).div(scale[cols],axis=1)
                elif self.type_group[name_group.index(g)] == "n":
                    dummies_cols = model[g].call_.dummies.columns
                    dummies_ind_sup = disjunctive(X_ind_sup[cols],cols=dummies_cols)
                    Zcod_ind_sup[dummies_cols] = dummies_ind_sup.sub(center[dummies_cols],axis=1).div(scale[dummies_cols],axis=1)
                elif self.type_group[name_group.index(g)] == "m":
                    #split X
                    split_Xcols_ind_sup = splitmix(X_ind_sup[cols])
                    Xcols_ind_sup_quanti, Xcols_ind_sup_quali, n_ind_sup_quanti, n_ind_sup_quali = split_Xcols_ind_sup.quanti, split_Xcols_ind_sup.quali, split_Xcols_ind_sup.k1, split_Xcols_ind_sup.k2
                    #initialization
                    Xcols_ind_sup = None
                    if n_ind_sup_quanti > 0:
                        if model[g].call_.k1 != n_ind_sup_quanti:
                            raise TypeError("The number of quantitative variables must be the same")
                        Xcols_ind_sup = concat_empty(Xcols_ind_sup,Xcols_ind_sup_quanti,axis=1)
                    if n_ind_sup_quali > 0:
                        if model[g].call_.k2 != n_ind_sup_quali:
                            raise TypeError("The number of qualitative variables must be the same")
                        Xcols_ind_sup = concat_empty(Xcols_ind_sup,disjunctive(X=Xcols_ind_sup_quali,cols=model[g].call_.dummies.columns),axis=1)
                    Zcod_ind_sup[Xcols_ind_sup.columns] = Xcols_ind_sup.sub(model[g].call_.center,axis=1).sub(model[g].call_.scale,axis=1)

            if num_group_freq is not None:
                if len(name_group_freq) > 0:
                    #frequencies in supplementary individuals
                    P_row_sup = X_ind_sup[freq_cols].div(total)
                    #supplementary rows margin
                    row_sup_m = P_row_sup.sum(axis=1)
                    #construction of recoded table
                    for g, cols in group_freq_dict.items():
                        #group rows margins
                        row_rowsup_m_g = P_row_sup[cols].sum(axis=1)
                        #normalize such sum is equal to 1
                        B_row_sup = row_rowsup_m_g/sum(row_rowsup_m_g)
                        #recoded columns
                        Zcod_ind_sup[cols] = P_row_sup[cols].div(col_m[cols],axis=1).sub(B_row_sup,axis=0).div(row_sup_m,axis=0).replace([nan,inf,-inf], 1e-15)

            #standardization : z_ik = (x_ik - m_k)/s_k - m_zk
            Z_ind_sup = Zcod_ind_sup.sub(z_center,axis=1)
            #statistics for supplementary individuals
            ind_sup_ = func_predict(X=Z_ind_sup,Y=self.svd_.V[:,:ncp],w=col_w,axis=0)
            #partiels coordinates for supplementary individuals
            ind_sup_coord_partiel = OrderedDict()
            for g, cols in columns_dict.items():
                data = DataFrame(tile(z_center,(len(ind_sup_label),1)),index=Z_ind_sup.index,columns=Z_ind_sup.columns)
                data[cols] = Z_ind_sup[cols]
                coord = ((data - z_center)*col_w).dot(self.svd_.V[:,:ncp])/alpha[g]
                coord.columns = self.eig_.index[:ncp]
                ind_sup_coord_partiel[g] = coord
            #update dictionary
            ind_sup_["coord_partiel"] = namedtuple("coord_partiel",ind_sup_coord_partiel.keys())(*ind_sup_coord_partiel.values())
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` is the number of rows and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def transform(self,X):
        """
        Apply dimensionality reduction to ``X``.

        ``X`` is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            New data, where ``n_rows`` is the number of rows and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Projection of ``X`` in the first principal components, where ``n_rows`` is the number of rows and ``ncp`` is the number of the components.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an object of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set index name as None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X contains original columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not set(self.call_.X.columns).issubset(X.columns): 
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the {} result".format(self.__class__.__name__))
        X = X[self.call_.X.columns]

        model = self.separate_analyses_
        # active group
        name_group = list(self.group_.coord.index)
        group_dict = OrderedDict({g : list(model[g].call_.X.columns) for g in name_group})
        #Data preparation
        Zcod = DataFrame(index=X.index,columns=self.call_.Z.columns).astype(float)
        for g, cols in group_dict.items():
            if self.type_group[self.call_.name_group.index(g)] in ("c","s"):
                Zcod[cols] = (X[cols] - model[g].call_.center)/model[g].call_.scale
            elif self.type_group[self.call_.name_group.index(g)] == "n":
                dummies_cols = model[g].call_.dummies.columns
                dummies = disjunctive(X[cols],cols=dummies_cols)
                Zcod[dummies_cols] = (dummies - model[g].call_.center)/model[g].call_.scale
            elif self.type_group[self.call_.name_group.index(g)] == "m":
                #split X
                split_Xcols = splitmix(X[cols])
                Xcols_quanti, Xcols_quali, n_quanti, n_quali = split_Xcols.quanti, split_Xcols.quali, split_Xcols.k1, split_Xcols.k2
                #initialization
                Xcols = None
                if n_quanti > 0:
                    if model[g].call_.k1 != n_quanti:
                        raise TypeError("The number of continuous variables must be the same")
                    Xcols = concat_empty(Xcols,Xcols_quanti,axis=1)
                if n_quali > 0:
                    if model[g].call_.k2 != n_quali:
                        raise TypeError("The number of categorical variables must be the same")
                    Xcols = concat_empty(Xcols,disjunctive(X=Xcols_quali,cols=model[g].call_.dummies.columns),axis=1)
                Zcod[Xcols.columns] = (Xcols - model[g].call_.center)/model[g].call_.scale 

        #which type_group is f
        num_group_freq = list(where(array(self.type_group) == "f")[0]) if any(x == "f" for x in self.type_group) else None
        if num_group_freq is not None:
            name_group_freq = [g for i, g in enumerate(self.call_.name_group) if i in num_group_freq] if self.call_.num_group_sup is None else [g for i, g in enumerate(self.call_.name_group) if i in num_group_freq and not i in self.call_.num_group_sup]
            if len(name_group_freq) > 0:
                group_freq_dict = OrderedDict({k : list(model[k].call_.X.columns) for k in name_group_freq})
                freq_cols = list(chain.from_iterable(group_freq_dict.values()))
                #frequencies
                P = X[freq_cols]/self.call_.total
                #supplementary rows margin
                row_m = P.sum(axis=1)
                #construction of recoded table
                for g, cols in group_freq_dict.items():
                    #group rows margins
                    row_m_g = P[cols].sum(axis=1)
                    #normalize such sum is equal to 1
                    B = row_m_g/sum(row_m_g)
                    #recoded columns
                    Zcod[cols] = (((P[cols]/model[g].call_.col_w).T - B)/row_m).T.replace([nan,inf,-inf], 1e-15)

        #standardize according to non normed PCA program
        Z = Zcod - self.call_.z_center
        #coordinates for the new nrows
        coord = (Z * self.call_.col_w).dot(self.svd_.V[:,:self.svd_.ncp])
        coord.columns = self.eig_.index[:self.svd_.ncp]
        return coord