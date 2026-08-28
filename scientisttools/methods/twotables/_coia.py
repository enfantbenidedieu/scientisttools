# -*- coding: utf-8 -*-
from numpy import array, ones, ndarray, nan, diag,sqrt,unique,sum, where, inf
from pandas import DataFrame, Series, concat
from collections import namedtuple
from itertools import repeat, chain
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..onetable._pca import PCA
from ..onetable._mca import MCA
from ..onetable._famd import FAMD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.func_coinertia import func_coinertia
from ..functions.func_eta2 import func_eta2
from ..functions.concat_empty import concat_empty
from ..functions.cov2corr import cov2corr
from ..functions.gfa import gFA
from ..functions.utils import cols_dtypes, check_is_dataframe
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix

class COIA(BaseEstimator,TransformerMixin):
    """
    CO-inertia Analysis (COIA)

    Performs CO-inertia Analysis (COIA) between two groups of variables in the sense of `Chessel-Dolédec <https://www.researchgate.net/publication/228011497_Co-inertia_analysis_an_alternative_method_for_studying_species-environment_relationships>`_.
    Groups of variables can be continuous, categorical, mixed or contingency tables.
    Missing values in continuous variables are replaced by the column mean. Missing values in categorical variables are replaced by the most freqent.

    Parameters
    ----------
    excl : None, list, default = None
        The "junk" categories. It can be a list or a tuple of the names of the categories or a list or a tuple of the indexes in the active disjunctive table.
    
    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : list, tuple
        The number of variables in each group.

    name_group : list, tuple, default = None
        The name of the groups. If None, the group are named Gr1 and Gr2.

    type_group : list, tuple
        The type of variables in each group. Possible values are: 

        * "c" or "s" for continuous variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (continuous and categorical variables)
        * "f" for frequency (from contingency tables)
    
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional individuals weights. The weights are given only for the active individuals.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional variables weights.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters, with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        Zcod : DataFrame of shape (n_rows, n_columns) 
            Standardized data from separate analyses. 
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data for COIA.
        tab : DataFrame of shape (n_rows, n_columns)
            Data used for GSVD.
        total : int
            The sum of all freqencies table.
        ind_w : Series of shape (n_rows,) 
            The individuals weights.
        row_w : Series of shape (n_xcolumns,)
            The rows weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_ycolumns,)
            The columns weights.
        ncp : int
            The number of components kepted.
        group : list
            The number of variables in each group.
        type_group : list
            The type of variables in each group.
        name_group : list
            The name of groups.
        ind_sup : None, list
            The names of the supplementary individuals.

    eig_ : DataFrame of shape (rank, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    freq_ : freq
        An object containing all the results for the active frequencies, with the following attributes:

        coord : DataFrame of shape (n_freq, ncp)
            The coordinates of the frequencies.
        cos2 : DataFrame of shape (n_freq, ncp)
            The squared cosinus of the frequencies.
        contrib : DataFrame of shape (n_freq, ncp)
            The relative contributions of the frequencies.
        infos : DataFrame of shape (n_freq, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the frequencies.

    group_ : group
        An object containing all the results for the groups with the following attributes:

        coinertia : DataFrame of shape (2,2)
            The coinertia coefficients.
        RV : DataFrame of shape (2,2)
            The \emph{RV} coefficients.

    ind_ : ind
        An object containing all the results for the active individuals with the following attributes:

        coord_partiel : DataFrame of shape (2*n_rows, ncp)
            The partiel coordinates of the active individuals.
        coord_partiel_n : DataFrame of shape (2*n_rows, ncp)
            The normed partiel coordinates of the active individuals.

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals with the following attributes:

        coord_partiel : DataFrame of shape (2*n_rows_sup, ncp)
            The partiel coordinates of the supplementary individuals.
        
    levels_ : levels_sup, optional
        An object containing all the results for the levels with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the levels.
        cos2 : DataFrame of shape (n_levels, ncp)
            The squared cosinus of the levels.
        contrib : DataFrame of shape (n_levels, ncp)
            The contributions of the levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the levels.

    quali_var_ : quali_var, optional
        An object containing all the results for the active categorical variables with the following attributes:

        coord_partiel : DataFrame(2*n_quali_var, ncp)
            The partiel coordinates for the categorical variables.

        contrib : DataFrame of shape (n_quali_var, ncp)
            The contributions of the categorical variables.
        
    quanti_var_ : quanti_var, optional
        An object containing all the results for the active variables with the following attributes:

        coord : DataFrame of shape (n_quanti_var, ncp)
            The coordinates of the variables.
        cos2 : DataFrame of shape (n_quanti_var, ncp)
            The squared cosinus of the variables.
        contrib : DataFrame of shape (n_quanti_var, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_quanti_var, 4)
            Additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the continuous variables.
    
    separate_analyses_ : dict
        The results for the separate analyses.

    svd_ : svd
        An object containing all the results for the generalized singular value decomposition (GSVD), with the following attributes:
        
        vs : 1d numpy array of shape (rank,)
            The singular values.
        d : 1d numpy array of shape (rank,)
            The eigen values.
        U : 2d numpy array of shape (n_xcolumns, rank)
            The left singular vectors.
        V : 2d numpy array of shape (n_ycolumns, rank)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.

    References
    ----------
    [1] Dolédec, S. and Chessel, D. (1994) Co-inertia analysis: an alternative method for studying species-environment relationships. Freshwater Biology, 31, 277-294.

    [2] Dray, S., Chessel, D. and J. Thioulouse (2003) Co-inertia analysis and the linking of the ecological data tables. Ecology, 84, 11, 3078-3089.
    
    See Also
    --------
    save : Print results for general factor analysis model in an Excel sheet
    sprintf : Print the analysis results
    summary : Printing summaries of general factor analysis model

    Examples
    --------
    >>> from scientisttools.datasets import decathlon, wine, poison, mortality, doubs, autos2005
    >>> from scientisttools import COIA
    >>> #co-inertia between two PCA
    >>> wine2 = wine.data.iloc[:,10:29]
    >>> clf = COIA(group=(10,9),type_group=("s","s"),name_group=("olfag","gust"))
    >>> clf.fit(wine2)
    COIA(group=(10,9),name_group=("olfag","gust"),type_group=("s","s"))
    >>>
    >>> #co-inertia between two MCA
    >>> poison2 = poison.data.iloc[:,4:15]
    >>> clf = COIA(group=(5,6),type_group=("n","n"),name_group=("symptom","eat"))
    >>> clf.fit(poison2)
    COIA(group=(5,6),name_group=("symptom","eat"),type_group=("n","n"))
    >>>
    >>> #co-inertia between PCA and MCA
    >>> clf = COIA(group=(9,3),type_group=("s","n"),ind_sup=range(38,45))
    >>> clf.fit(autos2005.data.iloc[:,:12])
    COIA(group=(9,3),ind_sup=range(38,45),type_group=("s","n"))
    >>>
    >>> #co-inertia between two contingency tables
    >>> clf = COIA(group=(9,9),type_group=("f","f"),name_group=("y1979","y2006"))
    >>> clf.fit(mortality.data)
    COIA(group=(9,9),name_group=("y1979","y2006"),type_group=("f","f"))
    >>>
    >>> #co-inertia between a contingency table and PCA
    >>> clf = COIA(group=(27,11),type_group=("f","s"),name_group=("species","environmental"))
    >>> clf.fit(doubs)
    COIA(group=(27,11),name_group=("species","environmental"),type_group=("f","s"))
    """
    def __init__(
            self, 
            excl = None, 
            ncp = 5, 
            group = None, 
            type_group = None, 
            name_group = None, 
            row_w = None, 
            col_w = None, 
            ind_sup = None,  
            tol = 1e-7
    ):
        self.excl = excl
        self.ncp = ncp
        self.group = group
        self.type_group = type_group
        self.name_group = name_group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.tol = tol

    def fit(self,X,y=None):
        """Fit the model to X

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None: 
            raise ValueError("group must be assigned.")
        elif not isinstance(self.group, (list,tuple,ndarray,Series)):
            raise ValueError("group must be a 1d array-like with the number of variables in each group")
        elif len(self.group) != 2:
            raise ValueError("group must be a 1d array-like with lenght 2.")
        else: 
            group = [int(x) for x in self.group]

        # check if group definition
        if sum(group) != X.shape[1]:
            raise TypeError("Not convenient group definition")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if any group has only one /columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if group[0] == 1 or group[1] == 1:
            raise ValueError("groups should have at least two columns")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if type_group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.type_group is None: 
            raise ValueError("type_group must be assigned.")
        elif not isinstance(self.type_group, (list,tuple,ndarray,Series)): 
            raise ValueError("type_group must be a 1d array-like with the type of variables in each group")
        elif len(self.type_group) != 2: 
            raise ValueError("type_group must be a 1d array-like with lenght 2.")
        else:
            type_group = [str(x) for x in self.type_group]
    
        # check if type_group definition is convenient
        if any(x not in ("c","f","m","n","s") for x in type_group):
            raise ValueError("Not convenient type_group definition")
        
        if len(self.group) != len(self.type_group):
            raise TypeError("Not convenient group definition")

        # which type_group is f
        num_group_freq = list(where(array(type_group) == "f")[0]) if any(x == "f" for x in type_group) else None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned name_group
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.name_group is None:
            name_group = ["X","Y"]
        elif not isinstance(self.name_group,(list,tuple,ndarray,Series)):
            raise TypeError("name_group must be a 1d array-like with names of group")
        elif len(self.name_group) != 2:
            raise ValueError("name_group must be a 1d array-like with lenght 2.")
        else:
            name_group = [x for x in self.name_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group name to label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_dict, k = {}, 0
        for i, g in zip(range(len(group)),name_group):
            group_dict[g] = X.columns[k:(k+group[i])]
            k += group[i]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # type_group_var
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        type_group_var, k = None, 0
        for i in range(len(group)):
            colnames = X.columns[k:(k+group[i])]
            if type_group[i] in ("c","s"):
                type_var = Series(repeat("quanti",group[i]),index=colnames)
            if type_group[i] == "f":
                type_var = Series(repeat("freq",group[i]),index=colnames)
            if type_group[i] == "n":
                type_var = Series(repeat("quali",group[i]),index=colnames)
            if type_group[i] == "m":
                type_var = Series(cols_dtypes(X.loc[:,colnames]),index=colnames)
            type_group_var = concat_empty(type_group_var,type_var,axis=0)
            k += group[i]

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coinertia analysis (COIA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # number of samples, columns
        n_rows, n_vars = X.shape
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set rows and columnss weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # individuals weights
        if self.row_w is None: 
            ind_w = Series(ones(n_rows)/n_rows,index=X.index,name="weights")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("row_w must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
        else: 
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weights")

        #columns weights
        if self.col_w is None: 
            var_w = Series(ones(n_vars),index=X.columns,name="weights")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("col_w must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_vars: 
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_vars},).")
        else: 
            var_w = Series(array(self.col_w),index=X.columns,name="weights")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis model for active group
        Xcod, total = X.copy(), 0
        if num_group_freq is not None:
            name_group_freq = [g for i, g in enumerate(name_group) if i in num_group_freq]
            
            if len(name_group_freq) > 0:
                group_freq_dict ={k : group_dict[k] for k in name_group_freq}
                freq_cols = list(chain.from_iterable(group_freq_dict.values()))
                #select frequencies data
                N = X.loc[:,freq_cols]
                #sum of all elements 
                total = N.sum().sum()
                #proportional table
                P = N/total
                #set global row margin and columns margin
                row_m, col_m = P.sum(axis=1), P.sum(axis=0)
                #construction of recoded table
                for g, cols in group_freq_dict.items():
                    #row margin for group
                    row_m_g = P[cols].sum(axis=1)
                    #normalize such as sum equal to 1
                    row_w_g = row_m_g/sum(row_m_g)
                    #recoded columns and fill NA, +/-inf if 1e-15
                    Xcod[cols] = (((P[cols]/col_m[cols]).T - row_w_g)/row_m).T.replace([nan,inf,-inf], 1e-15)
                #update weights for rows and columns
                ind_w, var_w[freq_cols] = row_m, col_m

        #rows weights
        row_w = ind_w.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # separate general factor analysis model for active group
        model = {}
        for g, cols in group_dict.items():
            if self.type_group[name_group.index(g)] in ("c","f","s"):
                scale_unit = False if self.type_group[name_group.index(g)] in ("c","f") else True
                fa = PCA(scale_unit=scale_unit,ncp=self.ncp,row_w=row_w,col_w=var_w[cols],tol=self.tol)
            if self.type_group[name_group.index(g)] == "m":
                fa = FAMD(ncp=self.ncp,row_w=row_w,col_w=var_w[cols],tol=self.tol)
            if self.type_group[name_group.index(g)] == "n":
                fa = MCA(excl=self.excl,ncp=self.ncp,row_w=ind_w,col_w=var_w[cols],tol=self.tol)
            model[g]= fa.fit(Xcod[cols])

        #store separate analysis
        self.separate_analyses_ = model

        #extract elements
        xmodel, ymodel = model[name_group[0]], model[name_group[1]]
        row_w, col_w = xmodel.call_.col_w, ymodel.call_.col_w

        # concatenate
        Zcod = concat((model[g].call_.Z for g in name_group),axis=1)

        #co-inertia table : Z = Z_x'DZ_y
        Z = (xmodel.call_.Z.T * ind_w).dot(ymodel.call_.Z)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # fit factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gFA(X=Z,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
        # extract elements
        self.svd_, self.eig_, col_ = fit_.svd, fit_.eig, {k : concat((fit_.row[k],fit_.col[k]),axis=0) for k in list(fit_.col.keys())}
        # reset number of components
        ncp = self.svd_.ncp

        #set call_ informations
        call_ = {"Xtot": Xtot, "X": X, "Zcod": Zcod, "Z": Z, "tab": Z, "total": total, 
                 "ind_w": ind_w, "row_w": row_w, "var_w": var_w, "col_w": col_w, "ncp": ncp, 
                 "group": group, "name_group": name_group, "type_group": type_group, "ind_sup": ind_sup_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #convert to ordered dictionary
        V = {name_group[0] : self.svd_.U, name_group[1] : self.svd_.V}
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for the individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partiel and normed partiel coordinates for the individuals
        ind_coord_partiel, ind_coord_partiel_n  = None, None
        for g in name_group:
            # partiel coordinates for the individuals
            coord_partiel = (model[g].call_.Z * model[g].call_.col_w).dot(V[g][:,:ncp])
            # set index and columns
            coord_partiel.columns = self.eig_.index[:ncp]
            coord_partiel.index = [f"{x}.{g}" for x in coord_partiel.index]
            # normed partiel coordinates for the individuals
            coord_partiel_n = coord_partiel.apply(lambda x : x/sqrt(sum(ind_w*x**2)),axis=0)
            # concatenate
            ind_coord_partiel = concat_empty(ind_coord_partiel,coord_partiel,axis=0)
            ind_coord_partiel_n = concat_empty(ind_coord_partiel_n,coord_partiel_n,axis=0)

        # convert to dictionary
        ind_ = {"coord_partiel" : ind_coord_partiel, "coord_partiel_n":ind_coord_partiel_n}
        # convert to namedtuple and add to model attribute
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quanti" in type_group_var.values:
            #select all  names
            quanti_var_cols = type_group_var[type_group_var=="quanti"].index
            # convert to dictionary
            quanti_var_ = {k : col_[k].loc[quanti_var_cols,:] for k in list(col_.keys())}
            # convert to namedtuple and add to model attribute
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # statistics for levels/qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quali" in type_group_var.values:
            #select all qualitative variables names
            quali_var_cols = type_group_var[type_group_var == "quali"].index
            #select all qualitative variables
            X_quali_var = X[quali_var_cols]
            #select all levels
            levels_ = {k : col_[k].loc[col_[k].index.isin(unique(X_quali_var)),:] for k in list(col_.keys())}
            #proportion for the slevels
            p_k = (disjunctive(X_quali_var).T * ind_w).sum(axis=1)
            #vtest for the levels
            levels_["vtest"] = (levels_["coord"].T * sqrt((n_rows-1)/((1/p_k) - 1))).T/self.svd_.vs[:ncp]
            #convert to namedtuple and add to model attribute
            self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

            #statistics for qualitative variables
            #partiel coordinates for the qualitative variables
            quali_var_coord_partiel = None
            for g in name_group:
                # extract individuals partiel coordinates
                coord_partiel = ind_coord_partiel.iloc[i:(i+n_rows),:]
                coord_partiel.index = model[g].call_.Z.index
                # partiel coordinates for the qualitative variables
                coord = func_eta2(X=coord_partiel,by=X_quali_var,w=ind_w,excl=None)
                # set index
                coord.index = [f"{x}.{g}" for x in coord.index]
                # concatenate
                quali_var_coord_partiel = concat_empty(quali_var_coord_partiel,coord,axis=0)
            
            # contributions for the qualitative variables
            quali_var_contrib = concat((levels_["contrib"].loc[levels_["contrib"].index.isin(X_quali_var[j].unique()),:].sum(axis=0).to_frame(j) for j in X_quali_var.columns),axis=1).T
            # convert to dictionary
            quali_var_ = {"coord_partiel" : quali_var_coord_partiel,"contrib":quali_var_contrib}
            # convert to namedtuple and add to model attribute
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # frequency informations : coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "freq" in type_group_var.values:
            # select all frequencies columns
            freq_cols = type_group_var[type_group_var=="freq"].index
            # coordinates for frequencies
            freq_ = {k : col_[k].loc[freq_cols,:] for k in list(col_.keys())}
            # convert to namedtuple
            self.freq_ = namedtuple("freq",freq_.keys())(*freq_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlations with coinertia axis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlations between X and coinertia axis
        ax = DataFrame(xmodel.svd_.V[:,:xmodel.call_.ncp].T.dot(self.svd_.U[:,:ncp].T.dot(diag(row_w)).T),
                       index=[f"Dim{x+1}.{name_group[0]}" for x in range(xmodel.call_.ncp)],columns=self.eig_.index[:ncp])
        #correlations between Y and coinertia axis
        ay = DataFrame(ymodel.svd_.V[:,:ymodel.call_.ncp].T.dot(self.svd_.V[:,:ncp].T.dot(diag(col_w)).T),
                       index=[f"Dim{x+1}.{name_group[1]}" for x in range(ymodel.call_.ncp)],columns=self.eig_.index[:ncp])
        # concatenate
        partiel_axes_coord = concat((ax,ay),axis=0)
        # convert to dictionary
        partiel_axes_ = {"coord":partiel_axes_coord}
        # convert to namedtuple
        self.partial_axes_ = namedtuple("partial_axes",partiel_axes_.keys())(*partiel_axes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group informations : coinertia and RV
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coinertia
        coinertia = DataFrame(index=name_group,columns=name_group).astype("float")
        coinertia.iloc[0,0] = func_coinertia(X=xmodel.call_.Z,Y=xmodel.call_.Z,xcol_w=xmodel.call_.col_w,ycol_w=xmodel.call_.col_w,row_w=ind_w)
        coinertia.iloc[1,1] = func_coinertia(X=ymodel.call_.Z,Y=ymodel.call_.Z,xcol_w=ymodel.call_.col_w,ycol_w=ymodel.call_.col_w,row_w=ind_w)
        coinertia.iloc[1,0] = coinertia.iloc[0,1] = func_coinertia(X=xmodel.call_.Z,Y=ymodel.call_.Z,xcol_w=xmodel.call_.col_w,ycol_w=ymodel.call_.col_w,row_w=ind_w)
        # RV-coefficients
        RV = cov2corr(X=coinertia)
        #convert to ordered dictionary
        group_ = {"coinertia": coinertia, "RV": RV}
        # convert to namedtuple
        self.group_ = namedtuple("group",group_.keys())(*group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #Data preparation
            Zcod_ind_sup = DataFrame(index=X_ind_sup.index,columns=Zcod.columns).astype("float")
            for g, cols in group_dict.items():
                if self.type_group[name_group.index(g)] in ("c","s"):
                    Zcod_ind_sup[cols] = (X_ind_sup[cols] - model[g].call_.center)/model[g].call_.scale
                elif self.type_group[name_group.index(g)] == "n":
                    dummies_cols = model[g].call_.dummies.columns
                    dummies_ind_sup = disjunctive(X_ind_sup[cols],cols=dummies_cols)
                    Zcod_ind_sup[dummies_cols] = (dummies_ind_sup - model[g].call_.center)/model[g].call_.scale
                elif self.type_group[name_group.index(g)] == "m":
                    #split X
                    split_Xcols_ind_sup = splitmix(X_ind_sup[cols])
                    Xcols_ind_sup_quanti, Xcols_ind_sup_quali = split_Xcols_ind_sup.quanti, split_Xcols_ind_sup.quali
                    n_ind_sup_quanti, n_ind_sup_quali  = split_Xcols_ind_sup.k1, split_Xcols_ind_sup.k2
                    
                    #initialization
                    Xcols_ind_sup = None
                    if n_ind_sup_quanti > 0:
                        if model[g].call_.k1 != n_ind_sup_quanti:
                            raise TypeError("The number of continuous variables must be the same")
                        Xcols_ind_sup = concat_empty(Xcols_ind_sup,Xcols_ind_sup_quanti,axis=1)
                    if n_ind_sup_quali > 0:
                        if model[g].call_.k2 != n_ind_sup_quali:
                            raise TypeError("The number of categoricals variables must be the same")
                        Xcols_ind_sup = concat_empty(Xcols_ind_sup,disjunctive(X=Xcols_ind_sup_quali,cols=model[g].call_.dummies.columns),axis=1)
                    Zcod_ind_sup[Xcols_ind_sup.columns] = (Xcols_ind_sup - model[g].call_.center)/model[g].call_.scale

            if num_group_freq is not None:
                if len(name_group_freq) > 0:
                    #frequencies in supplementary individuals
                    P_row_sup = X_ind_sup[freq_cols]/total
                    #supplementary rows margin
                    row_sup_m = P_row_sup.sum(axis=1)
                    #construction of recoded table
                    for g, cols in group_freq_dict.items():
                        #group rows margins
                        row_rowsup_m_g = P_row_sup[cols].sum(axis=1)
                        #normalize such sum is equal to 1
                        B_row_sup = row_rowsup_m_g/sum(row_rowsup_m_g)
                        #recoded columns
                        Zcod_ind_sup[cols] = (((P_row_sup[cols]/col_m[cols]).T - B_row_sup)/row_sup_m).T.replace([nan,inf,-inf], 1e-15)

            # partiel coordinates for the supplementary individuals
            ind_sup_coord_partiel = None
            for g in name_group:
                coord = DataFrame((Zcod_ind_sup[model[g].call_.col_w.index] * model[g].call_.col_w).dot(V[g][:,:ncp]).to_numpy(),
                                  index=[f"{x}.{g}" for x in ind_sup_label],columns=self.eig_.index[:ncp]) 
                # concatenate
                ind_sup_coord_partiel = concat_empty(ind_sup_coord_partiel,coord,axis=0)
            
            # convert todictionary
            ind_sup_ = {"coord_partiel" : ind_sup_coord_partiel}
            # convert to namedtuple and add to model attribute
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self  
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` is the number of rows 
            and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (2*n_rows, ncp)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord_partiel
    
    def transform(self,X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            New data, where ``n_rows`` is the number of rows 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (2*n_rows, ncp)
            Projection of X in the first principal components, where ``n_rows`` is the number of rows 
            and ``ncp`` is the number of the components.
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
        name_group = self.call_.name_group
        group_dict = {g : model[g].call_.X.columns for g in name_group}
        #Data preparation
        Zcod = DataFrame(index=X.index,columns=self.call_.Z.columns).astype("float")
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
                Xcols_quanti, Xcols_quali = split_Xcols.quanti, split_Xcols.quali
                n_quanti, n_quali = split_Xcols.k1, split_Xcols.k2
                
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
        num_group_freq = where(array(self.type_group) == "f")[0] if any(x == "f" for x in self.type_group) else None
        if num_group_freq is not None:
            name_group_freq = [g for i, g in enumerate(self.call_.name_group) if i in num_group_freq] 
            if len(name_group_freq) > 0:
                group_freq_dict = {k : model[k].call_.X.columns for k in name_group_freq}
                freq_cols = list(chain.from_iterable(group_freq_dict.values()))
                # frequencies
                P = X[freq_cols]/self.call_.total
                # supplementary rows margin
                row_m = P.sum(axis=1)
                # construction of recoded table
                for g, cols in group_freq_dict.items():
                    # group rows margins
                    row_m_g = P[cols].sum(axis=1)
                    # normalize such sum is equal to 1
                    B = row_m_g/sum(row_m_g)
                    # recoded columns
                    Zcod[cols] = (((P[cols]/model[g].call_.col_w).T - B)/row_m).T.replace([nan,inf,-inf], 1e-15)
        
        # convert to ordered dictionary
        V = {self.call_.name_group[0] : self.svd_.U, self.call_.name_group[1] : self.svd_.V}
        # partiel coordinates for the new individuals
        coord_partiel = None
        for g in name_group:
            coord = DataFrame((Zcod[model[g].call_.col_w.index] * model[g].call_.col_w).dot(V[g][:,:self.call_.ncp]).to_numpy(),
                              index=[f"{x}.{g}" for x in X.index],columns=self.eig_.index[:self.call_.ncp]) 
            # concatenate
            coord_partiel = concat_empty(coord_partiel,coord,axis=0)
        return coord_partiel