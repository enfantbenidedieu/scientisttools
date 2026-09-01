# -*- coding: utf-8 -*-
from numpy import ndarray,where, unique, ones, array,sqrt,inf,nan,outer,array,sum,vstack,c_,insert,cumsum,diff,nan
from pandas import DataFrame, Series, concat
from itertools import chain, repeat
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from ..onetable._famd import FAMD
from ..onetable._mca import MCA
from ..onetable._pca import PCA
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.concat_empty import concat_empty
from ..functions.gsvd import gSVD
from ..functions.func_eta2 import func_eta2
from ..functions.cov2corr import cov2corr
from ..functions.func_coinertia import func_coinertia
from ..functions.statistics import func_groupby, wmean
from ..functions.utils import cols_dtypes
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class MCOIA(BaseEstimator,TransformerMixin):
    """
    Multiple CO-Inertia Analysis (MCOIA)

    Performs Multiple CO-inertia Analysis in the sense of `Chessel-Hanafi <https://www.numdam.org/article/RSA_1996__44_2_35_0.pdf>`_.
    Groups of variables can be continuous, categorical, mixed or contingency table. Missing values in numeric variables are replaced by the column mean.

    Parameters
    ----------
    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : list, tuple
        The number of variables in each group.

    type_group : list, tuple
        The type of variables in each group. Possible values are: 

        * "c" or "s" for continuous variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (continuous and categorical variables)
        * "f" for frequency (from contingency tables)

    name_group : list, tuple, default = None
        The name of the groups. If ``None``, the group are named Gr1, Gr2 and so on.

    option : str, default = "lambda1"
        A string for the weightings of the variables.

        * 'inertia': weighting of group :math:`k` by the inverse of the total inertia of the group :math:`k`.
        * 'lambda1': weighting of group :math:`k` by the inverse of the first eigenvalue of the :math:`k`analysis.
        * 'uniform': uniform weighting of groups.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

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
        ind_sup : None, list
            The names of the supplementary individuals.

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    freq_ : freq, optional
        An object containing all the results for the active frequencies, with the following attributes:

        coord : DataFrame of shape (n_freq, ncp)
            The coordinates (onto synthetic scores) of the frequencies.

    group_ : group
        An object containing all the results for the groups, with the following attributes:

        lambd : DataFrame of shape (n_groups, ncp)
            All eigenvalues (computed on the separate analyses) after normalisation.
        coinertia : DataFrame of shape (n_groups, n_groups)
            The trace \emph{coinertia} coefficients.
        RV : DataFrame of shape (n_groups, n_groups)
            The \emph{RV} coefficients.
        cov2 : DataFrame of shape (n_groups, ncp)
            All pseudo eigenvalues (synthetic analysis).
    
    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows,ncp)
            The synthetic scores of the individuals.
        coord_partiel : coord_partiel
            An object containing the co-inertia coordinates of the individuals.
        coord_partiel_n : coord_partiel_n
            An object containing the co-inertia normed scores of the individuals.
        
    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord_partiel : coord_partiel
            An object containing all the co-inertia coordinates for the supplementary individuals.

    levels_ : levels_sup, optional
        An object containing all the results for the active levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The synthetic scores of the levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the levels.
        coord_partiel : coord_partiel
            An object containing the co-inertia coordinates of the levels.

    quali_var_ : quali_var, optional
        An object containing all the results for the active qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_quali_var, ncp)
            The synthetic scores of the qualitative variables, which is eta2, the square correlation corefficient between a qualitative variable and a dimension.
        coord_partiel : coord_partiel
            An object containing the co-inertia coordinates for the qualitatve variables.

    quanti_var_ : quanti_var, optional
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates (onto synthetic scores) of the variables.

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

    See also
    --------
    save : Print results for general factor analysis model in an Excel sheet.
    sprintf : Print the analysis results.
    summary : Printing summaries of general factor analysis model.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MCOIA
    >>> clf = MCOIA(group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name,num_group_sup=(0,5))
    >>> clf.fit(wine.data)
    MCOIA(group=wine.group,type_group=("n","s","s","s","s","s"),name_group=wine.name,num_group_sup=(0,5))
    """
    def __init__(
            self, 
            excl = None, 
            ncp = 5, 
            group = None, 
            type_group = None, 
            name_group = None, 
            option = "lambda1", 
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
        self.option = option
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.tol = tol
        
    def fit(self,X,y=None):
        """
        Fit the model to X

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns
        -------
        self : object
            Returns the instance itself
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
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if sum(group) != X.shape[1]:
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
        #check if option is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.option in ("lambda1","inertia","uniform")):
            raise ValueError("option must be one of 'lambda1', 'inertia', 'uniform'")

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
        # set type_group_var
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

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign each assign to group
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign each variable to group
        var_group = DataFrame({
            "variable" : X.columns,
            "group" : list(chain(*[repeat(i,k) for i, k in zip(name_group,group)]))
        })

        # assign column to group
        cols_group, k = None, 0
        for i in range(len(group)):
            vars_names = X.columns[k:(k+group[i])]
            if type_group[i] in ("c","s","f"):
                col_group = Series(repeat(name_group[i],group[i]),index=vars_names,name="group")
            if type_group[i] == "n":
                col_names = unique(X.loc[:,vars_names])
                col_group = Series(repeat(name_group[i],len(col_names)),index=col_names,name="group")
            if type_group[i] == "m":
                col_dtypes = Series(cols_dtypes(X.loc[:,vars_names]),index=vars_names)
                col_names1 = col_dtypes[col_dtypes=="quanti"].index
                col_names2 = unique(X.loc[:,col_dtypes[col_dtypes=="quali"].index])
                col_group1 = Series(repeat(name_group[i],len(col_names1)),index=col_names1,name="group")
                col_group2 = Series(repeat(name_group[i],len(col_names2)),index=col_names2,name="group")
                col_group = concat((col_group1,col_group2),axis=0)
            cols_group = concat_empty(cols_group,col_group,axis=0)
            k += group[i]
        # reset index and rename
        cols_group = cols_group.to_frame().reset_index().rename(columns={"index" : "variable"})

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #multiple CO-inertia analysis (MCOA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of samples and columns
        n_rows, n_vars = X.shape
        
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
            var_w = Series(ones(n_vars),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)):
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_vars:
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_vars},).")
        else:
            var_w = Series(array(self.col_w),index=X.columns,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Data Preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Xcod, total = X.copy(), 0
        if num_group_freq is not None:
            name_group_freq = [g for i, g in enumerate(name_group) if i in num_group_freq]
            
            if len(name_group_freq) > 0:
                group_freq_dict = {k : group_dict[k] for k in name_group_freq}
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
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis model for active group
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

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardized data
        Zcod = concat((model[g].call_.Z for g in name_group),axis=1)
        # weighted average and standard deviation
        center = concat((model[g].call_.center for g in name_group),axis=0)
        scale = concat((model[g].call_.scale for g in name_group),axis=0)
        # number of components in all models
        mncp = Series([model[g].call_.ncp for g in name_group],index=name_group)
        #active columns dictionary and columns weights
        columns_dict = {g : model[g].call_.Z.columns for g in name_group}
        #columns index
        columns_index = {g : [Zcod.columns.tolist().index(k) for k in cols] for g, cols in columns_dict.items()}
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set columns weights for multiple co-inertia analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set alpha - weighted
        if self.option == "lambda1":
            alpha = Series([1/model[g].eig_.iloc[0,0] for g in name_group],index=name_group)
        elif self.option == "inertia":
            alpha = Series([1/sum(model[g].eig_.iloc[:,0]) for g in name_group],index=name_group)
        else:
            alpha = Series(ones(len(name_group)),index=name_group)
        # set columns weights for multiple co-inertia analysis
        col_w = concat((model[g].call_.col_w*alpha[g] for g in name_group),axis=0)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #non-normed principal component analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center according to non normed PCA
        z_center = wmean(X=Zcod,w=row_w)
        #center
        Z = Zcod - z_center
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #run multiple generalized singular values decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #make a copy of standardized data
        tab = Z.copy()
        #initialisation
        U, V, vs = [], [], [] 
        for i in range(int(min(20, n_rows - 1, n_vars))):
            #generalized singular value decomposition
            fit_ = gSVD(tab,row_w=row_w,col_w=col_w,ncp=self.ncp,tol=self.tol)
            #normalization right matrix using mcoia variables weights by group and substract projection
            if (i == 0) or ((i > 0) and ((fit_.vs[0]/vs[0])**2) > self.tol):
                v = fit_.V[:,0]/sqrt(col_w.to_numpy())
                for g, cols in columns_dict.items():
                    idx = columns_index[g]
                    v1 = v[idx]
                    v2 = sqrt(sum(v1*v1))
                    #normalisation of v1
                    if v2 > self.tol:
                        v1 = v1/v2
                    #substract projection
                    tab[cols] = tab[cols] - outer((tab[cols] * v1).sum(axis=1),v1)
                    v[idx] = v1
                U.append(fit_.U[:,0]), V.append(v), vs.append(float(fit_.vs[0]))
            else:
                break

        #convert to array, update U and V with number of components
        U, V, vs = vstack(U).T, vstack(V).T, array(vs)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #update maximum number of components
        rank = U.shape[1]
        #set number of components
        if self.ncp is None:
            ncp = rank
        elif not isinstance(self.ncp,int):
            raise TypeError("'ncp' must be an integer.")
        elif self.ncp < 1:
            raise ValueError("'ncp' must be equal or greater than 1.")
        else:
            ncp = min(self.ncp,rank)

        #set call_ informations
        call_ = {"Xtot":Xtot,"X":X,"Xcod":Xcod,"Zcod":Zcod,"Z":Z,"total":total,"ind_w":ind_w,"row_w":row_w,
                 "var_w":var_w,"col_w":col_w,"center":center,"scale":scale,"z_center":z_center,"alpha":alpha,"ncp":ncp,
                 "group":group,"type_group":type_group,"name_group":name_group,"ind_sup":ind_sup_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #add to model attributes
        self.svd_ = namedtuple("svd",["U","V","vs","rank","ncp"])(U,V,vs,rank,ncp)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eigvals = self.svd_.vs**2
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(len(eigvals))])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals co-inertia informations: synthetic scores, partial scores and partial normed scores
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #synthetic scores for the individuals
        ind_synvar = DataFrame(self.svd_.U[:,:ncp],index=Z.index,columns=self.eig_.index[:ncp])
        # partial coordinates and normed partial coordinates for individuals
        ind_coord_partiel, ind_coord_partiel_n = None, None
        for g, cols in columns_dict.items():
            #partial scores of the individuals
            coord_partiel = (Z[cols] * sqrt(col_w[cols])).dot((self.svd_.V[columns_index[g],:ncp].T * model[g].call_.col_w.to_numpy()).T)
            #normed partial scores of the individuals
            coord_partiel_n = coord_partiel/sqrt(((coord_partiel.T * sqrt(row_w))**2).sum(axis=1))
            #set columns
            coord_partiel.columns, coord_partiel_n.columns = self.eig_.index[:ncp], self.eig_.index[:ncp]
            # set index
            coord_partiel.index, coord_partiel_n.index = [f"{x}.{g}" for x in Z.index], [f"{x}.{g}" for x in Z.index]
            #concatenate
            ind_coord_partiel = concat_empty(ind_coord_partiel,coord_partiel,axis=0) 
            ind_coord_partiel_n = concat_empty(ind_coord_partiel_n,coord_partiel_n,axis=0)
        #convert to ordered dictionary
        ind_ = {"coord":ind_synvar,"coord_partiel":ind_coord_partiel,"coord_partiel_n":ind_coord_partiel_n}
        #convert to namedtuple - add to model attributes
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group informations : coinertia, RV
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #lambda : eigen values informations - after normalization
        lambd, mincp = None, ncp
        for g, cols in columns_dict.items():
            nbcol = min(ncp,mncp[g])
            # squared singular values
            sqvs = gSVD(X=Z[cols],ncp=nbcol,row_w=row_w,col_w=col_w[cols],tol=self.tol).vs[:nbcol]**2
            lambd = concat_empty(lambd,Series(sqvs,index=[f"Dim{x+1}" for x in range(nbcol)]).to_frame(g).T,axis=0)
            mincp = min(mincp, nbcol)
        lambd = lambd.iloc[:,:mincp]

        #covariance
        group_sqcov, i = DataFrame(index=name_group,columns=self.eig_.index[:ncp]).astype("float"), 0
        for g in name_group:
            # partiel coordinates for individuals
            coord_partiel = ind_coord_partiel.iloc[i:(i+n_rows),:ncp]
            group_sqcov.loc[g,:] = (((coord_partiel * self.svd_.U[:,:ncp]).T * row_w.to_numpy()).sum(axis=1)**2)
            # update i
            i += n_rows
        
        #coinertia coefficients
        coinertia = DataFrame(index=name_group,columns=name_group).astype("float")
        for g1, cols1 in columns_dict.items():
            for g2, cols2 in columns_dict.items():
                coinertia.loc[g1,g2] = func_coinertia(X=Z[cols1],Y=Z[cols2],xcol_w=col_w[cols1],ycol_w=col_w[cols2],row_w=row_w)
        
        # add MCOA coinertia coefficients
        if self.option == "lambda1":
            den = self.eig_.iloc[0,0]
        elif self.option == "inertia":
            den = sum(self.eig_.iloc[:,0])
        else:
            den = 1
        coinertia.loc["MCOA",:] = coinertia.loc[:,"MCOA"] = coinertia.loc[name_group,:].sum(axis=0)/den
        coinertia.loc["MCOA","MCOA"] = coinertia.loc[name_group,"MCOA"].sum()/den
        #RV Coefficients
        RV = cov2corr(X=coinertia)
        #convert to ordered dictionary
        group_ = {"lambd":lambd,"coinertia":coinertia,"RV":RV,"cov2":group_sqcov}
        #convert to namedtuple
        self.group_ = namedtuple("group",group_.keys())(*group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partial inertia 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates of partial axes - inertia axes onto co-inertia axis
        partial_axes_coord = None
        for g, cols in columns_dict.items():
            nbcol = min(ncp,mncp[g])
            coord_partial = model[g].svd_.V[:,:nbcol].T.dot((self.svd_.V[columns_index[g],:nbcol].T * model[g].call_.col_w.to_numpy()).T)
            for i in range(nbcol):
                if coord_partial[i,i] < 0:
                    for j in range(nbcol):
                        coord_partial[i,j] = - coord_partial[i,j]
            coord_partial = DataFrame(coord_partial,index=model[g].eig_.index[:nbcol],columns= self.eig_.index[:nbcol])
            # concatenate
            partial_axes_coord = concat_empty(partial_axes_coord,coord_partial,axis=0)
        # convert dictionary
        partial_axes_ = {"coord":partial_axes_coord,"cos2":partial_axes_coord**2}
        # convert to namedtuple
        self.partial_axes_ = namedtuple("partial_axes",partial_axes_.keys())(*partial_axes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for continuous variables (coordinates on synthetic scores, contributions and cos2)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quanti" in type_group_var.to_numpy():
            # continuouos variables columns
            quanti_var_cols = type_group_var[type_group_var == "quanti"].index
            #coordinates on synthetic scores of the quantitative variables
            quanti_var_coord  = (Z[quanti_var_cols] * sqrt(col_w[quanti_var_cols])).T.dot((self.svd_.U[:,:ncp].T * row_w.to_numpy()).T)
            quanti_var_coord.columns = self.eig_.index[:ncp]
            #convert to ordered dictionary
            quanti_var_ = {"coord":quanti_var_coord, "cos2" : quanti_var_coord**2}
            #convert to namedtuple
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for levels (coordinates, partiel coordinates & value-test) and qualitative variables (coordinates & partial coordinates)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "quali" in type_group_var.to_numpy():
            # categorical variable columns
            quali_var_cols = type_group_var[type_group_var == "quali"].index
            # select all categorical variables
            X_quali_var = X[quali_var_cols]
            # coordinates for the levels
            levels_coord = func_groupby(X=self.ind_.coord,by=X_quali_var,func="mean",w=row_w)
            # proportion for the levels
            p_k = (disjunctive(X_quali_var).T * row_w).sum(axis=1)
            # vtest for the levels
            levels_vtest = (levels_coord.T * sqrt((n_rows-1)/((1/p_k) - 1))).T/self.svd_.vs[:ncp]
            # coordinates for the categorical variables on synthetic scores - Eta-squared
            quali_var_coord = func_eta2(X=self.ind_.coord,by=X_quali_var,w=row_w,excl=None)
            # partial coordinates of the levels
            levels_coord_partiel, quali_var_coord_partiel, i = None, None, 0
            for g in list(columns_dict.keys()):
                # extract individuals partiel coordinates
                coord_partiel = ind_coord_partiel.iloc[i:(i+n_rows),:]
                coord_partiel.index = ind_["coord"].index
                coord1 = func_groupby(X=coord_partiel,by=X_quali_var,func="mean",w=row_w)
                coord2 = func_eta2(X=coord_partiel,by=X_quali_var,w=row_w,excl=None)
                # set index
                coord1.index, coord2.index = [f"{x}.{g}" for x in coord1.index], [f"{x}.{g}" for x in coord2.index]
                # concatenate
                levels_coord_partiel = concat((levels_coord_partiel,coord1),axis=0)
                quali_var_coord_partiel = concat((quali_var_coord_partiel,coord2),axis=0)
                # update i
                i += n_rows
            #convert to ordered dictionary
            levels_ = {"coord":levels_coord,"coord_partiel":levels_coord_partiel,"vtest":levels_vtest}
            #conver to namedtuple
            self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

            ##statistics for the categorical variables
            # convert to ordered dictionary
            quali_var_ = {"coord":quali_var_coord,"coord_partiel":quali_var_coord_partiel}
            # convert to namedtuple
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for the frequencies : coordinates, cos2, contributions & infos
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "freq" in type_group_var.to_numpy():
            #frequencies columns
            freq_cols = type_group_var[type_group_var == "freq"].index
            #coordinates on synthetic scores of the frequencies
            freq_coord = (Z[freq_cols] * sqrt(col_w[freq_cols])).T.dot((self.svd_.U[:,:ncp].T * row_w.to_numpy()).T)
            freq_coord.columns = self.eig_.index[:ncp]
            #convert to ordered dictionary
            freq_ = {"coord":freq_coord, "cos2" : freq_coord**2}
            #convert to namedtuple
            self.freq_ = namedtuple("freq",freq_.keys())(*freq_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #Data preparation
            Z_ind_sup = DataFrame(index=ind_sup_label,columns=Z.columns).astype(float)
            for g, cols in group_dict.items():
                if self.type_group[name_group.index(g)] in ("c","s"):
                    Z_ind_sup[cols] = (X_ind_sup[cols] - model[g].call_.center)/model[g].call_.scale
                elif self.type_group[name_group.index(g)] == "n":
                    dummies_cols = model[g].call_.dummies.columns
                    dummies_ind_sup = disjunctive(X_ind_sup[cols],cols=dummies_cols)
                    Z_ind_sup[dummies_cols] = (dummies_ind_sup - model[g].call_.center)/model[g].call_.scale
                elif self.type_group[name_group.index(g)] == "m":
                    #split X
                    split_Xcols_ind_sup = splitmix(X_ind_sup[cols])
                    Xcols_ind_sup_quanti, Xcols_ind_sup_quali, n_ind_sup_quanti, n_ind_sup_quali = split_Xcols_ind_sup.quanti, split_Xcols_ind_sup.quali, split_Xcols_ind_sup.k1, split_Xcols_ind_sup.k2
                    #initialization
                    Xcols_ind_sup = None
                    if n_ind_sup_quanti > 0:
                        if model[g].call_.k1 != n_ind_sup_quanti:
                            raise TypeError("The number of continuous variables must be the same")
                        Xcols_ind_sup = concat_empty(Xcols_ind_sup,Xcols_ind_sup_quanti,axis=1)
                    if n_ind_sup_quali > 0:
                        if model[g].call_.k2 != n_ind_sup_quali:
                            raise TypeError("The number of categorical variables must be the same")
                        Xcols_ind_sup = concat_empty(Xcols_ind_sup,disjunctive(X=Xcols_ind_sup_quali,cols=model[g].call_.dummies.columns),axis=1)
                    Z_ind_sup[Xcols_ind_sup.columns] = (Xcols_ind_sup - model[g].call_.center)/model[g].call_.scale

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
                        Z_ind_sup[cols] = (((P_row_sup[cols]/col_m[cols]).T - B_row_sup)/row_sup_m).T.replace([nan,inf,-inf],1e-15)
            
            #partial scores of the supplementary individuals
            ind_sup_coord_partiel = None
            for g, cols in columns_dict.items():
                nbcol = min(ncp,mncp[g])
                # partial coordinates
                coord_partiel = (Z_ind_sup[cols] * sqrt(col_w[cols])).dot((self.svd_.V[columns_index[g],:nbcol].T * model[g].call_.col_w.to_numpy()).T)
                # set index and columns
                coord_partiel.columns, coord_partiel.index = self.eig_.index[:nbcol], [f"{x}.{g}" for x in ind_sup_label]
                # concatenate
                ind_sup_coord_partiel = concat_empty(ind_sup_coord_partiel,coord_partiel,axis=0)
            #convert to dictionary
            ind_sup_ = {"coord_partiel" : ind_sup_coord_partiel}
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X

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