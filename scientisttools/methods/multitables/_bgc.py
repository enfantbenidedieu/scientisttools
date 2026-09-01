# -*- coding: utf-8 -*-
from numpy import ones, array, repeat, ndarray, linalg, diff, insert, cumsum, c_,nan, diag, sum,sqrt
from pandas import DataFrame, Series, concat, CategoricalDtype
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# intern functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd
from ..functions.spca import sPCA
from ..functions.rvstats import RVstats
from ..functions.utils import check_is_bool, is_all_numeric_dtype, is_all_object_or_category_dtype, check_is_dataframe
from ..others._disjunctive import disjunctive

class BGC(BaseEstimator,TransformerMixin):
    """
    Between Group Comparison (BGC)

    Performns Between Group Comparison (BGC) with supplementary individuals.

    Parameters
    ----------
    scale_unit : bool, default = True
        If True, then the data are scaled to unit variance.

    ncp : int, default = 2
        The number of dimensions kept in the results.

    sncp : int, default = None
        The number of dimensions kept in separate principal component analysis (sPCA). If None, then sncp is equal 
        to :math:`min(K-1,p)` where p is the number of columns and K the number of groups.

    group : int, str
        The indexe or name of the categorical variable which allows to make the group of individuals.

    row_w : 1d array-like of shape (n_samples,), default = None
        An optional individuals weights. The weights are given only for the active individuals.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters, with the following attributes:

        Xtot : DataFrame of shape (n_samples + n_samples_sup, n_columns)
            Input data.
        X : DataFrame of shape (n_samples, n_columns)
            Active data.
        x : DataFrame of shape (n_samples, n_columns - 1)
            The Data
        y : Series of shape (n_samples,)
            The vector of factors associated with group structure
        Xcod : DataFrame of shape (n_samples, n_columns)
            Recoded data.
        dummies : DataFrame of shape (n_samples, n_levels)
            Disjunctive table.
        M : DataFrame of shape (n_groups, n_levels)
            The 1-proportion of levels associated to each group.
        Zcod : DataFrame of shape (n_samples, n_columns)
            The concatenated standardized data
        Z : DataFrame of shape (n_samples, n_columns) 
            Standardized data.
        W : DataFrame of shape (n_columns, n_columns)
            The compromise loadings.
        center : DataFrame of shape (n_groups, n_columns)
            The concatenated variables weighted average.
        scale : DataFrame of shape (n_groups, n_columns)
            The concatenate variables standard deviation.
        z_center : Series of shape (n_columns,)
            The weighted average of concatenate standardized data.
        z_scale : Series of shape (n_columns,)
            The weighted standard deviation of concatenate standardized data.
        ncp : int
            The number of dimensions kept in the results.
        sncp : int 
            The number of dimensions kept in separate principal component analysis.
        row_w : Series of shape (n_samples,)
            The individuals weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        group : list
            The name of the group variables used to make the group of individuals.
        ind_sup : None, list, default = None
            The names of the supplementary individuals.

    eig_ : DataFrame of shape (rank, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    evd_ : evdResult
        An object containing all the results for the eigen value decomposition (EVD), with the following attributes:

        V : 2d numpy array of shape (n_columns, rank)
            The eigen vectors.
        d : 1d numpy array of shape (rank,)
            The eigen values.
        vs : 1d numpy array of shape (rank,)
            The singular values, 
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.
    
    group_ : group
        An object containing all the results for the groups, with the following attributes:

        traceRV : DataFrame of shape (n_groups, n_groups)
            The trace RV between groups.
        RV : DataFrame of shape (n_groups, n_groups)
            The RV coefficient between groups.
        eig : DataFrame of shape (rank_rv, 4)
            The eigenvalue of RV matrix, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.
        coord : DataFrame of shape (n_groups, n_groups)
            The coordinates of the groups.
        infos : DataFrame of shape (n_groups, 3)
            Additionals informations (weight, inertia and percentage of inertia) of the groups.
        lambd : DataFrame of shape (n_groups, ncp)
            The specific variances of groups.
        expl_var : DataFrame of shape (n_groups, ncp)
            Percentages of total variance recovered associated with each dimension.

    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_samples, ncp)
            The coordinates of the individuals.

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_samples_plus, ncp)
            The coordinates of the supplementary individuals.

    quanti_var_ : quanti_var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.

    separate_analyses_ : dict
        The results for the separates Principal Component Analysis (sPCA).

    References
    ----------
    [1] W. J. Krzanowski (1979). Between-groups comparison of principal components, *Journal of the American Statistical Association*, 74, 703-707. `https://doi.org/10.2307/2286995 <https://doi.org/10.2307/2286995>`_.
    
    [2] A. Eslami, E. M. Qannari, A. Kohler and S. Bougeard (2013). `General overview of methods of analysis of multi-group datasets <https://editions-rnti.fr/render_pdf.php?p=1001883>`_, *Revue des Nouvelles Technologies de l'Information*, 25, 108-123.
    
    [3] A. Eslami, E. M. Qannari, A. Kohler and S. Bougeard (2013). `Analyses factorielles de donnees structurees en groupes d'individus <https://www.numdam.org/item/JSFS_2013__154_3_44_0.pdf>`_, *Journal de la Societe Francaise de Statistique*, 154(3), 44-57.
    
    See also
    --------
    save : Print results for general factor analysis model in an Excel sheet.
    sprintf : Print the analysis results.
    summary : Printing summaries of general factor analysis model.

    Examples
    --------
    >>> from scientisttools.datasets import iris, housevotes84
    >>> from scientisttools import BGC
    >>> # between group comparison with continuous variables.
    >>> clf = BGC(group=4,scale_unit=True,ncp=2,ind_sup=[0,1,2,50,51,52,100,101,102])
    >>> clf.fit(iris)
    BGC(group=4,ind_sup=[0,1,2,50,51,52,100,101,102],ncp=2,scale_unit=True)
    >>> # between group comparison with categorical variables
    >>> clf = BGC(scale_unit=False,ncp=2,group=0,ind_sup=range(400,435))
    >>> clf.fit(housevotes84)
    BGC(group=0,ind_sup=range(400,435),ncp=2,scale_unit=False)
    """
    def __init__(
            self, 
            scale_unit = True, 
            ncp = 2, 
            sncp = None,  
            group = None, 
            row_w = None, 
            col_w = None, 
            ind_sup = None, 
            tol = 1e-7
    ):  
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.sncp = sncp
        self.group = group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.tol = tol

    def fit(self,X,y=None):
        """Fit the model to X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if scale_unit is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.scale_unit)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group validation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.group,(int,str)):
            raise TypeError("'group' must be either an objet of type int or str")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_label, ind_sup_label = get_sup_label(X=X, indexes=self.group, axis=1), get_sup_label(X=X,indexes=self.ind_sup,axis=0)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # between group comparison (BGC)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into x and y
        y, x = X[group_label[0]], X.drop(columns=group_label)

        #check if all columns are either continuous or categorical.
        if not (is_all_numeric_dtype(x) or is_all_object_or_category_dtype(x)):
            raise TypeError("Not applied to mixed data") 

        # unique element in y
        name_group = sorted(y.unique())
        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=name_group,ordered=True))

        #number of rows and number of columns
        n_rows, n_vars = x.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None:
            row_w = Series(ones(n_rows)/n_rows,index=x.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)):
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows:
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else:
            row_w = Series(array(self.row_w)/sum(self.row_w),index=x.index,name="weight")

        #set variables weights
        if self.col_w is None:
            var_w = Series(ones(n_vars),index=x.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)):
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_vars:
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_vars},).")
        else:
            var_w = Series(array(self.col_w),index=x.columns,name="weight")

        #group index
        group_dict = {k : y[y==k].index for k in name_group}
     
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set variables xcod - reorder 
        Xcod, col_w, dummies, M = x.copy(), var_w.copy(), None, None
        if is_all_object_or_category_dtype(x):
            # disjunctive table
            dummies = disjunctive(x)
            # transformation of the indicator variables
            M = concat(((1 - ((dummies.loc[r,:].T * row_w[r]/sum(row_w[r])).sum(axis=1))).to_frame(g) for g, r in group_dict.items()),axis=1).T
            # recode data
            Xcod = dummies*M.loc[y.to_numpy(),:].to_numpy()
            # columns weights for variable categories
            col_w = Series(repeat(var_w.to_numpy(),x.nunique().to_numpy()),index=dummies.columns,name="weight")
            
        # number of columns
        n_cols = Xcod.shape[1]
        
        # set number of components in separate principal component analysis
        if self.sncp is None:
            sncp = int(min(n_cols,len(name_group)-1))
        elif not isinstance(self.sncp,int):
            raise TypeError("sncp must be an integer")
        elif self.sncp < 1: 
            raise ValueError("sncp must be strictly positive")
        else: 
            sncp = self.sncp

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # separate principal component analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # separate principal component analysis
        model = sPCA(X=Xcod,y=y,scale_unit=self.scale_unit,ncp=sncp,row_w=row_w,col_w=col_w,tol=self.tol)
            
        # store separate analysis
        self.separate_analyses_ = model

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # standardized data
        Zcod = concat((model[g].call_.Z for g in list(model.keys())),axis=0,ignore_index=False).loc[y.index,:]
        #weighted average
        center = concat((model[g].call_.center.to_frame(g) for g in list(model.keys())),axis=1).T
        scale = concat((model[g].call_.scale.to_frame(g) for g in list(model.keys())),axis=1).T
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization according to normed principal components analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        z_center, z_scale = wmean(X=Zcod,w=row_w), wstd(X=Zcod,w=row_w)
        #standardization : z_ik = (x_ik - m_k)/s_k
        Z = (Zcod - z_center)/z_scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # compromise matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        W = DataFrame(sum([model[g].svd_.V[:,:model[g].svd_.ncp].dot(model[g].svd_.V[:,:model[g].svd_.ncp].T) for g in name_group],axis=0),
                      index=Xcod.columns,columns=Xcod.columns)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # eigen value decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # eigen value decomposition (=singular values decomposition of hermitian matrix)
        evd = linalg.svd(W,hermitian=True)
        # set maximum number of components
        rank = sum(evd[1]/evd[1][0] > self.tol)
        
        #set number of components
        if self.ncp is None:
            ncp = rank
        elif not isinstance(self.ncp,int):
            raise TypeError("ncp must be an integer")
        elif self.ncp < 1: 
            raise ValueError("ncp must be strictly positive")
        else: 
            ncp = int(min(self.ncp,rank))

        #Store call informations
        call_ = {"Xtot":Xtot,"X":X,"x":x,"y":y,"Xcod":Xcod,"dummies":dummies,"M":M,"Zcod":Zcod,"Z":Z,"W":W,
                 "center":center,"scale":scale,"z_center":z_center,"z_scale":z_scale,"ncp":ncp,"sncp": sncp,
                 "row_w":row_w,"var_w":var_w,"col_w":col_w,"group":group_label,"name_group":name_group,"ind_sup":ind_sup_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #convert to ordered dictionary
        evd_ = {"V":evd[0][:,:rank], "d":evd[1][:rank], "vs":sqrt(evd[1][:rank]), "rank":rank, "ncp":ncp}
        #convert to namedtuple
        self.evd_ = namedtuple("evdResult",evd_.keys())(*evd_.values())
    
        #eigen values informations
        eigvals = evd[1][:rank]
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],
                              index = [f"Dim{x+1}" for x in range(rank)])  

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables in compromises spaces
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compromise loadings - columns coordinates
        quanti_var_coord = DataFrame(self.evd_.V[:,:ncp]*self.evd_.vs[:ncp],index=Xcod.columns,columns=self.eig_.index[:self.ncp])
        #convert to ordered dictionary
        quanti_var_ = {"coord":quanti_var_coord}
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #groups informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_ = RVstats(model=model,tol=self.tol)
        # lambda - specific variances of group
        lambd =  concat((Series(diag(self.evd_.V[:,:ncp].T.dot(model[g].call_.Vb).dot(self.evd_.V[:,:ncp])),index=self.eig_.index[:ncp]).to_frame(g) for g in name_group),axis=1).T
        # explained variance
        expl_var = concat((100*lambd.loc[g,:]/sum(diag(model[g].call_.Vb)) for g in name_group),axis=1).T
        # add to dictionary
        group_ = {**group_, **{"lambd":lambd,"expl_var":expl_var}}
        #store all group informations
        self.group_ = namedtuple("group",group_.keys())(*group_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals in compromises spaces
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = (Z * col_w).dot(self.evd_.V[:,:ncp])
        ind_coord.columns = self.eig_.index[:ncp]
        #convert to ordered dictionary
        ind_ = {"coord":ind_coord}
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split in x and y
            y_ind_sup, X_ind_sup = X_ind_sup[group_label[0]], X_ind_sup.drop(columns=group_label)

            Xcod_ind_sup = X_ind_sup
            if is_all_object_or_category_dtype(X_ind_sup):
                Xcod_ind_sup = disjunctive(X_ind_sup,cols=dummies.columns) * M.loc[y_ind_sup.to_numpy(),:].to_numpy()
            
            #standardization
            Z_ind_sup = (((Xcod_ind_sup - center.loc[y_ind_sup.to_numpy(),:].to_numpy())/scale.loc[y_ind_sup.to_numpy(),:].to_numpy()) - z_center)/z_scale
            #coordinates for supplementary individuals
            ind_sup_coord = (Z_ind_sup * col_w).dot(self.evd_.V[:,:ncp])
            ind_sup_coord.columns = self.eig_.index[:ncp]
            #convert to ordered dictionary
            ind_sup_ = {"coord":ind_sup_coord}
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        return self
        
    def fit_transform(self,X,y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.
        
        y : Ignored
            Ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, ncp)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def transform(self,X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted 
        from a training set.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_samples, ncp)
            Projection of X in the first principal components, where ``n_samples`` 
            is the number of samples and ``ncp`` is the number of the components.
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

        #split in x and y
        y, X = X[self.call_.group[0]], X.drop(columns=self.call_.group)

        Xcod = X.copy()
        if is_all_object_or_category_dtype(X):
            Xcod = disjunctive(X,cols=self.call_.dummies.columns) * self.call_.M.loc[y.to_numpy(),:].to_numpy()
        
        #standardization
        Z = (((Xcod - self.call_.center.loc[y.to_numpy(),:].to_numpy())/self.call_.scale.loc[y.to_numpy(),:].to_numpy()) - self.call_.z_center)/self.call_.z_scale
        # coordinates for news individuals
        coord = (Z * self.call_.col_w).dot(self.evd_.V[:,:self.evd_.ncp])
        coord.columns = self.eig_.index[:self.evd_.ncp]
        return coord