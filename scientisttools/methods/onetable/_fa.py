# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, concat
from numpy import array, ndarray, ones, diag,linalg, fill_diagonal, insert, diff, nan, cumsum, c_, sqrt
from collections import namedtuple, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd, func_groupby, wcov
from ..functions.func_eta2 import func_eta2
from ..functions.utils import check_is_bool
from ..functions.func_predict import predict_first_check
from ..functions.cov2corr import cov2corr
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class FA(BaseEstimator,TransformerMixin):
    """
    Factor Analysis (FA)
    
    Performs Iterative and Non Iterative Principal Factor Analysis (PFA), Non Iterative Harris Component Analysis (HCA) with supplementary individuals.

    Parameters
    ----------
    method : str, default = "principal"
        A string indicating the factor method. Possible values are: 
        
        * 'principal' for principal (exploratory) factor analysis (PFA or EFA), 
        * 'harris' for harris component analysis (HCA).

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.

    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (quantitative and/or qualitative).

    min_error : float, default = 1e-3
        iterate until the change in communalities is less than min_error.

    max_iter : int, default = 50
        Maximum number of iterations for convergence.

    warn_message : bool, default = True
        Warn if number of components is too many.

    Returns
    -------
    call_ : call
        An object with the following attributes:

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        M : DataFrame of shape (n_columns, n_columns)
            Correlation/Covariance matrix.
        row_w : Series of shape (n_rows,) 
            The individuals weights.
        col_w : Series of shape (n_columns,)
            The variables weights.
        center : Series of shape (n_columns,)
            The variables weighted average.
        scale : Series of shape (n_columns,)
            The variables standard deviation:
            
            - If `scale_unit = True`, then standard deviation are computed using variables weighted standard deviation
            - If `scale_unit = False`, then standard deviation are a vector of ones with length number of variables.
        ncp : int
            The number of components kepted.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables.

    eig_ : DataFrame of shape (maxcp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    evd_ : evd
        An object containing the results of eigen values decomposition with the following attributes:

        eigenvalues : 1d array-like of shape
            The eigenvalues
        eigenvectors : 2d array-like of shape
            The eigenvectors

    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows,ncp)
            The coordinates of the individuals.

    ind_sup_ : ind_sup
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.

    levels_sup_ : levels_sup 
        An object containing all the results for the supplementary levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the supplementary levels.

    quali_var_sup_ : quali_var_sup 
        An object containing all the results for the supplementary qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary qualitative variables. The squared correlation ratio of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension

    quanti_var_ : quanti_var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.
        contrib : DataFrame of shape (n_columns, ncp)
            The relative contributions of the variables.
        infos : DataFrame of shape (n_columns, 4)
            Additionals informations (weight, prior communality, final communality) of the variables.

    quanti_var_sup_ : quanti_var_sup
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_sup, ncp)
            The coordinates of the supplementary quantitative variables.

    References
    ----------
    [1] Berger J-L (2021), Analyse factorielle exploratoire et analyse en composantes principales : guide pratique, hal-03436771v1

    [2] D. Suhr Diana (), Prinicpal Component Analysis vs. Exploratory Factor Analysis, University of Northern Colorado. See : https://support.sas.com/resources/papers/proceedings/proceedings/sugi30/203-30.pdf

    [3] Lawley, D.N., Maxwell, A.E. (1963), Factor Analysis as a Statistical Method, Butterworths Mathematical Texts, England

    [4] Marley W. Watkins (2018), Exploratory Factor Analysis : A guide to best practice, Journal of Black Psychology, Vol. 44(3) 219-246

    [5] Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0

    [6] wikipedia, https://en.wikipedia.org/wiki/Exploratory_factor_analysis

    [7] datalab, https://datatab.fr/tutorial/exploratory-factor-analysis

    [8] https://jmeunierp8.github.io/ManuelJamovi/s15.html

    [9] https://stats.oarc.ucla.edu/sas/output/factor-analysis/

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
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA
    >>> #non-iterative principal factor analysis (NIPFA)
    >>> clf = FA(warn_message=False)
    >>> clf.fit(beer)
    FA(warn_message=False)
    >>> #iterative principal factor analysis (IPFA)
    >>> clf = FA(max_iter=50,warn_message=False)
    >>> clf.fit(beer)
    FA(max_iter=50,warn_message=False)
    >>> #harris component analysis (HCA)
    >>> clf = FA(method="harris",warn_message=False)
    >>> clf.fit(beer)
    FA(method="harris",warn_message=False)
    """
    def __init__(
            self, method = "principal", scale_unit=True, ncp = 2,  row_w=None, col_w=None, ind_sup=None, sup_var=None, min_error = 1e-3, max_iter = 0, warn_message = True
    ):
        self.method = method
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
        self.min_error = min_error
        self.max_iter = max_iter
        self.warn_message = warn_message

    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : Dataframe of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check max_iter is an integer
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.max_iter < 0: 
            raise ValueError("'max_iter' must be equal to or greater than 0.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if method not equal to 'principal' and 'harris'
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method not in ("principal","harris"): 
            raise ValueError("'method' must be one either 'principal' or 'harris'.")
        
        #set harris 
        if self.method == "harris" and self.max_iter > 1: 
            raise ValueError("Harris Component Analysis is a non-iterative factor analysis approach")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if scale_unit is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.scale_unit)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X) 
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label, sup_var_label = get_sup_label(X=X,indexes=self.ind_sup,axis=0), get_sup_label(X=X,indexes=self.sup_var,axis=1)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary columns
        if self.sup_var is not None: 
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
            if self.ind_sup is not None: 
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

        #drop supplementary rows
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal Factor Analysis (PFA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #number of rows/columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None: 
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else: 
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
        
        #set variables weights
        if self.col_w is None: 
            col_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
        else: 
            col_w = Series(array(self.col_w),index=X.columns,name="weight")
 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and weighted standard deviation
        center, scale = wmean(X=X,w=row_w), wstd(X=X,w=row_w) if self.scale_unit else Series(ones(n_cols),index=X.columns,name="scale")
        #standardization: z_ik = (x_ik - m_k)/s_k
        Z = (X - center)/scale
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #covariance matrix of Z
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #covariance matrix
        M = (Z.T * row_w).dot(Z)

        #Init communality - Prior
        try: 
            init_comm = 1 - 1/diag(linalg.inv(M))
        except: 
            init_comm = 1 - 1/diag(linalg.pinv(M))

        #replace diagonal of correlation matrix with initial communality
        M_c = M.copy(deep=True)
        for i, c in enumerate(X.columns):
            M_c.loc[c,c] = init_comm[i]

        # Harris correlation matrix
        if self.method == "harris": 
            M_c = (M_c.T/sqrt(1-init_comm)).T/sqrt(1-init_comm)
            
        #eigen decomposition
        eigenvals, eigenvects = linalg.eigh(M_c)
        #sorted in descending order
        eigenvals, eigenvects = eigenvals[::-1], eigenvects[:,::-1]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        maxcp = int((eigenvals > 0).sum())
        if self.ncp is None: 
            ncp = maxcp
        elif not isinstance(self.ncp,int): 
            raise ValueError("'ncp' must be an integer.")
        elif self.ncp < 1: 
            raise ValueError("'ncp' must be equal or greater than 1.")
        else: 
            ncp = int(min(self.ncp,maxcp))

        #update according maximum number of components
        eigenvals, eigenvects = eigenvals[:maxcp], eigenvects[:,:maxcp]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables: - loadings
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables loadings
        quanti_var_coord = eigenvects[:,:ncp] * sqrt(eigenvals[:ncp])
        #apply harris correction
        if self.method == "harris": 
            quanti_var_coord = (quanti_var_coord.T * sqrt(1 - init_comm)).T

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #iterative 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        comm, comm_ls = sum(diag(M_c)), list()
        error = comm
        i = 0
        while error > self.min_error:
            eigenvals, eigenvects = linalg.eigh(M_c)
            #sorted
            eigenvals, eigenvects = eigenvals[::-1], eigenvects[:,::-1]

            maxcp = int((eigenvals > 0).sum())
            if self.ncp is None: 
                ncp = maxcp
            elif not isinstance(self.ncp,int): 
                raise ValueError("'ncp' must be an integer.")
            elif self.ncp < 1: 
                raise ValueError("'ncp' must be equal or greater than 1.")
            else: 
                ncp = int(min(self.ncp,maxcp))

            #update according maximum number of components
            eigenvals, eigenvects = eigenvals[:maxcp], eigenvects[:,:maxcp]
            #loadings
            quanti_var_coord = eigenvects[:,:ncp] * sqrt(eigenvals[:ncp])
            #apply harris correction
            if self.method == "harris": 
                quanti_var_coord = (quanti_var_coord.T * sqrt(1 - init_comm)).T
            #
            model = (quanti_var_coord.T * col_w.values).T.dot(quanti_var_coord.T)
            new_comm = diag(model)
            comm1 = sum(new_comm)
            M_c = M.copy()
            for i, c in enumerate(X.columns):
                M_c.loc[c,c] = new_comm[i]

            #harris correlation matrix
            if self.method == "harris": 
                M_c = (M_c.T/sqrt(1-new_comm)).T/sqrt(1-new_comm)

            error = abs(comm - comm1)
            comm = comm1
            comm_ls.append(comm1)
            i += 1
            if i >= self.max_iter:
                if self.warn_message: 
                    print("maximum iteration exceeded")
                error = 0

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,M=M,row_w=row_w,col_w=col_w,center=center,scale=scale,ncp=ncp,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set eigenvalues
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigenvalues
        eigvals = eigenvals
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals) 
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(len(eigvals))])
        
        #sorte eigen value decomposition
        self.evd_ = namedtuple("evd",["V","d"])(eigenvects,eigenvals)
       
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics of the variables: quanti_var_coord, contributions, coefficients of projections
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to DataFrame
        quanti_var_coord = DataFrame(quanti_var_coord,columns = [f"Dim{x+1}" for x in range(ncp)],index=X.columns)
        #factor score : coefficient of projection
        self.coef_ = DataFrame(linalg.inv(M).dot(quanti_var_coord),columns = [f"Dim{x+1}" for x in range(ncp)],index=X.columns)
        #contribution of variables
        den = ((self.coef_ **2).T * col_w).T.sum(axis=0)
        quanti_var_ctr = 100*((self.coef_ ** 2).T * col_w).T/den
        #infos for quantitative variables
        #final communality
        final_comm = ((quanti_var_coord **2).T * col_w).T.sum(axis=1)
        #communality
        quanti_var_infos = DataFrame(c_[col_w,init_comm,final_comm],columns=["Weight","Prior","Final"],index=Z.columns)
        #convert to ordered citionary
        quanti_var_ = OrderedDict(coord=quanti_var_coord, contrib=quanti_var_ctr,infos=quanti_var_infos)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for the individuals: coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals factor coordinates
        ind_coord = (Z * col_w).dot(self.coef_)
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Statistics for supplementary individuals                                      
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #supplementary individuals factor coordinates
            ind_sup_coord = (((X_ind_sup - center)/scale) * col_w).dot(self.coef_)
            #convert to ordered dictionary
            ind_sup_ = OrderedDict(coord=ind_sup_coord)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_var_sup, X_quali_var_sup, n_quanti_var_sup, n_quali_var_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2
            
            #statistics for supplementary quantitative variables
            if n_quanti_var_sup > 0:
                #compute weighted average and weighted standard deviation
                center_sup, scale_sup = wmean(X=X_quanti_var_sup,w=row_w), wstd(X=X_quanti_var_sup,w=row_w) if self.scale_unit else ones(n_quanti_var_sup)
                #standardization: z_ik = (x_ik - m_k)/s_k
                Z_quanti_var_sup = (X_quanti_var_sup - center_sup)/scale_sup
                #coordinates of supplementary quantitative variables
                quanti_var_sup_coord = wcov(concat((Z_quanti_var_sup,ind_coord),axis=1),w=row_w,ddof=0).iloc[:n_quanti_var_sup,n_quanti_var_sup:]
                #convert to ordered dictionary
                quanti_var_sup_ = OrderedDict(coord=quanti_var_sup_coord)
                #convert to namedtuple
                self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_var_sup > 0:
                #conditional mean - Barycenter of original data
                levels_sup_coord = func_groupby(X=self.ind_.coord,by=X_quali_var_sup,w=row_w,func="mean")
                #proportion of supplementary levels
                p_k_sup = (disjunctive(X_quali_var_sup).T * row_w).sum(axis=1)
                #vtest for the supplementary levels
                levels_sup_vtest = (levels_sup_coord.T * sqrt((n_rows-1)/((1/p_k_sup) - 1))).T/sqrt(self.evd_.d[:ncp])
                #convert to ordered dictionary
                levels_sup_ = OrderedDict(coord=levels_sup_coord,vtest=levels_sup_vtest)
                #convert to namedtuple
                self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

                #coordinates for the supplementary qualitative variables - Eta-squared
                quali_var_sup_coord = func_eta2(X=ind_coord,by=X_quali_var_sup,w=row_w,excl=None)
                #convert to ordered dictionary
                quali_var_sup_ = OrderedDict(coord=quali_var_sup_coord)
                #convert to namedtuple
                self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` is the number of samples and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, ncp)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
        
    def transform(self,X):
        """
        Apply the dimensionality reduction on ``X``
        
        X is projected on the principal factor previously extracted from a training set.

        Parameters
        ----------
        X : Dataframe of shape (n_rows, n_columns)
            New data, where ``n_rows`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : Dataframe of shape (n_rows, ncp)
            Projection of ``X`` in the principal factor.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #prediction input check
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = predict_first_check(self,X)
        return (((X - self.call_.center)/self.call_.scale) * self.call_.col_w).dot(self.coef_)
    
def statsFA(
        obj
):
    """
    Statistics with Factor Analysis

    Performs statistics with Factor Analysis.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.FA`.

    Returns
    -------
    result : statsFAResult
        An object with the following attributes:

        corr_ : corr
            An object with the following attributes:
            
            corrcoef : DataFrame of shape (n_columns, n_columns)
                Pearson correlation coefficients.
            pcorrcoef : DataFrame of shape (n_columns, n_columns)
                partial pearson correlation coefficients.
            reconst : DataFrame of shape (n_columns, n_columns)
                Correlation matrix estimated by the model.
            residual : DataFrame of shape (n_columns, n_columns)
                Residual correlations after the factor model is applied.
            
        others_ : others
            An object with the following attributes:

            vaccounted : DataFrame of shape (7, ncp)
                Variance acconted.
            explained_variance : DataFrame of shape (ncp, 3)
                Variance explained by each factor (weighted, unweighted) and R2, which is the multiple R-square between the factors and factor score estimates.
            communalities : float
                The communalities reflecting the total amount of common variance. They will exceed the communality (above) which is the model estimated common variance.
            inertia : float
                The total inertia.

    Examples
    --------
    >>> from scientisttools.datasets import beer
    >>> from scientisttools import FA, statsFA
    >>> clf = FA(ncp=2,warn_message=False)
    >>> clf.fit(beer)
    FA(ncp=2,warn_message=False)
    >>> stats = statsFA(clf)     
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class FA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("FA","FArot")): 
        raise TypeError("'obj' must be an object of class FA, FArot")

    #set number of columns and number of components kepted
    n_cols, ncp = obj.quanti_var_.coord.shape
    colnames = obj.quanti_var_.coord.index

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #correlation matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #inverse of M
    try: 
        inv_M = DataFrame(linalg.inv(obj.call_.M),index=colnames,columns=colnames)
    except linalg.LinAlgError: 
        inv_M = DataFrame(linalg.pinv(obj.call_.M),index=colnames,columns=colnames)
    #weighted partial correlation matrix and reconst covariance/correlation
    partial_M, reconst_M = -1*cov2corr(inv_M), obj.quanti_var_.coord.mul(obj.call_.col_w,axis=0).dot(obj.quanti_var_.coord.T)
    for c in partial_M.columns:
        partial_M.loc[c,c] = 1
    #residual covariance/correlation
    resid_M = obj.call_.M - reconst_M.values
    for c in resid_M.columns:
        resid_M.loc[c,c] = nan
    #convert to ordered dictionary
    corr_ = OrderedDict(corrcoef=obj.call_.M,pcorrcoef=partial_M,reconst=reconst_M,residual=resid_M)
    #convert to namedtuple
    corr_ = namedtuple("corr",corr_.keys())(*corr_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #variance accounted
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #eigen value decomposition of the original matrix
    eigvals = linalg.eigh(obj.call_.M)[0][::-1]
    #sum of squared loadings - common eigen values
    ss_loadings = ((obj.quanti_var_.coord**2).T * obj.call_.col_w).T.sum(axis=0)
    #proportion
    prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
    #convert to pd.DataFrame
    vaccounted = DataFrame(c_[eigvals[:ncp],obj.eig_.iloc[:ncp,0],ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],index = [f"Dim{x+1}" for x in range(ncp)],
                            columns=["Original","Common","SS loadings","Proportion Var (%)","Cumulative Var (%)","Proportion Explained (%)","Cumulative Proportion (%)"]).T
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##others statistics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fidélité des facteurs - variance of the scores - R2
    r2_score = (obj.quanti_var_.coord * obj.coef_.values).sum(axis=0)
    #Variance explained by each factor
    explained_variance = DataFrame(c_[obj.eig_.iloc[:ncp,0],ss_loadings,r2_score],index=[f"Dim{x+1}" for x in range(ncp)],columns=["Weighted","Unweighted","R2"])
    #total inertia and communalities
    inertia, communalities = sum(obj.quanti_var_.infos.iloc[:,1]), sum(obj.quanti_var_.infos.iloc[:,2])
    #convert to ordered dictionary
    others_ = OrderedDict(vaccounted=vaccounted,explained_variance=explained_variance,communalities=communalities,inertia=inertia)
    #convert to namedtuple
    others_ = namedtuple("others",others_.keys())(*others_.values())
    return namedtuple("statsFAResult",["corr_","others_"])(corr_,others_)