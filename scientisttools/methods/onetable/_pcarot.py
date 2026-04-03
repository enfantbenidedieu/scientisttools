# -*- coding: utf-8 -*-
from numpy import linalg, cumsum,c_,nan, sqrt
from collections import namedtuple, OrderedDict
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from ..functions.cov2corr import cov2corr
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import predict_first_check
from ..others._varimax import varimax
from ..others._disjunctive import disjunctive

class PCArot(TransformerMixin,BaseEstimator):
    """
    Varimax rotation in Principal Component Analysis (PCArot)
    
    Performs varimax rotation in Principal Component Analysis (PCArot).

    Parameters
    ----------
    ncp : int, default = 2
        The number of rotated principal components.

    normalize : bool, default = True
        To perform Kaiser normalization and de-normalization prior to and following rotation. Used for 'varimax' and 'promax' rotations.
        If ``None``, default for 'promax' is ``False``, and default for 'varimax' is ``True``.
        
    max_iter : int, optional, default = 1000
        The maximum number of iterations.
        
    tol : float, optional, default = 1e-5
        The convergence threshold.

    Returns
    -------
    call_ : call
        An object with the following attributes

        Xtot : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup + n_quanti_sup + n_quali_sup)
            Input data.
        X : DataFrame of shape (n_rows, n_columns)
            Active data.
        Xcod : DataFrame of shape (n_rows, n_columns)
            Recoded data.
        Z : DataFrame of shape (n_rows, n_columns) 
            Standardized data.
        bary : None or DataFrameof shape (n_groups, n_columns)
            Barycenter of rows points.
        tab : DataFrame of shape (n_rows, n_columns) or (n_groups, n_columns)
            Data used for GSVD.
        x_center : Series of shape (n_columns,)
            The columns weighted average.
        x_scale : Series of shape (n_columns)
            The columns weighted standard deviation.
        center : Series of shape (n_columns,)
            The variables weighted average.
        scale : Series of shape (n_columns,)
            The variables standard deviation:

            - If `scale_unit = True`, then standard deviation are computed using variables weighted standard deviation
            - If `scale_unit = False`, then standard deviation are a vector of ones with length number of variables.
        ind_w : Series of shape (n_rows,) 
            The individuals weights.
        row_w : Series of shape (n_rows,) or (n_groups,)
            The rows weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        ncp : int
            The number of components kepted.
        features : None, list
            The names of the explanatory variables (instrumental variables) or partial variables.
        group : None, list
            The name of the group variables used for between/within - class analysis.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).
        zcod : DataFrame of shape (n_rows, n_features), optional
            The standardized features data.
        z_center : Series of shape (n_feature_quanti_var,), optional
            The weighted average of features variables.
        z_scale : Series of shape (n_feature_quanti_var,), optional
            The weighted standard deviation of features variables
        obj : class
            An object of class :class:`scientisttools.PCA`.

    eig_ : DataFrame of shape (ncp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    ind_ : ind
        An object containing all the results for the active individuals, with the following attributes:

        coord : DataFrame of shape (n_rows, ncp)
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

    quanti_var_sup_ : quanti_var_sup
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_sup, ncp)
            The coordinates of the supplementary quantitative variables.

    rotmat_ : DataFrame of shape (n_components, n_components)
        The rotation matrix and factor correlations matrix.

    See also
    --------
    

    Examples
    --------
    >>> from scientisttools.datasets import load_dataset
    >>> from scientisttools import PCA, PCArot
    >>> autos2006 = load_dataset("autos2006")
    >>> pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> pca.fit(autos2006)
    PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> pcarot = PCArot()
    >>> pcarot.fit(pca)
    PCArot()
    """
    def __init__(
            self, ncp=2, normalize=True, max_iter=1000, tol=1e-5
    ):
        self.ncp = ncp
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,obj,y=None):
        """
        Fit the model to ``obj``

        Parameters
        ----------
        obj : class 
            An object of class :class:`~scientisttools.PCA`.

        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check max_iter is an integer
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.max_iter < 0:
            raise ValueError("'max_iter' must be equal to or greater than 0.")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if obj is an object of class PCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.__class__.__name__ != "PCA":
            raise ValueError("`obj` must be an object of class PCA")
        
        #set number of columns and maximum number of components
        n_cols, max_ncp = obj.quanti_var_.coord.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncp is None:
            ncp = int(max_ncp)
        elif not isinstance(self.ncp,int):
            raise TypeError("'ncp' must be an integer.")
        elif self.ncp < 1:
            raise ValueError("'ncp' must be equal or greater than 1.")
        else:
            ncp = int(min(self.ncp,max_ncp))

        #store call informations
        call_ = {**obj.call_._asdict(), **OrderedDict(obj=obj,ncp=ncp)}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of quantitative variables
        quanti_var_coord, self.rotmat_ = varimax(obj.quanti_var_.coord.iloc[:,:ncp],normalize=self.normalize,max_iter=self.max_iter,tol=self.tol) 
        #convert to ordered dictionary
        quanti_var_ = OrderedDict(coord=quanti_var_coord)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
        
        #sum of squared loadings
        ss_loadings = ((quanti_var_coord**2).T * obj.call_.col_w).sum(axis=1)
        proportion = 100*ss_loadings/n_cols
        #convert to DataFrame
        self.eig_ = DataFrame(c_[ss_loadings,proportion,cumsum(proportion)],columns=["Eigenvalue","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(ncp)])
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals: coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = obj.ind_.coord.iloc[:,:ncp].dot(self.rotmat_.values)
        ind_coord.columns = self.eig_.index[:ncp]
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "ind_sup_"):
            #coordinates for supplementary individuals after rotation
            ind_sup_coord = obj.ind_sup_.coord.iloc[:,:ncp].dot(self.rotmat_.values)
            ind_sup_coord.columns = self.eig_.index[:ncp]
            #convert to dictionary
            ind_sup_ = OrderedDict(coord=ind_sup_coord)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quanti_var_sup_"):
            #coordinates for supplementary quantitative variables after rotation
            quanti_var_sup_coord = obj.quanti_var_sup_.coord.iloc[:,:ncp].dot(self.rotmat_.values)
            quanti_var_sup_coord.columns = self.eig_.index[:ncp]
            #convert to dictionary
            quanti_var_sup_ = OrderedDict(coord=quanti_var_sup_coord)
            #convert to namedtuple
            self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary levels and 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "levels_sup_"):
            X_quali_var_sup = obj.call_.Xtot.loc[:,obj.quali_var_sup_.coord.index]
            if hasattr(obj, "ind_sup_"): 
                X_quali_var_sup = X_quali_var_sup.drop(index=obj.ind_sup_.coord.index)
            n_rows = X_quali_var_sup.shape[0]
            #coordinates for supplementary levels after rotation
            levels_sup_coord = obj.levels_sup_.coord.iloc[:,:ncp].dot(self.rotmat_.values)
            levels_sup_coord.columns = self.eig_.index[:ncp]
            #proportion for supplementary levels
            p_k_sup = (disjunctive(X_quali_var_sup).T * self.call_.ind_w).sum(axis=1)
            #vtest for the supplementary levels
            levels_sup_vtest = (levels_sup_coord.T * sqrt((n_rows-1)/((1/p_k_sup) - 1))).T/sqrt(ss_loadings[:self.call_.ncp])
            #convert to dictionary
            levels_sup_ = OrderedDict(coord=levels_sup_coord,vtest=levels_sup_vtest)
            #convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

            #coordinates for the supplementary qualitative variables - Eta-squared
            quali_var_sup_coord = func_eta2(X=ind_coord,by=X_quali_var_sup,w=self.call_.ind_w,excl=None)
            #convert to ordered dictionary
            quali_var_sup_ = OrderedDict(coord=quali_var_sup_coord)
            #convert to namedtuple
            self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())

        return self
    
    def fit_transform(self,obj,y=None):
        """
        Fit the model with ``obj`` and apply the dimensionality reduction

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`.

        y : None
            y is ignored.
            
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(obj)
        return self.ind_.coord
        
    def transform(self,X):
        """
        Apply the dimensionality reduction on ``X``

        ``X`` is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : Dataframe of shape (n_samples, n_components)
            Projection of ``X`` in the principal components.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #prediction input check
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = predict_first_check(self,X)

        #standardisation: z_ik = (x_ik - m_k)/s_k
        Z = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1)
        #apply transition relation
        coord = Z.mul(self.call_.col_w,axis=1).dot(self.call_.obj.svd_.V[:,:self.call_.ncp]).dot(self.rotmat_.values)
        coord.columns = self.eig_.index[:self.call_.ncp]
        return coord
    
def statsPCArot(
        obj
):
    """
    Statistics with varimax in Principal Component Analysis

    Performs statistics with varimax in principal component analysis

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.PCArot`.

    Returns
    -------
    result : statPCArotResult
        A object with the following attributes

        corr_ : corr
            An object containing all the results for the correlation with the following attributes:  

            corrcoef: DataFrame of shape (n_columns, n_columns) 
                The pearson correlation coefficient matrix.
            pcorrcoef: DataFrame of shape (n_columns, n_columns) 
                The partial pearson correlation coefficient matrix
            reconst: DataFrame of shape (n_columns, n_columns) 
                The reconstitution pearson correlation coefficient matrix after rotation
            residual: DataFrame of shape (n_columns, n_columns) 
                The residual correlation matrix after rotation

        others_ : others
            An object with the following attributes:  

            vaccounted: DataFrame of shape (6, n_components)
                The variance accounted

            explained_variance: DataFrame of shape (n_components, 3)
                The explained variance.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class PCArot
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PCArot": 
        raise TypeError("'obj' must be an object of class PCArot")

    #set number of rows and columns
    n_cols, ncp = obj.quanti_var_.coord.shape
    colnames = obj.quanti_var_.coord.index

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #correlation matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #covariance/correlation of Z and reconst
    M = obj.call_.Z.mul(obj.call_.ind_w,axis=0).T.dot(obj.call_.Z)
    #inverse of M
    try: 
        inv_M = DataFrame(linalg.inv(M),index=colnames,columns=colnames)
    except linalg.LinAlgError: 
        inv_M = DataFrame(linalg.pinv(M),index=colnames,columns=colnames)
    #weighted partial correlation matrix and reconst
    partial_M, reconst_M = cov2corr(inv_M).mul(-1), (obj.quanti_var_.coord.T * obj.call_.col_w).T.dot(obj.quanti_var_.coord.T)
    for c in partial_M.columns:
        partial_M.loc[c,c] = 1
    #residual covariance/correlation
    resid_M = M - reconst_M.values
    for c in resid_M.columns:
        resid_M.loc[c,c] = nan
    #convert to ordered dictionary
    corr_ = OrderedDict(corrcoef=M,pcorrcoef=partial_M,reconst=reconst_M,resid=resid_M)
    #convert to namedtuple
    corr_ = namedtuple("corr",corr_.keys())(*corr_.values())

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #variance accounted
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #sum of squared loadings
    ss_loadings = ((obj.quanti_var_.coord**2).T * obj.call_.col_w).sum(axis=1)
    #proportion
    prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
    #convert to DataFrame
    vaccounted = DataFrame(c_[obj.call_.obj.eig_.iloc[:ncp,0],ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],index = [f"Dim{x+1}" for x in range(ncp)],
                            columns=["Eigenvalue","SS loadings","Proportion Var","Cumulative Var","Proportion Explained","Cumulative Proportion"]).T
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##others informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #fidélité des facteurs - variance of the scores - R2
    r2_score = obj.quanti_var_.coord.mul(inv_M.dot(obj.quanti_var_.coord)).sum(axis=0)
    #Variance explained by each factor
    explained_variance = DataFrame(c_[obj.call_.obj.eig_.iloc[:obj.call_.ncp,0],ss_loadings,r2_score],index=[f"Dim{x+1}" for x in range(obj.call_.ncp)],columns=["Weighted","Unweighted","R2"])

    #convert to ordered dictionary
    others_ = OrderedDict(vaccounted=vaccounted,explained_variance=explained_variance)
    #convert to namedtuple
    others_ = namedtuple("others",others_.keys())(*others_.values())
    return namedtuple("statsPCArotResult",["corr_","others_"])(corr_,others_)