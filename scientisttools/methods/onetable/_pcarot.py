# -*- coding: utf-8 -*-
from numpy import cumsum,c_,sqrt
from collections import namedtuple
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
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
        To perform Kaiser normalization and de-normalization prior to and following rotation.
        
    max_iter : int, optional, default = 1000
        The maximum number of iterations.
        
    tol : float, optional, default = 1e-5
        The convergence threshold.

    Attributes
    ----------
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
            The variables standard deviation.
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

    ind_sup_ : ind_sup, optional
        An object containing all the results for the supplementary individuals, with the following attributes:

        coord : DataFrame of shape (n_rows_plus, ncp)
            The coordinates of the supplementary individuals.

    levels_sup_ : levels_sup, optional
        An object containing all the results for the supplementary levels, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary levels.
        vtest : DataFrame of shape (n_levels, ncp)
            The value-test (which is a criterion with a Normal distribution) of the supplementary levels.

    quali_var_sup_ : quali_var_sup, optional
        An object containing all the results for the supplementary qualitative variables, with the following attributes:

        coord : DataFrame of shape (n_levels, ncp)
            The coordinates of the supplementary qualitative variables. The squared correlation ratio of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension

    quanti_var_ : quanti_var
        An object containing all the results for the active variables, with the following attributes:

        coord : DataFrame of shape (n_columns, ncp)
            The coordinates of the variables.

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing all the results for the supplementary quantitative variables, with the following attributes:
        
        coord : DataFrame of shape (n_quanti_sup, ncp)
            The coordinates of the supplementary quantitative variables.

    rotmat_ : DataFrame of shape (ncp, ncp)
        The rotation matrix and factor correlations matrix.

    See Also
    --------
    save : Print results for general factor analysis model in an Excel sheet
    sprintf : Print the analysis results
    summary : Printing summaries of general factor analysis model

    Examples
    --------
    >>> from scientisttools.datasets import load_dataset
    >>> from scientisttools import PCA, PCArot
    >>> pca = PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> pca.fit(autos2006)
    PCA(ind_sup=(18,19),sup_var=(6,7,8))
    >>> pcarot = PCArot()
    >>> pcarot.fit(pca)
    PCArot()
    """
    def __init__(
            self, 
            ncp = 2, 
            normalize = True, 
            max_iter = 1000, 
            tol = 1e-5
    ):
        self.ncp = ncp
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,obj,y=None):
        """Fit the model to obj

        Parameters
        ----------
        obj : class 
            An object of class :class:`~scientisttools.PCA`.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check max_iter is an integer
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.max_iter < 0:
            raise ValueError("max_iter must be positive.")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if obj is an object of class PCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.__class__.__name__ != "PCA":
            raise ValueError("obj must be an object of class PCA")
        
        #set number of columns and maximum number of components
        n_cols, rank = obj.quanti_var_.coord.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncp is None:
            ncp = rank
        elif self.ncp < 1:
            raise ValueError("ncp must be strictly positive.")
        else:
            ncp = min(self.ncp,rank)

        #store call informations
        call_ = {**obj.call_._asdict(), **{"obj": obj, "ncp": ncp}}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of quantitative variables
        quanti_var_coord, self.rotmat_ = varimax(obj.quanti_var_.coord.iloc[:,:ncp],normalize=self.normalize,max_iter=self.max_iter,tol=self.tol) 
        #convert to ordered dictionary
        quanti_var_ = {"coord": quanti_var_coord}
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
        ind_coord = obj.ind_.coord.iloc[:,:ncp].dot(self.rotmat_.to_numpy())
        ind_coord.columns = self.eig_.index[:ncp]
        #convert to ordered dictionary
        ind_ = {"coord": ind_coord}
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "ind_sup_"):
            #coordinates for supplementary individuals after rotation
            ind_sup_coord = obj.ind_sup_.coord.iloc[:,:ncp].dot(self.rotmat_.to_numpy())
            ind_sup_coord.columns = self.eig_.index[:ncp]
            #convert to dictionary
            ind_sup_ = {"coord": ind_sup_coord}
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quanti_var_sup_"):
            #coordinates for supplementary quantitative variables after rotation
            quanti_var_sup_coord = obj.quanti_var_sup_.coord.iloc[:,:ncp].dot(self.rotmat_.to_numpy())
            quanti_var_sup_coord.columns = self.eig_.index[:ncp]
            #convert to dictionary
            quanti_var_sup_ = {"coord": quanti_var_sup_coord}
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
            levels_sup_coord = obj.levels_sup_.coord.iloc[:,:ncp].dot(self.rotmat_.to_numpy())
            levels_sup_coord.columns = self.eig_.index[:ncp]
            #proportion for supplementary levels
            p_k_sup = (disjunctive(X_quali_var_sup).T * self.call_.ind_w).sum(axis=1)
            #vtest for the supplementary levels
            levels_sup_vtest = (levels_sup_coord.T * sqrt((n_rows-1)/((1/p_k_sup) - 1))).T/sqrt(ss_loadings[:self.call_.ncp])
            #convert to dictionary
            levels_sup_ = {"coord": levels_sup_coord, "vtest": levels_sup_vtest}
            #convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

            #coordinates for the supplementary qualitative variables - Eta-squared
            quali_var_sup_coord = func_eta2(X=ind_coord,by=X_quali_var_sup,w=self.call_.ind_w,excl=None)
            #convert to ordered dictionary
            quali_var_sup_ = {"coord": quali_var_sup_coord}
            #convert to namedtuple
            self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())

        return self
    
    def fit_transform(self,obj,y=None):
        """Fit the model with obj and apply the dimensionality reduction

        Parameters
        ----------
        obj : class
            An object of class :class:`~scientisttools.PCA`.

        y : Ignored
            Ignored.
            
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(obj)
        return self.ind_.coord
        
    def transform(self,X):
        """Apply the dimensionality reduction on X

        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : Dataframe of shape (n_samples, ncp)
            Projection of X in the principal components.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # prediction input check
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = predict_first_check(self,X)

        # apply transition relation
        coord = X.iloc[:,:self.call_.ncp].dot(self.rotmat_.to_numpy())
        coord.columns = self.eig_.index[:self.call_.ncp]
        return coord