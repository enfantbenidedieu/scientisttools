# -*- coding: utf-8 -*-
from numpy import linalg, cumsum, c_, sqrt
from pandas import DataFrame
from collections import namedtuple, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.func_eta2 import func_eta2
from ..others._varimax import varimax
from ..others._disjunctive import disjunctive

class FArot(BaseEstimator,TransformerMixin):
    """
    Varimax rotation in Factor Analysis (FArot)
    
    Performs varimax rotation in Iterative and Non Iterative Principal Factor Analysis (PFA), Non Iterative Harris Component Analysis (HCA).

    Parameters
    ----------
    ncp : int, default = 2
        The number of rotated principal factors.

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
        obj : class
            An object of class FA.

    eig_ : DataFrame of shape (ncp, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

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

    [6] wikipedia (en), <Exploratory factor analysis https://en.wikipedia.org/wiki/Exploratory_factor_analysis>_.

    [7] datalab, <Exploratory factor analysis https://datatab.fr/tutorial/exploratory-factor-analysis>_.

    [8] Manuel Jamovi, <Analyse factorielle https://jmeunierp8.github.io/ManuelJamovi/s15.html>_.

    [9] SAS, <Factor analysis - SAS annotated output https://stats.oarc.ucla.edu/sas/output/factor-analysis/>_.

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
    >>> from scientisttools import FA, FArot
    >>> #non-iterative principal factor analysis (NIPFA)
    >>> clf1 = FA(warn_message=False)
    >>> clf1.fit(beer)
    FA(max_iter=0,warn_message=False)
    >>> clf = FArot()
    >>> clf.fit(clf1)
    FArot()
    >>> #iterative principal factor analysis (IPFA)
    >>> clf1 = FA(max_iter=50,warn_message=False)
    >>> clf1.fit(beer)
    FA(warn_message=False)
    >>> clf = FArot()
    >>> clf.fit(clf1)
    FArot()
    >>> #harris component analysis (HCA)
    >>> clf1 = FA(method="harris",warn_message=False)
    >>> clf1.fit(beer)
    FA(method="harris",warn_message=False)
    >>> clf = FArot()
    >>> clf.fit(clf1)
    FArot()
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
            An object of class :class:`~scientisttools.FA`.

        y : None
            y is ignored.

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
        #check if obj is an object of class FA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.__class__.__name__ != "FA": 
            raise ValueError("`obj` must be an object of class FA")
        
        #set scale unit
        self.scale_unit = obj.scale_unit
        #set number of columns and maximum number of components kepted
        n_cols, maxncp = obj.quanti_var_.coord.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncp is None: 
            ncp = int(maxncp)
        elif not isinstance(self.ncp,int): 
            raise TypeError("'ncp' must be an integer.")
        elif self.ncp < 1: 
            raise ValueError("'ncp' must be equal or greater than 1.")
        else: 
            ncp = int(min(self.ncp,maxncp))

        #store call informations
        call_ = {**obj.call_._asdict(), **OrderedDict(obj=obj,ncp=ncp)}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of loadings
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of loading
        quanti_var_coord, self.rotmat_ = varimax(obj.quanti_var_.coord.iloc[:,:ncp],normalize=self.normalize,max_iter=self.max_iter,tol=self.tol)
        #factor score : coefficient of projection
        self.coef_ = DataFrame(linalg.inv(self.call_.M).dot(quanti_var_coord),columns = [f"Dim{x+1}" for x in range(ncp)],index=self.call_.X.columns)
        #contribution of variables
        den = ((self.coef_**2).T * self.call_.col_w).sum(axis=1)
        quanti_var_ctr = 100* ((self.coef_**2).T * self.call_.col_w).T/den   
        #infos for quantitative variables
        quanti_var_infos = obj.quanti_var_.infos
        #update final communality
        quanti_var_infos["Final"] = quanti_var_coord.pow(2).mul(obj.call_.col_w,axis=0).sum(axis=1) 
        #convert to ordered citionary
        quanti_var_ = OrderedDict(coord=quanti_var_coord, contrib=quanti_var_ctr,infos=quanti_var_infos)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #sum of squared loadings
        ss_loadings = ((quanti_var_coord**2).T*obj.call_.col_w).sum(axis=1)
        #proportion
        proportion = 100*ss_loadings/n_cols
        #convert to DataFrame
        self.eig_ = DataFrame(c_[ss_loadings,proportion,cumsum(proportion)],columns=["Eigenvalue","Proportion (%)","Cumulative (%)"],index = [f"Dim{x+1}" for x in range(ncp)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: coordinates
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
            p_k_sup = (disjunctive(X_quali_var_sup).T * self.call_.row_w).sum(axis=1)
            #vtest for the supplementary levels
            levels_sup_vtest = (levels_sup_coord.T * sqrt((n_rows - 1)/((1/p_k_sup) - 1))).T/sqrt(ss_loadings[:self.call_.ncp])
            #convert to dictionary
            levels_sup_ = OrderedDict(coord=levels_sup_coord,vtest=levels_sup_vtest)
            #convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

            #coordinates for the supplementary qualitative variables - Eta-squared
            quali_var_sup_coord = func_eta2(X=self.ind_.coord,by=X_quali_var_sup,w=self.call_.row_w,excl=None)
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
            An object of class :class:`~scientisttools.FA`.

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
        
        X is projected on the principal factor previously extracted from a training set.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : Dataframe of shape (n_samples, ncp)
            Projection of ``X`` in the principal factor.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #apply transition relation
        coord = self.call_.obj.transform(X).iloc[:,:self.call_.ncp].dot(self.rotmat_.values)
        coord.columns = self.eig_.index[:self.call_.ncp]
        return coord