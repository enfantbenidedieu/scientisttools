# -*- coding: utf-8 -*-
from numpy import linalg, cumsum, sqrt, c_, tril,apply_along_axis, diag,array
from collections import namedtuple, OrderedDict
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from .functions.varimax import varimax
from .functions.utils import is_dataframe

class PCArot(BaseEstimator,TransformerMixin):
    """
    Varimax rotation in Principal Component Analysis (PCArot)
    ---------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs varimax rotation in Principal Component Analysis (PCArot)

    Usage
    -----
    ```
    >>> PCArot(n_components = None, normalize = True, max_iter = 1000, tol = 1e-5)
    ```

    Parameters
    ----------
    `n_components`: an integer indicating the number of rotated principal components (by default None)

    `normalize`: a boolean or None indicating whether to perform Kaiser normalization and de-normalization prior to and following rotation (default True). Used for 'varimax' and 'promax' rotations.
        If ``None``, default for 'promax' is ``False``, and default for 'varimax' is ``True``.
        
    `max_iter`: optional, an integer indicating the maximum number of iterations (by default 1000).
        
    `tol`: optional, a numeric indicating the convergence threshold (by default 1e-5).

    Attributes
    ----------
    `call_`:

    `svd_`: 

    `rotmat_`: a pandas DataFrame containing the rotation matrix and factor correlations matrix



    `vaccounted_`: a pandas DataFrame containing the variance accounted

    See Also
    --------
    `predictPCArot`, `supvarPCArot`, `get_pcarot_ind`, `get_pcarot_var`, `get_pcarot`, `summaryPCArot`, `fviz_pcarot_ind`, `fviz_pcarot_var`, `fviz_pcarot_biplot`

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, PCA, PCArot, summaryPCArot
    >>> autos2006 = load_dataset("autos2006")
    >>> res_pca = PCA(n_components=2,ind_sup=(18,19),sup_var=(6,7,8))
    >>> res_pca.fit(autos2006)
    >>> res_pcarot = PCArot(n_components=2)
    >>> res_pcarot.fit(res_pca)
    >>> summaryPCArot(res_pcarot)
    ```
    """
    def __init__(self,
                 n_components = None,
                 normalize = True,
                 max_iter = 1000,
                 tol = 1e-5):
        self.n_components = n_components
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,obj):
        """
        Fit the model to obj
        --------------------

        Parameters
        ----------
        `obj`: an object of class PCA

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself
        """ 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if self is an object of class PCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.model_ != "pca":
            raise ValueError("`obj` must be an object of class PCA.")
        
        #set number of columns
        n_cols, max_components = obj.var_.coord.shape

        #set number of components
        if self.n_components is None:
            n_components = int(max_components)
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))

        #store call informations
        call_ = OrderedDict(X=obj.call_.X,Z=obj.call_.Z,ind_weights=obj.call_.ind_weights,var_weights=obj.call_.var_weights,center=obj.call_.center,scale=obj.call_.scale,
                            standardize=obj.standardize,n_components=n_components,max_components=max_components,normalize = self.normalize, max_iter = self.max_iter, tol = self.tol)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of loading
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #varimax rotation of loading
        rot = varimax(loadings=obj.var_.coord.iloc[:,:n_components],normalize = self.normalize,max_iter = self.max_iter, tol = self.tol)
    
        #sum of squared loadings
        ss_loadings = rot.loadings.mul(obj.call_.var_weights,axis=0).pow(2).sum(axis=0)
        
        #update V
        V =  apply_along_axis(func1d=lambda x : x*sqrt(ss_loadings),axis=1,arr=linalg.inv(obj.corr_.corrcoef).dot(rot.loadings))

        #update generalized singular value decomposition (GSVD)
        self.svd_ = namedtuple("gsvdResult",["U","vs","V"])(obj.svd_.U[:,:n_components],array(sqrt(ss_loadings)),V)
        
        #convert to namedtuple
        self.rotmat_ = rot.rotmat

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables informations: coordinates, relative contributions, squared cosinus and additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #contributions of the variables
        var_ctr = rot.loadings.pow(2).mul(100).mul(obj.call_.var_weights,axis=0).div(ss_loadings[:n_components],axis=1)
        #cos2 of the variables
        var_sqcos = rot.loadings.pow(2).div(obj.var_.infos.iloc[:,1],axis=0)
        #convert to ordered dictionary
        var_ = OrderedDict(coord=rot.loadings,cos2=var_sqcos,contrib=var_ctr,infos=obj.var_.infos)
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: coordinates, relative contributions, squared cosinus and additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = obj.call_.Z.mul(obj.call_.var_weights,axis=1).dot(rot.loadings).div(sqrt(ss_loadings[:n_components]),axis=1)
        #contributions of the individuals
        ind_ctr = ind_coord.pow(2).mul(100).mul(obj.call_.ind_weights,axis=0).div(ss_loadings[:n_components],axis=1)
        #cos2 of the rows
        ind_sqcos = ind_coord.pow(2).div(obj.ind_.infos.iloc[:,1],axis=0)
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord,cos2=ind_sqcos,contrib=ind_ctr,infos=obj.ind_.infos)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #reproduced correlations
        rcorr =  rot.loadings.dot(rot.loadings.T)
        #residual correlation
        residual_corr = obj.corr_.corrcoef.sub(rcorr)
        #error
        error = (tril(residual_corr,-1)**2).sum().sum()
        #convert to ordered dictionary
        corr_ = OrderedDict(corrcoef=obj.corr_.corrcoef,reconst=rcorr,residual=residual_corr,error=error)
        self.corr_ = namedtuple("correlation",corr_.keys())(*corr_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variance accounted
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #proportion
        prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
        #convert to DataFrame
        self.vaccounted_ = DataFrame(c_[obj.eig_.iloc[:n_components,0],ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],index = ["Dim."+str(x+1) for x in range(n_components)],
                                     columns=["Eigenvalue","SS loadings","Proportion Var","Cumulative Var","Proportion Explained","Cumulative Proportion"])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##others informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fidélité des facteurs - variance of the scores - R2
        f_fidelity = rot.loadings.mul(V).sum(axis=0)
        f_fidelity.name = "R2"
        #Variance explained by each factor
        explained_variance = DataFrame(c_[obj.eig_.iloc[:n_components,0],ss_loadings],index=["Dim."+str(x+1) for x in range(n_components)],columns=["Weighted","Unweighted"])
        #Initial community
        init_comm = 1 - 1/diag(linalg.inv(obj.corr_.corrcoef))
        #estimated communalities
        final_comm = rot.loadings.mul(obj.call_.var_weights,axis=0).pow(2).sum(axis=1)
        #communality
        communality = DataFrame(c_[init_comm,final_comm],columns=["Prior","Final"],index=rot.loadings.index)
        #communalities
        communalities = sum(final_comm)
        #uniquenesses
        uniquenesses = 1 - final_comm
        uniquenesses.name = "Uniquenesses"
        #convert to ordered dictionnary
        others_ = OrderedDict(r2_score=f_fidelity,explained_variance=explained_variance,communality=communality,communalities=communalities,uniquenesses=uniquenesses)
        #convert to namedtuple
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        self.model_ = "pcarot"
        return self

    def fit_transform(self,obj,y=None) -> DataFrame:
        """
        Fit the model with obj and apply the dimensionality reduction on X
        ------------------------------------------------------------------

        Parameters
        ----------
        `obj`: an object of class PCA
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: a pandas DataFrame of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(obj)
        return self.ind_.coord
    
    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        is_dataframe(X=X) #check if X is an instance of pd.DataFrame class
        
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
            raise ValueError("'columns' aren't aligned")
        
        if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
            raise TypeError("All columns in X must be numerics.")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns

        #standardisation: Z = (X - mu)/sigma
        Z = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1)
        #apply transition relation
        coord = Z.mul(self.call_.var_weights,axis=1).dot(self.var_.coord).div(self.svd_.vs[:self.call_.n_components],axis=1)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord