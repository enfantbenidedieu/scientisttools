# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, outer, diag, sum, dot, sqrt, vstack
from pandas import DataFrame, Series, concat, CategoricalDtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from ..onetable._pca import PCA
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.concat_empty import concat_empty
from ..functions.statistics import wmean, wstd, wcorr, func_groupby
from ..functions.gsvd import gSVD
from ..functions.func_eta2 import func_eta2
from ..functions.func_predict import func_predict
from ..functions.cov2corr import cov2corr
from ..functions.utils import check_is_bool, is_all_numeric_dtype, is_all_object_or_category_dtype
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix

class DMCOIA(BaseEstimator,TransformerMixin):
    """
    Dual Multiple CO-Inertia Analysis (DMCOIA)

    Performns Dual Multiple CO-Inertia Analysis in the sense of `Kissitia and al. <https://pphmjopenaccess.com/aas/article/view/527/488>`_ with supplementary individuals, and/or supplementary variables (continuous and/or categorical).

    Parameters
    ----------

    References
    ----------
    [1] Lavie Phanie Moulogho Issayaba, Léonard Niere, Bernédy NelMessie Kodia Banzouzi and Gabriel Kissita, Proposal of the dual multiple co-inertia analysis (DMCOA), Advances and Applications in Statistics 73 (2022), 17-46.
    """

    def __init__(
            self, scale_unit = True, ncp = 5,  group = None, row_w = None, col_w = None, ind_sup = None, sup_var = None, tol = 1e-7
    ):  
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.group = group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
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
        group_label, ind_sup_label, sup_var_label = get_sup_label(X=X, indexes=self.group, axis=1), get_sup_label(X=X,indexes=self.ind_sup,axis=0), get_sup_label(X=X,indexes=self.sup_var,axis=1)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary variables
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
            if self.ind_sup is not None: 
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

        #drop supplementary individuals
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #dual multiple factor analysis (DMFA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into x and y
        y, x = X[group_label[0]], X.drop(columns=group_label)

        #check if all columns are either continuous or categorical.
        if not (is_all_numeric_dtype(x) or is_all_object_or_category_dtype(x)):
            raise TypeError("Not applied to mixed data") 

        #unique element in y
        uq_classe = sorted(list(y.unique()))
        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=uq_classe,ordered=True))

        #number of rows and number of columns
        n_rows, n_vars = x.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.row_w is None:
            ind_w = Series(ones(n_rows)/n_rows,index=x.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)):
            raise TypeError("'row_w' must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows:
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else:
            ind_w = Series(array(self.row_w)/sum(self.row_w),index=x.index,name="weight")

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
        group_dict = OrderedDict({k : list(y[y==k].index) for k in uq_classe})
     
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set variables xcod - reorder 
        Xcod, dummies, M = None, None, None
        if is_all_numeric_dtype(x):
            Xcod, col_w = x.copy(), var_w.copy()
        elif is_all_object_or_category_dtype(x):
            dummies = disjunctive(x)
            M = concat((dummies.loc[rows,:].mul(ind_w[rows]/sum(ind_w[rows]),axis=0).sum(axis=0).mul(-1).add(1).to_frame(g) for g, rows in group_dict.items()),axis=1).T    
            Xcod = dummies.astype(float).mul(M.loc[y.values,:].values,axis=1)
            nb_moda = array([x[j].nunique() for j in x.columns])
            col_w = Series(array([x*y for x,y in zip(ones(dummies.shape[1]),array(list(chain(*[repeat(i,k) for i, k in zip(var_w,nb_moda)]))))]),index=dummies.columns,name="weight")
        else:
            raise NotImplementedError("'DMFA' for mixed data is not yet implemented")
        #set number of columns
        n_cols = Xcod.shape[1]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #separate general factor analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #run separate general factor analysis
        model = OrderedDict()
        for g, rows in group_dict.items():
            model[g] = PCA(scale_unit=self.scale_unit,ncp=self.ncp,row_w=ind_w[rows],col_w=col_w,tol=self.tol).fit(Xcod.loc[rows,:])
            
        #store separate analysis
        self.separate_analyses_ = model

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #scale_unitd data
        Zcod = concat((model[g].call_.Z for g in list(model.keys())),axis=0,ignore_index=False).loc[y.index,:]
        #weighted average
        center, scale = concat((model[g].call_.center.to_frame(g) for g in list(model.keys())),axis=1).T, concat((model[g].call_.scale.to_frame(g) for g in list(model.keys())),axis=1).T
        #number of final components
        mncp = Series([model[g].call_.ncp for g in list(model.keys())],index=list(model.keys()))
        #number of columns
        #nb_cols = Series([len(columns_dict[g]) for g in list(group_dict.keys())],index=list(group_dict.keys()))
        #set rows weights
        row_w = concat((model[g].call_.row_w/model[g].eig_.iloc[0,0] for g in list(model.keys())),axis=0)
        #rows index
        rows_index = {g : [list(y.index).index(k) for k in rows] for g, rows in group_dict.items()}

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization according to normed principal components analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        z_center, z_scale = wmean(X=Zcod,w=row_w), wstd(X=Zcod,w=ind_w)
        #standardization : z_ik = (x_ik - m_k)/s_k
        Z = Zcod.sub(z_center,axis=1).div(z_scale,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #run multiple generalized singular values decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #make a copy of standardized data
        tab = Z.copy()
        #initialize
        U, V, vs = [], [], [] 
        maxcp = min(self.ncp,10)
        ncp = self.ncp
        for i in range(maxcp):
            #generalized singular value decomposition
            fit_ = gSVD(tab,row_w=row_w,col_w=col_w,ncp=ncp)
            #normalization right matrix using mcoa variables weights by group and substract projection
            u = fit_.U[:,0]/sqrt(row_w)
            for g, rows in group_dict.items():
                idx = rows_index[g]
                u1 = u[idx]
                u2 = sqrt(sum(u1*u1))
                u1 = u1/u2 if u2 > self.tol else u1
                #substract projection
                tab.loc[rows,:] = tab.loc[rows,:].sub(outer(tab.loc[rows,:].mul(u1,axis=0).sum(axis=0),u1))
                tab.loc[rows,:] = u1
            V.append(fit_.V[:,0]), U.append(U), vs.append(float(fit_.vs[0]))

        #convert to array, update U and V with number of components
        U, V, vs = vstack(U).T, vstack(V).T, array(vs)
        #add to model attributes
        self.svd_ = namedtuple("svd",["U","V","vs"])(U[:,:ncp],V[:,:ncp],vs)

        print(self.svd_vs**2)




        return self