# -*- coding: utf-8 -*-
from numpy import array, ones, ndarray, sum,diag, sqrt,tile, where, nan,inf, linalg, insert, diff,  cumsum, c_, real
from collections import namedtuple, OrderedDict
from itertools import chain, repeat
from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from ..onetable._pca import PCA
from ..onetable._mca import MCA
from ..onetable._famd import FAMD
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.gfa import gFA
from ..functions.statistics import wmean, wcorr, func_groupby
from ..functions.cov2corr import cov2corr
from ..functions.concat_empty import concat_empty
from ..functions.func_predict import func_predict
from ..functions.func_eta2 import func_eta2
from ..functions.func_lg import func_Lg
from ..functions.utils import cols_dtypes
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix

class PTA(BaseEstimator,TransformerMixin):
    """
    Partial Triadic Analysis

    Performns a Partial Triadic Analysis in the sense of <>_
    
    
    
    
    """
    def __init__(
            self, excl=None, ncp=5, group=None, type_group=None, name_group=None, option="lambda1",  row_w=None, col_w=None, ind_sup=None, num_group_sup=None, tol = 1e-7
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
        elif not isinstance(self.type_group, (list,tuple,ndarray,Series)): 
            raise ValueError("'type' must be a 1d array-like with the type of variables in each group")
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
        elif not isinstance(self.name_group,(list,tuple,ndarray,Series)):
            raise TypeError("'name_group' must be a 1d array-like with name of group")
        else:
            name_group = [x for x in self.name_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if option is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.option in ("lambda1","inertia","uniform")):
            raise ValueError("'option' must be one of 'lambda1', 'inertia', 'uniform'")

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
        #multiple factor analysis (MFA)
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
                total = N.sum(axis=0).sum()
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