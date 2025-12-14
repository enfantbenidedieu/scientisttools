# -*- coding: utf-8 -*-
from numpy import number, ones, linalg, array, ndarray, outer, diag, real, sum, dot, corrcoef, sqrt
from pandas import DataFrame, Series, api, concat
from statsmodels.stats.weightstats import DescrStatsW
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.recodecont import recodecont
from .functions.revalue import revaluate_cat_variable
from .functions.get_indices import get_indices
from .functions.conditional_wmean import conditional_wmean
from .functions.conditional_wstd import conditional_wstd
from .functions.gfa import gfa
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.function_eta2 import function_eta2

class DMFA(BaseEstimator,TransformerMixin):
    """
    Dual Multiple Factor Analysis (DMFA)
    ------------------------------------

    Description
    -----------
    Performs Dual Multiple Factor Analysis (DMFA) with supplementary individuals and/or supplementary variables (quantitative and/or qualitative)

    Usage
    -----
    ```python
    >>> DMFA(group = None, standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, sup_var = None)
    ```

    Parameters
    ----------
    `group`: a string or an integer indicating the qualitative variable to make the group of individuals

    `standardize`: a boolean, default = True
        * If `True`: the data are scaled to unit variance.
        * If `False`: the data are not scaled to unit variance.
    
    `n_components`: None or an integer indicating the number of dimensions kept in the results (by default 5)

    `ind_weights`: an optional individuals weights (by default, a list/tuple/1darray/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional quantitative variables weights (by default, a list/tuple/1darray/Series of 1 for uniform variables weights), the weights are given only for the active variables

    `ind_sup`: an integer or a string or a list or a tuple indicating the indexes/names of the supplementary individuals

    `sup_var`: an integer or a string or a list or a tuple indicating the indexes/names of the supplementary variables (quantitative and/or qualitative)

    Attributes
    ----------
    
    `model_`: a string specifying the model fitted = 'famd'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    * Husson F., Le S. and Pagès J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Husson F., Josse L, Lê S. & Mazet J. (2009). FactoMineR : Factor Analysis and Data Mining iwith R. R package version 2.11

    * Lebart L., Piron M. & Morineau A. (2006). Statistique exploratoire multidimensionelle. Dunod Paris 4ed

    * Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1–18. https://doi.org/10.18637/jss.v025.i01

    * Lê, S. & Pagès J. (2003). Deux extensions de l'Analyse Factorielle Multiple
    
    * Pagès J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    * Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    `get_dmfa_ind`, `get_dmfa_var`, `get_dmfa`, `summaryDMFA`, `dimdesc`, `predictDMFA`, `supvarDMFA`, `fviz_dmfa_ind`, `fviz_dmfa_var`

    Examples
    --------
    ```python
    >>> from scientisttools import DMFA
    >>> iris = load_dataset('iris')
    >>> res_dmfa = DMFA(group=4)
    >>> res_dmfa.fit(iris)
    ```
    """
    def __init__(self,
                 group = None,
                 standardize = True,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 sup_var = None):
        self.group = group
        self.standardize = standardize
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.sup_var = sup_var

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of pd.DataFrame class
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
           raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #check if group is assigned
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif isinstance(self.group,int):
            #grp = self.group
            grp_label = X.columns[self.group]
        elif isinstance(self.group,str):
            grp_label = self.group
            #grp = X.columns.tolist().index(grp)
        else:
            raise TypeError("'group' must be either a string or an integer.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fill NA with mean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=number)
        if not is_quanti.empty:
            is_quanti = recodecont(X=is_quanti).X
            for k in is_quanti.columns:
                X[k] = is_quanti[k]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert categorical variables to factor
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if not is_quali.empty:
            is_quali = revaluate_cat_variable(is_quali)
            for j in is_quali.columns:
                X[j] = is_quali[j]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            if isinstance(self.sup_var,str):
                sup_var_label = [self.sup_var]
            elif isinstance(self.sup_var,(int,float)):
                sup_var_label = [X.columns[int(self.sup_var)]]
            elif isinstance(self.sup_var,(list,tuple)):
                if all(isinstance(x,str) for x in self.sup_var):
                    sup_var_label = [str(x) for x in self.sup_var]
                elif all(isinstance(x,(int,float)) for x in self.sup_var):
                    sup_var_label = X.columns[[int(x) for x in self.sup_var]].tolist()
        else:
            sup_var_label = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            if isinstance(self.ind_sup,str):
                ind_sup_label = [self.ind_sup]
            elif isinstance(self.ind_sup,(int,float)):
                ind_sup_label = [X.index[int(self.ind_sup)]]
            elif isinstance(self.ind_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.ind_sup):
                    ind_sup_label = [str(x) for x in self.ind_sup]
                elif all(isinstance(x,(int,float)) for x in self.ind_sup):
                    ind_sup_label = X.index[[int(x) for x in self.ind_sup]].tolist()
        else:
            ind_sup_label = None

        #extract group name
        grp_name = X.loc[:,grp_label].unique().tolist()
        
        #check if group name is an integer or float
        if all(isinstance(x, (int, float)) for x in grp_name):
            grp_name = ["Gr".format(x+1) for x in grp_name]
            X.loc[:,grp_label] = X.map(zip(X.loc[:,grp_label].unique().tolist(),grp_name))

        #make a copy of the data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X = X.drop(columns=sup_var_label)
        
        #drop supplementary individuls  
        if self.ind_sup is not None:
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #dual multiple factor analysis (DMFA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X into x and y
        y, x = X.loc[:,grp_label], X.drop(columns=grp_label)

        #set number of rows and number of column
        n_rows, n_cols = x.shape

        #group index
        grp_index = OrderedDict()
        for g in grp_name:
            grp_index[g] = get_indices(x=y,value=g)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list/tuple/1darray/Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/1darray/Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list/tuple/1darray/Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/1darray/Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=x.index,name="weight"), Series(var_weights,index=x.columns,name="weight")

        #conditional weighted average
        center = conditional_wmean(X=x,Y=y,weights=ind_weights)
        #conditional weighted standard deviation
        if self.standardize:
            scale = conditional_wstd(X=x,Y=y,weights=ind_weights)
        else:
            scale = DataFrame(ones((len(grp_name),n_cols)),columns=x.columns,index=grp_name)
        #standardize by group
        Zs = x.sub(center.loc[y.values,:].values).div(scale.loc[y.values,:].values)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize according to normed principal components analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        d_zs = DescrStatsW(Zs,weights=ind_weights,ddof=0)
        if self.standardize:
            zs_scale = d_zs.std
        else:
            zs_scale = ones(Zs.shape[1])
        #convert to Series
        zs_center, zs_scale = Series(d_zs.mean,index=x.columns,name="center"), Series(zs_scale,index=x.columns,name="scale")
        
        #standardization : Z = (X - mu)/sigma
        Z = Zs.sub(zs_center,axis=1).div(zs_scale,axis=1)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1, n_cols))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Zs=Zs,Z=Z,group=grp_label,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,zs_center=zs_center,zs_scale=zs_scale,
                            n_components=n_components,max_components=max_components,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(Z,ind_weights,var_weights,max_components,n_components)
        
        #extract elements
        self.svd_, self.eig_, self.ind_, self.var_ = fit_.svd, fit_.eig, namedtuple("ind",fit_.row.keys())(*fit_.row.values()), namedtuple("var",fit_.col.keys())(*fit_.col.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #group informations : coordinates, cos2
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        grp_coord, grp_coord_n = DataFrame(index=grp_name,columns=self.var_.coord.columns).astype(float),DataFrame(index=grp_name,columns=self.var_.coord.columns).astype(float)
        grp_sqcos = DataFrame(index=grp_name,columns=self.var_.coord.columns).astype(float)

        for i, g in enumerate(grp_name):
            if self.standardize:
                Cov_g = Zs.iloc[grp_index[g],:].corr(method="pearson")
            else:
                Cov_g = Zs.iloc[grp_index[g],:].cov(ddof=0)
            for j in range(n_components):
                grp_coord.iloc[i,j] = sum(diag(outer(self.var_.coord.iloc[:,j],dot(self.var_.coord.iloc[:,j],Cov_g))))/self.eig_.iloc[j,0]
            
            eigen = real(linalg.eig(Cov_g)[0])
            grp_coord_n.iloc[i,:], grp_sqcos.iloc[i,:] = grp_coord.iloc[i,:].div(eigen[0]), grp_coord.iloc[i,:].pow(2).div(sum(eigen**2))  

        #store all group informations
        self.group_ = namedtuple("group",["coord","coord_n","cos2"])(grp_coord,grp_coord_n,grp_sqcos) 

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partiel factor coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        var_partiel = OrderedDict()
        for g in grp_name:
            cor_g = corrcoef(self.ind_.coord.iloc[grp_index[g],:], Zs.iloc[grp_index[g],:],rowvar=False)[n_components:,:n_components]
            var_partiel[g] = DataFrame(cor_g,index=x.columns,columns=["Dim."+str(x+1) for x in range(n_components)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split in x and y
            y_ind_sup, x_ind_sup = X_ind_sup.loc[:,grp_label], X_ind_sup.drop(columns=grp_label)
            #standardize using conditional average and standard deviation by group
            Zs_ind_sup = x_ind_sup.sub(center.loc[y_ind_sup.values,:].values).div(scale.loc[y_ind_sup.values,:].values)
            #standardize accordind to normed principal component analysis
            Z_ind_sup = Zs_ind_sup.sub(zs_center,axis=1).div(zs_scale,axis=1)
            #coordinates for the supplementary individuals
            ind_sup_coord = Z_ind_sup.mul(var_weights,axis=1).dot(fit_.svd.V)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            #dist2 for the supplementary individuals
            ind_sup_sqdisto = Z_ind_sup.pow(2).mul(var_weights,axis=1).sum(axis=1)
            ind_sup_sqdisto.name = "Sq. Dist."
            #cos2 for the supplementary individuals
            ind_sup_sqcos = ind_sup_coord.pow(2).div(ind_sup_sqdisto,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",["coord","cos2","dist2"])(ind_sup_coord, ind_sup_sqcos, ind_sup_sqdisto)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            X_sup_var = Xtot.loc[:,sup_var_label]
            if self.ind_sup is not None:
                X_sup_var = X_sup_var.drop(index=ind_sup_label)
            
            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0:
                #conditional weighted average
                center_sup = conditional_wmean(X=X_quanti_sup,Y=y,weights=ind_weights)
                #conditional weighted standard deviation
                if self.standardize:
                    scale_sup = conditional_wstd(X=X_quanti_sup,Y=y,weights=ind_weights)
                else:
                    scale_sup = DataFrame(ones((len(grp_name),n_quanti_sup)),columns=X_quanti_sup.columns,index=grp_name)
                #standardize using conditional weighted average and conditional standard deviation
                Zs_quanti_sup = X_quanti_sup.sub(center_sup.loc[y.values,:].values).div(scale_sup.loc[y.values,:].values)
                #weighted average and weighted standard deviation
                d_zs_quanti_sup = DescrStatsW(Zs_quanti_sup,weights=ind_weights,ddof=0)
                if self.standardize:
                    zs_quanti_sup_scale = d_zs_quanti_sup.std
                else:
                    zs_quanti_sup_scale = ones(n_quanti_sup)
                #convert to Series
                zs_quanti_sup_center, zs_quanti_sup_scale = Series(d_zs_quanti_sup.mean,index=X_quanti_sup.columns,name="center"), Series(zs_quanti_sup_scale,index=X_quanti_sup.columns,name="scale")
                #standardization using weighted average and weighted standard deviation
                Z_quanti_sup = Zs_quanti_sup.sub(zs_quanti_sup_center,axis=1).div(zs_quanti_sup_scale,axis=1)
                #coordinates for the supplementary quantitative variables
                quanti_sup_coord = Z_quanti_sup.mul(ind_weights,axis=0).T.dot(fit_.svd.U)
                quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
                #dist2 for the supplementary quantitative variables
                quanti_sup_sqdisto  = Z_quanti_sup.pow(2).mul(ind_weights,axis=0).sum(axis=0)
                quanti_sup_sqdisto.name = "Sq. Dist."
                #cos2 for the supplementary quantitative variables
                quanti_sup_sqcos = quanti_sup_coord.pow(2).div(quanti_sup_sqdisto,axis=0)
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",["coord","cos2"])(quanti_sup_coord,quanti_sup_sqcos)

                #add supplementary partiel coordinates
                for g in grp_name:
                    cor_g = corrcoef(self.ind_.coord.iloc[grp_index[g],:], Zs_quanti_sup.iloc[grp_index[g],:],rowvar=False)[n_components:,:n_components]
                    var_partiel[g] = concat((var_partiel[g],DataFrame(cor_g,index=X_quanti_sup.columns,columns=["Dim."+str(x+1) for x in range(n_components)])),axis=0)

            #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #create new qualitative columns
                X_quali_sup_new = concat((concat((X_quali_sup[x],y),axis=1).apply(lambda x: '_'.join(x),axis=1) for x in X_quali_sup.columns),axis=1)
                X_quali_sup_new.columns = [x+"_" + grp_label for x in X_quali_sup.columns]
                #concatenate
                X_quali_sup = concat((X_quali_sup,X_quali_sup_new),axis=1)
                #recode qualitative variables
                rec2 = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec2.X, rec2.dummies
                #compute conditional weighted average
                bary = conditional_wmean(X=Zs,Y=X_quali_sup,weights=ind_weights)
                #standardize according to principal component analysis
                Z_quali_sup = bary.sub(zs_center,axis=1).div(zs_scale,axis=1)
                #coordinates for the supplementary levels
                levels_sup_coord = Z_quali_sup.mul(var_weights,axis=1).dot(fit_.svd.V)
                levels_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
                #vtest for the supplementary levels
                p_k_sup = dummies_sup.mul(ind_weights,axis=0).sum(axis=0)
                levels_sup_vtest = levels_sup_coord.mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(fit_.svd.vs[:n_components],axis=1)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=ind_weights)
                #dist2 for the supplementary levels
                levels_sup_sqdisto = Z_quali_sup.pow(2).mul(var_weights,axis=1).sum(axis=1)
                levels_sup_sqdisto.name = "Sq. Dist."
                #cos2 for the supplementary levels
                levels_sup_sqcos = levels_sup_coord.pow(2).div(levels_sup_sqdisto,axis=0)
                #convert to dictionary
                quali_sup_ = OrderedDict(coord=levels_sup_coord,cos2=levels_sup_sqcos,vtest=levels_sup_vtest,dist2=levels_sup_sqdisto,eta2=quali_sup_sqeta)
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

        #convert to namedtuple
        self.var_partiel_ = namedtuple("var_partiel",var_partiel.keys())(*var_partiel.values())

        #all quantitative variables in original dataframe
        is_quanti = Xtot.select_dtypes(include=number)
        #drop supplementary individuals
        if self.ind_sup is not None:
            is_quanti = is_quanti.drop(index=ind_sup_label)
        corr_partiel, cov_partiel = OrderedDict(), OrderedDict()
        for g in grp_name:
            cov_partiel[g], corr_partiel[g] = is_quanti.iloc[grp_index[g],:].cov(ddof=0), is_quanti.iloc[grp_index[g],:].corr(method="pearson")
        #convert to namedtuple
        self.cov_partiel_, self.corr_partiel_ = namedtuple("cov_partiel",cov_partiel.keys())(*cov_partiel.values()), namedtuple("corr_partiel",corr_partiel.keys())(*corr_partiel.values())
        self.model_ = "dmfa"
        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord
    
    def inverse_transform(self,X:DataFrame) -> DataFrame:
        """
        Transform data back to its original space
        -----------------------------------------

        Description
        -----------
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_components+group).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas DataFrame of shape (n_samples, n_columns)
            Original data, where `n_samples` is the number of samples and `n_columns` is the number of columns
        """
        if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #split X into x and y
        y, x = X.loc[:,self.call_.group], X.drop(columns=self.call_.group)
        #set number of components
        n_components = min(x.shape[1],self.call_.n_components)
        eigvals = self.var_.coord.pow(2).T.dot(self.call_.var_weights)[:n_components]
        #inverse transform
        Z = x.iloc[:,:n_components].dot(self.var_.coord.iloc[:,:n_components].div(sqrt(eigvals),axis=1).T).mul(self.call_.zs_scale,axis=1).add(self.call_.zs_center)
        #add conditional informations
        X_original = Z.mul(self.call_.scale.loc[y.values,:].values,axis=1).add(self.call_.center.loc[y.values,:].values,axis=1)
        return concat((X_original,y),axis=1)
    
    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
            raise ValueError("'columns' aren't aligned")
        
        intersect = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the DMFA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns

        #split X into x and y
        y, x = X.loc[:,self.call_.group], X.drop(columns=self.call_.group)

        all_num = all(api.types.is_numeric_dtype(x[k]) for k in x.columns)
        if not all_num: #check if all variables in x are numerics
            raise TypeError("All columns must be numeric")
        
        #standardize using conditional average and conditional standard deviation
        Zs = x.sub(self.call_.center.loc[y.values,:].values).div(self.call_.scale.loc[y.values,:].values)
        #coordinates for the new individuals
        coord = Zs.sub(self.call_.zs_center,axis=1).div(self.call_.zs_scale,axis=1).mul(self.call_.var_weights).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord

def predictDMFA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Dual Multiple Factor Analysis (DMFA)
    --------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Dual Multiple Factor Analysis (DMFA)

    Usage
    -----
    ```python
    >>> predictDMFA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class DMFA

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    a namedtuple of pandas DataFrame/Series containing all the results for the new individuals, including:
    
    `coord`: coordinates for the new individuals,

    `cos2`: squared cosinus for the new individuals,

    `dist2`: squared distance to origin for the new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from seaborn import load_dataset
    >>> from scientisttools import DMFA, predictDMFA
    >>> iris = load_dataset('iris')
    >>> res_dmfa = DMFA(group=4)
    >>> res_dmfa.fit(iris)
    >>> #predict for the new individuals
    >>> predict = predictDMFA(res_dmfa,iris)
    >>> predict.coord #coordinates for the new individuals
    >>> predict.cos2 #squared cosinus for the new individuals
    >>> predict.dist2 #squared distance to origin for the new individuals
    ```
    """
    if self.model_ != "dmfa": #check if self is an object of class DMFA
        raise TypeError("'self' must be an object of class DMFA")

    if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")
    
    intersect = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the DMFA result")
    X = X.loc[:,self.call_.X.columns] #reorder columns

    #split X into x and y
    y, x = X.loc[:,self.call_.group], X.drop(columns=self.call_.group)

    all_num = all(api.types.is_numeric_dtype(x[k]) for k in x.columns)
    if not all_num: #check if all variables in x are numerics
        raise TypeError("All columns must be numeric")
    
    #standardize using conditional average and conditional standard deviation
    Zs = x.sub(self.call_.center.loc[y.values,:].values).div(self.call_.scale.loc[y.values,:].values)
    #standardize according to principal component analysis
    Z = Zs.sub(self.call_.zs_center,axis=1).div(self.call_.zs_scale,axis=1)
    #coordinates for the new individuals
    coord = Z.mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the new individuals
    sqdisto = Z.pow(2).mul(self.call_.var_weights,axis=1).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #cos2 for the new individuals
    sqcos = coord.pow(2).div(sqdisto,axis=0)
    #convert to namedtuple
    return namedtuple("predictDMFAResult",["coord","cos2","dist2"])(coord,sqcos,sqdisto)
    
def supvarDMFA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables in Dual Multiple Factor Analysis (DMFA)
    ---------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Dual Multiple Factor Analysis (DMFA)

    Usage
    -----
    ```python
    >>> supvarDMFA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class DMFA

    `X`: a pandas DataFrame of supplementary variables (quantitative and/or qualitative)

    Returns
    -------
    a namedtuple of namedtuple containing all the results for supplementary variables, including: 

    `quanti`: a namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables, including:
        * `coord`: coordinates for the supplementary quantitative variables,
        * `cos2`: squared cosinus for the supplementary quantitative variables.
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results of the supplementary qualitative variables/levels, including:
        * `coord`: coordinates for the supplementary levels,
        * `cos2`: squared cosinus for the supplementary levels,
        * `vtest`: value-test for the supplementary levels
        * `dist2`: squared distance to origin for the supplementary levels,
        * `eta2`: squared correlation ratio for the supplementary qualitative variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import DMFA, supvarDMFA
    >>> from seaborn import load_dataset
    >>> iris = load_dataset('iris')
    >>> res_dmfa = DMFA(group=4)
    >>> res_dmfa.fit(iris)
    >>> #supplementary quantitative variables
    >>> sup_var_predict = supvarDMFA(res_dmfa, iris.iloc[:,:4])
    >>> quanti_sup = sup_var_predict.quanti
    >>> quanti_sup.coord #coordinates for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    ```
    """
    if self.model_ != "dmfa": #check if self is an object of class DMFA
        raise TypeError("'self' must be an object of class DMFA")
    
    if isinstance(X,Series): #if pandas series, transform to pandas dataframe
        X = X.to_frame()
        
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")
    
    #split X
    split_X = splitmix(X=X)
    X_quanti, X_quali, n_rows, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

    #statistics for supplementary quantitative variables
    if n_quanti > 0:
        #conditional weighted average
        center = conditional_wmean(X=X_quanti,Y=self.call_.X.loc[:,self.call_.group],weights=self.call_.ind_weights)
        #conditional weighted standard deviation
        if self.standardize:
            scale = conditional_wstd(X=X_quanti,Y=self.call_.X.loc[:,self.call_.group],weights=self.call_.ind_weights)
        else:
            scale = DataFrame(ones((self.group_.coord.shape[1],n_quanti)),columns=X_quanti.columns,index=self.group_.coord.index)
        #standardize using conditional weighted average and conditional standard deviation
        Zs = X_quanti.sub(center.loc[self.call_.X.loc[:,self.call_.group].values,:].values).div(scale.loc[self.call_.X.loc[:,self.call_.group].values,:].values)
        #weighted average and weighted standard deviation
        d_zs = DescrStatsW(Zs,weights=self.call_.ind_weights,ddof=0)
        if self.standardize:
            zs_scale = d_zs.std
        else:
            zs_scale = ones(n_quanti)
        #convert to Series
        zs_center, zs_scale = Series(d_zs.mean,index=X_quanti.columns,name="center"), Series(zs_scale,index=X_quanti.columns,name="scale")
        #standardization using weighted average and weighted standard deviation
        Z_quanti_sup = Zs.sub(zs_center,axis=1).div(zs_scale,axis=1)
        #coordinates for the supplementary quantitative variables
        quanti_sup_coord = Z_quanti_sup.mul(self.call_.ind_weights,axis=0).T.dot(self.svd_.U)
        quanti_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        #dist2 for the supplementary quantitative variables
        quanti_sup_sqdisto  = Z_quanti_sup.pow(2).mul(self.call_.ind_weights,axis=0).sum(axis=0)
        quanti_sup_sqdisto.name = "Sq. Dist."
        #cos2 for the supplementary quantitative variables
        quanti_sup_sqcos = quanti_sup_coord.pow(2).div(quanti_sup_sqdisto,axis=0)
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",["coord","cos2"])(quanti_sup_coord,quanti_sup_sqcos)
    else:
        quanti_sup = None

    #statistics for supplementary qualitative variables/levels
    if n_quali > 0:
        #create new qualitative columns
        X_quali_new = concat((concat((X_quali[x],self.call_.X.loc[:,self.call_.group]),axis=1).apply(lambda x: '_'.join(x),axis=1) for x in X_quali.columns),axis=1)
        X_quali_new.columns = [x+"_" + self.call_.group for x in X_quali.columns]
        #concatenate
        X_quali = concat((X_quali,X_quali_new),axis=1)
        #recode qualitative variables
        rec = recodecat(X=X_quali)
        X_quali, dummies = rec.X, rec.dummies
        #compute conditional weighted average
        bary = conditional_wmean(X=self.call_.Zs,Y=X_quali,weights=self.call_.ind_weights)
        #standardize according to principal component analysis
        Z_quali = bary.sub(self.call_.zs_center,axis=1).div(self.call_.zs_scale,axis=1)
        #coordinates for the supplementary levels
        levels_coord = Z_quali.mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        levels_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        #vtest for the supplementary levels
        p_k = dummies.mul(self.call_.ind_weights,axis=0).sum(axis=0)
        levels_vtest = levels_coord.mul(sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0).div(self.svd_.vs[:self.call_.n_components],axis=1)
        #eta2 for the supplementary qualitative variables
        quali_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=self.call_.ind_weights)
        #dist2 for the supplementary levels
        levels_sqdisto = Z_quali.pow(2).mul(self.call_.var_weights,axis=1).sum(axis=1)
        levels_sqdisto.name = "Sq. Dist."
        #cos2 for the supplementary levels
        levels_sqcos = levels_coord.pow(2).div(levels_sqdisto,axis=0)
        #convert to dictionary
        quali_sup_ = OrderedDict(coord=levels_coord,cos2=levels_sqcos,vtest=levels_vtest,dist2=levels_sqdisto,eta2=quali_sqeta)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None

    #convert to namedtuple
    return namedtuple("supvarDMFAResult",["quanti","quali"])(quanti_sup,quali_sup)