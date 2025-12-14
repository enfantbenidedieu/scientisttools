# -*- coding: utf-8 -*-
from numpy import number,array,ndarray,average,sqrt,linalg,zeros,ones, diag, dot
from pandas import DataFrame,concat,Series, get_dummies
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.gfa import fitfa
from .functions.splitmix import splitmix
from .functions.recodecont import recodecont
from .functions.recodecat import recodecat
from .functions.summarize import summarize
from .functions.association import association
from .functions.revalue import revaluate_cat_variable
from .functions.conditional_wmean import conditional_average
from .functions.function_eta2 import function_eta2
from .functions.corrmatrix import corrmatrix

class MPCA(BaseEstimator,TransformerMixin):
    """
    Mixed Principal Components Analysis (MPCA)
    ------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs principal component analysis of a set of individuals (observations) described by a mixture of qualitative and quantitative variables with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    >>> MPCA(n_components = 5, ind_weights = None, quanti_weights = None, quali_weights = None, ind_sup = None, sup_var = None, parallelize = False)
    ```

    Parameters
    ----------
    
    Attributes
    ----------
    
    `model_`: a string indicating the model fitted = 'mpca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Abdesselam R. (2006), Analyse en Composantes Principales Mixtes, CREM UMR CNRS 6211
    
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Pages J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Rakotomalala, R (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_mpca_ind, get_mpca_var, get_mpca, summaryMPCA, dimdesc, predictMPCA, supvarMPCA, fviz_mpca_ind, fviz_mpca_col, fviz_mpca_mod, fviz_mpca_var, 

    Examples
    --------
    ```python
    
    ```
    """
    def __init__(self,
                 n_components = 5,
                 ind_weights = None,
                 quanti_weights = None,
                 quali_weights = None,
                 ind_sup = None,
                 sup_var = None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.quanti_weights = quanti_weights
        self.quali_weights  = quali_weights
        self.ind_sup = ind_sup
        self.sup_var = sup_var
        self.parallelize = parallelize

    def fit(self,X:DataFrame, y=None):
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
        #check if parallelize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.parallelize,bool):
            raise TypeError("'parallelize' must be a boolean.")

        #set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fill NA with mean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=number)
        is_quanti = recodecont(X=is_quanti).X
        for k in is_quanti.columns:
            X[k] = is_quanti[k]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert categorical variables to factor
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
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

        #make a copy of original data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X = X.drop(columns=sup_var_label)
        
        #drop supplementary individuls  
        if self.ind_sup is not None:
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Mixed Principal Component Analysis (MPCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split X
        split_X = splitmix(X=X)
        X_quanti, X_quali, n_rows, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

        #check if no quantitative variables
        if n_quanti == 0:
            raise TypeError("No quantitative variables in X. X must be a mixed data")

        #check if no qualitative variables
        if n_quali == 0:
            raise TypeError("No qualitative variables in X. X must be a mixed data")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list/tuple/array/Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array/Series with length {n_rows}.")
        else:
            ind_weights = array(list(map(lambda x : x/sum(self.ind_weights),self.ind_weights)))
        
        #convert to Series
        ind_weights = Series(ind_weights,index=X.index,name="weight")

        #recode qualitative variables
        rec = recodecat(X=X_quali)
        X_quali, dummies = rec.X, rec.dummies

        #compute weighted average mean and standard deviation
        center1 = Series(average(X_quanti,axis=0,weights=ind_weights),index=X_quanti.columns,name="center")

        #center quantitatives variables
        X1c = mapply(X_quanti, lambda x : x - center1, axis=1,progressbar=False,n_workers=n_workers)

        #diagonal matrix of individuals weights
        Vx, Vylx = dot(dot(X1c.T,diag(ind_weights)),X1c), dot(dot(dummies.T,diag(ind_weights)),X1c)
        #compute the mean
        center2 = Series(dot(dot(dot(dot(Vylx,linalg.pinv(Vx,hermitian=True)),X1c.T),diag(ind_weights)),ones(X.shape[0])).T[0],index=dummies.columns,name="center")
        #center the dummies table
        X2c = mapply(dummies, lambda x : x - center2, axis=1,progressbar=False,n_workers=n_workers)
        
        #set quantitative variables weights
        if self.quanti_weights is None:
            quanti_weights = ones(n_quanti)
        elif not isinstance(self.quanti_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'quanti_weights' must be a list/tuple/1darray/Series of quantitative variables weights")
        elif len(self.quanti_weights) != n_quanti:
            raise TypeError(f"'quanti_weights' must be a list/tuple/1darray/Series with length {n_quanti}.")
        else:
            quanti_weights = array(self.quanti_weights)
        
        #convert to Series
        quanti_weights = Series(quanti_weights,index=X_quanti.columns,name="weight")

        #set qualitative variables weights
        if self.quali_weights is None:
            quali_weights = ones(n_quali)
        elif not isinstance(self.quali_weights,(list,tuple,ndarray,Series)):
            raise ValueError("'quali_weights' must be a list/tuple/1darray/Series of qualitative variables weights")
        elif len(self.quali_weights) != n_quali:
            raise TypeError(f"'quali_weights' must be a list/tuple/1darray/Series with length {n_quali}.")
        else:
            quali_weights = array(self.quali_weights)
        #duplicate according to number of levels
        nb_moda = Series([X_quali[j].nunique() for j in X_quali.columns],index=X_quali.columns)
        quali_weights = Series(array(list(chain(*[repeat(i,k) for i, k in zip(quali_weights,nb_moda)]))),index=dummies.columns,name="weight")

        #concatenate
        Xc, center, var_weights = concat((X1c,X2c),axis=1), concat((center1,center2),axis=0), concat((quanti_weights,quali_weights),axis=0)

        #compute weighted mean and standard deviation
        d_xc = DescrStatsW(Xc,weights=ind_weights,ddof=0)
        #standardize Z
        Z = mapply(Xc,lambda x : (x - d_xc.mean)/d_xc.std, axis=1,progressbar=False,n_workers=n_workers)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1, Z.shape[1] - n_quali))
        #set number of components
        if self.n_components is None:
            n_components =  max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Xc=Xc,Z=Z,k1=n_quanti,k2=n_quali,ind_weights=ind_weights,var_weights=var_weights,center=center,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,sup_var=sup_var_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit normed principal component analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis (GFA)
        fit_ = fitfa(Z,ind_weights,var_weights,max_components,n_components,n_workers)

        #extract elements
        self.svd_, self.eig_, self.ind_ = fit_.svd, fit_.eig, namedtuple("ind",fit_.row.keys())(*fit_.row.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        quanti_var_coord, quanti_var_ctr, quanti_var_sqcos, quanti_var_infos = fit_.col["coord"].iloc[:n_quanti,:], fit_.col["contrib"].iloc[:n_quanti,:], fit_.col["cos2"].iloc[:n_quanti,:], fit_.col["infos"].iloc[:n_quanti,:]
        #convert to ordered dictionary
        quanti_var_ = OrderedDict(coord=quanti_var_coord, contrib=quanti_var_ctr, cos2=quanti_var_sqcos,infos=quanti_var_infos)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates, cos2 and cos2 of active levels
        levels_coord, levels_ctr, levels_sqcos, levels_infos = fit_.col["coord"].iloc[n_quanti:,:], fit_.col["contrib"].iloc[n_quanti:,:], fit_.col["cos2"].iloc[n_quanti:,:], fit_.col["infos"].iloc[n_quanti:,:]
        #coordinates for the levels as barycenter of individuals
        levels_coord_n = conditional_average(X=self.ind_.coord,Y=X_quali,weights=ind_weights)
        #proportion for the levels
        p_k = mapply(dummies,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        #vtest for the levels
        levels_vtest = mapply(mapply(levels_coord_n,lambda x : x*sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0,progressbar=False,n_workers=n_workers),
                                    lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        #convert to OrderedDict
        quali_var_ = OrderedDict(coord=levels_coord,coord_n=levels_coord_n,contrib=levels_ctr,cos2=levels_sqcos,vtest=levels_vtest,infos=levels_infos)
        #convert to namedtuple
        self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eta2 for the qualitative variables
        quali_var_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=ind_weights,n_workers=n_workers) 
        #contributions for the qualitative variables
        quali_var_ctr = mapply(quali_var_sqeta,lambda x : 100*(x/self.eig_.iloc[:n_components,0]),axis=1,progressbar=False,n_workers=n_workers)
        #cos2 for the qualitative variables 
        quali_var_sqcos = mapply(quali_var_sqeta,lambda x : (x**2)/(nb_moda -1),axis=0,progressbar=False,n_workers=n_workers)
        #concatenate
        var_coord, var_ctr, var_sqcos = concat((quanti_var_sqcos,quali_var_sqeta),axis=0), concat((quanti_var_ctr,quali_var_ctr),axis=0), concat((quanti_var_sqcos.pow(2),quali_var_sqcos),axis=0)
        #convert to namedtuple
        self.var_ = namedtuple("var",["coord","contrib","cos2"])(var_coord,var_ctr,var_sqcos)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split data
            split_ind_sup = splitmix(X_ind_sup)
            X_ind_sup_quanti, X_ind_sup_quali, n_ind_sup = split_ind_sup.quanti, split_ind_sup.quali, split_ind_sup.n
            
            #create disjunctive table
            dummies_ind_sup = DataFrame(zeros((n_ind_sup,dummies.shape[1])),columns=dummies.columns,index=ind_sup_label)
            for i in range(n_ind_sup):
                values = [X_ind_sup_quali.iloc[i,j] for j in range(n_quali)]
                for k in range(dummies.shape[1]):
                    if dummies.columns[k] in values:
                        dummies_ind_sup.iloc[i,k] = 1

            #concatenate
            Xcod_ind_sup = concat((X_ind_sup_quanti,dummies_ind_sup),axis=1)
            #standardize the data
            Z_ind_sup = mapply(Xcod_ind_sup,lambda x : ((x - center) - d_xc.mean)/d_xc.std,axis=1,progressbar=False,n_workers=n_workers)
            #coordinates for the supplementary individuals
            ind_sup_coord = mapply(Z_ind_sup,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(self.svd_.V)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            #dist2 for the supplementary individuals
            ind_sup_sqdisto = mapply(Z_ind_sup, lambda x : (x**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            ind_sup_sqdisto.name = "Sq. Dist."
            #cos2 for the supplementary individuals
            ind_sup_sqcos = mapply(ind_sup_coord,lambda x : (x**2)/ind_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",["coord","cos2","dist2"])(ind_sup_coord, ind_sup_sqcos, ind_sup_sqdisto)

        #----------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            X_sup_var = Xtot.loc[:,sup_var_label]
            if self.ind_sup is not None:
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #initialisation
            Xcod_sup_var, center_sup = DataFrame().astype("float"), Series(name="center")
            if n_quanti_sup > 0: 
                #average on supplementary quantitative variables
                center1_sup = Series(average(X_quanti,axis=0,weights=ind_weights),index=X_quanti_sup.columns,name="center")
                #concatenate
                Xcod_sup_var, center_sup = concat((Xcod_sup_var,X_quanti_sup),axis=1), concat((center_sup,center1_sup),axis=0)

            if n_quali_sup > 0:
                #recode supplementary qualitative variables
                rec2 = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec2.X, rec2.dummies
                #cov matrix
                Vylx_sup = dot(dot(dummies_sup.T,diag(ind_weights)),X1c)
                #compute the average
                center2_sup = Series(dot(dot(dot(dot(Vylx_sup,linalg.pinv(Vx,hermitian=True)),X1c.T),diag(ind_weights)),ones(X.shape[0])).T[0],index=dummies_sup.columns,name="center")
                #concatenate
                Xcod_sup_var, center_sup = concat((Xcod_sup_var,dummies_sup),axis=1), concat((center_sup,center2_sup),axis=0)

            #center the supplementary variables (quantitative and/or disjunctive)
            Xc_sup_var = mapply(Xcod_sup_var, lambda x : x - center_sup, axis=1,progressbar=False,n_workers=n_workers)
            #compute weighted mean and standard deviation
            d_sup_var = DescrStatsW(Xc_sup_var,weights=ind_weights,ddof=0)
            #standardize Xcod_sup_var
            Z_sup_var = mapply(Xc_sup_var,lambda x : (x - d_sup_var.mean)/d_sup_var.std, axis=1,progressbar=False,n_workers=n_workers)
            #coordinates for the supplementary columns
            col_sup_coord = mapply(Z_sup_var,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(self.svd_.U)
            col_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            #dist2 for the supplementary columns
            col_sup_sqdisto  = mapply(Z_sup_var, lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            col_sup_sqdisto.name = "Sq. Dist."
            #cos2 for the supplementary columns
            col_sup_sqcos = mapply(col_sup_coord, lambda x : (x**2)/col_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)

            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0:
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",["coord","cos2"])(col_sup_coord.iloc[:n_quanti_sup,:],col_sup_sqcos.iloc[:n_quanti_sup,:])

            #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #coordinates, dist2 and cos2 for the supplementary levels
                levels_sup_coord, levels_sup_sqdisto, levels_sup_sqcos = col_sup_coord.iloc[n_quanti_sup:,:], col_sup_sqdisto.iloc[n_quanti_sup:], col_sup_sqcos.iloc[n_quanti_sup:,:]
                #coordinates as barycenter of individuals
                levels_sup_coord_n = conditional_average(X=self.ind_.coord,Y=X_quali_sup,weights=ind_weights)
                #proportion for the supplementary levels
                p_k_sup = mapply(dummies_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
                #vtest for the supplementary levels
                levels_sup_vtest = mapply(mapply(levels_sup_coord_n,lambda x : x*sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0,progressbar=False,n_workers=n_workers),
                                          lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                #eta2 for supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=ind_weights,n_workers=n_workers) 
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(coord=levels_sup_coord,coord_n=levels_sup_coord_n,cos2=levels_sup_sqcos,vtest=levels_sup_vtest,dist2=levels_sup_sqdisto,eta2=quali_sup_sqeta)
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

        #descriptive statistics for the quantitative variables
        if self.ind_sup is not None:
            is_quanti = is_quanti.drop(index=ind_sup_label)
        self.summary_quanti_ = summarize(X=is_quanti)

        #multivariate goodness of fit
        if self.ind_sup is not None:
            is_quali = is_quali.drop(index=ind_sup_label)
        self.summary_quali_ = summarize(X=is_quali)
        if is_quali.shape[1] > 1:
            self.goodness_ = association(X=is_quali,alpha=0.05)
        
        #correlation tests
        is_all = Xtot.copy()
        if self.ind_sup is not None:
            is_all = is_all.drop(index=ind_sup_label)
        self.corrtest_ = corrmatrix(X=is_all,weights=ind_weights)

        self.model_ = "mpca"
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
        `X_new`: pandas dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
            raise ValueError("'columns' aren't aligned")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the MPCA result")
        
        split_X = splitmix(X)
        X_quanti, X_quali, n_rows = split_X.quanti, split_X.quali, split_X.n
        
        #initial and extract active elements
        dummies, n_quanti, n_quali = self.call_.dummies, self.call_.k1, self.call_.k2

        if n_quanti != X_quanti.shape[1]:
            raise TypeError("The number of quantitative variables must be the same")
        
        if n_quali != X_quali.shape[1]:
            raise TypeError("The number of qualitative variables must be the same")
        
        X_quali = revaluate_cat_variable(X_quali)
        #create disjunctive table
        dummies_sup = DataFrame(zeros((n_rows,dummies.shape[1])),index=X.index,columns=dummies.columns)
        for i in range(n_rows):
            values = [X_quali.iloc[i,j] for j in range(n_quali)]
            for k in range(dummies.shape[1]):
                if dummies.columns[k] in values:
                    dummies_sup.iloc[i,k] = 1
        #concatenate
        Xcod = concat((X_quanti,dummies_sup),axis=1)
        #weighted average and standard deviation in Xc
        d_xc = DescrStatsW(self.call_.Xc,weights=self.call_.ind_weights,ddof=0)
        #coordinates for the new individuals
        coord = mapply(Xcod,lambda x : ((x - self.call_.center - d_xc.mean)/d_xc.std)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord

def predictMPCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Mixed Principal Component Analysis (MPCA)
    -------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin for the new individuals with Mixed Principal Component Analysis (MPCA)

    Usage
    -----
    ```python
    >>> predictMPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class MPCA

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    a namedtuple of pandas DataFrame/Series containing all the results for the new individuals including:
    
    `coord`: coordinates for the new individuals,

    `cos2`: square cosinus for the new individuals,

    `dist2`: squared distance to origin for the new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos1990
    >>> from scientisttools import MPCA, predictMPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(autos1990)
    >>> #predict for the individuals
    >>> predict = predictMPCA(res_mpca, autos1990)
    >>> predict.coord #coordinates of the individuals
    >>> predict.cos2 #cos2 of the individuals
    >>> predict.dist2 #dist2 of the individuals.
    ```
    """
    if self.model_ != "mpca": #check if self is an object of class MPCA
        raise TypeError("'self' must be an object of class MPCA")
    
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
        raise ValueError("'columns' aren't aligned")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the MPCA result")
    
    split_X = splitmix(X)
    X_quanti, X_quali, n_rows = split_X.quanti, split_X.quali, split_X.n
    
    #initial and extract active elements
    dummies, n_quanti, n_quali = self.call_.dummies, self.call_.k1, self.call_.k2

    if n_quanti != split_X.k1:
        raise TypeError("The number of quantitative variables must be the same")
    
    if n_quali != split_X.k2:
        raise TypeError("The number of qualitative variables must be the same")
    
    X_quali = revaluate_cat_variable(X_quali)
    #create disjunctive table
    dummies_sup = DataFrame(zeros((n_rows,dummies.shape[1])),index=X.index,columns=dummies.columns)
    for i in range(n_rows):
        values = [X_quali.iloc[i,j] for j in range(n_quali)]
        for k in range(dummies.shape[1]):
            if dummies.columns[k] in values:
                dummies_sup.iloc[i,k] = 1
    #concatenatE
    Xcod = concat((X_quanti,dummies_sup),axis=1)
    #weighted average and standard deviation in Xc
    d_xc = DescrStatsW(self.call_.Xc,weights=self.call_.ind_weights,ddof=0)
    #standardize the data
    Z = mapply(Xcod,lambda x : (x - self.call_.center - d_xc.mean)/d_xc.std,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #coordinates for the new individuals
    coord = mapply(Z,lambda x : x*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the new individuals
    sqdisto = mapply(Z,lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #cos2 for the new individuals
    sqcos = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)
    return namedtuple("predictMPCAResult",["coord","cos2","dist2"])(coord,sqcos,sqdisto)

def supvarMPCA(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables in Mixed Principal Components Analysis (MPCA)
    ---------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin for the supplementary variables (quantitative and/or qualitative) with Mixed Principal Components Analysis (MPCA)

    Usage
    -----
    ```python
    >>> supvarMPCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class MPCA

    `X`: pandas DataFrame/Series of supplementary variables (quantitative and/or qualitative)

    Returns
    -------
    a namedtuple of namedtuple containing all the results for supplementary variables including: 

    `quanti`: a namedtuple of pandas DataFrame containing all the results for the supplementary quantitative variables including:
        * `coord`: coordinates for the supplementary quantitative variables,
        * `cos2`: squared cosinus for the supplementary quantitative variables.
    
    `quali`: a namedtuple of pandas DataFrame/Series containing all the results of the supplementary qualitative variables/levels including:
        * `coord`: coordinates for the supplementary levels,
        * `cos2`: squared cosinus of the supplementary levels,
        * `vtest`: value-test of the supplementary levels,
        * `dist2`: squared distance to origin of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos1990
    >>> from scientisttools import MPCA, supvarMPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(autos1990)
    >>> #predict for the supplementary variables (quantitative and qualitative)
    >>> sup_var_predict = supvarMPCA(res_mpca, autos1990)
    >>> quanti_sup = sup_var_predic.quanti
    >>> quanti_sup.coord #coord for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    >>> quali_sup = sup_var_predict.quali
    >>> quali_sup.coord #coord for the supplementary levels
    >>> quali_sup.cos2 #cos2 for the supplementary levels
    >>> quali_sup.vtest #vtest for the supplementary levels
    >>> quali_sup.dist2 #dist2 for the supplementary levels
    >>> quali_sup.eta2 #eta2 for the supplementary qualitative variables
    ```
    """
    if self.model_ != "mpca": #check if self is and object of class MPCA
        raise TypeError("'self' must be an object of class MPCA")
    
    if isinstance(X,Series): #if pandas Series, transform to pandas dataframe
        X = X.to_frame()
        
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")
    
    #split X
    split_X = splitmix(X=X)
    X_quanti, X_quali, n_rows, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

    #initialisation
    Xcod, center = DataFrame().astype("float"), Series(name="center")
    if n_quanti > 0: 
        #average on supplementary quantitative variables
        center1 = Series(average(X_quanti,axis=0,weights=self.call_.ind_weights),index=X_quanti.columns,name="center")
        #concatenate
        Xcod, center = concat((Xcod,X_quanti),axis=1), concat((center,center1),axis=0)

    if n_quali > 0:
        X1c = self.call_.Xc.loc[:,self.quanti_var_.coord.index]
        #recode supplementary qualitative variables
        rec = recodecat(X=X_quali)
        X_quali, dummies = rec.X, rec.dummies
        #cov matrix
        Vx, Vylx = dot(dot(X1c.T,diag(self.call_.ind_weights)),X1c), dot(dot(dummies.T,diag(self.call_.ind_weights)),X1c)
        #compute the average
        center2 = Series(dot(dot(dot(dot(Vylx,linalg.pinv(Vx,hermitian=True)),X1c.T),diag(self.call_.ind_weights)),ones(X.shape[0])).T[0],index=dummies.columns,name="center")
        #concatenate
        Xcod, center = concat((Xcod,dummies),axis=1), concat((center,center2),axis=0)

    #center the supplementary variables (quantitative and/or disjunctive)
    Xc = mapply(Xcod, lambda x : x - center, axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #compute weighted mean and standard deviation
    d_xc = DescrStatsW(Xc,weights=self.call_.ind_weights,ddof=0)
    #standardize Xc
    Z = mapply(Xc,lambda x : (x - d_xc.mean)/d_xc.std, axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #coordinates for the supplementary columns
    col_coord = mapply(Z,lambda x : x*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).T.dot(self.svd_.U)
    col_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the supplementary columns
    col_sqdisto  = mapply(Z, lambda x : (x**2)*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
    col_sqdisto.name = "Sq. Dist."
    #cos2 for the supplementary columns
    col_sup_sqcos = mapply(col_coord, lambda x : (x**2)/col_sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)

    #statistics for supplementary quantitative variables
    if n_quanti > 0:
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",["coord","cos2"])(col_coord.iloc[:n_quanti,:],col_sup_sqcos.iloc[:n_quanti,:])
    else:
        quanti_sup = None

    #statistics for supplementary qualitative variables/levels
    if n_quali > 0:
        #coordinates, dist2 and cos2 for the supplementary levels
        levels_coord, levels_sqdisto, levels_sup_sqcos = col_coord.iloc[n_quanti:,:], col_sqdisto.iloc[n_quanti:], col_sup_sqcos.iloc[n_quanti:,:]
        #coordinates as barycenter of individuals
        levels_coord_n = conditional_average(X=self.ind_.coord,Y=X_quali,weights=self.call_.ind_weights)
        #proportion for the supplementary levels
        p_k = mapply(dummies,lambda x : x*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
        #vtest for the supplementary levels
        levels_vtest = mapply(mapply(levels_coord_n,lambda x : x*sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0,progressbar=False,n_workers=self.call_.n_workers),
                              lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #eta2 for supplementary qualitative variables
        quali_var_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=self.call_.ind_weights,n_workers=self.call_.n_workers) 
        #convert to ordered dictionary
        quali_sup_ = OrderedDict(coord=levels_coord,coord_n=levels_coord_n,cos2=levels_sup_sqcos,vtest=levels_vtest,dist2=levels_sqdisto,eta2=quali_var_sqeta)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None

    #convert to namedtuple
    return namedtuple("supvarMPCAResult",["quanti","quali"])(quanti_sup,quali_sup)