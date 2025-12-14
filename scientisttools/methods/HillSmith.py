# -*- coding: utf-8 -*-
from numpy import number,array,ndarray,average,sqrt,linalg,zeros,ones
from pandas import DataFrame, concat,Series
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.gfa import fitfa
from .functions.splitmix import splitmix
from .functions.recodevar import recodevarhillsmith
from .functions.recodecont import recodecont
from .functions.summarize import summarize
from .functions.association import association
from .functions.revalue import revaluate_cat_variable
from .functions.function_eta2 import function_eta2
from .functions.corrmatrix import corrmatrix

class HillSmith(BaseEstimator,TransformerMixin):
    """
    Hill and Smith Analysis of Mixed Data (HillSmith)
    -------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Hill and Smith Analysis of Mixed Data (HillSmith) with supplementary individuals and/or supplementary variables (quantitative and/or qualitative). 
    
    Details
    -------
    HillSmith includes standard Principal Component Analysis (PCA) and Multiple Correspondence Analysis (MCA) as special cases. If all variables are quantitative, standard PCA is performed.
    if all variables are qualitative, then standard MCA is performed.

    #Missing values are replaced by means for quantitative variables. Note that, when all the variables are qualitative, the factor coordinates of the individuals are equal to the factor scores
    #of standard MCA times squares root of J (the number of qualitatives variables) and the eigenvalues are then equal to the usual eigenvalues of MCA times J.
    #When all the variables are quantitative, HillSmith gives exactly the same results as standard PCA.

    Usage
    -----
    ```python 
    >>> HillSmith(n_components = 5, ind_weights = None, quanti_weights = None, quali_weights = None, ind_sup = None, sup_var = None, parallelize = False)
    ```
    
    Parameters
    ----------
    `n_components` : None or an integer indicating the number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `quanti_weights`: an optional quantitative variables weights (by default, a list/tuple/array/Series of 1 for uniform quantitative variables weights), the weights are given only for the active quantitative variables

    `quali_weights`: an optional qualitative variables weights (by default, a list/tuple/array/Series of 1 for uniform qualitative variables weights), the weights are given only for the active qualitative variables

    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `sup_var`: an integer/string/list/tuple indicating the indexes/names of the supplementary variables (quantitative and/or qualitative)

    `parallelize` : boolean, default = False. If model should be parallelize
        * If True : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If False : parallelize using pandas apply
    
    Attributes
    ----------
    

    `model_`: a string indicating the model fitted = 'hillsmith'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    * Hill M., Smith A. (1976). Principal Component Analysis of taxonomic data withmulti-state discrete characters. Taxon, 25, pp. 249-255

    * Husson F., Le S. and Pagès J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Husson F., Josse L, Lê S. & Mazet J. (2009). FactoMineR : Factor Analysis and Data Mining with R. R package version 2.11

    * Kiers H.A.L (1991). Simple structure in Component Analysis Techniques for mixtures of qualitative and quantitative variables. Psychometrika, 56, pp. 197-212.
    
    * Lebart L., Piron M. & Morineau A. (2006). Statistique exploratoire multidimensionelle. Dunod Paris 4ed

    * Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1–18. https://doi.org/10.18637/jss.v025.i01

    * Pagès J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    * Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_hillsmith_ind, get_hillsmith_var, get_hillsmith, summaryHillSmith, dimdesc, predictHillSmith, supvarHillSmith 

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2005, decathlon, canines
    >>> from scientisttools import HillSmith
    >>> #PCA with HillSmith function
    >>> res_pca = HillSmith(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> res_pca.fit(decathlon)
    >>> #MCA with HillSmith function
    >>> res_mca = MillSmith(ind_sup=(27,28,29,30,31,32),sup_var=(6,7))
    >>> res_mca.fit(canines)
    >>> #Mixed Data with HillSmith function
    >>> res_mix = HillSmith(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_mix.fit(autos2005)
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
        self.quali_weights = quali_weights
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
        if is_quanti.shape[1]>0:
            for k in is_quanti.columns:
                X[k] = recodecont(X[k]).X
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert categorical variables to factor
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
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
        #Hill and Smith Analysis of Mixed Data (HillSmith)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of individuals
        n_rows = X.shape[0]

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
            ind_weights = array(list(map(lambda x : x/sum(self.ind_weights),self.ind_weights)))
            
        #convert to Series
        ind_weights = Series(ind_weights,index=X.index,name="weight")
        
        #recode variables
        rec = recodevarhillsmith(X=X,weights=ind_weights)

        #extract elements
        X, Z, X_quanti, X_quali, dummies, nb_moda, n_quanti, n_quali, center, scale  = rec.X, rec.Z, rec.quanti ,rec.quali, rec.dummies, rec.nb_moda, rec.k1, rec.k2, rec.center, rec.scale

        #set variables weights
        var_weights = Series(name="weight").astype("float")

        #set quantitative variables weights
        if n_quanti > 0:
            if self.quanti_weights is None:
                quanti_weights = ones(n_quanti)
            elif not isinstance(self.quanti_weights,(list,tuple,ndarray,Series)):
                raise TypeError("'quanti_weights' must be a list/tuple/1darray/Series of quantitative variables weights")
            elif len(self.quanti_weights) != n_quanti:
                raise TypeError(f"'quanti_weights' must be a list/tuple/1darray/Series with length {n_quanti}.")
            else:
                quanti_weights = array(self.quanti_weights)
            
            #convert to Series
            quanti_weights = Series(quanti_weights,index=X_quanti.columns)
            #concatenate
            var_weights = concat((var_weights,quanti_weights),axis=0)
        
        #set categoricals variables weights
        if n_quali > 0:
            #set qualitative variables weights
            if self.quali_weights is None:
                quali_weights = ones(n_quali)
            elif not isinstance(self.quali_weights,(list,tuple,ndarray,Series)):
                raise ValueError("'quali_weights' must be a list/tuple/1darray/Series of qualitative variables weights")
            elif len(self.quali_weights) != n_quali:
                raise TypeError(f"'quali_weights' must be a list/tuple/1darray/Series with length {n_quali}.")
            else:
                quali_weights = array(self.quali_weights)
        
            quali_weights2 = array(list(chain(*[repeat(i,k) for i, k in zip(quali_weights,nb_moda)])))
            p_k = mapply(dummies, lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            mod_weights = array([x*y for x,y in zip(p_k,quali_weights2)])
            #convert to Series
            mod_weights = Series(mod_weights,index=dummies.columns,name="weight")
            #concatenate
            var_weights = concat((var_weights,mod_weights),axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_rows - 1,  Z.shape[1] - n_quali))
        #et number of components
        if self.n_components is None:
            n_components =  max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,k1=n_quanti,k2=n_quali,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,sup_var=sup_var_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        z_center = average(Z,axis=0,weights=ind_weights)
        Zc = mapply(Z,lambda x : x - z_center, axis=1,progressbar=False,n_workers=n_workers)
        fit_ = fitfa(Zc,ind_weights,var_weights,max_components,n_components,n_workers)
        
        #extract elements
        self.svd_, self.eig_, self.ind_ = fit_.svd, fit_.eig, namedtuple("ind",fit_.row.keys())(*fit_.row.values())

        #statistics for active quantitative variables
        if n_quanti > 0:
            quanti_var_coord, quanti_var_ctr, quanti_var_sqcos, quanti_var_infos = fit_.col["coord"].iloc[:n_quanti,:], fit_.col["contrib"].iloc[:n_quanti,:], fit_.col["cos2"].iloc[:n_quanti,:], fit_.col["infos"].iloc[:n_quanti,:]
            #convert to ordered dictionary
            quanti_var_ = OrderedDict(coord=quanti_var_coord, contrib=quanti_var_ctr, cos2=quanti_var_sqcos,infos=quanti_var_infos)
            #convert to namedtuple
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        ##statistics for active levels - active qualitative variables
        if n_quali > 0:
            levels_coord, levels_ctr, levels_sqcos, levels_infos = fit_.col["coord"].iloc[n_quanti:,:], fit_.col["contrib"].iloc[n_quanti:,:], fit_.col["cos2"].iloc[n_quanti:,:], fit_.col["infos"].iloc[n_quanti:,:]
            #vtest for the active levels
            n_k = p_k*n_rows
            levels_vtest = mapply(mapply(levels_coord,lambda x : x*sqrt(((n_rows - 1)*n_k)/(n_rows - n_k)),axis=0,progressbar=False,n_workers=n_workers),
                                  lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
            #eta2 for the qualitative variables
            quali_var_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=ind_weights,n_workers=n_workers)
            #convert to OrderedDict
            quali_var_ = OrderedDict(coord=levels_coord,contrib=levels_ctr,cos2=levels_sqcos,vtest=levels_vtest,infos=levels_infos)
            #add squared correlation ratio if no quantitative variables
            if n_quanti == 0:
                quali_var_ = OrderedDict(**quali_var_, **OrderedDict(eta2=quali_var_sqeta))
            #convert to namedtuple
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())
    
        #statistics for variables
        if all(x > 0 for x in [n_quanti,n_quali]):
            #contributions of the qualitative variables
            quali_var_ctr = mapply(quali_var_sqeta,lambda x : 100*(x/self.eig_.iloc[:n_components,0]),axis=1,progressbar=False,n_workers=n_workers)
            #cos2 of the qualitative variables
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
            X_ind_sup_quanti, X_ind_sup_quali = split_ind_sup.quanti, split_ind_sup.quali
            #initialize the data
            Xcod_ind_sup = DataFrame().astype("float")
            if n_quanti > 0 and X_ind_sup_quanti is not None:
                Xcod_ind_sup = concat((Xcod_ind_sup,X_ind_sup_quanti),axis=1)
            if n_quali > 0 and X_ind_sup_quali is not None:
                #create disjunctive table
                dummies_ind_sup = DataFrame(zeros((X_ind_sup.shape[0],dummies.shape[1])),columns=dummies.columns,index=X_ind_sup.index)
                for i in range(X_ind_sup.shape[0]):
                    values = [X_ind_sup_quali.iloc[i,j] for j in range(n_quali)]
                    for k in range(dummies.shape[1]):
                        if dummies.columns[k] in values:
                            dummies_ind_sup.iloc[i,k] = 1
                Xcod_ind_sup = concat((Xcod_ind_sup, dummies_ind_sup),axis=1)
            
            #standardize the data (Z=(X-center)/scale)
            Z_ind_sup = mapply(Xcod_ind_sup,lambda x : ((x - center)/scale) - z_center,axis=1,progressbar=False,n_workers=n_workers)
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

            #recode supplementary variables
            rec_sup_var = recodevarhillsmith(X=X_sup_var,weights=ind_weights)
            #extract elements
            Z_sup_var, X_quali_sup, n_quanti_sup, n_quali_sup, dummies_sup  = rec_sup_var.Z, rec_sup_var.quali, rec_sup_var.k1, rec_sup_var.k2, rec_sup_var.dummies

            #coordinates for the supplementary columns
            col_sup_coord = mapply(Z_sup_var,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).T.dot(self.svd_.U)
            col_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            #dist2 for the supplementary levels
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
                #count levels
                n_k_sup = mapply(dummies_sup,lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)*n_rows
                #coordinates, dist2 and cos2 for the supplementary levels
                levels_sup_coord, levels_sup_sqdisto, levels_sup_sqcos = col_sup_coord.iloc[n_quanti_sup:,:],col_sup_sqdisto.iloc[n_quanti_sup:], col_sup_sqcos.iloc[n_quanti_sup:,:]
                #vtest for the supplementary levels
                levels_sup_vtest = mapply(mapply(levels_sup_coord,lambda x : x*sqrt(((n_rows - 1)*n_k_sup)/(n_rows - n_k_sup)),axis=0,progressbar=False,n_workers=n_workers),
                                          lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                #eta2 for supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=ind_weights,n_workers=n_workers) 
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(coord=levels_sup_coord,cos2=levels_sup_sqcos,vtest=levels_sup_vtest,dist2=levels_sup_sqdisto,eta2=quali_sup_sqeta)
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

        #descriptive statistics for the quantitative variables
        if is_quanti.shape[1] > 0:
            if self.ind_sup is not None:
                is_quanti = is_quanti.drop(index=ind_sup_label)
            self.summary_quanti_ = summarize(X=is_quanti)

        #multivariate goodness of fit
        if is_quali.shape[1] > 0:
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

        self.model_ = "hillsmith"
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

    def transform(self,X:DataFrame,y=None) -> DataFrame:
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

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == ncols
            raise ValueError("'columns' aren't aligned")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the HillSmith result")

        split_x = splitmix(X) #split X
        X_quanti, X_quali = split_x.quanti, split_x.quali
        
        #initial and extract active elements
        Xcod, dummies, n_quanti, n_quali = DataFrame().astype("float"), self.call_.dummies, self.call_.k1, self.call_.k2

        if X_quanti is not None :
            if n_quanti != X_quanti.shape[1]:
                raise TypeError("The number of quantitative variables must be the same")
            Xcod = concat((Xcod,X_quanti),axis=1)
        
        if X_quali is not None:
            if n_quali != X_quali.shape[1]:
                raise TypeError("The number of qualitative variables must be the same")
            
            X_quali = revaluate_cat_variable(X_quali)
            #create disjunctive table
            dummies_sup = DataFrame(zeros((X.shape[0],dummies.shape[1])),index=X.index,columns=dummies.columns)
            for i in range(X.shape[0]):
                values = [X_quali.iloc[i,j] for j in range(n_quali)]
                for k in range(dummies.shape[1]):
                    if dummies.columns[k] in values:
                        dummies_sup.iloc[i,k] = 1
            Xcod = concat((Xcod,dummies_sup),axis=1)
        #weighted average of Z
        z_center = average(self.call_.Z,axis=0,weights=self.call_.ind_weights)
        #standardize the data
        coord = mapply(Xcod,lambda x : (((x - self.call_.center)/self.call_.scale) - z_center)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord
    
def predictHillSmith(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Hill and Smith Analysis of Mixed Data (HillSmith)
    ---------------------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Hill and Smith Analysis of Mixed Data (HillSmith)

    Usage
    -----
    ```python
    >>> predictHillSmith(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class HillSmith

    `X`: pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Returns
    -------
    namedtuple of pandas DataFrames/Series containing all the results for the new individuals including:
    
    `coord`: coordinates for the new individuals,

    `cos2`: squared cosinus for the new individuals,

    `dist2`: squared distance to origin for the new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_autos2005
    >>> from scientisttools import HillSmith, predictHillSmith
    >>> autos2005 = load_autos200()
    >>> res_hillsmith = HillSmith(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_hillsmith.fit(autos2005)
    >>> #prediction on new individuals
    >>> ind_sup = load_autos2005("ind_sup")
    >>> predict = predictHillSmith(res_hillsmith,X=ind_sup)
    >>> predict.coord.head() #coordinates of the new individuals
    >>> predict.cos2.head() #cos2 of the new individuals
    >>> predict.dist2.head() #dist2 of the new individuals.
    ```
    """
    if self.model_ != "hillsmith": #check if self is an object of class HillSmith
        raise TypeError("'self' must be an object of class HillSmith")
    
    if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == ncols
        raise ValueError("'columns' aren't aligned")
    
    #find intersect
    intersect_col = list(set(X.columns.tolist()) & set(self.call_.X.columns.tolist()))
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables are not the same as the ones in the active variables of the PCA result")
    #reorder columns
    X = X.loc[:,self.call_.X.columns]

    #split X
    split_x = splitmix(X)
    X_quanti, X_quali = split_x.quanti, split_x.quali
    
    #initial and extract active elements
    Xcod, dummies, n_quanti, n_quali = DataFrame().astype("float"), self.call_.dummies, self.call_.k1, self.call_.k2

    if X_quanti is not None :
        if n_quanti != X_quanti.shape[1]:
            raise TypeError("The number of quantitative variables must be the same")
        Xcod = concat((Xcod,X_quanti),axis=1)
    
    if X_quali is not None:
        if n_quali != X_quali.shape[1]:
            raise TypeError("The number of qualitative variables must be the same")
        
        X_quali = revaluate_cat_variable(X_quali)
        #create disjunctive table
        dummies_sup = DataFrame(zeros((X.shape[0],dummies.shape[1])),index=X.index,columns=dummies.columns)
        for i in range(X.shape[0]):
            values = [X_quali.iloc[i,j] for j in range(n_quali)]
            for k in range(dummies.shape[1]):
                if dummies.columns[k] in values:
                    dummies_sup.iloc[i,k] = 1
        Xcod = concat((Xcod,dummies_sup),axis=1)
    #weighted average of Z
    z_center = average(self.call_.Z,axis=0,weights=self.call_.ind_weights)
    #standardize the data
    Z = mapply(Xcod,lambda x : ((x - self.call_.center)/self.call_.scale) - z_center,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #coordinates for the new individuals
    coord = mapply(Z,lambda x : x*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the new individuals
    sqdisto = mapply(Z,lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #cos2 for the new individuals
    sqcos = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)
    return namedtuple("predictHillSmithResult",["coord","cos2","dist2"])(coord,sqcos,sqdisto)
        
def supvarHillSmith(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables in Hill and Smith Analysis of Mixed Data (HillSmith)
    ----------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Hill and Smith Analysis of Mixed Data (HillSmith).

    Usage
    -----
    ```python
    >>> supvarHillSmith(self,X=None)
    ```

    Parameters
    ----------
    `self`: an object of class HillSmith

    `X`: pandas DataFrame of supplementary variables (quantitative and/or qualitative)

    Returns
    -------
    namedtuple of namedtuple containing all the results for supplementary variables including: 

    `quanti`: namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables including:
        * `coord`: coordinates for the supplementary quantitative variables,
        * `cos2`: squared cosinus for the supplementary quantitative variables
    
    `quali`: namedtuple of pandas DataFrames/Series containing all the results for the supplementary qualitative variables/levels including:
        * `coord`: coordinates for the supplementary levels,
        * `cos2`: squared cosinus for the supplementary levels,
        * `vtest`: value-test for the supplementary levels,
        * `dist2`: squared distance to origin of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_autos2005
    >>> from scientisttools import HillSmith, supvarHillSmith
    >>> autos2005 = load_autos2005()
    >>> res_hillsmith = HillSmith(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_hillsmith.fit(autos2005)
    >>> #prediction on supplementary variables (quantitative and qualitative)
    >>> X_sup_var = load_autos2005("sup_var")
    >>> sup_var_predict = supvarHillSmith(res_hillsmith,X=X_sup_var)
    >>> quanti_sup = sup_var_predict.quanti
    >>> quanti_sup.coord.head() #coordinates of the supplementary quantitative variables
    >>> quanti_sup.vos2.head() #cos2 of the supplementary quantitative variables
    >>> quali_sup = sup_var_predict.quali
    >>> quali_sup.coord.head() #coordinates of the supplementary levels
    >>> quali_sup.cos2.head() #cos2 of the supplementary levels
    >>> quali_sup.vtest.head() #vtest of the supplementary levels
    >>> quali_sup.dist2.head() #dist2 of the supplementary levels
    >>> quali_sup.eta2.head() #eta2 of the supplementary qualitative variables
    ```
    """
    if self.model_ != "hillsmith": #check if self is and object of class HillSmith
        raise TypeError("'self' must be an object of class HillSmith")
    
    if isinstance(X,Series): #if pandas series, transform to pandas dataframe
        X = X.to_frame()
        
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")
    
    #recode supplementary variables
    rec = recodevarhillsmith(X=X,weights=self.call_.ind_weights)
    #extract elements
    Z, X_quali, n_rows, n_quanti, n_quali, dummies  = rec.Z, rec.quali, rec.n, rec.k1, rec.k2, rec.dummies

    #coordinates for the supplementary columns
    col_coord = mapply(Z,lambda x : x*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).T.dot(self.svd_.U)
    col_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the supplementary levels
    col_sqdisto  = mapply(Z, lambda x : (x**2)*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
    col_sqdisto.name = "Sq. Dist."
    #cos2 for the supplementary columns
    col_sqcos = mapply(col_coord, lambda x : (x**2)/col_sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)

    #statistics for supplementary quantitative variables
    if n_quanti > 0:
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",["coord","cos2"])(col_coord.iloc[:n_quanti,:],col_sqcos.iloc[:n_quanti,:])
    else:
        quanti_sup = None

    #statistics for supplementary qualitative variables/levels
    if n_quali > 0:
        #count levels
        n_k = mapply(dummies,lambda x : x*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)*n_rows
        #coordinates, cos2 and dist2 for the supplementary levels
        levels_coord, levels_sqdisto, levels_sqcos = col_coord.iloc[n_quanti:,:], col_sqdisto.iloc[n_quanti:], col_sqcos.iloc[n_quanti:,:]
        #vtest for the supplementary levels
        levels_vtest = mapply(mapply(levels_coord,lambda x : x*sqrt(((n_rows - 1)*n_k)/(n_rows - n_k)),axis=0,progressbar=False,n_workers=self.call_.n_workers),
                             lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #eta2 for supplementary qualitative variables
        quali_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=self.call_.ind_weights,n_workers=self.call_.n_workers) 
        #convert to ordered dictionary
        quali_sup_ = OrderedDict(coord=levels_coord,cos2=levels_sqcos,vtest=levels_vtest,dist2=levels_sqdisto,eta2=quali_sqeta)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarHillSmithResult",["quanti","quali"])(quanti_sup,quali_sup)