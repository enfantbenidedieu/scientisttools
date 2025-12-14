# -*- coding: utf-8 -*-
from numpy import number,array,ndarray,average,sqrt,linalg,zeros,ones
from pandas import DataFrame,concat,Series
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.gfa import gfa
from .functions.splitmix import splitmix
from .functions.recodevar import recodevarfamd
from .functions.recodecont import recodecont
from .functions.summarize import summarize
from .functions.association import association
from .functions.revalue import revaluate_cat_variable
from .functions.conditional_wmean import conditional_wmean
from .functions.function_eta2 import function_eta2
from .functions.corrmatrix import corrmatrix

class FAMD(BaseEstimator,TransformerMixin):
    """
    Factor Analysis of Mixed Data (FAMD)
    ------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Factor Analysis of Mixed Data (FAMD) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    FAMD is a principal component method dedicated to explore data with both continuous and categorical variables. 
    It can be seen roughly as a mixed between PCA and MCA. More precisely, 
    the continuous variables are scaled to unit variance and the categorical variables are transformed 
    into a disjunctive data table (crisp coding) and then scaled using the specific scaling of MCA. 
    This ensures to balance the influence of both continous and categorical variables in the analysis. 
    It means that both variables are on a equal foot to determine the dimensions of variability. 
    This method allows one to study the similarities between individuals taking into account mixed 
    variables and to study the relationships between all the variables.

    Details
    -------
    FAMD includes standard Principal Component Analysis (PCA) and Multiple Correspondence Analysis (MCA) as special cases. If all variables are quantitative, standard PCA is performed.
    if all variables are qualitative, then standard MCA is performed.

    Missing values are replaced by means for quantitative variables. Note that, when all the variable are qualitative, the factor coordinates of the individuals are equal to the factor scores
    of standard MCA times squares root of J (the number of qualitatives variables) and the eigenvalues are then equal to the usual eigenvalues of MCA times J.
    When all the variables are quantitative, FAMD gives exactly the same results as normed PCA.

    Usage
    -----
    ```python
    >>> FAMD(n_components = 5, ind_weights = None, quanti_weights = None, quali_weights = None, ind_sup = None, sup_var = None, parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : None or an integer indicating the number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `quanti_weights`: an optional quantitative variables weights (by default, a list/tuple/array/Series of 1 for uniform quantitative variables weights), the weights are given only for the active quantitative variables

    `quali_weights`: an optional qualitative variables weights (by default, a list/tuple/array/Series of 1 for uniform qualitative variables weights), the weights are given only for the active qualitative variables

    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `sup_var`: an integer/string/list/tuple indicating the indexes/names of the supplementary variables (quantitative and/or qualitative)

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

    * Pagès J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    * Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_famd_ind, get_famd_var, get_famd, summaryFAMD, dimdesc, predictFAMD, supvarFAMD, fviz_famd_ind, fviz_famd_col, fviz_famd_mod, fviz_famd_var

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import autos2005, decathlon, canines
    >>> from scientisttools import FAMD
    >>> #PCA with FAMD function
    >>> res_pca = FAMD(ind_sup=(41,42,43,44,45),sup_var=(10,11,12))
    >>> res_pca.fit(decathlon)
    >>> #MCA with FAMD function
    >>> res_mca = FAMD(ind_sup=(27,28,29,30,31,32),sup_var=(6,7))
    >>> res_mca.fit(canines)
    >>> #Mixed Data with FAMD function
    >>> res_mix = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_mix.fit(autos2005)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 ind_weights = None,
                 quanti_weights = None,
                 quali_weights = None,
                 ind_sup = None,
                 sup_var = None):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.quanti_weights = quanti_weights
        self.quali_weights = quali_weights
        self.ind_sup = ind_sup
        self.sup_var = sup_var

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
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fill NA with mean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=number)
        if is_quanti.shape[1]>0:
            is_quanti = recodecont(X=is_quanti).X
            for k in is_quanti.columns:
                X[k] = is_quanti[k]
        
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
            elif isinstance(self.sup_var,range): #this is a range
                sup_var_label = X.columns[list(self.sup_var)].tolist()
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
            elif isinstance(self.ind_sup,range): #this is a range
                ind_sup_label = X.index[list(self.ind_sup)].tolist()
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
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)


        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Factor Analysis of Mixed Data (FAMD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of individuals
        n_rows = X.shape[0]

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

        #recode variables
        rec = recodevarfamd(X=X,weights=ind_weights)

        #extract elements
        X, Z, X_quanti, X_quali, dummies, nb_moda, n_quanti, n_quali, center, scale  = rec.X, rec.Z, rec.quanti ,rec.quali, rec.dummies, rec.nb_moda, rec.k1, rec.k2, rec.center, rec.scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
            
        #set levels weights
        if n_quali > 0:
            if self.quali_weights is None:
                quali_weights = ones(n_quali)
            elif not isinstance(self.quali_weights,(list,tuple,ndarray,Series)):
                raise ValueError("'quali_weights' must be a list/tuple/1darray/Series of qualitative variables weights")
            elif len(self.quali_weights) != n_quali:
                raise TypeError(f"'quali_weights' must be a list/tuple/1darray/Series with length {n_quali}.")
            else:
                quali_weights = array(self.quali_weights)
            #duplicate according to number of levels
            quali_weights = Series(array(list(chain(*[repeat(i,k) for i, k in zip(quali_weights,nb_moda)]))),index=dummies.columns)
            #concatenate
            var_weights = concat((var_weights,quali_weights),axis=0)

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
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,k1=n_quanti,k2=n_quali,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,sup_var=sup_var_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        z_center = average(Z,axis=0,weights=ind_weights)
        Zc = Z.sub(z_center,axis=1)
        fit_ = gfa(Zc,ind_weights,var_weights,max_components,n_components,n_workers)
        
        #extract elements
        self.svd_, self.eig_, self.ind_ = fit_.svd, fit_.eig, namedtuple("ind",fit_.row.keys())(*fit_.row.values())

        #statistics for active quantitative variables
        if n_quanti > 0:
            quanti_var_coord, quanti_var_ctr, quanti_var_sqcos = fit_.col["coord"].iloc[:n_quanti,:], fit_.col["contrib"].iloc[:n_quanti,:], fit_.col["cos2"].iloc[:n_quanti,:]
            #convert to ordered dictionary
            quanti_var_ = OrderedDict(coord=quanti_var_coord, contrib=quanti_var_ctr, cos2=quanti_var_sqcos,infos=fit_.col["infos"].iloc[:n_quanti,:])
            #convert to namedtuple
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #statistics for active levels - active qualitative variables
        if n_quali > 0:
            #proportion  of levels
            p_k = mapply(dummies, lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            #coordinates for the levels
            levels_coord = mapply(mapply(fit_.col["coord"].iloc[n_quanti:,:],lambda x : x/sqrt(p_k),axis=0,progressbar=False,n_workers=n_workers),
                                  lambda x : x*self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
            #vtest for the levels
            levels_vtest = mapply(mapply(levels_coord,lambda x : x*sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0,progressbar=False,n_workers=n_workers),
                                  lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
            #eta2 for the qualitative variables
            quali_var_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=ind_weights,n_workers=n_workers)
            #dist2 for levels
            barycentre = conditional_average(X=Z,Y=X_quali,weights=ind_weights)
            levels_sqdisto = mapply(barycentre, lambda x : ((x - z_center)**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            levels_sqdisto.name = "Sq. Dist."
            #cos2 for the levels
            levels_sqcos = mapply(levels_coord, lambda x : (x**2)/levels_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
            #convert to dictionary
            quali_var_ = OrderedDict(coord=levels_coord,contrib=fit_.col["contrib"].iloc[n_quanti:,:],cos2=levels_sqcos,vtest=levels_vtest,dist2=levels_sqdisto)
            # Add to qualitatives/categoricals variables if not continuous variables
            if n_quanti == 0:
                quali_var_["eta2"] = quali_var_sqeta
            #convert to namedtuple
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #statistics for variables
        if all(x > 0 for x in [n_quanti,n_quali]):
            #contrib of the qualitative variables
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
                dummies_ind_sup = DataFrame(zeros((len(ind_sup_label),dummies.shape[1])),columns=dummies.columns,index=ind_sup_label)
                for i in range(len(ind_sup_label)):
                    values = [X_ind_sup_quali.iloc[i,j] for j in range(n_quali)]
                    for k in range(dummies.shape[1]):
                        if dummies.columns[k] in values:
                            dummies_ind_sup.iloc[i,k] = 1
                Xcod_ind_sup = concat((Xcod_ind_sup,dummies_ind_sup),axis=1)
            
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
            rec2 = recodevarfamd(X=X_sup_var,weights=ind_weights)

            #extract elements
            Z_sup_var, X_quali_sup, dummies_sup, n_quanti_sup, n_quali_sup  = rec2.Z, rec2.quali, rec2.dummies, rec2.k1, rec2.k2

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

            #statistics for supplementary levels - supplementary qualitative variables
            if n_quali_sup > 0:
                #proportion of supplementary levels
                p_k_sup = mapply(dummies_sup, lambda x : x*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
                #coordinates for the supplementary levels
                levels_sup_coord = mapply(mapply(col_sup_coord.iloc[n_quanti_sup:,:],lambda x : x/sqrt(p_k_sup),axis=0,progressbar=False,n_workers=n_workers),
                                          lambda x : x*self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                #vtest for the supplementary levels
                levels_sup_vtest = mapply(mapply(levels_sup_coord,lambda x : x*sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0,progressbar=False,n_workers=n_workers),
                                          lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=ind_weights,n_workers=n_workers)
                #dist2 for supplementary levels
                bary_sup = conditional_average(X=Z,Y=X_quali_sup,weights=ind_weights)
                levels_sup_sqdisto = mapply(bary_sup, lambda x : ((x - z_center)**2)*var_weights,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
                levels_sup_sqdisto.name = "Sq. Dist."
                #cos2 for the supplementary levels
                levels_sup_sqcos = mapply(levels_sup_coord, lambda x : (x**2)/levels_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
                #convert to dictionary
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
                  
        self.model_ = "famd"
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
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")

        split_x = splitmix(X) #split X
        X_quanti, X_quali = split_x.quanti, split_x.quali
        
        #extract active elements
        dummies, n_quanti, n_quali = self.call_.dummies, self.call_.k1, self.call_.k2

        #create code variables
        Xcod = DataFrame().astype("float")
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
        #average of Z
        z_center = average(self.call_.Z,axis=0,weights=self.call_.ind_weights)
        #standardize the data
        coord = mapply(Xcod,lambda x : (((x - self.call_.center)/self.call_.scale) - z_center)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord

def predictFAMD(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Factor Analysis of Mixed Data (FAMD)
    --------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin for new individuals with Factor Analysis of Mixed Data (FAMD)

    Usage
    -----
    ```python
    >>> predictFAMD(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    namedtuple of pandas Dataframes/Series containing all the results for the new individuals including:
    
    `coord`: coordinates for the new individuals,

    `cos2`: squared cosinus for the new individuals,

    `dist2`: squared distance to origin for the new individuals.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_autos2005
    >>> from scientisttools import FAMD, predictFAMD
    >>> autos2005 = load_autos2005()
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_famd.fit(autos2005)
    >>> #prediction for the new individuals
    >>> X_ind_sup = load_autos2005("ind_sup")
    >>> predict = predictFAMD(res_famd,X=X_ind_sup)
    >>> predict.coord.head() #coord of new individuals
    >>> predict.cos2.head() #cos2 of new individuals
    >>> predict.contrib.head() #contrib of new individuals
    ```
    """
    if self.model_ != "famd": #check if self is an object of class FAMD
        raise TypeError("'self' must be an object of class FAMD")

    if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    X.index.name = None #set index name as None

    if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == ncols
        raise ValueError("'columns' aren't aligned")
    
    intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the FAMD result")

    split_x = splitmix(X) #split X
    X_quanti, X_quali = split_x.quanti, split_x.quali

    # Extract active elements
    dummies, n_quanti, n_quali = self.call_.dummies, self.call_.k1, self.call_.k2

    #create code variables
    Xcod = DataFrame().astype("float")
    if X_quanti is not None:
        if n_quanti != X_quanti.shape[1]:
            raise TypeError("The number of continuous columns must be the same")
        Xcod = concat((Xcod,X_quanti),axis=1)
    
    if X_quali is not None:
        if n_quali != X_quali.shape[1]:
            raise TypeError("The number of qualitatives columns must be the same")
        
        X_quali = revaluate_cat_variable(X_quali)
        #create disjunctive table
        dummies_sup = DataFrame(zeros((X.shape[0],dummies.shape[1])),index=X.index,columns=dummies.columns)
        for i in range(X.shape[0]):
            values = [X_quali.iloc[i,j] for j in range(n_quali)]
            for k in range(dummies.shape[1]):
                if dummies.columns[k] in values:
                    dummies_sup.iloc[i,k] = 1
        Xcod = concat((Xcod,dummies_sup),axis=1)

    #average of Z
    z_center = average(self.call_.Z,axis=0,weights=self.call_.ind_weights)
    #standardize the data
    Z = mapply(Xcod,lambda x : ((x - self.call_.center)/self.call_.scale) - z_center,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #coord for the new individuals
    coord = mapply(Z,lambda x : x*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the new individuals
    sqdisto = mapply(Z, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #cos2 for the new individuals
    sqcos = mapply(coord,lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)
    #return namedtuple
    return namedtuple("predictFAMDResult",["coord","cos2","dist2"])(coord,sqcos,sqdisto)

def supvarFAMD(self,X:DataFrame) -> NamedTuple:
    """
    Supplementary variables in Factor Analysis of Mixed Data (FAMD)
    ---------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables (quantitative and/or qualitative) with Factor Analysis of Mixed Data (FAMD)

    Usage
    -----
    ```python
    >>> supvarFAMD(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class FAMD

    `X`: pandas DataFrame of supplementary variables (quantitative and/or qualitative)

    Returns
    -------
    namedtuple of namedtuple containing all the results for supplementary variables (quantitative and/or qualitative) including : 

    `quanti`: namedtuple of pandas DataFrame containing all the results for the supplementary quantitative variables including :
        * `coord`: coordinates for the supplementary quantitative variables,
        * `cos2`: square cosinus for the supplementary quantitative variables.
    
    `quali`: namedtuple pf pandas DataFrame/Series containing all the results for the supplementary qualitative variables including :
        * `coord`: coordinates for the supplementary levels,
        * `cos2`: square cosinus for the supplementary levels,
        * `vtest`: value-test for the supplementary levels,
        * `dist2`: squared distance to origin for the supplementary levels,
        * `eta2`: squared correlation ratio for the supplementary qualitative variables.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools.datasets import load_autos2005
    >>> from scientisttools import FAMD, supvarFAMD
    >>> autos2005 = load_autos2005()
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),sup_var=(12,13,14,15))
    >>> res_famd.fit(autos2005)
    >>> #predict for the supplementary variables (quantitative and qualitative)
    >>> X_sup_var = load_autos2005("sup_var")
    >>> sup_var_predict = supvarFAMD(res_famd, X=X_sup_var)
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
    if self.model_ != "famd": #check if self is an object of class FAMD
        raise TypeError("'self' must be an object of class FAMD")
    
    if isinstance(X,Series): #if pandas series, transform to pandas dataframe
        X = X.to_frame()
        
    if not isinstance(X,DataFrame): #check if X is an instance of pd.DataFrame class
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    if X.shape[0] != self.call_.X.shape[0]: #check if X.shape[0] = nrows
        raise ValueError("'rows' aren't aligned")
    
    #recode variables
    rec = recodevarfamd(X=X,weights=self.call_.ind_weights)

    #extract elements
    Z, X_quali, dummies, n_rows, n_quanti, n_quali  = rec.Z, rec.quali, rec.dummies, rec.n, rec.k1, rec.k2

    #coordinates for the supplementary columns
    coord = mapply(Z,lambda x : x*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).T.dot(self.svd_.U)
    coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
    #dist2 for the supplementary levels
    sqdisto  = mapply(Z, lambda x : (x**2)*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
    sqdisto.name = "Sq. Dist."
    #cos2 for the supplementary columns
    sqcos = mapply(coord, lambda x : (x**2)/sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)

    #statistics for supplementary quantitative variables
    if n_quanti > 0:
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",["coord","cos2"])(coord.iloc[:n_quanti,:],sqcos.iloc[:n_quanti,:])
    else:
        quanti_sup = None

    #statistics for supplementary levels - supplementary qualitative variables
    if n_quali > 0:
        #proportion of supplementary levels
        p_k = mapply(dummies, lambda x : x*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
        #coordinates for the supplementary levels
        levels_coord = mapply(mapply(coord.iloc[n_quanti:,:],lambda x : x/sqrt(p_k),axis=0,progressbar=False,n_workers=self.call_.n_workers),
                                    lambda x : x*self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #vtest for the supplementary levels
        levels_vtest = mapply(mapply(levels_coord,lambda x : x*sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0,progressbar=False,n_workers=self.call_.n_workers),
                                    lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #eta2 for the supplementary qualitative variables
        quali_var_sqeta = function_eta2(X=X_quali,Y=self.ind_.coord,weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #dist2 for supplementary levels
        bary = conditional_average(X=self.call_.Z,Y=X_quali,weights=self.call_.ind_weights)
        z_center = average(self.call_.Z,axis=0,weights=self.call_.ind_weights)
        levels_sqdisto = mapply(bary, lambda x : ((x - z_center)**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
        levels_sqdisto.name = "Sq. Dist."
        #cos2 for the supplementary levels
        levels_sqcos = mapply(levels_coord, lambda x : (x**2)/levels_sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)
        #convert to dictionary
        quali_sup_ = OrderedDict(coord=levels_coord,cos2=levels_sqcos,vtest=levels_vtest,dist2=levels_sqdisto,eta2=quali_var_sqeta)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None

    #convert to namedtuple
    return namedtuple("supvarFAMDResult",["quanti","quali"])(quanti_sup,quali_sup)