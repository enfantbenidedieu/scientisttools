# -*- coding: utf-8 -*-
from numpy import array,ones,number,ndarray,c_,cumsum,sqrt,zeros, unique
from pandas import DataFrame,Series,Categorical,concat,crosstab,get_dummies,api
from itertools import chain, repeat
from scipy.stats import chi2, chi2_contingency, contingency
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.association import association
from .functions.summarize import summarize
from .functions.revaluate_cat_variable import revaluate_cat_variable
from .functions.fitfa import fitfa
from .functions.function_eta2 import function_eta2
from .functions.predict_sup import predict_ind_sup, predict_quanti_sup, predict_quali_sup
from .functions.recodecont import recodecont

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)
    -------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Correspondence Analysis (MCA) or Specific Multiple Correspondence Analysis (SpecificMCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    >>> MCA(n_components = 5, excl = None, ind_weights = None, var_weights = None, ind_sup = None, quali_sup = None, quanti_sup = None, parallelize = False)
    ```

    Parameters
    ----------
    `n_components`: number of dimensions kept in the results (by default 5)

    `excl`: an integer or a list indicating the "junk" categories (by default None). It can be a list/tuple of the names of the categories or a list/tuple of the indexes in the disjunctive table.

    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `quali_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary categorical variables

    `quanti_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary quantitative variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Atttributes
    -----------
    `call_`: a namedtuple with some informations
        * `Xtot`: pandas DataFrame with all data (active and supplementary)
        * `X`: pandas dataframe with active data
        * `dummies`: pandas DataFrame with disjunctive table
        * `Z`: pandas DataFrame with standardized data:
        * `ind_weights`: pandas Series containing individuals weights
        * `var_weights`: pandas Series containing variables weights
        * `mod_weights`: pandas Series containing modalities weights
        * `excl`: None or a list of string indicating names of the excluded categories
        * `n_components`: an integer indicating the number of components kept
        * `n_workers`: an integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals
        * `quali_sup`: None or a list of string indicating names of the supplementary qualitative variables
        * `quanti_sup`: None or a list of string indicating names of the supplementary quantitative variables
    
    `svd_`: a namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD)
        * `vs`: 1D numpy array containing the singular values
        * `U`: 2D numpy array whose columns contain the left singular vectors
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: a pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `eig_correction_`: a namedtuple of pandas DataFrame containing eigenvalues correction.
        * `benzecri`: Benzecri correction
        * `greenacre`: Greenacre correction

    `ind_`: a namedtuple of pandas Dataframe containing all the results for the active individuals.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `contrib`: relative contributions
        * `infos`: additionals informations (weight, squared distance to origin and inertia)

    `var_`: a namedtuple of pandas DataFrame containing all the results for the active variables.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `contrib`: relative contributions
        * `infos`: additionnal informations (weight, squared distance to origin, inertia)
        * `coord_n`: Normalized factor coordinates
        * `vtest`: value-test (v-test)
        * `eta2`: squared correlation ratio
        * `var_inertia`: inertia of qualitative
        * `var_contrib`: contributions of qualitative

    `others_`: a namedtuple of others statistics.
        * `inertia`: global multiple correspondence analysis inertia
        * `kaiser`: namedtuple of numerics values containing the kaiser threshold:
            * `threshold`: kaiser threshold
            * `proportion`: kaiser proportion threshold

    `ind_sup_`: if ind_sup is not None, a namedtuple of pandas Dataframe containing all the results for the supplementary individuals.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `dist`: squared distance to origin

    `quali_sup_`: if quali_sup is not None, a namedtuple of pandas DataFrame containing all the results for the supplementary categorical variables.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `vtest`: value-test
        * `dist`: squared distance to origin
        * `eta2`: squared correlation ratio

    `quanti_sup_`: if quanti_sup is not None, a namedtuple of pandas DataFrame containing all the results for the supplementary quantitative variables.
        * `coord`: factor coordinates
        * `cos2`: squared cosinus

    `summary_quanti_` : summary statistics for quantitative variables if quanti_sup is not None
    
    `summary_quali_`: summary statistics for qualitative variables (active and supplementary)

    `chi2_test_`: chi-squared test.

    `model_`: string specifying the model fitted = 'mca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    * Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Le Roux B. and Rouanet H., Geometric Data Analysis: From Correspondence Analysis to Stuctured Data Analysis, Kluwer Academic Publishers, Dordrecht (June 2004).

    * Le Roux B. and Rouanet H., Multiple Correspondence Analysis, SAGE, Series: Quantitative Applications in the Social Sciences, Volume 163, CA:Thousand Oaks (2010).

    * Le Roux B. and Jean C. (2010), Développements récents en analyse des correspondances multiples, Revue MODULARD, Numéro 42

    * Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    * Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    `predictMCA`, `supvarMCA`, `get_mca_ind`, `get_mca_var`, `get_mca`, `summaryMCA`, `dimdesc`, `fviz_mca_ind`, `fviz_mca_var`, `fviz_mca_quali_var`, `fviz_mca`

    Examples
    --------
    ```python
    >>> #load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> #multiple correspondence analysis (MCA)
    >>> from scientisttools import MCA, summaryMCA
    >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
    >>> res_mca.fit(poison)
    >>> summaryMCA(res_mca)
    >>> #specific multiple correspondence analysis (SpecificMCA)
    >>> res_specmca = MCA(excl=(0,2),quali_sup = (13,14),quanti_sup=(0,1))
    >>> res_specmca.fit(poison)
    >>> summaryMCA(res_specmca)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 excl = None,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 quali_sup = None,
                 quanti_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.excl = excl
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.quali_sup = quali_sup
        self.quanti_sup = quanti_sup
        self.parallelize = parallelize

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
        
        Examples
        --------
        ```python
        >>> #load poison dataset
        >>> from scientisttools import load_poison
        >>> poison = load_poison()
        >>> #multiple correspondence analysis (MCA)
        >>> from scientisttools import MCA
        >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
        >>> res_mca.fit(poison)
        >>> #specific multiple correspondence analysis (SpecificMCA)
        >>> res_specmca = MCA(excl=(0,2),quali_sup=(13,14),quanti_sup=(0,1))
        >>> res_specmca.fit(poison)
        ```
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
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
        #checks if quantitatives variables are in X
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=number)
        if is_quanti.shape[1]>0:
            for j in is_quanti.columns.tolist():
                X[j] = X[j].astype("float")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert categorical variables to factor
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        for q in is_quali.columns:
            X[q] = Categorical(X[q],categories=sorted(X[q].dropna().unique().tolist()),ordered=True)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary qualitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            if isinstance(self.quali_sup,str):
                quali_sup_label = [self.quali_sup]
            elif isinstance(self.quali_sup,(int,float)):
                quali_sup_label = [X.columns[int(self.quali_sup)]]
            elif isinstance(self.quali_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.quali_sup):
                     quali_sup_label = [str(x) for x in self.quali_sup]
                elif all(isinstance(x,(int,float)) for x in self.quali_sup):
                    quali_sup_label = X.columns[[int(x) for x in self.quali_sup]].tolist()
        else:
            quali_sup_label = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary quantitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            if isinstance(self.quanti_sup,str):
                quanti_sup_label = [self.quanti_sup]
            elif isinstance(self.quanti_sup,(int,float)):
                quanti_sup_label = [X.columns[int(self.quanti_sup)]]
            elif isinstance(self.quanti_sup,(list,tuple)):
                if all(isinstance(x,str) for x in self.quanti_sup):
                    quanti_sup_label = [str(x) for x in self.quanti_sup]
                elif all(isinstance(x,(int,float)) for x in self.quanti_sup):
                    quanti_sup_label = X.columns[[int(x) for x in self.quanti_sup]].tolist()
        else:
            quanti_sup_label = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary individuals
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

        ##Drop supplementary qualitative variables
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        #Drop supplementary quantitatives columns
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
        #drop supplementary individuls  
        if self.ind_sup is not None:
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##Multiple Correspondence Analysis (MCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if all columns are categoricals
        all_cat = all(api.types.is_string_dtype(X[q]) for q in X.columns)
        if not all_cat:
            raise TypeError("All active columns in `X` must be categoricals")
        
        #number of rows/columns
        n_rows, n_cols = X.shape

        #check if two categorical variables have same categories
        X = revaluate_cat_variable(X)

        #disjunctive table
        dummies = concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
        #number of categories, count and proportion
        n_cat, n_k, p_k = dummies.shape[1], dummies.sum(axis=0), dummies.mean(axis=0)
        n_k.name , p_k.name = "count", "proportion"

        #standardize the data
        Z = mapply(dummies,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=n_workers)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##set exclusion
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.excl is not None:
            if isinstance(self.excl,str):
                excl_label = [self.excl]
            elif isinstance(self.excl,(int,float)):
                excl_label = [Z.columns[int(self.excl)]]
            elif isinstance(self.excl,(list,tuple)):
                if all(isinstance(x,str) for x in self.excl):
                    excl_label = [str(x) for x in self.excl]
                elif all(isinstance(x,(int,float)) for x in self.excl):
                    excl_label = Z.columns[[int(x) for x in self.excl]].tolist()
            #set exclusion index
            excl_idx = [Z.columns.tolist().index(x) for x in excl_label]
        else:
            excl_label = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list/tuple/array/Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array/Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##set variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = ones(n_cols)/n_cols
        elif not isinstance(self.var_weights,(list,tuple,ndarray)):
            raise ValueError("'var_weights' must be a list/tuple/array of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array with length {n_cols}.")
        else:
            var_weights = array([x/sum(self.var_weights) for x in self.var_weights])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##set categories weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        nb_mod = array([X[j].nunique() for j in X.columns])
        var_weights2 = array(list(chain(*[repeat(i,k) for i, k in zip(var_weights,nb_mod)])))
        mod_weights = array([x*y for x,y in zip(p_k,var_weights2)])

        #replace excluded categories weights by 0
        if self.excl is not None:
            for i in excl_idx:
                mod_weights[i] = 0
        
        #convert weights to Series
        ind_weights, var_weights, mod_weights =  Series(ind_weights,index=X.index,name="weight"), Series(var_weights,index=X.columns,name="weight"), Series(mod_weights,index=Z.columns,name="weight")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        max_components = n_cat - n_cols
        if self.n_components is None:
            n_components =  int(max_components)
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))

        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Z=Z, ind_weights=ind_weights,var_weights=var_weights,mod_weights=mod_weights,
                            excl=excl_label,n_components=n_components,n_workers=n_workers,ind_sup=ind_sup_label,quali_sup=quali_sup_label,quanti_sup=quanti_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## fit factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z,ind_weights,mod_weights,max_components,n_components,n_workers)
        
        # Extract elements
        self.svd_, self.eig_, ind_, var_ = fit_.svd, fit_.eig, fit_.row, fit_.col

        #replace nan or inf by 0
        if self.excl is not None:
            self.svd_.V[excl_idx,:] = 0

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##Eigenvalues corrections
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # save eigen value grather than threshold
        kaiser_threshold = 1/n_cols
        lambd = self.eig_.iloc[:,0][self.eig_.iloc[:,0]>kaiser_threshold]

        #Add elements
        if self.excl is not None:
            # Add modified rated
            self.eig_["modified rates"] = 0.0
            self.eig_["cumulative modified rates"] = 100.0
            pseudo = (n_cols/(n_cols-1)*(lambd-1/n_cols))**2
            self.eig_.iloc[:len(lambd),4] = 100*pseudo/sum(pseudo)
            self.eig_.iloc[:,5] = cumsum(self.eig_.iloc[:,4])

        #benzecri correction
        lambd_tilde = ((n_cols/(n_cols-1))*(lambd - kaiser_threshold))**2
        s_tilde = 100*(lambd_tilde/sum(lambd_tilde))
        benzecri = DataFrame(c_[lambd_tilde,s_tilde,cumsum(s_tilde)],columns=["eigenvalue","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])
        #greenacre correction
        s_tilde_tilde = n_cols/(n_cols-1)*(sum(self.eig_.iloc[:,0]**2)-(n_cat-n_cols)/(n_cols**2))
        tau = 100*(lambd_tilde/s_tilde_tilde)
        greenacre = DataFrame(c_[lambd_tilde,tau,cumsum(tau)],columns=["eigenvalue","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])
        #convert to namedtuple
        self.eig_correction_ = namedtuple("correction",["benzecri","greenacre"])(benzecri,greenacre)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals - convert to NamedTuple
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        var_coord_n = mapply(var_["coord"],lambda x: x*self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)
        #categoris variables value - test
        var_vtest = mapply(var_["coord"],lambda x : x*sqrt(((n_rows-1)*n_k)/(n_rows-n_k)),axis=0,progressbar=False,n_workers=n_workers)
        #variables squared correlation ratio
        quali_var_eta2 = function_eta2(X=X,Y=ind_["coord"],weights=ind_weights,excl=excl_label,n_workers=n_workers)
        #contribution des variables
        quali_var_contrib = DataFrame().astype("float")
        for j in X.columns:
            contrib = var_["contrib"].loc[X[j].unique(),:].sum(axis=0).to_frame(j).T
            quali_var_contrib = concat((quali_var_contrib,contrib),axis=0)
        #inertia for the variables
        quali_var_inertia = Series((nb_mod - 1)/n_rows,index=X.columns,name="inertia")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.excl is not None:
            var_ = OrderedDict(coord=var_["coord"].drop(index=excl_label),cos2=var_["cos2"].drop(index=excl_label),contrib=var_["contrib"].drop(index=excl_label),infos=var_["infos"].drop(index=excl_label),
                               coord_n=var_coord_n.drop(index=excl_label),vtest=var_vtest.drop(index=excl_label),eta2=quali_var_eta2,var_inertia=quali_var_inertia,var_contrib=quali_var_contrib)
        else:
            var_ = OrderedDict(**var_,**OrderedDict(coord_n=var_coord_n,vtest=var_vtest,eta2=quali_var_eta2,var_inertia=quali_var_inertia,var_contrib=quali_var_contrib))  
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##multiple correspondence analysis additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #inertia
        inertia = (n_cat/n_cols) - 1
        #eigenvalue threshold
        kaiser_proportion_threshold = 100/inertia
        #convert to namedtuple
        self.others_ = namedtuple("others",["inertia","kaiser"])(inertia,namedtuple("kaiser",["threshold","proportion_threshold"])(kaiser_threshold,kaiser_proportion_threshold))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #disjunctive table
            Y = DataFrame(zeros((len(ind_sup_label),n_cat)),columns=dummies.columns,index=ind_sup_label)
            for i in range(len(ind_sup_label)):
                values = [X_ind_sup.iloc[i,j] for j in range(n_cols)]
                for k in range(n_cat):
                    if dummies.columns[k] in values:
                        Y.iloc[i,k] = 1
            #standardize the data and exclude the data
            Z_ind_sup = mapply(Y,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #square distance to origin for supplementary individuals
            ind_sup_sqdisto = mapply(Z_ind_sup, lambda x : (x**2)*self.call_.mod_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
            ind_sup_sqdisto.name = "Sq. Dist."
            #statistics for supplementary individuals
            ind_sup_ = predict_ind_sup(Z=Z_ind_sup,V=self.svd_.V,sqdisto=ind_sup_sqdisto,col_weights=self.call_.mod_weights,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)
            #reevaluate if two variables have the same level
            X_quali_sup = revaluate_cat_variable(X_quali_sup)
            #compute disjunctive tables
            X_quali_dummies = concat((get_dummies(X_quali_sup[j],dtype=int) for j in quali_sup_label),axis=1)
            quali_sup_n_k, quali_sup_p_k = X_quali_dummies.sum(axis=0), X_quali_dummies.mean(axis=0)
            #standardiz data
            Z_quali_sup = mapply(X_quali_dummies,lambda x : (x/quali_sup_p_k)-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #supplementary categories factor coordinates
            quali_sup_coord = mapply(mapply(X_quali_dummies,lambda x : x/sum(x),axis=0,progressbar=False,n_workers=self.call_.n_workers).T.dot(self.ind_.coord),lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #supplementary categories squared distance to origin
            quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : (x**2)*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
            quali_sup_sqdisto.name = "Sq. Dist."
            #coefficients
            coef_k = sqrt(((n_rows-1)*quali_sup_n_k)/(n_rows-quali_sup_n_k))
            #statistics for supplementary categories
            quali_sup_ = predict_quali_sup(X=X_quali_sup,Y=self.ind_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary quantitatives variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            #fill missing with mean
            X_quanti_sup = recodecont(X_quanti_sup.astype("float")).X
            # Compute weighted average and and weighted standard deviation
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            # Standardization
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=self.call_.n_workers)
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_quanti_sup(Z=Z_quanti_sup,U=self.svd_.U,row_weights=ind_weights,n_workers=self.call_.n_workers)
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            #descriptive statistics for quantitative variables
            self.summary_quanti_ = summarize(X=X_quanti_sup)

        #multivariate goodness of fit
        if self.ind_sup is not None:
            is_quali = is_quali.drop(index=ind_sup_label)
        self.goodness_ = association(X=is_quali,alpha=0.05)
        self.summary_quali_ = summarize(X=is_quali)
        self.model_ = "mca"
        
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
        `X_new`: pandas dataframe of shape (n_samples, n_components)
            Transformed values.

        Examples
        --------
        ```python
        >>> #load poison dataset
        >>> from scientisttools import load_poison
        >>> poison = load_poison()
        >>> #multiple correspondence analysis (MCA)
        >>> from scientisttools import MCA
        >>> res_mca = MCA(quali_sup=(2,3),quanti_sup =(0,1))
        >>> ind_coord = res_mca.fit_transform(poison)
        >>> #specific multiple correspondence analysis (SpecificMCA)
        >>> res_specmca = MCA(excl=(0,2),quali_sup = (13,14),quanti_sup=(0,1))
        >>> ind_coord = res_specmca.fit_transform(poison)
        ```
        """
        self.fit(X)
        return self.ind_.coord
    
    def inverse_transform(self,X:DataFrame) -> DataFrame:
        """
        Transform data back to its original disjunctive 
        -----------------------------------------------

        Description
        -----------
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: pandas DataFrame of shape (n_samples, n_categories)
            Original data, where `n_samples` is the number of samples and `n_categories` is the number of categories in original disjunctive table
        
        Examples
        --------
        ```
        >>> #load poison dataset
        >>> from scientisttools import load_poison
        >>> poison = load_poison()
        >>> from scientisttools import MCA
        >>> res_mca = MCA(n_components=None,quali_sup=(2,3), quanti_sup=(0,1))
        >>> res_mca.fit(poison)
        >>> X_disjunctive = res_pca.inverse_transform(res_mca.ind_.coord)
        ```
        """
        # Check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #set number of components
        n_components = min(X.shape[1],self.call_.n_components)
        #inverse transform
        X_original = X.iloc[:,:n_components].dot(mapply(self.var_.coord.iloc[:,:n_components],lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers).T)
        #estimation of standardize data
        X_original = mapply(X_original.add(1),lambda x : x*self.call_.dummies.mean(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #disjunctive table
        X_original = (X_original > (self.call_.X.shape[1]/self.call_.dummies.shape[1])).astype(int)
        return X_original

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameter
        ----------
        `X`: pandas DataFrame of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: pandas DataFrame of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.

        Examples
        --------
        ```python
        >>> #load poison dataset
        >>> from scientisttools import load_poison
        >>> poison = load_poison()
        >>> #multiple correspondence analysis (MCA)
        >>> from scientisttools import MCA
        >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
        >>> res_mca.fit(poison)
        >>> ind_coord = res_mca.transform(res_mca.call_.X)
        >>> #specific multiple correspondence analysis (SpecificMCA)
        >>> res_specmca = MCA(excl=(0,2),quali_sup = (13,14),quanti_sup=(0,1))
        >>> res_specmca.fit(poison)
        >>> ind_coord = res_specmca.transform(res_specmca.call_.X)
        ```
        """
        #check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
           raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None
        
        #check if X.shape[1] == n_cols
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")
     
        #check if all columns are categoricals
        all_cat = all(api.types.is_string_dtype(X[q]) for q in X.columns)
        if not all_cat:
            raise TypeError("All columns in `X` must be categoricals")
        
        #find intersect
        intersect_col = [x for x in X.columns if x in self.call_.X.columns]
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the MCA result")
        #reorder columns
        X = X.loc[:,self.call_.X.columns]

        #create disjunctive table
        Y = DataFrame(zeros((X.shape[0],self.call_.dummies.shape[1])),columns=self.call_.dummies.columns,index=X.index)
        for i in range(X.shape[0]):
            values = [X.iloc[i,j] for j in range(X.shape[1])]
            for k in range(self.call_.dummies.shape[1]):
                if self.call_.dummies.columns[k] in values:
                    Y.iloc[i,k] = 1
        #standardization (z_ik = (x_ik/pk)-1) and apply transition relation
        coord = mapply(Y,lambda x : ((x/self.call_.dummies.mean(axis=0))-1)*self.call_.mod_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns  = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord
    
def predictMCA(self,X:DataFrame) -> NamedTuple:
    """
    Predict projection for new individuals with Multiple Correspondence Analysis (MCA)
    ----------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Multiple Correspondence Analysis (MCA)

    Usage
    -----
    ```python
    >>> predictMCA(self,X)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `X`: a pandas DataFrame in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Returns
    -------
    namedtuple of pandas DataFrames containing all the results for the new individuals including:
    
    `coord`: factor coordinates

    `cos2`: squared cosines

    `dist`: squared distance
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    #check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")

    #check if X is an instance of pd.DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #set index name as None
    X.index.name = None

    #check if X.shape[1] == n_cols
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")

    #check if all columns are categoricals
    all_cat = all(api.types.is_string_dtype(X[q]) for q in X.columns)
    if not all_cat:
        raise TypeError("All columns in `X` must be categoricals")
    
    #find intersect
    intersect_col = [x for x in X.columns if x in self.call_.X.columns]
    if len(intersect_col) != self.call_.X.shape[1]:
        raise ValueError("The names of the variables is not the same as the ones in the active variables of the MCA result")
    #reorder columns
    X = X.loc[:,self.call_.X.columns]
    
    #create disjunctive table for new individuals
    Y = DataFrame(zeros((X.shape[0],self.call_.dummies.shape[1])),columns=self.call_.dummies.columns,index=X.index)
    for i in range(X.shape[0]):
        values = [X.iloc[i,j] for j in range(X.shape[1])]
        for k in range(self.call_.dummies.shape[1]):
            if self.call_.dummies.columns[k] in values:
                Y.iloc[i,k] = 1
    
    #standardization z_ik = (x_ik/pk)-1
    Z = mapply(Y,lambda x : (x/self.call_.dummies.mean(axis=0))-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #squared distance to origin for new individuals
    sqdisto = mapply(Z, lambda x : (x**2)*self.call_.mod_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #statistic for new individuals
    predict_ = predict_ind_sup(Z=Z,V=self.svd_.V,sqdisto=sqdisto,col_weights=self.call_.mod_weights,n_workers=self.call_.n_workers)
    #convert to NamedTuple
    return namedtuple("predictMCAResult",predict_.keys())(*predict_.values())

def supvarMCA(self,X_quanti_sup=None,X_quali_sup=None) -> NamedTuple:
    """
    Supplementary variables in Multiple Correspondence Analysis (MCA)
    -----------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Multiple Correspondence Analysis (MCA)

    Usage
    -----
    ```python
    >>> supvarMCA(self,X_quanti_sup=None,X_quali_sup=None)
    ```

    Parameters
    ----------
    `self`: an object of class MCA

    `X_quanti_sup`: pandas DataFrame of supplementary quantitative variables

    `X_quali_sup`: pandas DataFrame of supplementary qualitative variables

    Returns
    -------
    a namedtuple of namedtuple containing all the results for supplementary variables including : 

    `quanti`: a namedtuple of pandas DataFrames containing all the results of the supplementary quantitative variables including:
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
    
    `quali`: a namedtuple of pandas DataFrames/Series containing all the results of the supplementary qualitative/categories variables including:
        * `coord`: factor coordinates
        * `cos2`: squared cosinus
        * `vtest`: value-test
        * `dist`: squared distance to origin
        * `eta2`: squared correlation ratio

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    #check if self is and object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        #if pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        #check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if X_quanti_sup.shape[0] = nrows
        if X_quanti_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")
        
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X_quanti_sup[k]) for k in X_quanti_sup.columns)
        if not all_num:
            raise TypeError("All columns in `X_quanti_sup` must be numeric")
        
        #fill missing with mean
        X_quanti_sup = recodecont(X_quanti_sup).X
        # Compute weighted average and and weighted standard deviation
        d_quanti_sup = DescrStatsW(X_quanti_sup,weights=self.call_.ind_weights,ddof=0)
        # Standardization
        Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(Z=Z_quanti_sup,U=self.svd_.U,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary qualitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        #if pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        #check if X is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #check if X_quali_sup.shape[0] = nrows
        if X_quali_sup.shape[0] != self.call_.X.shape[0]:
            raise ValueError("'rows' aren't aligned")
        
        #check if all columns are categoricals
        all_cat = all(api.types.is_string_dtype(X_quali_sup[j]) for j in X_quali_sup.columns)
        if not all_cat:
            raise TypeError("All columns in `X_quali_sup` must be categoricals")
        #convert to factor
        for q in X_quali_sup.columns:
            X_quali_sup[q] = Categorical(X_quali_sup[q],categories=sorted(X_quali_sup[q].dropna().unique().tolist()),ordered=True)
        #check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        #compute dummies tables
        dummies = concat((get_dummies(X_quali_sup[q],dtype=int) for q in X_quali_sup.columns),axis=1)
        #standardization
        Z_quali_sup = mapply(dummies,lambda x : (x/dummies.mean(axis=0))-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #supplementary categories factor coordinates
        quali_sup_coord = mapply(mapply(dummies,lambda x : x/sum(x),axis=0,progressbar=False,n_workers=self.call_.n_workers).T.dot(self.ind_.coord),lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #supplementary categories squared distance to origin
        quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : (x**2)*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
        quali_sup_sqdisto.name = "Sq. Dist."
        #statistics for qualitative variables
        coef_k = sqrt(((X_quali_sup.shape[0] - 1)*dummies.sum(axis=0))/(X_quali_sup.shape[0] - dummies.sum(axis=0)))
        quali_sup_ = predict_quali_sup(X=X_quali_sup,Y=self.ind_.coord,coord=quali_sup_coord,sqdisto=quali_sup_sqdisto,col_coef=coef_k,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarMCAResult",["quanti","quali"])(quanti_sup,quali_sup)