# -*- coding: utf-8 -*-
from numpy import array,ones,number,ndarray,c_,cumsum,sqrt,zeros
from pandas import DataFrame,Series,Categorical,concat,crosstab,get_dummies
from itertools import chain, repeat
from scipy.stats import chi2_contingency
from collections import OrderedDict, namedtuple
from typing import NamedTuple
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from scientisttools.others.revaluate_cat_variable import revaluate_cat_variable
from scientisttools.others.fitfa import fitfa
from scientisttools.others.function_eta2 import function_eta2
from scientisttools.others.predict_sup import predict_ind_sup, predict_quanti_sup
from scientisttools.others.recodecont import recodecont

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
    >>> MCA(n_components = 5, excl = None, ind_weights = None, var_weights = None, ind_sup = None,quali_sup = None,quanti_sup = None,parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `excl` : an integer or a list indicating the "junk" categories (by default None). It can be a list/tuple of the names of the categories or a list/tuple of the indexes in the disjunctive table.

    `ind_weights` : an optional individuals weights (by default, a list/tuple/ndarray of 1/(number of active individuals) for uniform row weights); the weights are given only for the active individuals
    
    `var_weights` : an optional variables weights (by default, a list/tuple/ndarray of 1/(number of active variables) for uniform row weights); the weights are given only for the active variables
    
    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `quali_sup` : an integert or a list/tuple indicating the indexes of the categorical supplementary variables

    `quanti_sup` : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Atttributes
    -----------
    `call_` : namedtuple with some informations.

    `svd_` : namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD).

    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `eig_correction_` : namedtuple of eigenvalues correction (benzecri and greenacre eigenvalues correction).

    `ind_` : `namedtuple` of pandas dataframe containing all the results for the active individuals (coordinates, square cosine, contributions)

    `var_` : `namedtuple` of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    `others_` : namedtuple of others statistics (Kaiser threshold, ...)

    `ind_sup_` : `namedtuple` of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `quali_sup_` : `namedtuple` of pandas dataframe containing all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a criterion with a Normal distribution, and eta2 which is the square correlation coefficient between a qualitative variable and a dimension)

    `quanti_sup_` : `namedtuple` of pandas dataframe containing all the results for the supplementary quantitative variables (coordinates, correlation between variables and axes)

    `summary_quanti_` : summary statistics for quantitative variables if quanti_sup is not None
    
    `summary_quali_` : summary statistics for qualitative variables (active and supplementary)

    `chi2_test_` : chi-squared test.

    `model_` : string specifying the model fitted = 'mca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Le Roux B. and Rouanet H., Geometric Data Analysis: From Correspondence Analysis to Stuctured Data Analysis, Kluwer Academic Publishers, Dordrecht (June 2004).

    Le Roux B. and Rouanet H., Multiple Correspondence Analysis, SAGE, Series: Quantitative Applications in the Social Sciences, Volume 163, CA:Thousand Oaks (2010).

    Le Roux B. and Jean C. (2010), Développements récents en analyse des correspondances multiples, Revue MODULARD, Numéro 42

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    `get_mca_ind`, `get_mca_var`, `get_mca`, `summaryMCA`, `dimdesc`, `predictMCA`, `supvarMCA`, `fviz_mca_ind`, `fviz_mca_var`, `fviz_mca_quali_var`, `fviz_mca`

     Examples
    --------
    ```python
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> 
    >>> # Multiple Correspondence Analysis (MCA)
    >>> from scientisttools import MCA
    >>> res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1],parallelize=True)
    >>> res_mca.fit(poison)
    >>> 
    >>> # Specific Multiple Correspondence Analysis (SpecificMCA)
    >>> res_specmca = MCA(n_components=5,excl=(0,2),ind_sup=[50,51,52,53,54],quali_sup = [13,14],quanti_sup =[0,1],parallelize=True)
    >>> res_specmca.fit(poison)
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

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y` : None
            y is ignored

        Returns
        -------
        `self` : object
            Returns the instance itself
        """
        #check if X is an instance of class pd.DataFrame
        if not isinstance(X,DataFrame):
           raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Set index name as None
        X.index.name = None

        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##checks if quantitatives variables are in X
        #----------------------------------------------------------------------------------------------------------------------------------------
        is_quanti = X.select_dtypes(include=number)
        if is_quanti.shape[1]>0:
            for j in is_quanti.columns.tolist():
                X[j] = X[j].astype("float")
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##convert categorical variables to factor
        #----------------------------------------------------------------------------------------------------------------------------------------
        is_quali = X.select_dtypes(include=["object","category"])
        for j in is_quali.columns:
            X[j] = Categorical(X[j],categories=sorted(X[j].dropna().unique().tolist()),ordered=True)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary qualitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
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
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary quantitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
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
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Check if supplementary individuals
        #----------------------------------------------------------------------------------------------------------------------------------------
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

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##Multiple Correspondence Analysis (MCA)
        #----------------------------------------------------------------------------------------------------------------------------------------
        # Number of rows/columns
        n_rows, n_cols = X.shape

        #check if two categorical variables have same categories
        X = revaluate_cat_variable(X)

        # Compute statistiques
        summary_quali = DataFrame()
        for j in X.columns:
            eff = X[j].value_counts().to_frame("count").reset_index().rename(columns={j : "categorie"}).assign(proportion = lambda x : x["count"]/x["count"].sum())
            eff.insert(0,"variable",j)
            summary_quali = concat([summary_quali,eff],axis=0,ignore_index=True)
        summary_quali["count"] = summary_quali["count"].astype("int")

        #Chi2 statistic test
        chi2_test = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
        idx = 0
        for i in range(n_cols-1):
            for j in range(i+1,n_cols):
                chi = chi2_contingency(crosstab(X.iloc[:,i],X.iloc[:,j]),correction=False)
                row_chi2 = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                chi2_test = concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                idx = idx + 1
        # Transform to int
        chi2_test["dof"] = chi2_test["dof"].astype("int")

        #dummies tables
        dummies = concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)

        #number of categories, count and proportion
        n_cat, n_k, p_k = dummies.shape[1], dummies.sum(axis=0), dummies.mean(axis=0)
        n_k.name , p_k.name = "count","proportion"

        #standardize the data
        Z = mapply(dummies,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=n_workers)
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set exclusion
        #----------------------------------------------------------------------------------------------------------------------------------------
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

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set individuals weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray)):
            raise TypeError("'ind_weights' must be a list/tuple/array of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list/tuple/array with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set variables weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.var_weights is None:
            var_weights = ones(n_cols)/n_cols
        elif not isinstance(self.var_weights,(list,tuple,ndarray)):
            raise ValueError("'var_weights' must be a list/tuple/array of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array with length {n_cols}.")
        else:
            var_weights = array([x/sum(self.var_weights) for x in self.var_weights])

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set categories weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        nb_mod = array([X[j].nunique() for j in X.columns])
        var_weights2 = array(list(chain(*[repeat(i,k) for i, k in zip(var_weights,nb_mod)])))
        mod_weights = array([x*y for x,y in zip(p_k,var_weights2)])

        #replace excluded categories weights by 0
        if self.excl is not None:
            for i in excl_idx:
                mod_weights[i] = 0
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set number of components
        #----------------------------------------------------------------------------------------------------------------------------------------
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
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Z=Z, 
                            ind_weights=Series(ind_weights,index=X.index,name="weight"),
                            var_weights=Series(var_weights,index=X.columns,name="weight"),
                            mod_weights=Series(mod_weights,index=Z.columns,name="weight"),
                            excl=excl_label,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,quali_sup=quali_sup_label,quanti_sup=quanti_sup_label)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## fit factor analysis model and extract all elements
        #----------------------------------------------------------------------------------------------------------------------------------------
        fit_ = fitfa(Z,ind_weights,mod_weights,max_components,n_components,n_workers)
        
        # Extract elements
        self.svd_, self.eig_, ind_, var_ = fit_.svd, fit_.eig, fit_.row, fit_.col

        #replace nan or inf by 0
        if self.excl is not None:
            self.svd_.V[excl_idx,:] = 0

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##Eigenvalues corrections
        #----------------------------------------------------------------------------------------------------------------------------------------
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

        # Benzecri correction
        lambd_tilde = ((n_cols/(n_cols-1))*(lambd - kaiser_threshold))**2
        s_tilde = 100*(lambd_tilde/sum(lambd_tilde))
        benzecri = DataFrame(c_[lambd_tilde,s_tilde,cumsum(s_tilde)],columns=["eigenvalue","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])

        # Greenacre correction
        s_tilde_tilde = n_cols/(n_cols-1)*(sum(self.eig_.iloc[:,0]**2)-(n_cat-n_cols)/(n_cols**2))
        tau = 100*(lambd_tilde/s_tilde_tilde)
        greenacre = DataFrame(c_[lambd_tilde,tau,cumsum(tau)],columns=["eigenvalue","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])

        #convert to namedtuple
        self.eig_correction_ = namedtuple("correction",["benzecri","greenacre"])(benzecri,greenacre)

        #----------------------------------------------------------------------------------------------------------------------------------------
        #Convert to NamedTuple
        #----------------------------------------------------------------------------------------------------------------------------------------
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## variables additionals informations
        #----------------------------------------------------------------------------------------------------------------------------------------
        # Normalized columns coordinates : see (Saporta, p235)  or (Husson, 138)
        var_coord_n = mapply(var_["coord"],lambda x: x*self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)

        ##categoris variables value - test
        var_vtest = mapply(var_["coord"],lambda x : x*sqrt(((n_rows-1)*n_k)/(n_rows-n_k)),axis=0,progressbar=False,n_workers=n_workers)
        
        # Variables squared correlation ratio
        quali_var_eta2 = function_eta2(X=X,Y=ind_["coord"],weights=ind_weights,excl=excl_label,n_workers=n_workers)

        # Contribution des variables
        quali_var_contrib = DataFrame().astype("float")
        for j in X.columns:
            contrib = var_["contrib"].loc[X[j].unique(),:].sum(axis=0).to_frame(j).T
            quali_var_contrib = concat((quali_var_contrib,contrib),axis=0)

        # Inertia for the variables
        quali_var_inertia = Series((nb_mod - 1)/n_rows,index=X.columns,name="inertia")

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.excl is not None:
            var_ = OrderedDict(coord=var_["coord"].drop(index=excl_label),cos2=var_["cos2"].drop(index=excl_label),contrib=var_["contrib"].drop(index=excl_label),infos=var_["infos"].drop(index=excl_label),
                               coord_n=var_coord_n.drop(index=excl_label),vtest=var_vtest.drop(index=excl_label),eta2=quali_var_eta2,var_inertia=quali_var_inertia,var_contrib=quali_var_contrib)
        else:
            var_ = OrderedDict(**var_,**OrderedDict(coord_n=var_coord_n,vtest=var_vtest,eta2=quali_var_eta2,var_inertia=quali_var_inertia,var_contrib=quali_var_contrib))
            
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##multiple correspondence analysis additionals informations
        #----------------------------------------------------------------------------------------------------------------------------------------
        #inertia
        inertia = (n_cat/n_cols) - 1

        # Eigenvalue threshold
        kaiser_proportion_threshold = 100/inertia

        #convert to namedtuple
        self.others_ = namedtuple("others",["inertia","kaiser"])(inertia,namedtuple("kaiser",["threshold","proportion_threshold"])(kaiser_threshold,kaiser_proportion_threshold))

        #----------------------------------------------------------------------------------------------------------------------------------------
        ## Statistics for supplementary individuals
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            # Create dummies table for supplementary individuals
            Y = DataFrame(zeros((len(ind_sup_label),n_cat)),columns=dummies.columns,index=ind_sup_label)
            for i in range(len(ind_sup_label)):
                values = [X_ind_sup.iloc[i,j] for j in range(n_cols)]
                for k in range(n_cat):
                    if dummies.columns[k] in values:
                        Y.iloc[i,j] = 1
            
            #standardize the data and exclude the data
            Z_ind_sup = mapply(Y,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=n_workers)
            #statistics for supplementary individuals
            ind_sup_ = predict_ind_sup(Z_ind_sup,self.svd_.V[:,:n_components],mod_weights,n_workers)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary qualitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)

            # Reevaluate if two variables have the same level
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            # Compute dummies tables
            X_quali_dummies = concat((get_dummies(X_quali_sup[j],dtype=int) for j in quali_sup_label),axis=1)
            quali_sup_n_k, quali_sup_p_k = X_quali_dummies.sum(axis=0), X_quali_dummies.mean(axis=0)

            #standardiz data
            Z_quali_sup = mapply(X_quali_dummies,lambda x : (x/quali_sup_p_k)-1,axis=1,progressbar=False,n_workers=n_workers)
        
            #square correlation Ratio
            quali_sup_eta2 = function_eta2(X=X_quali_sup,Y=ind_["coord"],weights=ind_weights,n_workers=n_workers)
            
            #supplementary categories factor coordinates
            quali_sup_coord = mapply(mapply(X_quali_dummies,lambda x : x/sum(x),axis=0,progressbar=False,n_workers=n_workers).T.dot(ind_["coord"]),lambda x : x/self.svd_.vs[:n_components],axis=1,progressbar=False,n_workers=n_workers)

            #supplementary categories squared distance to origin
            quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : (x**2)*ind_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            quali_sup_sqdisto.name = "Sq. Dist."

            #supplementary categories square cosinus
            quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=n_workers)
            
            #supplementary categories value-test
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*sqrt(((n_rows-1)*quali_sup_n_k)/(n_rows-quali_sup_n_k)),axis=0,progressbar=False,n_workers=n_workers)
            
            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",["coord","cos2","vtest","dist","eta2"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_sqdisto,quali_sup_eta2)

            #Summary supplementary qualitatives variables
            # Compute statistiques
            summary_quali_sup = DataFrame()
            for i in quali_sup_label:
                eff = X_quali_sup[i].value_counts().to_frame("count").reset_index().rename(columns={i : "categorie"}).assign(proportion = lambda x : x["count"]/x["count"].sum())
                eff.insert(0,"variable",i)
                summary_quali_sup = concat((summary_quali_sup,eff),axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")
            summary_quali_sup.insert(0,"group","sup")

            # Concatenate with activate summary
            summary_quali.insert(0,"group","active")
            summary_quali = concat((summary_quali,summary_quali_sup),axis=0,ignore_index=True)

            #Chi2 statistic test
            chi2_test2 = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
            idx = 0
            for i in quali_sup_label:
                for j in X.columns:
                    chi = chi2_contingency(crosstab(X_quali_sup[i],X[j]),correction=False)
                    row_chi2 = DataFrame(OrderedDict(variable1=i,variable2=j,statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                    chi2_test2 = concat((chi2_test2,row_chi2),axis=0,ignore_index=True)
                    idx = idx + 1
            # Transform to int
            chi2_test2["dof"] = chi2_test2["dof"].astype("int")

            #concatenate
            chi2_test2.insert(0,"group","sup")
            chi2_test.insert(0,"group","active")
            chi2_test = concat((chi2_test,chi2_test2),axis=0,ignore_index=True)
            
            #Chi2 statistics between each supplementary qualitatives columns
            if len(quali_sup_label)>1:
                chi2_test3 = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for i in range(len(quali_sup_label)-1):
                    for j in range(i+1,len(quali_sup_label)):
                        chi = chi2_contingency(crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j]),correction=False)
                        row_chi2 = DataFrame(OrderedDict(variable1=quali_sup_label[i],variable2=quali_sup_label[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                        chi2_test3 = concat((chi2_test3,row_chi2),axis=0,ignore_index=True)
                        idx = idx + 1
                # Transform to int
                chi2_test3["dof"] = chi2_test3["dof"].astype("int")
                chi2_test3.insert(0,"group","sup")
                #concatenate
                chi2_test = concat((chi2_test,chi2_test3),axis=0,ignore_index=True)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary quantitatives variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)

            # Recode continuous variables : Fill NA if missing
            X_quanti_sup = recodecont(X_quanti_sup.astype("float")).Xcod
            
            # Compute weighted average and and weighted standard deviation
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)

            # Standardization
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=n_workers)
            #statistics for supplementary quantitative variables
            quanti_sup_ = predict_quanti_sup(Z_quanti_sup,self.svd_.U[:,:n_components],ind_weights,n_workers)
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            # Compute statistics
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            self.summary_quanti_ = summary_quanti_sup

        self.summary_quali_ = summary_quali
        self.chi2_test_ = chi2_test

        self.model_ = "mca"
        
        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y` : None
            y is ignored.
        
        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord

    def transform(self,X):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """

        # Check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
           raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Check if columns are aligned
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")
        
        # Set index name as None
        X.index.name = None
        
        # Add revaluate
        X = revaluate_cat_variable(X)
        n_rows, n_cols = X.shape

        # Extract some elements
        dummies = self.call_.dummies
        n_cat, p_k = dummies.shape[1], dummies.mean(axis=0)
        
        #create disjunctive table
        Y = DataFrame(zeros((n_rows,n_cat)),columns=dummies.columns,index=X.index)
        for i in range(n_rows):
            values = [X.iloc[i,j] for j in range(n_cols)]
            for k in range(n_cat):
                if dummies.columns[k] in values:
                    Y.iloc[i,k] = 1

        # Standardization z_ik = (x_ik/pk)-1
        Z =  mapply(Y,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        # Supplementary individuals Coordinates
        coord = mapply(Z,lambda x : x*self.call_.mod_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V[:,:self.call_.n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)] 
        return coord
    
def predictMCA(self,X) -> NamedTuple:
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
    `self` : an object of class MCA

    `X` : a pandas dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Returns
    -------
    namedtuple of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : squared cosines of the new individuals

    `dist` : distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is an object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")

    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Check if columns are aligned
    if X.shape[1] != self.call_.X.shape[1]:
        raise ValueError("'columns' aren't aligned")

    # Set index name as None
    X.index.name = None

    # Revaluate if at least two columns have same levels
    X = revaluate_cat_variable(X)
    n_rows, n_cols = X.shape

    # Extract some elements
    dummies  = self.call_.dummies 
    n_cat, p_k = dummies.shape[1], dummies.mean(axis=0)

    # Create dummies table for supplementary individuals
    Y = DataFrame(zeros((n_rows,n_cat)),columns=dummies.columns,index=X.index)
    for i in range(n_rows):
        values = [X.iloc[i,j] for j in range(n_cols)]
        for k in range(n_cat):
            if dummies.columns[k] in values:
                Y.iloc[i,k] = 1

    # Standardization z_ik = (x_ik/pk)-1
    Z = mapply(Y,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #statistic for supplementary rows
    ind_sup_ = predict_ind_sup(Z,self.svd_.V[:,:self.call_.n_components],self.call_.mod_weights,self.call_.n_workers)
    # convert to NamedTuple
    return namedtuple("predictMCAResult",ind_sup_.keys())(*ind_sup_.values())

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
    `self` : an object of class MCA

    `X_quanti_sup` : pandas dataframe of supplementary quantitatives variables

    `X_quali_sup` : pandas dataframe of supplementary qualitatives variables

    Returns
    -------
    namedtuple of namedtuple containing the results for supplementary variables including : 

    `quanti` : namedtuple containing the results of the supplementary quantitatives variables including :
        * coord : factor coordinates of the supplementary quantitatives variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : namedtuple containing the results of the supplementary qualitatives/categories variables including :
        * coord : factor coordinates of the supplementary categories
        * cos2 : square cosinus of the supplementary categories
        * vtest : value-test of the supplementary categories
        * dist : square distance to origin of the supplementary categories
        * eta2 : square correlation ratio of the supplementary qualitatives variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>>
    ```
    """
    # Check if self is and object of class MCA
    if self.model_ != "mca":
        raise TypeError("'self' must be an object of class MCA")

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary quantitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Recode continuous variables : Fill NA if missing
        X_quanti_sup = recodecont(X_quanti_sup.astype("float")).Xcod
        
        # Compute weighted average and and weighted standard deviation
        d_quanti_sup = DescrStatsW(X_quanti_sup,weights=self.call_.ind_weights,ddof=0)

        # Standardization
        Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #statistics for supplementary quantitative variables
        quanti_sup_ = predict_quanti_sup(Z_quanti_sup,self.svd_.U[:,:self.call_.n_components],self.call_.ind_weights,self.call_.n_workers)
        #convert to namedtuple
        quanti_sup = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X_quali_sup.index.name = None
        
        #convert to factor
        for j in X_quali_sup.columns:
            X_quali_sup[j] = Categorical(X_quali_sup[j],categories=sorted(X_quali_sup[j].dropna().unique().tolist()),ordered=True)
        
        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        n_rows = X_quali_sup.shape[0]

        # Compute dummies tables
        dummies = concat((get_dummies(X_quali_sup[j],dtype=int) for j in X_quali_sup.columns),axis=1)
        n_k, p_k = dummies.sum(axis=0), dummies.mean(axis=0)

        #standardization
        Z_quali_sup = mapply(dummies,lambda x : (x/p_k)-1,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        
        #supplementary qualitative variables square correlation ratio
        quali_sup_eta2 = function_eta2(X=X_quali_sup,Y=self.ind_.coord,weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        
        #supplementary categories factor coordinates
        quali_sup_coord = mapply(mapply(dummies,lambda x : x/sum(x),axis=0,progressbar=False,n_workers=self.call_.n_workers).T.dot(self.ind_.coord),lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)

        #supplementary categories squared distance to origin
        quali_sup_sqdisto = mapply(Z_quali_sup,lambda x : (x**2)*self.call_.ind_weights,axis=0,progressbar=False,n_workers=self.call_.n_workers).sum(axis=0)
        quali_sup_sqdisto.name = "Sq. Dist."

        #supplementary categories square cosinus
        quali_sup_cos2 = mapply(quali_sup_coord, lambda x : (x**2)/quali_sup_sqdisto,axis=0,progressbar=False,n_workers=self.call_.n_workers)
        
        #supplementary categories value-test
        quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*sqrt(((n_rows-1)*n_k)/(n_rows-n_k)),axis=0,progressbar=False,n_workers=self.call_.n_workers)

        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",["coord","cos2","vtest","dist","eta2"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_sqdisto,quali_sup_eta2)
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarMCAResult",["quanti","quali"])(quanti_sup,quali_sup)