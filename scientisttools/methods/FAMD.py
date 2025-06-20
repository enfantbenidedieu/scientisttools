# -*- coding: utf-8 -*-
from numpy import number,array,ndarray,average,sqrt,linalg,zeros,insert,diff,nan,c_,diag,ones,cumsum
from pandas import DataFrame,Categorical,concat,Series,crosstab
from collections import OrderedDict, namedtuple
from scipy.stats import chi2_contingency
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .PCA import PCA
from .functions.splitmix import splitmix
from .functions.recodevarfamd import recodevarfamd
from .functions.recodecont import recodecont
from .functions.revaluate_cat_variable import revaluate_cat_variable
from .functions.namedtuplemerge import namedtuplemerge
from .functions.predict_sup import predict_ind_sup, predict_quanti_sup, predict_quali_sup
from .functions.conditional_average import conditional_average

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
    When all the variables are quantitative, FAMD gives exactly the same results as standard PCA.

    Usage
    -----
    ```python
    >>> FAMD(n_components = 5, ind_weights = None,ind_sup=None,quanti_sup=None,quali_sup=None,parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `ind_weights` : an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `ind_sup` : an integer or a list/tuple indicating the indexes of the supplementary individuals

    `quanti_sup` : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    `quali_sup` : an integer or a list/tuple indicating the indexes of the categorical supplementary variables

    `disjunctive` : a pandas dataframe obtained from `imputeFAMD` function of the `missmdatools` package that allows to handle mmissing values.

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `svd_` : dictionary of matrices containing all the results of the singular value decomposition

    `var_`  : dictionary of pandas dataframe containing all the results for the variables considered as group (coordinates, square cosine, contributions)

    `ind_` : dictionary of pandas dataframe with all the results for the individuals (coordinates, square cosine, contributions)

    `ind_sup_` : dictionary of pandas dataframe containing all the results for the supplementary individuals (coordinates, square cosine)

    `quali_var_` : namedtuple of pandas dataframe with all the results for the categorical variables (coordinates, square cosine, contributions, v.test)

    `quali_sup_` : namedtuple of pandas dataframe with all the results for the supplementary categorical variables (coordinates, square cosine, v.test)
    
    `quanti_var_` : namedtuple of pandas dataframe with all the results for the quantitative variables (coordinates, correlation, square cosine, contributions)

    `quanti_sup_` : namedtuple of pandas dataframe with all the results for the supplementary quantitative variables (coordinates, correlation, square cosine)

    `call_` : dictionary with some statistics

    `summary_quanti_` : descriptive statistics of quantitatives variables

    `summary_quali_` : statistics of categories variables

    `chi2_test_` : chi2 statistics test

    `model_` : string specifying the model fitted = 'famd'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson F., Le S. and Pagès J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Husson F., Josse L, Lê S. & Mazet J. (2009). FactoMineR : Factor Analysis and Data Mining iwith R. R package version 2.11

    Lebart L., Piron M. & Morineau A. (2006). Statistique exploratoire multidimensionelle. Dunod Paris 4ed

    Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1–18. https://doi.org/10.18637/jss.v025.i01

    Pagès J. (2004). Analyse factorielle de donnees mixtes. Revue Statistique Appliquee. LII (4). pp. 93-111.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. edp sciences

    Rakotomalala, Ricco (2020), Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_famd_ind, get_famd_var, get_famd, summaryFAMD, dimdesc, predictFAMD, supvarFAMD, fviz_famd_ind, fviz_famd_col, fviz_famd_mod, fviz_famd_var

    Examples
    --------
    ```python
    >>> # Load gironde dataset
    >>> from scientisttools import load_gironde, splitmix
    >>> gironde = load_gironde()
    >>> # Split data
    >>> X_quant, X_qual = splitmix(gironde).quanti, splitmix(gironde).quali
    >>> from scientisttools import FAMD
    >>> # PCA with FAMD function
    >>> res_pca = FAMD().fit(X_quant)
    >>> # MCA with FAMD function
    >>> res_mca = FAMD().fit(X_qual)
    >>> # FAMD with FAMD function
    >>> res_famd = FAMD().fit(gironde)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 ind_weights = None,
                 ind_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 disjunctive = None,
                 parallelize = False):
        self.n_components = n_components
        self.ind_weights = ind_weights
        self.ind_sup = ind_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.disjunctive = disjunctive
        self.parallelize = parallelize

    def fit(self,X, y=None):
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
        
        #check if disjunctive is not None and is an instance of class pd.DataFrame
        if self.disjunctive is not None and not isinstance(self.disjunctive,DataFrame):
            raise TypeError(f"{type(self.disjunctive)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
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
        ##Factor Analysis of Mixed Data (FAMD)
        #----------------------------------------------------------------------------------------------------------------------------------------
        # recode variable for Factor Analysis of Mixed Data
        rec = recodevarfamd(X)

        # Extract elements
        X, X_quanti, X_quali, nb_moda = rec.X, rec.quanti, rec.quali, rec.nb_moda
        n_rows, n_quanti, n_quali = rec.n, rec.k1, rec.k2

        #disjunctive table
        if self.disjunctive is not None:
            dummies = self.disjunctive
        else:
            dummies = rec.dummies
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set individuals weights
        #----------------------------------------------------------------------------------------------------------------------------------------
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

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##Create Z, center and scale
        #----------------------------------------------------------------------------------------------------------------------------------------
        Xcod, center, scale = DataFrame().astype("float"), Series(name="center").astype("float"), Series(name="scale").astype("float")

        #for quantitative variables
        if n_quanti > 0:
            # Compute weighted average and standard deviation
            d_quanti = DescrStatsW(X_quanti,weights=ind_weights,ddof=0) 
            # Concatenate
            Xcod = concat((Xcod,X_quanti),axis=1)
            center = concat((center,Series(d_quanti.mean,index=X_quanti.columns,name="center")),axis=0)
            scale = concat((scale,Series(d_quanti.std,index=X_quanti.columns,name="scale")),axis=0)
            
            # Summary quantitatives variables 
            summary_quanti = X_quanti.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti["count"] = summary_quanti["count"].astype("int")
            self.summary_quanti_ = summary_quanti
        
        # Set categoricals variables weights
        if n_quali > 0:
            # Concatenate
            Xcod = concat((Xcod,dummies),axis=1)
            center = concat((center, Series(average(dummies,axis=0,weights=ind_weights),index=dummies.columns,name="center")),axis=0) 
            scale = concat((scale,Series(sqrt(average(dummies,axis=0,weights=ind_weights)),index=dummies.columns,name="scale")),axis=0)

            # Compute statistiques
            summary_quali = DataFrame()
            for j in X_quali.columns:
                eff = X_quali[j].value_counts().to_frame("count").reset_index().rename(columns={j : "categorie"}).assign(proportion = lambda x : x["count"]/sum(x["count"]))
                eff.insert(0,"variable",j)
                summary_quali = concat([summary_quali,eff],axis=0,ignore_index=True)
            summary_quali["count"] = summary_quali["count"].astype("int")
            self.summary_quali_ = summary_quali
            
            # Chi2 statistic test
            if n_quali >1:
                chi2_test = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for i in range(n_quali-1):
                    for j in range(i+1,n_quali):
                        chi = chi2_contingency(crosstab(X_quali.iloc[:,i],X_quali.iloc[:,j]),correction=False)
                        row_chi2 = DataFrame(OrderedDict(variable1=X_quali.columns[i],variable2=X_quali.columns[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[idx])
                        chi2_test = concat((chi2_test,row_chi2),axis=0,ignore_index=True)
                        idx = idx + 1
                # Transform to int
                chi2_test["dof"] = chi2_test["dof"].astype("int")
                self.chi2_test_ = chi2_test

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##standardization : Z = (Xcod - center)/scale
        #----------------------------------------------------------------------------------------------------------------------------------------
        Z = mapply(Xcod, lambda x : (x - center)/scale, axis=1, progressbar=False,n_workers=n_workers)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set variables weights
        #----------------------------------------------------------------------------------------------------------------------------------------
        var_weights = Series(ones(Z.shape[1]),index=Z.columns,name="weight")

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##set number of components
        #----------------------------------------------------------------------------------------------------------------------------------------
        # QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = min(linalg.matrix_rank(Q),linalg.matrix_rank(R))

        if self.n_components is None:
            n_components =  int(max_components)
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
        
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,n_workers=n_workers,
                            ind_sup=ind_sup_label,quanti_sup=quanti_sup_label,quali_sup=quali_sup_label,rec=rec)
        
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##Run PCA with active elements
        #----------------------------------------------------------------------------------------------------------------------------------------
        res = PCA(standardize=False,n_components=int(max_components),ind_weights=ind_weights,var_weights=var_weights).fit(Z)

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for active categories - active qualitative variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if n_quali > 0:
            Z_quali = concat((Z,X_quali),axis=1)
            index = [Z_quali.columns.tolist().index(x) for x in X_quali.columns]
            #Update PCA with supplementary qualitative variables
            res = PCA(standardize=False,n_components=int(max_components),ind_weights=ind_weights,var_weights=var_weights,quali_sup=index).fit(Z_quali)
            quali_var_eta2 = res.quali_sup_.eta2.iloc[:,:n_components]
            quali_var_ = OrderedDict(coord=res.quali_sup_.coord.iloc[:,:n_components],contrib=res.var_.contrib.iloc[n_quanti:,:n_components],cos2=res.quali_sup_.cos2.iloc[:,:n_components],
                                     vtest=res.quali_sup_.vtest.iloc[:,:n_components],dist=res.quali_sup_.dist)
            self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary individuals
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #split data
            split_ind_sup = splitmix(X_ind_sup)
            X_ind_sup_quanti, X_ind_sup_quali = split_ind_sup.quanti, split_ind_sup.quali

            # Initialize the data
            Xcod_ind_sup = DataFrame().astype("float")

            if n_quanti > 0:
                Xcod_ind_sup = concat((Xcod_ind_sup,X_ind_sup_quanti),axis=1)
            
            if n_quali > 0:
                #create disjunctive table
                dummies_ind_sup = DataFrame(zeros((len(ind_sup_label),dummies.shape[1])),columns=dummies.columns,index=ind_sup_label)
                for i in range(len(ind_sup_label)):
                    values = [X_ind_sup_quali.iloc[i,j] for j in range(n_quali)]
                    for k in range(dummies.shape[1]):
                        if dummies.columns[k] in values:
                            dummies_ind_sup.iloc[i,k] = 1
                Xcod_ind_sup = concat((Xcod_ind_sup,dummies_ind_sup),axis=1)
            
            # Standardize the data
            Z_ind_sup = mapply(Xcod_ind_sup,lambda x : (x - self.call_.center)/self.call_.scale,axis=1,progressbar=False,n_workers=n_workers)
            # Update PCA with supplementary individuals
            res = PCA(standardize=False,n_components=int(max_components),ind_weights=ind_weights,var_weights=var_weights,ind_sup=ind_sup_label).fit(concat((Z,Z_ind_sup),axis=0))
            #convert to ordered dictionary
            ind_sup_ = OrderedDict(coord=res.ind_sup_.coord.iloc[:,:n_components],cos2=res.ind_sup_.cos2.iloc[:,:n_components],dist=res.ind_sup_.dist)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary quantitative variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=ind_sup_label)
            
            #Recode to fill NA
            X_quanti_sup = recodecont(X_quanti_sup).X
            # Standardize
            d_quanti_sup = DescrStatsW(X_quanti_sup,weights=ind_weights,ddof=0)
            Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=n_workers)
            # Update PCA with supplementary quantitative variables
            res = PCA(standardize=False,n_components=int(max_components),ind_weights=ind_weights,var_weights=var_weights,quanti_sup=quanti_sup_label).fit(concat((Z,Z_quanti_sup),axis=1))
            #convert to ordered dictionary
            quanti_sup_ = OrderedDict(coord=res.quanti_sup_.coord.iloc[:,:n_components],cos2=res.quanti_sup_.cos2.iloc[:,:n_components])
            #convert to namedtuple
            self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            # Summary statistics with supplementary quantitatives variables
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")

            # Store
            if n_quanti > 0:
                self.summary_quanti_.insert(0,"group","active")
                summary_quanti_sup.insert(0,"group","sup")
                self.summary_quanti_ = concat((self.summary_quanti_,summary_quanti_sup),axis=0,ignore_index=True)
            elif n_quanti == 0:
                self.summary_quanti_ = summary_quanti_sup
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for supplementary qualitative variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=ind_sup_label)

            # Update PCA with supplementary qualitative variables
            res = PCA(standardize=False,n_components=int(max_components),ind_weights=ind_weights,var_weights=var_weights,quali_sup=quali_sup_label).fit(concat((Z,X_quali_sup),axis=1))
            #convert to ordered dictionary
            quali_sup_ = OrderedDict(coord=res.quali_sup_.coord.iloc[:,:n_components],cos2=res.quali_sup_.cos2.iloc[:,:n_components],vtest=res.quali_sup_.vtest.iloc[:,:n_components],
                                     eta2=res.quali_sup_.eta2.iloc[:,:n_components],dist=res.quali_sup_.dist)
            #convert to namedtuple
            self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
            
            # Chi-squared test between new categorie
            n_quali_sup = X_quali_sup.shape[1]
            if n_quali_sup > 1:
                chi2_sup_test = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                cpt = 0
                for i in range(n_quali_sup-1):
                    for j in range(i+1,n_quali_sup):
                        chi = chi2_contingency(crosstab(X_quali_sup.iloc[:,i],X_quali_sup.iloc[:,j]),correction=False)
                        row_chi2 = DataFrame(OrderedDict(variable1=X_quali_sup.columns[i],variable2=X_quali_sup.columns[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[cpt])
                        chi2_sup_test  = concat([chi2_sup_test,row_chi2],axis=0)
                        cpt = cpt + 1
                chi2_sup_test["dof"] = chi2_sup_test["dof"].astype("int")
            
            # Chi-squared between old and new qualitatives variables
            if n_quali > 0:
                chi2_sup_test2 = DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
                cpt = 0
                for i in range(n_quali_sup):
                    for j in range(n_quali):
                        chi = chi2_contingency(crosstab(X_quali_sup.iloc[:,i],X_quali.iloc[:,j]),correction=False)
                        row_chi2 = DataFrame(OrderedDict(variable1=X_quali_sup.columns[i],variable2=X_quali.columns[j],statistic=chi.statistic,dof=chi.dof,pvalue=chi.pvalue),index=[cpt])
                        chi2_sup_test2 = concat([chi2_sup_test2,row_chi2],axis=0,ignore_index=True)
                        cpt = cpt + 1
                chi2_sup_test2["dof"] = chi2_sup_test2["dof"].astype("int")
            
            if n_quali > 1:
                if n_quali_sup > 1 :
                    chi2_sup_test = concat([chi2_sup_test,chi2_sup_test2],axis=0,ignore_index=True)
                else:
                    chi2_sup_test = chi2_sup_test2
                self.chi2_test_ = concat((self.chi2_test_,chi2_sup_test),axis=0,ignore_index=True)
            else:
                if n_quali_sup > 1 :
                    self.chi2_test_ = chi2_sup_test

            # Compute statistiques
            summary_quali_sup = DataFrame()
            for j in X_quali_sup.columns:
                eff = X_quali_sup[j].value_counts().to_frame("count").reset_index().rename(columns={j : "categorie"}).assign(proportion=lambda x : x["count"]/sum(x["count"]))
                eff.insert(0,"variable",j)
                summary_quali_sup = concat([summary_quali_sup,eff],axis=0,ignore_index=True)
            summary_quali_sup["count"] = summary_quali_sup["count"].astype("int")

            if n_quali == 0:
                self.summary_quali_ = summary_quali_sup
            elif n_quali > 0:
                summary_quali_sup.insert(0,"group","sup")
                self.summary_quali_.insert(0,"group","active")
                self.summary_quali_ = concat([self.summary_quali_,summary_quali_sup],axis=0,ignore_index=True)

        #Generalized Singular Values Decomposition (GSVD)
        svd_ = res.svd_
        self.svd_ = namedtuple("svd",["vs","U","V"])(svd_.vs[:max_components], svd_.U[:,:n_components],svd_.V[:,:n_components])

        # Eigen - values
        eigen_values = self.svd_.vs**2
        difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values)
        cumulative = cumsum(proportion)
        self.eig_ = DataFrame(c_[eigen_values,difference,proportion,cumulative],columns=["eigenvalue","difference","proportion","cumulative"],index=list(map(lambda x : "Dim."+str(x+1),range(max_components))))

        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for individuals
        #----------------------------------------------------------------------------------------------------------------------------------------
        ind_ = OrderedDict(coord=res.ind_.coord.iloc[:,:n_components],contrib=res.ind_.contrib.iloc[:,:n_components],cos2=res.ind_.cos2.iloc[:,:n_components],infos=res.ind_.infos)
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for quantitative variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if n_quanti > 0:
            quanti_var_ctr, quanti_var_cos2 = res.var_.contrib.iloc[:n_quanti,:n_components], res.var_.cos2.iloc[:n_quanti,:n_components]
            #convert to ordered dictionary
            quanti_var_ = OrderedDict(coord=res.var_.coord.iloc[:n_quanti,:n_components],contrib=quanti_var_ctr,cos2=quanti_var_cos2,infos=res.var_.infos.iloc[:n_quanti,:])
            #convert to namedtuple
            self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        ##statistics for variables
        #----------------------------------------------------------------------------------------------------------------------------------------
        if n_quali > 0:
            # Add to qualitatives/categoricals variables if not continuous variables
            if n_quanti == 0:
                self.quali_var_ = namedtuplemerge("quali_var",self.quali_var_,namedtuple("quali_var",["eta2"])(quali_var_eta2))
            elif n_quanti > 0:
                # Qualitative variables contributions (contrib) in FAMD
                quali_var_ctr = mapply(quali_var_eta2,lambda x : 100*x/eigen_values[:n_components],axis=1,progressbar=False,n_workers=n_workers)
                # Qualitative variables square cosinus (cos2) in FAMD
                quali_var_cos2 = mapply(quali_var_eta2,lambda x : (x**2)/(nb_moda -1),axis=0,progressbar=False,n_workers=n_workers)
                #concatenate
                var_coord, var_ctr, var_cos2 = concat((quanti_var_cos2,quali_var_eta2),axis=0), concat((quanti_var_ctr,quali_var_ctr),axis=0), concat((quanti_var_cos2.pow(2),quali_var_cos2),axis=0)
                #convert to namedtuple
                self.var_ = namedtuple("var",["coord","contrib","cos2"])(var_coord,var_ctr,var_cos2)      

        self.model_ = "famd"

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
        # check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X.index.name = None

        #split X
        split_x = splitmix(X)
        X_quanti, X_quali = split_x.quanti, split_x.quali
        
        # Extract active elements
        dummies, n_quanti, n_quali = self.call_.rec.dummies, self.call_.rec.k1, self.call_.rec.k2

        #create code variables
        Xcod = DataFrame().astype("float")
        if X_quanti is not None :
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
        
        # Standardize the data
        coord = mapply(Xcod,lambda x : (((x - self.call_.center)/self.call_.scale) - self.call_.Z.mean(axis=0))*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.svd_.V)
        coord.columns = ["Dim."+str(x+1) for x in range(coord.shape[1])]
        return coord

def predictFAMD(self,X=None):
    """
    Predict projection for new individuals with Factor Analysis of Mixed Data (FAMD)
    --------------------------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and square distance to origin of new individuals with Factor Analysis of Mixed Data (FAMD)

    Usage
    -----
    ```python
    >>> predictFAMD(self,X=None)
    ```

    Parameters
    ----------
    `self` : an object of class FAMD

    `X` : pandas dataframe in which to look for variables with which to predict. X must contain columns with the same names as the original data.
    
    Return
    ------
    namedtuple of dataframes containing all the results for the new individuals including:
    
    `coord` : factor coordinates of the new individuals

    `cos2` : square cosinus of the new individuals

    `dist` : square distance to origin for new individuals
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> from scientisttools import FAMD, predictFAMD, load_gironde
    >>> gironde = load_gironde()
    >>> res_famd = FAMD().fit(gironde)
    >>> predict = predictFAMD(res_famd,X=gironde)
    ```
    """
    # Check if self is an object of class FAMD
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")

    # check if X is a pandas DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # Set index name as None
    X.index.name = None

    #split X
    split_x = splitmix(X)
    X_quanti, X_quali = split_x.quanti, split_x.quali

    # Extract active elements
    dummies, n_quanti, n_quali = self.call_.rec.dummies, self.call_.rec.k1, self.call_.rec.k2

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
    
    # Standardize the data
    Z = mapply(Xcod,lambda x : ((x - self.call_.center)/self.call_.scale) - self.call_.Z.mean(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
    #square distance to origin
    sqdisto = mapply(Z, lambda x : (x**2)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
    sqdisto.name = "Sq. Dist."
    #statistics for 
    ind_sup_ = predict_ind_sup(Z=Z,V=self.svd_.V,sqdisto=sqdisto,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
    return namedtuple("predictFAMDResult",ind_sup_.keys())(*ind_sup_.values())

def supvarFAMD(self,X_quanti_sup=None, X_quali_sup=None):
    """
    Supplementary variables in Factor Analysis of Mixed Data (FAMD)
    ---------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Factor Analysis of Mixed Data (FAMD)

    Usage
    -----
    ```python
    >>> supvarFAMD(self,X_quanti_sup=None, X_quali_sup=None)
    ```

    Parameters
    ----------
    `self` : an object of class FAMD

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives variables (default = None)

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives variables (default = None)

    Returns
    -------
    namedtuple of dictionary containing the results for supplementary variables including : 

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
    >>> from scientisttools import FAMD, supvarFAMD, load_gironde, splitmix
    >>> gironde = load_gironde()
    >>> res_famd = FAMD().fit(gironde)
    >>> X_quant, X_qual = splitmix(gironde).quanti, splitmix(gironde).quali
    >>> supvar_famd = supvaFAMD(res_famd, X_quali_sup=X_qual, X_quanti_sup=X_quant)
    ```
    """
    # Check if self is and object of class FAMD
    if self.model_ != "famd":
        raise TypeError("'self' must be an object of class FAMD")

    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary quantitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quanti_sup is not None:
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X_quanti_sup is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,DataFrame):
            raise TypeError(f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X_quanti_sup.index.name = None
        
        #fill missing with mean
        X_quanti_sup = recodecont(X_quanti_sup.astype("float")).X

        # Compute weighted average and standard deviation
        d_quanti_sup = DescrStatsW(X_quanti_sup,weights=self.call_.ind_weights,ddof=0)

        # Standardization data
        Z_quanti_sup = mapply(X_quanti_sup,lambda x : (x - d_quanti_sup.mean)/d_quanti_sup.std,axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #statistics for 
        quanti_sup_ = predict_quanti_sup(Z=Z_quanti_sup,U=self.svd_.U,row_weights=self.call_.ind_weights,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quanti_sup =  namedtuple("quanti_stp",quanti_sup_.keys())(*quanti_sup_.values())
    else:
        quanti_sup = None
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    ##statistics for supplementary qualitative variables
    #----------------------------------------------------------------------------------------------------------------------------------------
    if X_quali_sup is not None:
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X_quali_sup is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,DataFrame):
            raise TypeError(f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X_quali_sup.index.name = None

        #convert to factor - categorie
        for j in X_quali_sup.columns:
            X_quali_sup[j] = Categorical(X_quali_sup[j],categories=sorted(X_quali_sup[j].dropna().unique().tolist()),ordered=True)
        
        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)
        #barycenter of Z
        barycentre = conditional_average(X=self.call_.Z,Y=X_quali_sup,weights=self.call_.ind_weights)
        #center the data
        Z_quali_sup = mapply(barycentre,lambda x : x - self.call_.Z.mean(axis=0),axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #categories square distance
        quali_sup_sqdisto  = mapply(Z_quali_sup, lambda x : (x**2),axis=1,progressbar=False,n_workers=self.call_.n_workers).sum(axis=1)
        quali_sup_sqdisto.name = "Sq. Dist."

        #categories coefficients
        n_k = concat((X_quali_sup[j].value_counts().sort_index() for j in X_quali_sup.columns),axis=0)
        n_rows = X_quali_sup.shape[0]
        coef_k = sqrt(((n_rows-1)*n_k)/(n_rows-n_k))

        #statistics for supplementary categories
        quali_sup_ = predict_quali_sup(X=X_quali_sup,Z=Z_quali_sup,Y=self.ind_.coord,V=self.svd_.V,col_coef=coef_k,sqdisto=quali_sup_sqdisto,
                                       row_weights=self.call_.ind_weights,col_weights=self.call_.var_weights,n_workers=self.call_.n_workers)
        #update value-test with squared eigenvalues
        quali_sup_["vtest"] = mapply(quali_sup_["vtest"],lambda x : x/self.svd_.vs[:self.call_.n_components],axis=1,progressbar=False,n_workers=self.call_.n_workers)
        #convert to namedtuple
        quali_sup = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
    else:
        quali_sup = None
    
    #convert to namedtuple
    return namedtuple("supvarFAMDResult",["quanti","quali"])(quanti_sup,quali_sup)