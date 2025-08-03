# -*- coding: utf-8 -*-
from warnings import warn
from pandas import DataFrame, Series, concat, api
from numpy import array, ndarray, ones, diag,linalg, fill_diagonal, flip, insert, diff, nan, cumsum, c_, sqrt, apply_along_axis, fliplr, dot
from collections import namedtuple, OrderedDict
from mapply.mapply import mapply
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.summarize import summarize
from .functions.pcorrcoef import pcorrcoef
from .functions.rotate_factors import rotate_factors

class FactorAnalysis(BaseEstimator,TransformerMixin):
    """
    Factor Analysis (FA)
    --------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    ------------
    Performs Iterative and Non Iterative Principal Factor Analysis (PFA), Non Iterative Harris Component Analysis (HCA) with supplementary individuals.

    Usage
    -----
    ```python
    >>> FactorAnalysis(method = "principal", n_components = 2, ind_weights = None, var_weights = None, ind_sup = None, min_error = 1e-3, max_iter = 50, rotate = 'varimax', rotation_kwargs = None, parallelize = False)
    ```

    Parameters
    ----------
    `method`: a string indicating the factor method (by default 'principal'). Allowed values include: `principal` (principal factor analysis), `harris` (harris component analysis).

    `n_components`: number of dimensions kept in the results (by default 2)

    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables

    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `min_error`: iterate until the change in communalities is less than min_error (default = 1e-3)

    `max_iter`: Maximum number of iterations for convergence (default  = 50)

    `warnings`: warnings = True => warn if number of components is too many.

    `rotate`: the type of rotation to performn after fitting the factor analysis model (by default 'varimax'). 
        If set to `None`, no rotation will be performed. Allowed values include: `varimax`, `promax`, `oblimin`, `oblimax`, `quartimin`, `quartimax` and `equamax`.

    `rotate_kwargs`: optional
        Dictionary containing keyword arguments for the rotation method.
    
    `parallelize`: boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply
    
    Attrbutes
    ---------
    `call_`: a namedtuple with some informations
        * `Xtot`: a pandas DataFrame with all data (active and supplementary)
        * `X`: a pandas DataFrame with active data
        * `Z`: a pandas DataFrame with standardized data : Z = (X-center)/scale
        * `ind_weights`: a pandas Series containing individuals weights
        * `var_weights`: a pandas Series containing variables weights
        * `center`: a pandas Series containing variables means
        * `scale`: a pandas Series containing variables standard deviation : 
        * `n_components`: an integer indicating the number of components kept
        * `min_error`: iterate until the change in communalities is less than min_error
        * `max_iter`: Maximum number of iterations for convergence
        * `rotate`: None or a string specifying the rotation method
        * `rotate_kwargs`: empty dict or dictionary containing keyword arguments for the rotation method
        * `n_workers`: an integer indicating the maximum amount of workers (processes) to spawn. For more information see: https://mapply.readthedocs.io/en/0.1.28/_code_reference/mapply.html
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals

    `eig_`: a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `eigval_`: a namedtuple of numpy array containing eigen values
        * `original`: eigen values of the original matrix
        * `common`: eigen values of the common factor solution

    `vaccounted_`: a pandas DataFrame containing the variance accounted

    `rotate_`: a namedtuple containing the rotation matrix and factor correlations matrix
        * `rotmat`: numpy array of rotation matrix (if a rotation has been performed. None otherwise.)
        * `phi`: pandas DataFrame of factor correlations matrix

    `var_`: a namedtuple of pandas DataFrames containing all the results for the active variables
        * `coord`: factor coordinates of the variables
        * `contrib`: relative contribution of the variables
        * `f_score`: normalized factor coefficients of the variables (factor scores)

    `ind_`: a namedtuple of pandas DataFrames containing all the results for the active individuals
        * `coord`: factor coordinates of the individuals

    `corr_`: a namedtuple of pandas DataFrame containing all the results of correlations
        * `corrcoef`: pearson correlation coefficient
        * `model`: correlation matrix used by the model
        * `pcorrcoef`: partial pearson correlation coefficient
        * `reconst`: correlation matrix estimated by the model
        * `residual`: residual correlations after the factor model is applied.
        * `error`: sum of squared residuals

    `others_`: a namedtuple of pandas DataFrames/Series containing :
        * `communalities_iterations`: the history of the communality estimates. Probably only useful for teaching what happens in the process of iterative fitting.
        * `r2_score`: the multiple R square between the factors and factor score estimates.
        * "communality": communality estimates for each item. These are merely the sum of squared factor loadings for that item.
        * `communalities`: the communalities reflecting the total amount of common variance. They will exceed the communality (above) which is the model estimated common variance.
        * `ùniquenesses`: uniquenesses estimates for each item.s
        * "explained_variance": variance explained by each factor (weighted and unweighted)
        * "inertia": total inertia

    `ind_sup_`: a namedtuple of pandas DataFrame containing all the results for the supplementary individuals
        * `coord`: factor coordinates of the supplementary individuals

    `summary_quanti_`: a pandas DataFrame with summary statistics for quantitative variables

    `model_`: string specifying the model fitted = 'fa'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Berger J-L (2021), Analyse factorielle exploratoire et analyse en composantes principales : guide pratique, hal-03436771v1

    * D. Suhr Diana (), Prinicpal Component Analysis vs. Exploratory Factor Analysis, University of Northern Colorado. See : https://support.sas.com/resources/papers/proceedings/proceedings/sugi30/203-30.pdf

    * Lawley, D.N., Maxwell, A.E. (1963), Factor Analysis as a Statistical Method, Butterworths Mathematical Texts, England

    * Marley W. Watkins (2018), Exploratory Factor Analysis : A guide to best practice, Journal of Black Psychology, Vol. 44(3) 219-246

    * Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0

    Links
    -----
    * https://en.wikipedia.org/wiki/Exploratory_factor_analysis

    * https://datatab.fr/tutorial/exploratory-factor-analysis

    * https://jmeunierp8.github.io/ManuelJamovi/s15.html

    * https://stats.oarc.ucla.edu/sas/output/factor-analysis/

    See Also
    --------
    `get_fa_ind`, `get_fa_var`, `get_fa`, `summaryFA`

    Examples
    --------
    ```python
    >>> #load beer dataset
    >>> from scientisttools import load_beer
    >>> beer = load_beer()
    >>> from scientisttools import FactorAnalysis, summaryFA
    >>> #non iterative principal factor analysis (NIPFA)
    >>> res_fa = FactorAnalysis(max_iter=1,rotation=None).fit(beer)
    >>> #iterative principal factor analysis (IPFA)
    >>> res_fa = FactorAnalysis(rotation=None).fit(beer)
    >>> #harris component analysis (HCA)
    >>> res_fa = FactorAnalysis(method = "harris",max_iter=1,rotation=None).fit(beer)
    >>> summaryFA(res_fa)
    ```
    """
    def __init__(self,
                 method = "principal",
                 n_components = 2,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 min_error = 1e-3,
                 max_iter = 50,
                 warnings = True,
                 rotate = "varimax",
                 rotate_kwargs = dict(),
                 parallelize = False):
        self.method = method
        self.n_components =n_components
        self.ind_weights = ind_weights
        self.var_weights = var_weights
        self.ind_sup = ind_sup
        self.min_error = min_error
        self.max_iter = max_iter
        self.warnings = warnings
        self.rotate = rotate
        self.rotate_kwargs = rotate_kwargs
        self.parallelize = parallelize

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: pandas Dataframe of shape (n_samples, n_columns)
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
        >>> #load beer dataset
        >>> from scientisttools import load_beer
        >>> beer = load_beer()
        >>> from scientisttools import FactorAnalysis, summaryFA
        >>> #non iterative principal factor analysis (NIPFA)
        >>> res_fa = FactorAnalysis(max_iter=1,rotation=None).fit(beer)
        >>> #iterative principal factor analysis (IPFA)
        >>> res_fa = FactorAnalysis(rotation=None).fit(beer)
        >>> #harris component analysis (HCA)
        >>> res_fa = FactorAnalysis(method = "harris",max_iter=1,rotation=None).fit(beer)
        ```
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if X is an instance of pd.DataFrame class
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set index name as None
        X.index.name = None
         
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check max_iter is an integer
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.max_iter < 0:
            raise ValueError("'max_iter' must be equal to or greater than 0.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if method is a string
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.method,str):
            raise TypeError("'method' must be a string.")
        elif self.method not in ("principal","harris"):
            raise ValueError("'method' must be one either 'principal' or 'harris'.")
        
        #set harris 
        if self.method == "harris" and self.max_iter > 1:
            raise ValueError("Harris Component Analysis is a non-iterative factor analysis approach")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if parallelize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.parallelize,bool):
            raise TypeError("'parallelize' must be a boolean.")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if individuls supplementary
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
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Check if missing values in quantitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.isnull().any().any():
            for k in X.columns:
                if X.loc[:,k].isnull().any():
                    X.loc[:,k] = X.loc[:,k].fillna(X.loc[:,k].mean())
                    
        #save dataframe
        Xtot = X.copy()

        #drop supplementary individuls
        if self.ind_sup is not None:
            X_ind_sup = X.loc[ind_sup_label,:]
            X = X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##Principal Factor Analysis (PFA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")

        #number of rows/columns
        n_rows, n_cols = X.shape

        #descriptive statistics of quantitatives variables 
        summary_quanti = summarize(X=X)

        # Number of rows/columns
        n_rows, n_cols = X.shape

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
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list/tuple/array/Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list/tuple/array/Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=X.index,name="weight"), Series(var_weights,index=X.columns,name="weight")
 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## Standardize
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Compute weighted average and standard deviation
        d1 = DescrStatsW(X,weights=ind_weights,ddof=0)
        #convert to Series
        center, scale = Series(d1.mean,index=X.columns,name="center"), Series(d1.std,index=X.columns,name="scale")
        
        # Standardization : Z = (X - mu)/sigma
        Z = mapply(X,lambda x : (x - center)/scale,axis=1,progressbar=False,n_workers=n_workers)

        # weighted Pearson correlation
        wcorr = DataFrame(d1.corrcoef,index=X.columns,columns=X.columns)

        #eigen values of the original matrix
        eigvals, _ = linalg.eig(wcorr)
        
        #Init communality - Prior
        init_comm = 1 - 1/diag(linalg.inv(wcorr))

        #Replace Diagonal of correlation matrix with initial communality
        wcorr_c = wcorr.copy()
        fill_diagonal(wcorr_c.values,init_comm)
        # Harris correlation matrix
        if self.method == "harris":
            wcorr_c = apply_along_axis(func1d=lambda x : x/sqrt(1-init_comm),axis=0,arr=wcorr_c/sqrt(1-init_comm))
        
        #eigen decomposition
        eigenvals, eigenvec = linalg.eigh(wcorr_c)

        # Set n_components_
        max_components = int((eigenvals > 0).sum())
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
         
        #Store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,min_error=self.min_error,
                            max_iter=self.max_iter,rotate=self.rotate,rotate_kwargs=self.rotate_kwargs,n_workers=n_workers,ind_sup=ind_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #variables factor coordinates - loadings
        var_coord = apply_along_axis(func1d=lambda x : x*sqrt(flip(eigenvals)[:n_components]),axis=1,arr=fliplr(eigenvec)[:,:n_components])
        #apply harris correction
        if self.method == "harris":
            var_coord = apply_along_axis(func1d=lambda x : x*sqrt(1 - init_comm),axis=0,arr=var_coord)

        #iterative 
        comm = sum(diag(wcorr_c))
        error = comm
        comm_list = list()
        i = 0   
        while error > self.min_error:
            eigenvals, eigenvec = linalg.eigh(wcorr_c)
            var_coord = apply_along_axis(func1d=lambda x : x*sqrt(flip(eigenvals)[:n_components]),axis=1,arr=fliplr(eigenvec)[:,:n_components])
            #apply harris correction
            if self.method == "harris":
                var_coord = apply_along_axis(func1d=lambda x : x*sqrt(1 - init_comm),axis=0,arr=var_coord)

            model = var_coord.dot(var_coord.T)
            new_comm = diag(model)
            comm1 = sum(new_comm)
            wcorr_c = wcorr.copy()
            fill_diagonal(wcorr_c.values,new_comm)
            # Harris correlation matrix
            if self.method == "harris":
                wcorr_c = apply_along_axis(func1d=lambda x : x/sqrt(1-new_comm),axis=0,arr=wcorr_c/sqrt(1-new_comm))

            error = abs(comm - comm1)
            comm = comm1
            comm_list.append(comm1)
            i += 1
            if i >= self.max_iter:
                if self.warnings:
                    warn("maximum iteration exceeded")
                error = 0

        #eigenvalues
        eigen_values = flip(eigenvals)
        difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values) 
        self.eig_ = DataFrame(c_[eigen_values,difference,proportion,cumsum(proportion)],columns=["Eigenvalue","Difference","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(len(eigen_values))])
        
        #eigen values comparison
        self.eigval_ = namedtuple("eigval",["original","common"])(eigvals,eigen_values)

        #variables factor coordinates
        var_coord = DataFrame(var_coord,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        #add rotation
        if self.rotate is not None:
            if n_components <= 1:
                warn("No rotation will be performed when the number of factors equals 1.")
            else:
                rot = rotate_factors(loadings=var_coord,method=self.rotate,**self.rotate_kwargs)
                var_coord, rotmat, phi = rot.loadings, rot.rotmat, rot.phi
                self.rotate_ = namedtuple("rotate",["rotmat","phi"])(rotmat,phi)

        #weights
        f_score = linalg.inv(wcorr).dot(var_coord)
        f_score = DataFrame(f_score,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        #contribution des variances
        var_contrib = f_score.pow(2).mul(100)/f_score.pow(2).sum(axis=0)
        var_contrib = DataFrame(var_contrib,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)
        
        #store variables informations
        var_ = OrderedDict(coord=var_coord,contrib=var_contrib,f_score=f_score)
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variance accounted
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ss_loadings = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
        self.vaccounted_ = DataFrame(c_[ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],index = ["Dim."+str(x+1) for x in range(n_components)],
                                     columns=["SS loadings","Proportion Var","Cumulative Var","Proportion Explained","Cumulative Proportion"])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##individuals informations : factor coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals factor coordinates
        ind_coord = mapply(Z,lambda x : x*var_weights,axis=1,progressbar=False,n_workers=n_workers).dot(f_score)
        ind_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        self.ind_ = namedtuple("ind",["coord"])(ind_coord)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##Correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #reconst and partial correlations
        rcorr, pcorr = self.var_.coord.dot(self.var_.coord.T), pcorrcoef(X=X)
        #residual correlation
        residual_corr = wcorr.sub(rcorr)
        #error
        error = residual_corr.pow(2).sum().sum()
        #convert to ordered dictionary
        corr_ = OrderedDict(corrcoef=wcorr,pcorrcoef=pcorr,model=wcorr_c,reconst=rcorr,residual=residual_corr,error=error)
        self.corr_ = namedtuple("correlation",corr_.keys())(*corr_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##others statistics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fidélité des facteurs - variance of the scores - R2
        f_fidelity = var_coord.mul(f_score).sum(axis=0)
        f_fidelity.name = "R2"
        #Variance explained by each factor
        explained_variance = DataFrame(c_[eigen_values[:n_components],ss_loadings],index=["Dim."+str(x+1) for x in range(n_components)],columns=["Weighted","Unweighted"])
        #final communality
        final_comm = mapply(var_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
        #communality
        communality = DataFrame(c_[init_comm,final_comm],columns=["Prior","Final"],index=X.columns)
        #communalities
        communalities = sum(final_comm)
        #uniquenesses
        uniquenesses = 1 - final_comm
        uniquenesses.name = "uniquenesses"
        #total inertia
        inertia = sum(init_comm)
        #convert to ordered dictionary
        others_ = OrderedDict(communality_iterations = comm_list,r2_score=f_fidelity,explained_variance=explained_variance,communality=communality,communalities=communalities,uniquenesses=uniquenesses,inertia=inertia)
        #convert to namedtuple
        self.others_ = namedtuple("others",others_.keys())(*others_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##Statistics for supplementary individuals                                      
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            ##supplementary individuals factor coordinates
            ind_sup_coord = mapply(X_ind_sup,lambda x : ((x - self.call_.center)/self.call_.scale)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.var_.f_score)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",["coord"])(ind_sup_coord)

        self.summary_quanti_ = summary_quanti
        self.model_ = "fa"

        return self

    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: pandas Dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: pandas Dataframe of shape (n_samples, n_components)
            Transformed values.
        
        Examples
        --------
        ```python
        >>> #load beer dataset
        >>> from scientisttools import load_beer
        >>> beer = load_beer()
        >>> from scientisttools import FactorAnalysis, summaryFA
        >>> #non iterative principal factor analysis (NIPFA)
        >>> ind_coord = FactorAnalysis(max_iter=1,rotation=None).fit_transform(beer)
        >>> #iterative principal factor analysis (IPFA)
        >>> ind_coord = FactorAnalysis(rotation=None).fit_transform(beer)
        >>> #harris component analysis (HCA)
        >>> ind_coord = FactorAnalysis(method = "harris",max_iter=1,rotation=None).fit_transform(beer)
        ```
        """
        self.fit(X)
        return self.ind_.coord

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal factor previously extracted from a training set.

        Parameters
        ----------
        `X`: pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal factor where `n_samples` is the number of samples and `n_components` is the number of the components.
        
        Examples
        --------
        ```python
        >>> #load beer dataset
        >>> from scientisttools import load_beer
        >>> beer = load_beer()
        >>> from scientisttools import FactorAnalysis, summaryFA
        >>> #non iterative principal factor analysis (NIPFA)
        >>> res_nipfa = FactorAnalysis(max_iter=1,rotation=None).fit(beer)
        >>> ind_coord = res_nipfa.transform(res_nipfa.call_.X)
        >>> #iterative principal factor analysis (IPFA)
        >>> res_ipfa = FactorAnalysis(rotation=None).fit(beer)
        >>> ind_coord = res_ifa.transform(res_ipfa.call_.X)
        >>> #harris component analysis (HCA)
        >>> res_hca = FactorAnalysis(method = "harris",max_iter=1,rotation=None).fit(beer)
        >>> ind_coord ) res_hca.transform(res_hca.call_.X)
        ```
        """
        #check if X is a pandas DataFrame
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        #set index name as None
        X.index.name = None

        #check if all variables are numerics
        all_num = all(api.types.is_numeric_dtype(X[k]) for k in X.columns)
        if not all_num:
            raise TypeError("All columns must be numeric")
        
        #check if X.shape[1] = ncols
        if X.shape[1] != self.call_.X.shape[1]:
            raise ValueError("'columns' aren't aligned")
        
        #standardize the data and apply transition relation
        coord = mapply(X,lambda x : ((x - self.call_.center)/self.call_.scale)*self.call_.var_weights,axis=1,progressbar=False,n_workers=self.call_.n_workers).dot(self.var_.f_score)
        coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord