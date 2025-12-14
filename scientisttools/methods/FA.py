# -*- coding: utf-8 -*-
from warnings import warn
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from numpy import array, ndarray, ones, diag,linalg, fill_diagonal, flip, insert, diff, nan, cumsum, c_, sqrt, apply_along_axis, fliplr, average, cov, tril, number
from collections import namedtuple, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.wcorrcoef import wcorrcoef
from .functions.utils import is_dataframe
from .functions.wpcorrcoef import wpcorrcoef
from .functions.varimax import varimax
from .functions.summarize import summarize
from .functions.corrmatrix import corrmatrix

class FA(BaseEstimator,TransformerMixin):
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
    >>> FA(method = "principal", n_components = 2, ind_weights = None, var_weights = None, ind_sup = None, min_error = 1e-3, max_iter = 50, rotate = False, rotation_kwargs = None)
    ```

    Parameters
    ----------
    `method`: a string indicating the factor method (by default 'principal'). Allowed values are: 
        * `principal` for principal (exploratory) factor analysis (PFA or EFA), 
        * `harris` for harris component analysis (HCA).

    `n_components`: number of dimensions kept in the results (by default 2)

    `ind_weights`: an optional individuals weights (by default, a list/tuple/array/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: an optional variables weights (by default, a list/tuple/array/Series of 1 for uniform variables weights), the weights are given only for the active variables

    `ind_sup`: an integer/string/list/tuple indicating the indexes/names of the supplementary individuals

    `min_error`: iterate until the change in communalities is less than min_error (default = 1e-3)

    `max_iter`: Maximum number of iterations for convergence (default  = 50)

    `warnings`: warnings = True => warn if number of components is too many.

    `rotate`: a boolean indicating if varimax rotation should be performs.
    
    `rotate_kwargs`: optional
        Dictionary containing keyword arguments for the rotation method.
    
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
        * `rotate`: a boolean
        * `rotate_kwargs`: empty dict or dictionary containing keyword arguments for the rotation method
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals

    `eig_`: a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `eigval_`: a namedtuple of numpy array containing eigen values
        * `original`: eigen values of the original matrix
        * `common`: eigen values of the common factor solution

    `vaccounted_`: a pandas DataFrame containing the variance accounted

    `rotmat_`: 

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
    >>> from scientisttools import load_dataset, FA, summaryFA
    >>> beer = load_dataset("beer")
    >>> #non-iterative principal factor analysis (NIPFA)
    >>> res_fa = FactorAnalysis(max_iter=1,rotate=False)
    >>> res_fa.fit(beer)
    >>> #iterative principal factor analysis (IPFA)
    >>> res_fa = FA(rotate=False)
    >>> res_fa.fit(beer)
    >>> #harris component analysis (HCA)
    >>> res_fa = FA(method = "harris",max_iter=1,rotate=False)
    >>> res_fa.fit(beer)
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
                 rotate = False,
                 rotate_kwargs = None):
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

    def fit(self,X:DataFrame,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns.

        `y`: None
            y is ignored

        Returns
        -------
        `self`: object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X) 
        
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
        #check if supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary individuals labels
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)
       
        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary individuls
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal Factor Analysis (PFA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all variables are numerics
        if not all(is_numeric_dtype(X[k]) for k in X.columns):
            raise TypeError("All active columns must be numerics")

        #number of rows/columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])
        
        #set variables weights
        if self.var_weights is None:
            var_weights = ones(n_cols)
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list or a tuple or a 1-D array or a pandas Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list or a tuple or a 1-D array or a pandas Series with length {n_cols}.")
        else:
            var_weights = array(self.var_weights)

        #convert weights to Series
        ind_weights, var_weights =  Series(ind_weights,index=X.index,name="weight"), Series(var_weights,index=X.columns,name="weight")
 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: Z = (X - mu)/sigma
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        center= Series(average(X,axis=0,weights=ind_weights),index=X.columns,name="center")
        scale = Series(array([sqrt(cov(X.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)]),index=X.columns,name="scale")
        #standardization : Z = (X - mu)/sigma
        Z = X.sub(center,axis=1).div(scale,axis=1)
        
        #weighted pearson correlation
        wcorr = wcorrcoef(X=X,weights=ind_weights)

        #eigen values of the original matrix
        eigvals, _ = linalg.eig(wcorr)
        
        #Init communality - Prior
        init_comm = 1 - 1/diag(linalg.inv(wcorr))

        #replace diagonal of correlation matrix with initial communality
        wcorr_c = wcorr.copy()
        fill_diagonal(wcorr_c.values,init_comm)

        # Harris correlation matrix
        if self.method == "harris":
            wcorr_c = wcorr_c.div(sqrt(1-init_comm),axis=1).div(sqrt(1-init_comm),axis=0)
            
            #apply_along_axis(func1d=lambda x : x/sqrt(1-init_comm),axis=0,arr=wcorr_c/sqrt(1-init_comm))
        
        #eigen decomposition
        eigenvals, eigenvec = linalg.eigh(wcorr_c)

        #set n_components_
        max_components = int((eigenvals > 0).sum())
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = int(min(self.n_components,max_components))
         
        #loadings or factor pattern
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
            #harris correlation matrix
            if self.method == "harris":
                wcorr_c = wcorr_c.div(sqrt(1-new_comm),axis=1).div(sqrt(1-new_comm),axis=0)

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

        #convert to DataFrame
        var_coord = DataFrame(var_coord,columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        #add rotation
        if self.rotate:
            if n_components <= 1:
                warn("No rotation will be performed when the number of factors equals 1.")
            else:
                #set parameters
                if self.rotate_kwargs is None:
                    self.rotate_kwargs = dict(normalize = True,max_iter = 1000,tol = 1e-5)
                #check if a dictionary
                if not isinstance(self.rotate_kwargs, (dict,OrderedDict)):
                    raise TypeError("`rotate_kwargs` must be a dictionary.")
                #add normalize
                if 'normalize' not in self.rotate_kwargs.keys():
                    self.rotate_kwargs = {**self.rotate_kwargs,**dict(normalize = True)}
                if 'max_iter' not in self.rotate_kwargs.keys():
                    self.rotate_kwargs = {**self.rotate_kwargs,**dict(max_iter = 1000)}
                if 'tol' not in self.rotate_kwargs.keys():
                    self.rotate_kwargs = {**self.rotate_kwargs,**dict(tol = 1e-5)}

                rot = varimax(loadings=var_coord,**self.rotate_kwargs)
                var_coord,  self.rotmat_ = rot.loadings, rot.rotmat

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,min_error=self.min_error,
                            max_iter=self.max_iter,rotate=self.rotate,rotate_kwargs=self.rotate_kwargs,ind_sup=ind_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #factor score
        f_score = DataFrame(linalg.inv(wcorr).dot(var_coord),columns = ["Dim."+str(x+1) for x in range(n_components)],index=X.columns)

        #contribution of variables
        var_ctr = f_score.pow(2).mul(var_weights,axis=0).mul(100).div(f_score.pow(2).mul(var_weights,axis=0).sum(axis=0),axis=1)
        
        #store variables informations
        var_ = OrderedDict(coord=var_coord,contrib=var_ctr,f_score=f_score)
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variance accounted
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ss_loadings = var_coord.pow(2).mul(var_weights,axis=0).sum(axis=0)
        prop_var, prop_expl = 100*ss_loadings/n_cols, 100*ss_loadings/sum(ss_loadings)
        self.vaccounted_ = DataFrame(c_[ss_loadings,prop_var,cumsum(prop_var),prop_expl,cumsum(prop_expl)],index = ["Dim."+str(x+1) for x in range(n_components)],
                                     columns=["SS loadings","Proportion Var","Cumulative Var","Proportion Explained","Cumulative Proportion"])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals factor coordinates
        ind_coord = Z.mul(var_weights,axis=1).dot(f_score)
        ind_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        self.ind_ = namedtuple("ind",["coord"])(ind_coord)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #reconst and partial correlations
        rcorr, pcorr = var_coord.dot(var_coord.T), wpcorrcoef(X=X,weights=ind_weights)
        #residual correlation
        residual_corr = wcorr.sub(rcorr)
        #error
        error = (tril(residual_corr,-1)**2).sum().sum()
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
        final_comm = var_coord.pow(2).sum(axis=1)
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
        #coordinates of supplementary individuals                                      
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #supplementary individuals factor coordinates
            ind_sup_coord = X_ind_sup.sub(center,axis=1).div(scale,axis=1).mul(var_weights,axis=1).dot(f_score)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",["coord"])(ind_sup_coord)

        #all quantitative variables in original dataframe
        all_quanti = Xtot.select_dtypes(include=number)
        #drop supplementary individuals
        if self.ind_sup is not None:
            all_quanti = all_quanti.drop(index=ind_sup_label)
        #descriptive statistics of quantitatives variables 
        self.summary_quanti_ = summarize(X=all_quanti)

        #correlation tests
        all_vars = Xtot.copy()
        if self.ind_sup is not None:
            all_vars = all_vars.drop(index=ind_sup_label)
        self.corrtest_ = corrmatrix(X=all_vars,weights=ind_weights)

        self.model_ = "fa"
        return self

    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
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
        X is projected on the principal factor previously extracted from a training set.

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal factor where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        is_dataframe(X=X) #check if X is an instance of pd.DataFrame classs
               
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] = ncols
            raise ValueError("'columns' aren't aligned")
        
        if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
            raise TypeError("All columns in X must be numerics")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the FA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns
        
        #standardization: Z = (X - mu)/sigma
        Z = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1)
        #apply transition relation
        coord = Z.mul(self.call_.var_weights,axis=1).dot(self.var_.f_score)
        #coord.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord