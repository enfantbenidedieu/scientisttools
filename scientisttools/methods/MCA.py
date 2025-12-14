# -*- coding: utf-8 -*-
from numpy import array, ones, ndarray, c_, cumsum, sqrt, zeros, linalg, average, cov
from pandas import DataFrame, Series, concat
from pandas.api.types import is_string_dtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_row_sup_label, get_col_sup_label
from .functions.recodecat import recodecat
from .functions.gfa import gfa
from .functions.splitmix import splitmix
from .functions.function_eta2 import function_eta2
from .functions.association import association
from .functions.summarize import summarize
from .functions.corrmatrix import corrmatrix

class MCA(BaseEstimator,TransformerMixin):
    """
    Multiple Correspondence Analysis (MCA)
    --------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Multiple Correspondence Analysis (MCA) or Specific Multiple Correspondence Analysis (SpecificMCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    >>> MCA(excl = None, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, sup_var = None)
    ```

    Parameters
    ----------
    `excl`: None or an integer or a list indicating the "junk" categories (by default None). It can be a list or a tuple of the names of the categories or a list or a tuple of the indexes in the disjunctive table.

    `n_components`: None or an integer indicating the number of dimensions kept in the results (by default 5)

    `ind_weights`: None or an optional individuals weights (by default, a list/tuple/1darray/Series of 1/(number of active individuals) for uniform individuals weights), the weights are given only for active individuals.
    
    `var_weights`: None or an optional variables weights (by default, a list/tuple/1darray/Series of 1 for uniform variables weights), the weights are given only for the active variables
    
    `ind_sup`: None or an integer or a string or a list or a tuple or a range indicating the indexes or names of the supplementary individuals

    `sup_var`: None or an integer or a string or a list or a tuple or a range indicating the indexes or names of the supplementary variables (quantitative and/or qualitative)

    Atttributes
    -----------
    `call_`: a namedtuple with some informations, including:
        * `Xtot`: a pandas DataFrame with all data (active and supplementary)
        * `X`: a pandas dataframe with active data
        * `dummies`: a pandas DataFrame with disjunctive table
        * `Z`: a pandas DataFrame with standardized data:
        * `ind_weights`: a pandas Series containing the individuals weights
        * `var_weights`: a pandas Series containing the qualitative variables weights
        * `levels_weights`: a pandas Series containing the levels weights
        * `excl`: None or a list of string indicating names of the excluded categories
        * `n_components`: an integer indicating the number of components kept
        * `ind_sup`: None or a list of string indicating names of the supplementary individuals
        * `sup_var`: None or a list of string indicating names of the supplementary variables (qualitative and/or quantitative)
    
    `svd_`: a namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD), including:
        * `vs`: 1D numpy array containing the singular values
        * `U`: 2D numpy array whose columns contain the left singular vectors
        * `V`: 2D numpy array whose columns contain the right singular vectors.

    `eig_`: a pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `eig_correction_`: a namedtuple of pandas DataFrames containing eigenvalues correction, including:
        * `benzecri`: a pandas DataFrame containing Benzecri correction
        * `greenacre`: a pandas DataFrame containing Greenacre correction

    `ind_`: a namedtuple of pandas Dataframes containing all the results for the active individuals, including:
        * `coord`: coordinates of the individuals,
        * `cos2`: squared cosinus of the individuals,
        * `contrib`: relative contributions of the individuals,
        * `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals, including:

    `var_`: a namedtuple of pandas DataFrames containing all the results for the active levels, including:
        * `coord`: coordinates of the levels,
        * `coord_n`: normalized coordinates of the levels,
        * `cos2`: squared cosinus of the levels,
        * `contrib`: relative contributions of the levels,
        * `infos`: additionnal informations (weight, squared distance to origin, inertia and percentage of inertia) of the levels,
        * `vtest`: value-test of the levels

    `quali_var_`: a namedtuple of pandas DataFrames/Seris containing all the results for the active qualitative variables, including:
        * `coord`: coordinates of the qualitative variables, which is eta2, the square correlation corefficient between a qualitative variable and a dimension
        * `contrib`: contributions of the qualitative variables.
        * `infos`: additionals informations (inertia and percentage of inertia) of the qualitative variables,
    
    `others_`: a namedtuple of others statistics, including:
        * `inertia`: global multiple correspondence analysis inertia
        * `kaiser`: namedtuple of numerics values containing the kaiser threshold, including:
            * `threshold`: kaiser threshold
            * `proportion`: kaiser proportion threshold

    `ind_sup_`: a namedtuple of pandas Dataframes/Series containing all the results for the supplementary individuals, including:
        * `coord`: coordinates of the supplementary individuals,
        * `cos2`: squared cosinus of the supplementary individuals,
        * `dist2`: squared distance to origin of the supplementary individuals.

    `quali_sup_`: a namedtuple of pandas DataFrame/Series containing all the results for the supplementary qualitative variables/levels, inclduing:
        * `coord`: coordinates of the supplementary levels,
        * `coord_n`: normalized coordinates of the supplementary levels,
        * `cos2`: squared cosinus of the supplementary levels,
        * `vtest`: value-test of the supplementary levels,
        * `dist2`: squared distance to origin of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables.

    `quanti_sup_`: a namedtuple of pandas DataFrame containing all the results for the supplementary quantitative variables, including:
        * `coord`: coordinates of the supplementary quantitative variables,
        * `cos2`: squared cosinus of the supplementary quantitative variables.

    `summary_quanti_`: a pandas DataFrame containing descriptive statistics for quantitative variables.
    
    `summary_quali_`: a pandas DataFrame containing frequencies distribution for levels.

    `goodness_`: a namedtuple of pandas DataFrame for multivariate goodness of fit test, including:
        * `chi2`: Pearson's chi-squared test
        * `gtest`: log-likelihood ratio (i.e the "G-test")
        * `association`: degree of association between two nominal variables ("cramer", "tschuprow", "pearson")

    `corrtest_`: a pandas DataFrame containing the correlation test.

    `model_`: a string indicating the model fitted = 'mca'

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
    >>> from scientisttools.datasets import poison
    >>> from scientisttools import MCA, summaryMCA
    >>> #multiple correspondence analysis (MCA)
    >>> res_mca = MCA(sup_var=(0,1,2,3))
    >>> res_mca.fit(poison)
    >>> summaryMCA(res_mca)
    >>> #specific multiple correspondence analysis (SpecificMCA)
    >>> res_specmca = MCA(excl=(0,2),sup_var = (0,1,13,14))
    >>> res_specmca.fit(poison)
    >>> summaryMCA(res_specmca)
    ```
    """
    def __init__(self,
                 excl = None,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 sup_var = None):
        self.excl = excl
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
        #preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary elements (individuals and/or variables)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary individuals labels
        ind_sup_label = get_row_sup_label(X=X,row_sup=self.ind_sup)
        
        #check if supplementary variables (quantitative and/or qualitative)
        sup_var_label = get_col_sup_label(X=X,col_sup=self.sup_var)
        
        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Multiple correspondence analysis (MCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not all(is_string_dtype(X[q]) for q in X.columns): #check if all active columns are categoricals
            raise TypeError("All active columns in `X` must be categoricals")
        
        #recode variables
        rec = recodecat(X=X)
        X, dummies = rec.X, rec.dummies
        
        #number of rows/columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        if self.ind_weights is None:
            ind_weights = ones(n_rows)/n_rows
        elif not isinstance(self.ind_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'ind_weights' must be a list or a tuple or a 1D array or a pandas Series of individuals weights.")
        elif len(self.ind_weights) != n_rows:
            raise ValueError(f"'ind_weights' must be a list or a tuple or a 1D array or a pandas Series with length {n_rows}.")
        else:
            ind_weights = array([x/sum(self.ind_weights) for x in self.ind_weights])

        #set variables weights
        if self.var_weights is None:
            var_weights = ones(n_cols)/n_cols
        elif not isinstance(self.var_weights,(list,tuple,ndarray,Series)):
            raise TypeError("'var_weights' must be a list or a tuple or a 1D array or a pandas Series of variables weights.")
        elif len(self.var_weights) != n_cols:
            raise ValueError(f"'var_weights' must be a list or a tuple or a 1D array or a pandas Series with length {n_cols}.")
        else:
            var_weights = array([x/sum(self.var_weights) for x in self.var_weights])

        #number of levels, count and proportion
        n_levels, p_k = dummies.shape[1], dummies.mul(ind_weights,axis=0).sum(axis=0)
        #standardization: z_ik = (y_ik/p_k) - 1
        Z = dummies.div(p_k,axis=1).sub(1)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set exclusion label(s)
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
        #set categories weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        nb_levels = array([X[j].nunique() for j in X.columns])
        var_weights2 = array(list(chain(*[repeat(i,k) for i, k in zip(var_weights,nb_levels)])))
        levels_weights = array([x*y for x,y in zip(p_k,var_weights2)])

        #replace excluded categories weights by 0
        if self.excl is not None:
            for i in excl_idx:
                levels_weights[i] = 0
        
        #convert weights to Series
        ind_weights, var_weights, levels_weights =  Series(ind_weights,index=X.index,name="weight"), Series(var_weights,index=X.columns,name="weight"), Series(levels_weights,index=Z.columns,name="weight")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_levels - n_cols))
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
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Z=Z, ind_weights=ind_weights,var_weights=var_weights,levels_weights=levels_weights,
                            excl=excl_label,n_components=n_components,ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Z,row_weights=ind_weights,col_weights=levels_weights,max_components=max_components,n_components=n_components)
        
        #extract elements
        self.svd_, self.eig_ = fit_.svd, fit_.eig

        #replace nan or inf by 0
        if self.excl is not None:
            self.svd_.V[excl_idx,:] = 0

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Eigenvalues corrections
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #save eigen value grather than threshold
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
        benzecri = DataFrame(c_[lambd_tilde,s_tilde,cumsum(s_tilde)],columns=["Eigenvalue","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])
        #greenacre correction
        s_tilde_tilde = n_cols/(n_cols-1)*(sum(self.eig_.iloc[:,0]**2)-(n_levels - n_cols)/(n_cols**2))
        tau = 100*(lambd_tilde/s_tilde_tilde)
        greenacre = DataFrame(c_[lambd_tilde,tau,cumsum(tau)],columns=["Eigenvalue","Proportion","Cumulative"],index = ["Dim."+str(x+1) for x in range(len(lambd))])
        #convert to namedtuple
        self.eig_correction_ = namedtuple("correction",["benzecri","greenacre"])(benzecri,greenacre)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals - convert to NamedTuple
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.ind_ = namedtuple("ind",fit_.row.keys())(*fit_.row.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #levels additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #normalized levels coordinates - conditional weighted mean
        levels_coord_n = fit_.col["coord"].mul(fit_.svd.vs[:n_components],axis=1)
        #vtest for the levels
        levels_vtest = fit_.col["coord"].mul(sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.excl is not None:
            var_ = OrderedDict(coord=fit_.col["coord"].drop(index=excl_label),coord_n=levels_coord_n.drop(index=excl_label),vtest=levels_vtest.drop(index=excl_label),
                               contrib=fit_.col["contrib"].drop(index=excl_label),cos2=fit_.col["cos2"].drop(index=excl_label),infos=fit_.col["infos"].drop(index=excl_label))
        else:
            var_ = OrderedDict(coord=fit_.col["coord"],coord_n=levels_coord_n,vtest=levels_vtest,contrib=fit_.col["contrib"],cos2=fit_.col["cos2"],infos=fit_.col["infos"])
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eta2 for the qualitative variables
        quali_var_eta2 = function_eta2(X=X,Y=fit_.row["coord"],weights=ind_weights,excl=excl_label)
        #contributions for the qualitative variables
        quali_var_contrib = concat((var_["contrib"].loc[X[j].unique(),:].sum(axis=0).to_frame(j).T for j in X.columns),axis=0)
        #inertia for the qualitative variables
        quali_var_inertia = (nb_levels - 1)/n_rows
        #percentage of inertia for the qualitative variables
        quali_var_inertia_pct = 100*quali_var_inertia/sum(quali_var_inertia)
        #convert to DataFrame
        quali_var_infos = DataFrame(c_[var_weights,quali_var_inertia,quali_var_inertia_pct],columns=["Weight","Inertia","% Inertia"],index=X.columns)
        #convert to namedtuple
        self.quali_var_ = namedtuple("quali_var",["coord","contrib","infos"])(quali_var_eta2,quali_var_contrib,quali_var_infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #multiple correspondence analysis additionals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #inertia
        inertia = (n_levels/n_cols) - 1
        #eigenvalue threshold
        kaiser_proportion_threshold = 100/inertia
        #convert to namedtuple
        self.others_ = namedtuple("others",["inertia","kaiser"])(inertia,namedtuple("kaiser",["threshold","proportion"])(kaiser_threshold,kaiser_proportion_threshold))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #disjunctive table
            dummies_ind_sup = DataFrame(zeros((len(ind_sup_label),n_levels)),columns=dummies.columns,index=ind_sup_label)
            for i in range(len(ind_sup_label)):
                values = [X_ind_sup.iloc[i,j] for j in range(n_cols)]
                for k in range(n_levels):
                    if dummies.columns[k] in values:
                        dummies_ind_sup.iloc[i,k] = 1
            #standardize the data and exclude the data
            Z_ind_sup = dummies_ind_sup.div(p_k,axis=1).sub(1)
            #coordinates for the supplementary individuals
            ind_sup_coord = Z_ind_sup.mul(levels_weights,axis=1).sum(fit_.svd.V)
            ind_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            #dist2 for the supplementary individuals
            ind_sup_sqdisto = Z_ind_sup.pow(2).mul(levels_weights,axis=1).sum(axis=1)
            ind_sup_sqdisto.name = "Sq. Dist."
            #cos2 for the supplemantary individuals
            ind_sup_sqcos = ind_sup_coord.pow(2).div(ind_sup_sqdisto,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",["coord","cos2","dist2"])(ind_sup_coord, ind_sup_sqcos, ind_sup_sqdisto)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary qualitative variables (qualitative and/or quantitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            if self.ind_sup is not None: #remove supplementary individuals
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2

            #initialize Z
            Z_col_sup = DataFrame().astype(float)
            if n_quanti_sup > 0:
                #compute weighted average and weighted standard deviation
                center_sup = Series(average(X_quanti_sup,axis=0,weights=ind_weights),index=X_quanti_sup.columns,name="center")
                scale_sup = Series([sqrt(cov(X_quanti_sup.iloc[:,k],aweights=ind_weights,ddof=0)) for k in range(n_quanti_sup)],index=X_quanti_sup.columns,name="scale")
                #standardization : Z = (X - mu)/sigma
                Z_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
                #concatenate
                Z_col_sup = concat((Z_col_sup, Z_quanti_sup),axis=1)

            if n_quali_sup > 0:
                #recode supplementary qualitative variables
                rec2 = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec2.X, rec2.dummies
                #proportion and count
                p_k_sup = dummies_sup.mul(ind_weights,axis=0).sum(axis=0)
                #standardization : z_ik = (y_ik/p_k) - 1
                Z_quali_sup = dummies_sup.div(p_k_sup,axis=1).sub(1)
                #concatenate
                Z_col_sup = concat((Z_col_sup,Z_quali_sup),axis=1)

            #coordinates for the supplementary variables
            col_sup_coord = Z_col_sup.mul(ind_weights,axis=0).T.dot(fit_.svd.U)
            col_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            #dist2 for the supplementary variables
            col_sup_sqdisto  = Z_col_sup.pow(2).mul(ind_weights,axis=0).sum(axis=0)
            col_sup_sqdisto.name = "Sq. Dist."
            #cos2 for the supplementary variables
            col_sup_sqcos = col_sup_coord.pow(2).div(col_sup_sqdisto,axis=0)

            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0 :
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",["coord","cos2"])(col_sup_coord.iloc[:n_quanti_sup,:], col_sup_sqcos.iloc[:n_quanti_sup,:])

                #descriptive statistics for quantitative variables
                self.summary_quanti_ = summarize(X=X_quanti_sup)

            #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #coordinates, cos2 and dist2 of the supplementary levels
                levels_sup_coord, levels_sup_sqcos, levels_sup_sqdisto = col_sup_coord.iloc[n_quanti_sup:,:], col_sup_sqcos.iloc[n_quanti_sup:,:], col_sup_sqdisto.iloc[n_quanti_sup:]
                #normalized coordinates of the supplementary levels
                levels_sup_coord_n = levels_sup_coord.mul(fit_.svd.vs[:n_components],axis=1)
                #vtest of the supplementary levels
                levels_sup_vtest = levels_sup_coord.mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0)
                #eta2 of the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=fit_.row["coord"],weights=ind_weights,excl=excl_label)
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(coord=levels_sup_coord,coord_n=levels_sup_coord_n,cos2=levels_sup_sqcos,vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=levels_sup_sqdisto)
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())

        #multivariate goodness of fit
        all_quali = Xtot.select_dtypes(include=["object","category"])
        if self.ind_sup is not None:
            all_quali = all_quali.drop(index=ind_sup_label)
        self.goodness_ = association(X=all_quali,alpha=0.05)
        #frequencies distribution of levels
        self.summary_quali_ = summarize(X=all_quali)

        #correlation tests
        all_vars = Xtot.copy()
        if self.ind_sup is not None:
            all_vars = all_vars.drop(index=ind_sup_label)
        self.corrtest_ = corrmatrix(X=all_vars,weights=ind_weights)

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
        """
        if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #set number of components
        n_components = min(X.shape[1],self.call_.n_components)
        eigvals = self.var_.coord.pow(2).T.dot(self.call_.levels_weights)[:n_components]
        p_k = self.call_.dummies.mul(self.call_.ind_weights,axis=0).sum(axis=0)
        #inverse transform
        X_original = X.iloc[:,:n_components].dot(self.var_.coord.iloc[:,:n_components].div(sqrt(eigvals),axis=1).T)
        #estimation of standardize data
        X_original = X_original.add(1).mul(p_k,axis=1)
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
        """
        if not isinstance(X,DataFrame): #check if X is a pandas DataFrame
           raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        X.index.name = None #set index name as None
        
        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
            raise ValueError("'columns' aren't aligned")
     
        if not all(is_string_dtype(X[q]) for q in X.columns): #check if all columns are categoricals
            raise TypeError("All columns in `X` must be categoricals")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the MCA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns

        #create disjunctive table for new individuals
        dummies_new = DataFrame(zeros((X.shape[0],self.call_.dummies.shape[1])),columns=self.call_.dummies.columns,index=X.index)
        for i in range(X.shape[0]):
            values = [X.iloc[i,j] for j in range(X.shape[1])]
            for k in range(self.call_.dummies.shape[1]):
                if self.call_.dummies.columns[k] in values:
                    dummies_new.iloc[i,k] = 1
        #proportion of levels
        p_k = self.call_.dummies.mul(self.call_.ind_weights,axis=0).sum(axis=0)
        #standardization (z_ik = (y_ik/pk)-1) and apply transition relation
        coord = dummies_new.div(p_k,axis=1).sub(1).mul(self.call_.levels_weights,axis=1).dot(self.svd_.V)
        coord.columns  = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return coord