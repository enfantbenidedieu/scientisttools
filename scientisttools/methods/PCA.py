# -*- coding: utf-8 -*-
from scipy.stats import chi2
from numpy import ndarray, array, ones, average, sqrt, linalg, log, flip,cumsum,mean, diag, c_, tril, number, average, cov
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from collections import namedtuple,OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.gfa import gfa
from .functions.wcorrcoef import wcorrcoef
from .functions.wpcorrcoef import wpcorrcoef
from .functions.kaiser_msa import kaiser_msa
from .functions.predict_sup import predict_sup
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.function_eta2 import function_eta2
from .functions.summarize import summarize, conditional_wmean
from .functions.association import association
from .functions.corrmatrix import corrmatrix
from .functions.utils import is_dataframe

class PCA(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA)
    ----------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Principal Component Analysis (PCA) with supplementary individuals and/or supplementary variables (numerics and/or categoricals). Missing values are replaced by the column mean.

    Usage
    -----
    ```python
    >>> PCA(standardize = True, n_components = 5, ind_weights = None, var_weights = None, ind_sup = None, sup_var = None)
    ```
    
    Parameters
    ----------
    `standardize`: a boolean indicating if the data should be scaled to unit variance (by default True)
        * If `True`: the data are scaled to unit variance.
        * If `False`: the data are not scaled to unit variance.

    `n_components`: an integer indicating the number of dimensions kept in the results (by default 5)
    
    `ind_weights`: None or a list or a tuple or a 1-D array or a pandas Series indicating individuals weights. The weights are given only for active individuals.
    
    `var_weights`: None or a list or a tuple or a 1-D arry or a pandas Series indicating variables weights. The weights are given only for the active variables
    
    `ind_sup`: None or an integer or a string or a list or a tuple or a range indicating the indexes or the names of the supplementary individuals

    `sup_var`: None or an integer or a string or a list or a tuple or a range indicating the indexes or the names of the supplementary variables

    Attributes
    ----------
    `call_`: a namedtuple with some informations, including:
        * `Xtot`: a pandas DataFrame with all data (active and supplementary)
        * `X`: a pandas DataFrame with active data
        * `Z`: a pandas DataFrame with standardized data : Z = (X - center)/scale
        * `ind_weights`: a pandas Series containing individuals weights
        * `var_weights`: a pandas Series containing variables weights
        * `center`: a pandas Series containing variables weighted average
        * `scale`: a pandas Series containing variables weighted standard deviation : 
            * If `standardize = True`, then standard deviation are computed using variables weighted standard deviation
            * If `standardize = False`, then standard deviation are a vector of ones with length number of variables.
        * `n_components`: an integer indicating the number of components kepted
        * `ind_sup`: None or a list of string containing names of the supplementary individuals
        * `sup_var`: None or a list of string containing names of the supplementary variables
    
    `svd_`: a namedtuple of matrices containing all the results of the generalized singular value decomposition (GSVD), including
        * `vs`: a 1-D numpy array containing the singular values
        * `U`: a 2-D numpy array whose columns contains the left singular vectors
        * `V`: a 2-D numpy array whose columns contains the right singular vectors.

    `eig_`: pandas DataFrame containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `ind_`: a namedtuple of pandas DataFrames containing all the results for the active individuals, including:
        * `coord`: coordinates of the individuals,
        * `cos2`: squared cosinus of the individuals,
        * `contrib`: relative contributions of the individuals,
        * `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the individuals.

    `var_`: a namedtuple of pandas DataFrames containing all the results for the active variables, including:
        * `coord`: coordinates of the variables,
        * `cos2`: squared cosinus of the variables,
        * `contrib`: relative contributions of the variables,
        * `infos`: additionals informations (weight, squared distance to origin, inertia and percentage of inertia) of the variables.

    `corr_`: a namedtuple of pandas DataFrame containing all the results for the correlation, including:  
        * `corrcoef`: Pearson correlation coefficient matrix
        * `pcorrcoef`: partial pearson correlation coefficient matrix
        * `reconst`: reconstitution pearson correlation coefficient matrix
        * `residual`: residual correlation matrix
        * `error`: error

    `others_`: a namedtuple of others statistics, including:
        * `r2_score`: the multiple R square between the factor score estimates,
        * `communality`: communality estimates for each variable. These are merely the sum of squared factor loadings for that variable,
        * `communalities`: total amount of common variance,
        * `uniquenesses`: uniquenesse estimates for each variable, 
        * `bartlett`: Bartlett's test of Spericity, 
        * `kaiser`: a namedtuple of numerics containing the Kaiser threshold, including:
            * `threshold`: kaiser threshold,
            * `proportion`: kaiser threshold in proportion
        * `kaiser_msa`: Kaiser measure of sampling adequacy,
        * `kss`: Karlis-Saporta-Spinaki,
        * `broken`: broken's stick threshold

    `ind_sup_`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary individuals, including:
        * `coord`: coordinates of the supplementary individuals, 
        * `cos2`: squared cosinus of the supplementary individuals,
        * `dist2`: squared distance to origin of the supplementary individuals.

    `quanti_sup_`: a namedtuple of pandas DataFrames containing all the results for the supplementary quantitative variables, including:
        * `coord`: coordinates of the supplementary quantitative variables, 
        * `cos2`: squared cosinus of the supplementary quantitative variables.

    `quali_sup_`: a namedtuple of pandas DataFrames/Series containing all the results for the supplementary qualitative variables/levels, including:
        * `coord`: coordinates of the supplementary levels,
        * `cos2`: squared cosinus of the supplementary levels,
        * `vtest`: value-test (which is a criterion with a Normal distribution) of the supplementary levels,
        * `eta2`: squared correlation ratio of the supplementary qualitative variables, which is the square correlation coefficient between a qualitative variable and a dimension
        * `dist2`: squared distance to origin of the supplementary levels.

    `summary_quali_`: a pandas DataFrame containing the frequencies distribution of supplementary levels

    `association_`; a nametuple of pandas DataFrames containing all the results of association between qualitative variables, including:
        * `chi2`: Pearson's chi-squared test
        * `gtest`: log-likelihood ratio (i.e the "G-test")
        * `association`: degree of association between two nominal variables ("cramer", "tschuprow", "pearson")

    `summary_quanti_`: a pandas DataFrame containing the descriptive statistics of quantitative variables

    `corrtest_`: a pandas DataFrame containing correlation tests

    `model_`: a string indicating the model fitted = 'pca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * Bry X. (1996), Analyses factorielles multiple, Economica

    * Bry X. (1999), Analyses factorielles simples, Economica

    * Escofier B., Pagès J. (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    * Saporta G. (2006). Probabilites, Analyse des données et Statistiques. Technip

    * Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    * Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    * Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    * Rakotomalala, R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    * Tenenhaus, M. (2006). Statistique : Méthodes pour décrire, expliquer et prévoir. Dunod.
    
    See Also
    --------
    `predictPCA`, `supvarPCA`, `get_pca_ind`, `get_pca_var`, `get_pca`, `summaryPCA`, `dimdesc`, `reconst`, `fviz_pca_ind`, `fviz_pca_var`, `fviz_pca_biplot`

    Examples
    --------
    ```python
    >>> from scientisttools import load_dataset, PCA, summaryPCA
    >>> decathlon = load_dataset("decathlon")
    >>> res_pca = PCA(ind_sup=range(41,46), sup_var=(10,11,12))
    >>> res_pca.fit(decathlon)
    >>> summaryPCA(res_pca)
    ```
    """
    def __init__(self,
                 standardize = True,
                 n_components = 5,
                 ind_weights = None,
                 var_weights = None,
                 ind_sup = None,
                 sup_var = None):
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
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` in the number of samples and `n_columns` is the number of columns (quantitative and/or qualitative).

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
        #check if standardize is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(self.standardize,bool):
            raise TypeError("'standardize' must be a boolean.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary individuals labels
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)

        #get supplementary variables labels
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)

        #make a copy of the original data
        Xtot = X.copy()

        #drop supplementary variables (quantitative and/or qualitative)
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
        
        #drop supplementary individuals
        if self.ind_sup is not None:
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #principal components analysis (PCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all active variables are numerics
            raise TypeError("All active variables must be numeric")

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
        #compute weighted average and weighted standard deviation
        center = average(X,axis=0,weights=ind_weights)
        if self.standardize:
            scale = array([sqrt(cov(X.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_cols)])
        else:
            scale = ones(n_cols)
        #convert to Series
        center, scale = Series(center,index=X.columns,name="center"), Series(scale,index=X.columns,name="scale")
        
        #standardization: Z = (X - mu)/sigma
        Z = X.sub(center,axis=1).div(scale,axis=1)
        
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
        call_ = OrderedDict(Xtot=Xtot,X=X,Z=Z,ind_weights=ind_weights,var_weights=var_weights,center=center,scale=scale,n_components=n_components,max_components=max_components,
                            ind_sup=ind_sup_label,sup_var=sup_var_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized factor analysis model and extract all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fit_ = gfa(X=Z,row_weights=ind_weights,col_weights=var_weights,max_components=max_components,n_components=n_components)

        #extract elements
        self.svd_, self.eig_, ind_, var_ = fit_.svd, fit_.eig, fit_.row, fit_.col

        #convert to namedtuple
        self.ind_, self.var_ = namedtuple("ind",ind_.keys())(*ind_.values()), namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #corrcoef, reproduces and partial corrcoef correlations
        wcorr, rcorr, pcorr = wcorrcoef(X=X,weights=ind_weights), var_["coord"].dot(var_["coord"].T), wpcorrcoef(X=X,weights=ind_weights)
        #residual correlation
        residual_corr = wcorr.sub(rcorr)
        #error
        error = (tril(residual_corr,-1)**2).sum().sum()
        #convert to ordered dictionary
        corr_ = OrderedDict(corrcoef=wcorr,pcorrcoef=pcorr,reconst=rcorr,residual=residual_corr,error=error)
        self.corr_ = namedtuple("correlation",corr_.keys())(*corr_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##others informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fidélité des facteurs - variance of the scores - R2
        f_fidelity = var_["coord"].mul(fit_.svd.V).sum(axis=0)
        f_fidelity.name = "R2"
        #Initial community
        init_comm = 1 - 1/diag(linalg.inv(wcorr))
        #estimated communalities
        final_comm = var_["coord"].pow(2).sum(axis=1)
        #communality
        communality = DataFrame(c_[init_comm,final_comm],columns=["Prior","Final"],index=Z.columns)
        #communalities
        communalities = sum(final_comm)
        #uniquenesses
        uniquenesses = 1 - final_comm
        uniquenesses.name = "Uniqueness"
        #Bartlett - statistics
        bartlett_stats = -(n_rows-1-(2*n_cols+5)/6)*sum(log(fit_.eig.iloc[:,0]))
        bs_dof = n_cols*(n_cols-1)/2
        bs_pvalue = 1 - chi2.cdf(bartlett_stats,df=bs_dof)
        bartlett = DataFrame([[sum(log(fit_.eig.iloc[:,0])),bartlett_stats,bs_dof,bs_pvalue]],columns=["|CORR.MATRIX|","CHISQ","dof","p-value"],index=["Bartlett's test"])
        #Kaiser threshold
        kaiser = namedtuple("kaiser",["threshold","proportion"])(mean(fit_.eig.iloc[:,0]),100/sum(fit_.col["infos"].iloc[:,2]))
        #Karlis - Saporta - Spinaki threshold
        kss_threshold =  1 + 2*sqrt((n_cols-1)/(n_rows-1))
        #broken-stick crticial values
        broken = Series(flip(cumsum([1/x for x in range(n_cols,0,-1)]))[:max_components],name="Broken-stick crit. val.",index=["Dim.".format(x+1) for x in range(max_components)])
        #convert to ordered dictionnary
        others_ = OrderedDict(r2_score=f_fidelity,communality=communality,communalities=communalities,uniquenesses=uniquenesses, bartlett=bartlett, kaiser=kaiser,kaiser_msa=kaiser_msa(X=X),kss=kss_threshold,broken=broken)
        #convert to namedtuple
        self.others_ = namedtuple("others",others_.keys())(*others_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup is not None:
            #standardization: Z = (X-mu)/sigma
            Z_ind_sup = X_ind_sup.sub(center,axis=1).div(scale,axis=1)
            #statistics for supplementary individuals
            ind_sup_ = predict_sup(X=Z_ind_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
            #convert to namedtuple
            self.ind_sup_ = namedtuple("ind_sup",ind_sup_.keys())(*ind_sup_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary variables (quantitative and/or qualitative)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            if self.ind_sup is not None:
                X_sup_var = X_sup_var.drop(index=ind_sup_label)

            #split X_sup_var
            split_X_sup_var = splitmix(X=X_sup_var)
            X_quanti_sup, X_quali_sup, n_quanti_sup, n_quali_sup = split_X_sup_var.quanti, split_X_sup_var.quali, split_X_sup_var.k1, split_X_sup_var.k2
            
            #statistics for supplementary quantitative variables
            if n_quanti_sup > 0:
                #compute weighted average for supplementary quantitative variables
                center_sup = average(X_quanti_sup,axis=0,weights=ind_weights)
                if self.standardize:
                    scale_sup = array([sqrt(cov(X_quanti_sup.iloc[:,k],rowvar=False,aweights=ind_weights,ddof=0)) for k in range(n_quanti_sup)])
                else:
                    scale_sup = ones(n_quanti_sup)
                #convert to pandas Series
                center_sup, scale_sup = Series(center_sup,index=X_quanti_sup.columns,name="center"), Series(scale_sup,index=X_quanti_sup.columns,name="scale")
                #standardization : Z = (X - mu)/sigma
                Z_quanti_sup = X_quanti_sup.sub(center_sup,axis=1).div(scale_sup,axis=1)
                #statistics for supplementary quantitative variables
                quanti_sup_ = predict_sup(X=Z_quanti_sup,Y=fit_.svd.U,weights=ind_weights,axis=1)
                del quanti_sup_['dist2'] #delete dist2
                #convert to namedtuple
                self.quanti_sup_ = namedtuple("quanti_sup",quanti_sup_.keys())(*quanti_sup_.values())

            #statistics for supplementary qualitative variables/levels
            if n_quali_sup > 0:
                #recode
                rec = recodecat(X=X_quali_sup)
                X_quali_sup, dummies_sup = rec.X, rec.dummies
                #conditional mean - Barycenter of original data
                X_levels_sup = conditional_wmean(X=X,Y=X_quali_sup,weights=ind_weights)
                #standardization: Z = (X - mu)/sigma
                Z_levels_sup = X_levels_sup.sub(center,axis=1).div(scale,axis=1)
                #statistics for supplementary levels
                quali_sup_ = predict_sup(X=Z_levels_sup,Y=fit_.svd.V,weights=var_weights,axis=0)
                #vtest for the supplementary levels
                p_k_sup = dummies_sup.mul(ind_weights,axis=0).sum(axis=0)
                levels_sup_vtest = quali_sup_["coord"].mul(sqrt((n_rows-1)/(1/p_k_sup).sub(1)),axis=0).div(fit_.svd.vs[:n_components],axis=1)
                #eta2 for the supplementary qualitative variables
                quali_sup_sqeta = function_eta2(X=X_quali_sup,Y=fit_.row["coord"],weights=ind_weights,excl=None)
                #convert to ordered dictionary
                quali_sup_ = OrderedDict(barycentre=X_levels_sup,coord=quali_sup_["coord"],cos2=quali_sup_["cos2"],vtest=levels_sup_vtest,eta2=quali_sup_sqeta,dist2=quali_sup_["dist2"])
                #convert to namedtuple
                self.quali_sup_ = namedtuple("quali_sup",quali_sup_.keys())(*quali_sup_.values())
                
                #descriptive descriptive of qualitative variables
                self.summary_quali_ = summarize(X=X_quali_sup)

                #degree of association - multivariate goodness
                if n_quali_sup > 1:
                    self.association_ = association(X=X_quali_sup) 

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

        self.model_ = "pca"
        return self
    
    def fit_transform(self,X:DataFrame,y=None) -> DataFrame:
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X`: a pandas DataFrame of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y`: None
            y is ignored.
        
        Returns
        -------
        `X_new`: a pandas DataFrame of shape (n_samples, n_components)
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
        `X`: a pandas DataFrame of shape (n_samples, n_components).
            New data, where `n_samples` is the number of samples and `n_components` is the number of components.

        Returns
        -------
        `X_original`: a pandas DataFrame of shape (n_samples, n_columns)
            Original data, where `n_samples` is the number of samples and `n_columns` is the number of columns
        """
        is_dataframe(X=X) #check if X is a pandas DataFrame
        
        n_components = min(X.shape[1],self.call_.n_components) #set number of components
        eigvals = self.var_.coord.pow(2).T.dot(self.call_.var_weights)[:n_components]
        #inverse transform
        X_original = X.iloc[:,:n_components].dot(self.var_.coord.iloc[:,:n_components].div(sqrt(eigvals),axis=1).T).mul(self.call_.scale,axis=1).add(self.call_.center,axis=1)
        return X_original

    def transform(self,X:DataFrame) -> DataFrame:
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X`: a pandas Dataframe of shape (n_samples, n_columns)
            New data, where `n_samples` is the number of samples and `n_columns` is the number of columns.

        Returns
        -------
        `X_new`: a pandas Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components where `n_samples` is the number of samples and `n_components` is the number of the components.
        """
        is_dataframe(X=X) #check if X is a pandas DataFrame
        
        X.index.name = None #set index name as None

        if X.shape[1] != self.call_.X.shape[1]: #check if X.shape[1] == n_cols
            raise ValueError("'columns' aren't aligned")
        
        if not all(is_numeric_dtype(X[k]) for k in X.columns): #check if all variables are numerics
            raise TypeError("All columns in X must be numerics")
        
        intersect_col = list(set(X.columns) & set(self.call_.X.columns)) #find intersect
        if len(intersect_col) != self.call_.X.shape[1]:
            raise ValueError("The names of the variables is not the same as the ones in the active variables of the PCA result")
        X = X.loc[:,self.call_.X.columns] #reorder columns

        #standardisation and apply transition relation
        X_new = X.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        X_new.columns = ["Dim."+str(x+1) for x in range(self.call_.n_components)]
        return X_new