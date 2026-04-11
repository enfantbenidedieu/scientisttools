# -*- coding: utf-8 -*-
from numpy import ndarray, ones, array,insert,diff,nan,sum,c_,cumsum
from pandas import Series, DataFrame, concat
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.gensvd import gensvd
from ..functions.statistics import wmean, wstd, wcorr, wcov
from ..functions.cancor_test import lrtest, pillai_test, hotelling_test,roy_test
from ..functions.utils import check_is_bool, check_is_all_numeric_dtype

class CANCORR(BaseEstimator,TransformerMixin):
    """
    Canonical Correlation Analysis (CANCORR)
    
    Performs Canonical Correlation Analysis (CANCORR) to highlight correlations between two dataframes, with supplementary individuals.
    Missing values are replaced by the column mean.

    Parameters
    ----------
    scale_unit : bool, default = True
        If ``True``, then the data are scaled to unit variance.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    group : list, tuple
        The number of variables in each group.

    name_group : list, tuple, default = None
        The name of the groups. If ``None``, the group are named X and Y.

    prefix_group : list, tuple, default = None
        The prefix of the groups. If ``None``, the group are prefixed Xcan, Ycan.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    Returns
    -------
    cancoef_ : cancoef
        An object with the following attributes

        X : DataFrame of shape (n_xcolumns, ncp)
            Raw canonical coefficients for ``X``.
        Y : DataFrame of shape (n_ycolumns, ncp)
            Raw canonical coefficients for ``Y``.
        
    cancorr_ : DataFrame of shape (ncp, 2)
        Canonical correlation.

    call_ : call 
        An object with the following attributes:

        Xtot : DataFrame of shape (n_rows, n_columns)
            Input data;
        X : DataFrame of shape (n_rows, n_xcolumns)
            First group data.
        Y : DataFrame of shape (n_rows, n_ycolumns)
            Second group data.
        Z : DataFrame of shape (n_rows, n_columns)
            Standardized data.
        row_w : Series of shape (n_rows,)
            Rows weights.
        center : Series of shape (n_columns,)
            The weighted average of ``X``.
        scale : Series of shape (n_columns,)
            The weighted standard deviation of ``X``.
        ncp : int
            The number of components kepted.
        group : list
            The number of variables in each group.
        name_group : list
            The name of the groups.
        prefix_group : list
            The prefix of the groups.

    corr_ : corr
        An object with the following attributes:

        xcorr : DataFrame of shape (n_xcolumns, n_ycolumns)
            Correlation coefficients among ``X``.
        ycorr :DataFrame of shape (n_ycolumns, n_ycolumns)
            Coerrelation coefficients among ``Y``.
        xycorr : DataFrame of shape (n_xcolumns, n_ycolumns)
            Correlation coefficients between ``X`` and ``Y``.

    cov_ : cov
        An object with the following attributes:

        xcov : DataFrame of shape (n_xcolumns, n_ycolumns)
            Covariance among ``X``.
        ycov :DataFrame of shape (n_ycolumns, n_ycolumns)
            Covariance among ``Y``.
        xycov : DataFrame of shape (n_xcolumns, n_ycolumns)
            Covariance between ``X`` and ``Y``.

    ind_ : ind
        An object containing the results for the individuals with the following attributes:

        X : DataFrame of shape (n_rows, ncp)
            Individuals scores for ``X``.
        Y : DataFrame of shape (n_rows, ncp)
            Individuals scores for ``Y``.

    ind_ : ind
        An object containing the resuts for the supplementary individuals with the following attributes:

        X : DataFrame of shape (n_rows_sup, ncp)
            Supplementary individuals scores for ``X``.
        Y : DataFrame of shape (n_rows_sup, ncp)
            Supplementary individuals scores for ``Y``.

    manova_ : manova
        An object with the following attributes:

        wilks : lrtestResult
            An object with the following attributes:

            header: str
                The test hypothesis.
            statistic : DataFrame of shape (ncp, 5)
                Likelihood test statistic.

        pillai : pillaitestResult
            An object with the following attributes:

            header: str
                The test hypothesis.
            statistic : DataFrame of shape (ncp, 5)
                Pillai test statistic.
        
        hotelling : hotellingtestResult
            An object with the following attributes:

            header: str
                The test hypothesis.
            statistic : DataFrame of shape (ncp, 5)
                Hotelling test statistic.

        roy : roytestResult
            An object with the following attributes:

            header: str
                The test hypothesis.
            statistic : DataFrame of shape (1,5)
                Roy's test statistic.

    quanti_var_ : quanti_var
        An object with the following attributes:

        X : X
            An object with the following attributes

            xscores : DataFrame of shape (n_xcolumns, ncp)
                Correlation between ``X`` and ``X`` scores. 
            yscores : DataFrame of shape (n_xcolumns, ncp)
                Correlation between ``X`` and ``Y`` scores. 

        Y : Y
            An object with the following attributes

            xscores : DataFrame of shape (n_ycolumns, ncp)
                Correlation between ``Y`` and ``X`` scores. 
            yscores : DataFrame of shape (n_ycolumns, ncp)
                Correlation between ``Y`` and ``Y`` scores. 

    sscp_ : sscp
        An object with the following attributes:

        xsscp : DataFrame of shape (n_xcolumns, n_xcolumns)
            Sum of squared cross product among ``X``.
        ysscp : DataFrame of shape (n_ycolumns, n_ycolumns)
            Sum of squared cross product among ``Y``.
        xysscp : DataFrame of shape (n_xcolumns, n_ycolumns)
            Sum of squared cross product between ``X`` and ``Y``.

    References
    ----------
    [1] Afifi, A, Clark, V and May, S. 2004. Computer-Aided Multivariate Analysis. 4th ed. Boca Raton, Fl: Chapman & Hall/CRC.

    [2] Gittins, R. (1985). Canonical Analysis: A Review with Applications in Ecology, Berlin: Springer.

    [3] Hotelling H. (1936). Relations between two sets of variables. Biometrika, 28, 321-327. doi:10.1093/biomet/28.3-4.321.

    [4] Mardia, K. V., Kent, J. T. and Bibby, J. M. (1979). Multivariate Analysis. London: Academic Press.

    [5] Seber, G. A. F. (1984). Multivariate Observations. New York: Wiley. Page 506f.

    Examples
    --------
    >>> from scientisttools.datasets import fitnessclub
    >>> from scientisttools import CANCOR
    >>> clf = CANCOR(scale_unit=False,ncp=3,group=(3,3),name_group=("Physiological","Exercises"))
    >>> clf.fit(fitnessclub)
    CANCOR(group=(3,3),name_group=("Physiological","Exercises"),ncp=3,scale_unit=False)
    """
    def __init__(
            self, scale_unit=False, ncp=None, group=None, name_group=None, prefix_group=None, row_w=None, ind_sup = None
    ):
        self.scale_unit = scale_unit
        self.ncp = ncp
        self.group = group
        self.name_group = name_group
        self.prefix_group = prefix_group
        self.row_w = row_w
        self.ind_sup = ind_sup

    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns),
            Training data, where ``n_rows`` in the number of rows and ``n_columns`` is the number of columns.

        y : None
            y is ignored.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if scale_unit is a boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.scale_unit)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None: 
            raise ValueError("'group' must be assigned.")
        elif not isinstance(self.group, (list,tuple,ndarray,Series)): 
            raise ValueError("'group' must be a 1d array-like with the number of variables in each group")
        elif len(self.group) != 2: 
            raise ValueError("'group' must be a 1d array-like of shape (2,).")
        else: 
            group = [int(x) for x in self.group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.name_group is None: 
            name_group = ["X","Y"]
        elif not isinstance(self.name_group,(list,tuple,ndarray,Series)): 
            raise TypeError("'name_group' must be a 1d array-like with the name of the groups")
        elif len(self.name_group) != 2: 
            raise ValueError("'name_group' must be a 1d array-like of shape (2,).")
        else: 
            name_group = [str(x) for x in self.name_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #assigned group prefix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.prefix_group is None: 
            prefix_group = ["Xcan","Ycan"]
        elif not isinstance(self.prefix_group,(list,tuple,ndarray,Series)): 
            raise TypeError("'prefix_group' must be a 1d array-like with prefix of group")
        elif len(self.prefix_group) != 2: 
            raise ValueError("'prefix_group' must be a 1d array-like of shape (2,).")
        else: 
            prefix_group = [str(x) for x in self.prefix_group]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if all values are numerics - all columns are continuous
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_all_numeric_dtype(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary individuals
        if self.ind_sup is not None: 
            X_ind_sup, X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correlation analysis (CANCOR)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #number of rows/columns
        n_rows = X.shape[0]
        n_xcols, n_ycols = group

        #split X into X1 and X2
        X1, X2 = X.iloc[:,:n_xcols], X.iloc[:,n_xcols:]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_w is None: 
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'row_w' must be a 1d array-like with the individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"'row_w' must be a 1d array-like of shape ({n_rows},).")
        else: 
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #weighted average of X
        center, scale = wmean(X=X,w=row_w), wstd(X=X,w=row_w) if self.scale_unit else Series(ones(X.shape[1]),index=X.columns,name="scale")
        #standardization: z_ik = (x_ik - m_k)/s_k
        Z = (X - center)/scale
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #maximum number of components
        maxncp = min(group)
        #set number of componenets
        if self.ncp is None: 
            ncp = maxncp
        elif not isinstance(self.ncp, int): 
            raise TypeError("'ncp' must be an integer")
        elif self.ncp < 1: 
            raise TypeError("'ncp' must be a positive integer")
        else:
            ncp = int(min(self.ncp, maxncp))

        #set call_ informations
        call_ = OrderedDict(Xtot=Xtot,X=X1,Y=X2,Z=Z,row_w=row_w,center=center,scale=scale,ncp=ncp,group=group,name_group=name_group,prefix_group=prefix_group,ind_sup=ind_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #covariance matrices
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #biased covariance matrices
        xcovb, ycovb, xycovb = wcov(X1,w=row_w,ddof=0), wcov(X2,w=row_w,ddof=0), wcov(X,w=row_w,ddof=0).iloc[:n_xcols,n_xcols:]
        #unbiaised covariance matrices
        xcov, ycov, xycov = wcov(X1,w=row_w,ddof=1), wcov(X2,w=row_w,ddof=1), wcov(X,w=row_w,ddof=1).iloc[:n_xcols,n_xcols:]
        #convert to namedtuple
        cov_ = OrderedDict(xcov=xcov,ycov=ycov,xycov=xycov,xcovb=xcovb,ycovb=ycovb,xycovb=xycovb)
        #convert to namedtuple
        self.cov_ = namedtuple("cov",cov_.keys())(*cov_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #sum of squared cross producs
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #unbiaised covariance matrices
        xsscp, ysscp, xysscp = xcovb * n_rows, ycovb * n_rows, xycovb * n_rows
        #convert to namedtuple
        sscp_ = OrderedDict(xsscp=xsscp,ysscp=ysscp,xysscp=xysscp)
        #convert to namedtuple
        self.sscp_ = namedtuple("sscp",sscp_.keys())(*sscp_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #pearson correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #pearson correlation of X, Y and XY
        xcorr, ycorr, xycorr = wcorr(X1,w=row_w), wcorr(X2,w=row_w), wcorr(X,w=row_w).iloc[:n_xcols,n_xcols:]
        #convert to ordered dictionary
        corr_ = OrderedDict(xcorr=xcorr,ycorr=ycorr,xycorr=xycorr)
        #add to model attributes
        self.corr_ = namedtuple("corr",corr_.keys())(*corr_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #generalized singular value decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        gensvd_ = gensvd(xycorr,xcorr,ycorr,nu=n_xcols,nv=n_ycols) if self.scale_unit else gensvd(xycov,xcov,ycov,nu=n_xcols,nv=n_ycols)
        #canonical correlation
        rho = gensvd_.cor[:maxncp]
        #raw canonical coefficients
        xcancoef = DataFrame(gensvd_.xcoef[:,:ncp],index=X1.columns,columns=[f"{prefix_group[0]}{x+1}" for x in range(ncp)])
        ycancoef = DataFrame(gensvd_.ycoef[:,:ncp],index=X2.columns,columns=[f"{prefix_group[1]}{x+1}" for x in range(ncp)])
        #add to model attributes
        self.cancoef_ = namedtuple("cancoef",name_group)(xcancoef,ycancoef)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigenvalue informations
        eigvals = array([x/(1-x) for x in rho**2])
        eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index=[f"Can{x+1}" for x in range(len(eigvals))])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correlations and multivariate statistics 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correlation
        self.cancorr_ = DataFrame(c_[rho,rho**2],columns=["Canonical Correlation","Squared Canonical Correlation"],index=self.eig_.index)

        #multivariate statistics and F approximations
        wilks, pillai  = lrtest(rho=rho,n_samples=n_rows,n_xcols=n_xcols,n_ycols=n_ycols), pillai_test(rho=rho,n_samples=n_rows,n_xcols=n_xcols,n_ycols=n_ycols)
        hotelling, roy = hotelling_test(rho=rho,n_samples=n_rows,n_xcols=n_xcols,n_ycols=n_ycols), roy_test(rho=rho,n_samples=n_rows,n_xcols=n_xcols,n_ycols=n_ycols)
        #convert to ordered dictionary
        manova_ = OrderedDict(wilks=wilks,pillai=pillai,hotelling=hotelling,roy=roy)
        #add to model attributes
        self.manova_ = namedtuple("manova",manova_.keys())(*manova_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for individuals - canonical scores 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #scores for the individuals in X and Y
        xscores, yscores = Z.iloc[:,:n_xcols].dot(xcancoef), Z.iloc[:,n_xcols:].dot(ycancoef)
        #add to model attributes
        self.ind_ = namedtuple("ind",name_group)(xscores,yscores)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute correlation
        corr_X_xscores, corr_X_yscores = wcorr(concat((X1,xscores),axis=1), w=row_w).iloc[:n_xcols,n_xcols:], wcorr(concat((X1,yscores),axis=1), w=row_w).iloc[:n_xcols,n_xcols:]
        corr_Y_xscores, corr_Y_yscores = wcorr(concat((X2,xscores),axis=1), w=row_w).iloc[:n_ycols,n_ycols:], wcorr(concat((X2,yscores),axis=1), w=row_w).iloc[:n_ycols,n_ycols:]
        #convert to namedtupkle
        X_scores, Y_scores = namedtuple(name_group[0],["xscores","yscores"])(corr_X_xscores,corr_X_yscores), namedtuple(name_group[1],["xscores","yscores"])(corr_Y_xscores,corr_Y_yscores)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",name_group)(X_scores,Y_scores)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ind_sup:
            #standardization
            Z_ind_sup = (X_ind_sup - center)/scale
            #scores in X and Y
            xscores_sup, yscores_sup = Z_ind_sup.iloc[:,:n_xcols].dot(xcancoef), Z_ind_sup.iloc[:,n_xcols:].dot(ycancoef)
            #add to model attributes
            self.ind_sup_ = namedtuple("ind",name_group)(xscores_sup,yscores_sup)

        return self