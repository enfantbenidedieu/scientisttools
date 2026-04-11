# -*- coding: utf-8 -*-
from numpy import ones,sqrt,outer,corrcoef
from scipy import linalg
from collections import namedtuple, OrderedDict
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

#interns functions
from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.model_matrix import model_matrix
from ..functions.gsvd import gSVD
from ..functions.statistics import wmean, wstd
from ..others._splitmix import splitmix
from ..others._disjunctive import disjunctive

class CCA(BaseEstimator,TransformerMixin):
    """
    Canonical Correspondence Analysis (CCA)

    Performs canonical (also known as constrained) correspondence analysis (CCA).
    Canonical (or constrained) correspondence analysis is a multivariate ordination technique. It appeared in community ecology [1] and relates community composition to the variation in the environment (or in other factors). It works from data on abundances or counts of samples and constraints variables, and outputs ordination axes that maximize sample separation among species.

    Parameters
    ----------
    ncp : int, default = 5
        The number of dimensions kept in the results.

    scaling : int, default = 1
        Scaling type 1 maintains :math:`\chi^{2}` distances between rows. Scaling type 2 preserves :math:`\chi^{2}` distances between columns. For a more detailed explanation of the interpretation, see [1] 
        
    env : int, str, list, tuple or range, default = None 
        The indexes or names of the environmental variables (continuous and/or categorical). Categorical variables are recoded into dummy variables without first category.

    row_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary rows points.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_rows, n_columns)
            Input data.
        X : DataFrame of shape (n_rows, n_xcolumns)
            The species data.
        Y : DataFrame of shape (n_rows, n_ycolumns)
            The environmental variables.
        Ycod : DataFrame of shape (n_rows, n_ycolumns)
            The recoded environmental variables.
        Z1 : DataFrame of shape (n_rows, n_xcolumns)
            The standardized residuals.
        Z2 : DataFrame of shape (n_rows, n_ycolumns)
            The standardized recoded variables.
        row_m : Series of shape (n_rows,)
            The rows margins.
        col_m : Series of shape (n_xcolumns)
            The columns margins.
        total : int
            The sum of all elements in ``X``.
        scaling : int
            The scaling type.
        env : list
            The names of the environmental variables.
        row_sup : None, list
            The names of the supplementary rows.

    coef_ : DataFrame of shape (n_xcolumns, n_zcolumns)
        The coefficients of canonical regression.

    col_ : col
        An object containing the results of the columns with the following attributes:

        coord : coord
            An object with the following attributes:

            CCA : DataFrame of shape (n_rows, ncp)
                The CCA columns coordinates.
            CA : DataFrame of shape (n_rows, ncp)
                The CA columns coordinates.

    eig_ : eig
        An object containing the results of the eigenvalues with the following attributes:

        CCA : Series of shape (ncp,)
            The CCA eigenvalues

        CA: Series of shape (ncp,)
            The CA eigenvalues

    quanti_var_ : quanti_var
        An object containing the results for the environmental variables with the following attributes:

        coord : DataFrame of shape (n_env, ncp)
            The coordinates of the environmental variables.

    row_ : row
        An object containing the results for the rows with the following attributes:

        coord : coord
            An object with the following attributes:

            CCA : DataFrame of shape (n_rows, ncp)
                The CCA rows coordinates.
            CA : DataFrame of shape (n_rows, ncp)
                The CA rows coordinates.

    row_sup_ : row_sup
        An object containing the results for the supplementary rows with the following attributes:

        coord : coord
            An object with the following attributes:

            CCA : DataFrame of shape (n_rows, ncp)
                The CCA supplementary rows coordinates.
            CA : DataFrame of shape (n_rows, ncp)
                The CA supplementary rows coordinates.
    
    svd_ : svd
        An object with the follwoing attributes:

        CCA : Series of shape (maxcp,)
            The CCA singular values decomposition.

        CA: Series of shape (maxcp,)
            The CA singular values decomposition.
    
    References
    ----------
    [1] Cajo J. F. Ter Braak, Canonical Correspondence Analysis: A New Eigenvector Technique for Multivariate Direct Gradient Analysis, Ecology 67.5 (1986), pp. 1167-1179.

    [2] Cajo J.F. Braak and Piet F.M. Verdonschot, Canonical correspondence analysis and related multivariate methods in aquatic ecology, Aquatic Sciences 57.3 (1995), pp. 255-289.

    [3] Legendre P. and Legendre L. 1998. Numerical Ecology. Elsevier, Amsterdam.

    Examples
    --------
    >>> from scientisttools.datasets dune
    >>> from scientisttools import CCA
    >>> clf = CCA(ncp=2,env=range(5),scaling=1)
    >>> clf.fit(dunedata)
    CCA(env=range(5),ncp=2,scaling=1)
    """
    def __init__(
            self, ncp=5, scaling=1, env=None, row_sup=None, tol = 1e-7
    ):
        self.ncp = ncp
        self.scaling = scaling
        self.env = env
        self.row_sup = row_sup
        self.tol = tol

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
        # check if scaling value is correct
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not self.scaling in (1,2):
            raise ValueError("'scaling' must be 1 or 2.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if env is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.env is None: 
            raise ValueError("'env' must be assigned.")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #get environment variables labels and supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        env_label, row_sup_label = get_sup_label(X=X, indexes=self.env, axis=1), get_sup_label(X=X, indexes=self.row_sup, axis=0)

        #make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #drop supplementary rows
        if self.row_sup is not None: 
            X_row_sup, X = X.loc[row_sup_label,:], X.drop(index=row_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correspondence analysis (CCA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #split between environmental and species
        Y, X = X.loc[:,env_label], X.drop(columns=env_label)

        #number of rows/columns
        n_rows, n_cols = X.shape[0], X.shape[1]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #species abundance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total 
        total = int(X.sum(axis=0).sum())
        #relative frequence distribution
        P = X/total
        #set rows and columns margins
        row_m, col_m = P.sum(axis=1), P.sum(axis=0)
        row_m.name, col_m.name = "fi.", "f.j"
        #set rows and columns weights
        row_w, col_w = row_m.copy(), col_m.copy()
        col_w.name, row_w.name = "weight", "weight"
        #relative expected frequency
        expected_freq = outer(row_m,col_m)
        #relative standardized residuals
        Z1 = (P - expected_freq)/sqrt(expected_freq)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #step 2: standardization of environmental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #recode categorical variable into disjunctive and drop first
        Ycod = model_matrix(X=Y)
        #compute weighted average and standard deviation
        center, scale = wmean(X=Ycod,w=row_m), wstd(X=Ycod,w=row_m)
        #standardization: z_ik = (x_ik - m_k)/s_k
        Z2 = (Ycod - center)/scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #step 3: weighted multiple regression
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X_w = (Z2.T * sqrt(row_m)).T
        #canonical regression
        beta, _, _, _ = linalg.lstsq(X_w, Z1)
        #fitted values and residuals
        Y_hat, Y_res = X_w.dot(beta), Z1 - X_w.dot(beta).values
        Y_hat.columns = Z1.columns
        #coefficients of weighted multiple regression
        self.coef_ = DataFrame(beta,index=X_w.columns,columns=Z1.columns)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #step 4: eigen value decomposition of fitted values
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        svd_hat = gSVD(X=Y_hat,ncp=self.ncp,row_w=ones(n_rows),col_w=ones(n_cols),tol=self.tol)
        U_hat = Z1.dot(svd_hat.V[:,:svd_hat.ncp])/svd_hat.vs[:svd_hat.ncp]
        #eigenvalues
        eigvals_hat = Series(svd_hat.vs**2,index=[f"Dim{x+1}" for x in range(svd_hat.rank)],name="Eigenvalue")

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,Y=Y,Ycod=Ycod,Z1=Z1,Z2=Z2,total=total,row_m=row_m,col_m=col_m,row_w=row_w,col_w=col_w,ncp=svd_hat.ncp,env=env_label,row_sup=row_sup_label)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #step 5: eigen value decomposition of residuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        svd_res = gSVD(X=Y_res,ncp=self.ncp,row_w=ones(n_rows),col_w=ones(n_cols),tol=self.tol)
        U_hat_res = Y_res.dot(svd_res.V[:,:svd_res.ncp])/svd_res.vs[:svd_res.ncp]
        #igenvalues
        eigvals_res = Series(svd_res.vs**2,index=[f"Dim{x+1}" for x in range(svd_res.rank)],name="Eigenvalue")
            
        #convert to namedtuple
        self.eig_, self.svd_ = namedtuple("eig",["CCA","CA"])(eigvals_hat,eigvals_res), namedtuple("svd",["CCA","CA"])(svd_hat,svd_res)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #columns estimations and residuals
        col_coord_hat = DataFrame((svd_hat.V[:,:svd_hat.ncp].T/sqrt(col_m.values)).T,index=X.columns,columns=self.eig_.CCA.index[:svd_hat.ncp])
        col_coord_res = DataFrame((svd_res.V[:,:svd_res.ncp].T/sqrt(col_m.values)).T,index=X.columns,columns=self.eig_.CA.index[:svd_res.ncp])
        if self.scaling == 2:
            col_coord_hat, col_coord_res = col_coord_hat * svd_hat.vs[:svd_hat.ncp], col_coord_res * svd_res.vs[:svd_res.ncp]
        #convert to namedtuple
        col_coord = namedtuple("coord",["CCA","CA"])(col_coord_hat,col_coord_res)
        #convert to ordered dictionary
        col_ = OrderedDict(coord=col_coord)
        #convert to namedtuple
        self.col_ = namedtuple("col",col_.keys())(*col_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #rows estimation scores
        row_coord_hat, row_coord_res = (U_hat.T/sqrt(row_m.values)).T, (U_hat_res.T/sqrt(row_m.values)).T
        if self.scaling == 1:
            row_coord_hat, row_coord_res = row_coord_hat * svd_hat.vs[:svd_hat.ncp], row_coord_res * svd_res.vs[:svd_res.ncp]
        row_coord_hat.columns, row_coord_res.columns = self.eig_.CCA.index[:svd_hat.ncp], self.eig_.CA.index[:svd_res.ncp]
        #convert to namedtuple
        row_coord = namedtuple("coord",["CCA","CA"])(row_coord_hat,row_coord_res)
        #convert to ordered dictionary
        row_ = OrderedDict(coord = row_coord)
        #convert to namedtuple
        self.row_ = namedtuple("row",row_.keys())(*row_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for features
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        quanti_var_coord = DataFrame(corrcoef(x=X_w,y=svd_hat.U[:,:svd_hat.ncp],rowvar=False)[:X_w.shape[1],X_w.shape[1]:],index= X_w.columns,columns=self.eig_.CCA.index[:self.svd_.CCA.ncp])
        #convert to ordered dictionary
        quanti_var_ = OrderedDict(coord = quanti_var_coord)
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #statistics for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.row_sup is not None:
            #split between environmental and species
            Y_row_sup, X_row_sup = X_row_sup.loc[:,env_label], X_row_sup.drop(columns=env_label)
            #standardized residuals for supplementary rows
            P_row_sup = X_row_sup/total
            #row margins for the supplementary rows
            row_sup_m = P_row_sup.sum(axis=1)
            #expected frequencies for supplementary rows
            expected_freq_row_sup = outer(row_sup_m,col_m)
            #relative standardized residuals for supplementary rows
            Z1_row_sup = (P_row_sup - expected_freq_row_sup)/sqrt(expected_freq_row_sup)

            #split Y_row_sup
            split_Y_row_sup = splitmix(Y_row_sup)
            #extract elements
            Y_row_sup_quanti, Y_row_sup_quali, n_quanti, n_quali = split_Y_row_sup.quanti, split_Y_row_sup.quali, split_Y_row_sup.k1, split_Y_row_sup.k2

            Ycod_row_sup = DataFrame(index=row_sup_label,columns=Ycod.columns).astype(float)
            #check if numerics variables
            if n_quanti > 0:
                #replace with numerics columns
                Ycod_row_sup.loc[:,Y_row_sup_quanti.columns] = Y_row_sup_quanti
            
            #check if categorical variables      
            if n_quali > 0:
                #active categorics
                categorics = [x for x in self.call_.Ycod.columns if x not in self.call_.Y.columns]
                #replace with dummies
                Ycod_row_sup.loc[:,categorics] = disjunctive(X=Y_row_sup_quali,cols=categorics,prefix=True,sep="")
            #standardization: z_ik = (x_ik - m_k)/s_k
            Z2_row_sup = (Ycod_row_sup - center)/scale
            #Multiply by supplementary rows margins
            X_w_row_sup = (Z2_row_sup.T * sqrt(row_sup_m)).T

            #supplementary rows
            U_hat_row_sup = Z1_row_sup.dot(self.svd_.CCA.V[:,:self.svd_.CCA.ncp])/self.svd_.CCA.vs[:self.svd_.CCA.ncp]
            U_hat_res_row_sup = (Z1_row_sup - X_w_row_sup.dot(self.coef_).values).dot(self.svd_.CA.V[:,:self.svd_.CA.ncp])/self.svd_.CA.vs[:self.svd_.CA.ncp]

            #supplementary rows estimation scores
            row_sup_coord_hat, row_sup_coord_res = (U_hat_row_sup.T/sqrt(row_sup_m.values)).T, (U_hat_res_row_sup.T/sqrt(row_sup_m.values)).T
            if self.scaling == 1:
                row_sup_coord_hat, row_sup_coord_res = row_sup_coord_hat * self.svd_.CCA.vs[:self.svd_.CCA.ncp], row_sup_coord_res * self.svd_.CA.vs[:self.svd_.CA.ncp]
            row_sup_coord_hat.columns, row_sup_coord_res.columns = self.eig_.CCA.index[:svd_hat.ncp], self.eig_.CA.index[:svd_res.ncp]

            #convert to namedtuple
            row_sup_coord = namedtuple("coord",["CCA","CA"])(row_sup_coord_hat,row_sup_coord_res)
            #convert to ordered dictionary
            row_sup_ = OrderedDict(coord = row_sup_coord)
            #convert to namedtuple
            self.row_sup_ = namedtuple("row_sup",row_sup_.keys())(*row_sup_.values())

        return self