# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .recodecont import recodecont
class CCA(BaseEstimator,TransformerMixin):
    """
    Canonical Correlation Analysis (CCA)
    ------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Canonical Correlation Analysis (CCA) to highlight correlations between two dataframes.
    
    Missing values are replaced by the column mean.

    Usage
    -----
    ```python
    >>> CCA(X = None,standardize=False,n_components=2,vars = None, vars_with = None)
    ```

    Parameters
    ----------
    `X` : pandas/polars dataframe of shape (n_samples, n_columns)

    `standardize` : a boolean, default = True
        * If True : the data are scaled to unit variance.
        * If False : the data are not scaled to unit variance.
    
    `n_components` : number of dimensions kept in the results (by default None)

    `vars` : The vars statement lists the variables in the first of the two sets of variables to be analyzed. The variables must be numeric. If you omit the vars statement, all numeric variables not mentioned in other statements make up the first set of variables.
    
    `vars_with` : The vars_with statement lists the variables in the second set of variables to be analyzed. The variables must be numeric. The vars_with statement is required.
    
    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `call_` : dictionary with some statistics

    `cov_` : dictionary of covariance matrix (unbiased and biased)

    `corr_` : dictionary of Pearson correlation matrix

    `standardized_coef_` : standardized canonical coefficients

    `coef_` : dictionary of raws canonical coefficients

    `can_coef_` : canonical coefficients

    `ind_` : dictionary of pandas dataframe containing all the results for the active individuals (coordinates for first group, coordinates for second group)

    `var_` : dictionary of pandas dataframe containing all the results for the active variables (canonical loadings)

    `tstat_` : tests of canonical dimensions

    `model_` : string specifying the model fitted = 'cca'

    References
    ----------
    Gittins, R. (1985). Canonical Analysis: A Review with Applications in Ecology, Berlin: Springer.

    Mardia, K. V., Kent, J. T. and Bibby, J. M. (1979). Multivariate Analysis. London: Academic Press.

    Afifi, A, Clark, V and May, S. 2004. Computer-Aided Multivariate Analysis. 4th ed. Boca Raton, Fl: Chapman & Hall/CRC.

    Links
    -----
    https://stats.oarc.ucla.edu/r/dae/canonical-correlation-analysis/

    https://stats.oarc.ucla.edu/r/dae/canonical-correlation-analysis/
    
    https://stats.oarc.ucla.edu/sas/dae/canonical-correlation-analysis/

    https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/viewer.htm#cancorr_toc.htm
    
    https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html
    
    https://cmdlinetips.com/2020/12/canonical-correlation-analysis-in-python/
    
    https://cmdlinetips.com/2020/12/canonical-correlation-analysis-in-r/
    
    https://www.geeksforgeeks.org/canonical-correlation-analysis-cca-using-sklearn/
    
    https://medium.com/@pozdrawiamzuzanna/canonical-correlation-analysis-simple-explanation-and-python-example-a5b8e97648d2
    
    https://stackoverflow.com/questions/26298847/canonical-correlation-analysis-in-python-with-sklearn
    
    https://fr.wikipedia.org/wiki/Analyse_canonique_des_corr%C3%A9lations
    
    https://www.cabannes.org/#4
    
    https://lemakistatheux.wordpress.com/2017/01/20/lanalyse-canonique-des-correlations-a-venir/
    
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html
    
    https://github.com/cran/CCP/blob/master/R/CCP.R
    
    https://github.com/cran/CCP
    
    https://github.com/friendly/candisc/tree/master
    
    https://davetang.github.io/muse/cca.html
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # load lifecyclesavings dataset
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    >>> from scientisttools import CCA
    >>> res_cca = CCA(lifecyclesavings,vars=[1,2])
    ```
    """
    def __init__(self,
                 X = None,
                 standardize=False,
                 n_components=2,
                 vars = None,
                 vars_with = None):
        
        def geigen(Amat,Bmat,Cmat):
            """
            Generalized eigenanalysis
            -------------------------

            Description
            -----------
            Solve the generalized eigenanalysis problem

            max{tr L'AM}

            Parameters
            ----------

            References
            ----------
            https://github.com/cran/fda/blob/master/R/geigen.R

            """
            Bdim = Bmat.shape
            Cdim = Cmat.shape
            if Bdim[0]!=Bdim[1]:
                raise TypeError("Bmat is not square")
            if Cdim[0]!=Cdim[1]:
                raise TypeError("Cmat is not square")
            p = Bdim[0]
            q = Cdim[0]
            s = min(p,q)
            if (np.max(np.abs(Bmat - Bmat.T))/np.max(np.abs(Bmat))) > 1e-10:
                raise TypeError("Bmat not symmetric")
            if (np.max(np.abs(Cmat - Cmat.T))/np.max(np.abs(Cmat))) > 1e-10:
                raise TypeError("Cmat not symmetric")
            # Update Bmat, Cmat
            Bmat = (Bmat + Bmat.T)/2
            Cmat = (Cmat + Cmat.T)/2
            Bfac = sp.linalg.cholesky(Bmat,lower=False)
            Cfac = sp.linalg.cholesky(Cmat, lower=False)
            Bfacinv = np.linalg.inv(Bfac)
            Cfacinv = np.linalg.inv(Cfac)
            Dmat = np.dot(np.dot(Bfacinv.T,Amat),Cfacinv)
            if p >= q:
                result = np.linalg.svd(Dmat)
                values = result[1]
                Lmat = np.dot(Bfacinv,result[0])
                Mmat = np.dot(Cfacinv, result[2].T)
            else:
                result = np.linalg.svd(Dmat.T)
                values = result[1]
                Lmat = np.dot(Bfacinv,result[2].T)
                Mmat = np.dot(Cfacinv,result[0])
            
            res = {"values" : values[:s], "Lmat" : Lmat[:,:s], "Mmat" : Mmat[:,:s]}
            return res

        def compute(X,Y,res,standardize):
            # Compute means
            xmeans, ymeans = X.mean(axis=0).values.reshape(1,-1), Y.mean(axis=0).values.reshape(1,-1)
            if standardize:
                xstd, ystd = X.std(axis=0,ddof=0).values.reshape(1,-1), Y.std(axis=0,ddof=0).values.reshape(1,-1)
            else:
                xstd, ystd = 1.0, 1.0
            
            Z1, Z2 = (X - xmeans)/xstd, (Y - ymeans)/ystd

            # Extract xscores and yscores
            xscores, yscores = Z1.dot(res["xcoef"]), Z2.dot(res["ycoef"])
            xscores.columns = ["V."+str(x+1) for x in range(xscores.shape[1])]
            yscores.columns = ["W."+str(x+1) for x in range(yscores.shape[1])]
            
            ##### Compute correlation
            X_xscores = np.corrcoef(X,xscores, rowvar=False)[:X.shape[1],X.shape[1]:]
            Y_xscores = np.corrcoef(Y,xscores, rowvar=False)[:Y.shape[1],Y.shape[1]:]
            X_yscores = np.corrcoef(X,yscores, rowvar=False)[:X.shape[1],X.shape[1]:]
            Y_yscores = np.corrcoef(Y,yscores, rowvar=False)[:Y.shape[1],Y.shape[1]:]

            # Put it in dataframe
            #xscores = pd.DataFrame(xscores,columns=)
            corr_X_xscores = pd.DataFrame(X_xscores,index=X.columns,columns=["V."+str(x+1) for x in range(X_xscores.shape[1])])
            corr_Y_xscores = pd.DataFrame(Y_xscores,index=Y.columns,columns=["V."+str(x+1) for x in range(Y_xscores.shape[1])])
            corr_X_yscores = pd.DataFrame(X_yscores,index=X.columns,columns=["W."+str(x+1) for x in range(X_yscores.shape[1])])
            corr_Y_yscores = pd.DataFrame(Y_yscores,index=Y.columns,columns=["W."+str(x+1) for x in range(Y_yscores.shape[1])])

            res = {"xscores" : xscores,"yscores" : yscores,"corr_X_xscores" : corr_X_xscores,"corr_Y_xscores" : corr_Y_xscores,
                   "corr_X_yscores" : corr_X_yscores,"corr_Y_yscores" : corr_Y_yscores}
            return res
        
        def rcc(X,Y,standardize):
            ################################ Define Biased Covariance Matrix
            xcovb, ycovb = X1.cov(ddof=1), X2.cov(ddof=1)
            xycovb = pd.DataFrame(np.cov(X1,X2,rowvar=False,ddof=1)[:X1.shape[1],X1.shape[1]:],index = X1.columns, columns=X2.columns)
            
            # Generalized eigenanalysis
            # https://github.com/cran/fda/blob/master/R/geigen.R
            res = geigen(xycovb,xcovb,ycovb)
            res = {"cor" : res["values"],"xcoef" : res["Lmat"], "ycoef" : res["Mmat"]}
            scores = compute(X=X,Y=Y,res=res,standardize=standardize)

            ###### Put in DataFrame
            xcoef = pd.DataFrame(res["xcoef"],index=X.columns,columns=["V."+str(x+1) for x in range(res["xcoef"].shape[1])])
            ycoef = pd.DataFrame(res["ycoef"],index=Y.columns,columns=["W."+str(x+1) for x in range(res["ycoef"].shape[1])])
            
            result = {"cor" : res["cor"],"xcoef" : xcoef,"ycoef" : ycoef,"scores" : scores}
            return result
        
        ############## Statistics tests
        # Lambda Wilks
        def WilksLambda(rho,p,q):
            minpq = min(p,q)
            value = np.zeros(minpq)
            for i in range(minpq):
                value[i] = np.prod((1-rho**2)[i:minpq])
            return value
        
        # HotellingLawleyTrace
        def HotellingLawleyTrace(rho,p,q):
            minpq = min(p,q)
            value = np.zeros(minpq)
            for i in range(minpq):
                rhosq = rho**2
                value[i] = np.sum((rhosq/(1-rhosq))[i:minpq])
            return value
        
        # PillaiBartlettTrace
        def PillaiBartlettTrace(rho,p,q):
            minpq = min(p,q)
            value = np.zeros(minpq)
            for i in range(minpq):
                value[i] = np.sum((rho**2)[i:minpq])
            return value
        
        # RaoF_stat
        def RaoF_stat(rho,N,p,q):
            minpq = min(p,q)
            wilkslambda = WilksLambda(rho,p,q)
            RaoF, df1, df2 = np.zeros(minpq),np.zeros(minpq),np.zeros(minpq)
            de = N - 1.5 - (p+q)/2
            for i in range(minpq):
                k = i
                df1[i] = (p-k)*(q-k)
                if (p-k)==1 or (q-k)==1:
                    nu = 1
                else:
                    nu = np.sqrt((df1[i]**2-4)/((p-k)**2+(q-k)**2-5))
                df2[i] = de*nu - df1[i]/2 + 1
                nu1 = 1/nu
                w = wilkslambda[i]**nu1
                RaoF[i] = (df2[i]/df1[i])*((1-w)/w)
            return {"stat" : wilkslambda,"approx" : RaoF,"df1" : df1,"df2" : df2}
        
        # Hotelling statistics
        def Hotelling_stat(rho,N,p,q):
            minpq = min(p,q)
            hotellinglawleytrace = HotellingLawleyTrace(rho,p,q)
            hotelling, df1, df2 = np.zeros(minpq),np.zeros(minpq),np.zeros(minpq)
            for i in range(minpq):
                k = i
                df1[i] = (p-k)*(q-k)
                df2[i] = minpq*(N  - 2 - p - q + 2*k) + 2
                hotelling[i] = hotellinglawleytrace[i]/minpq/df1[i]*df2[i]
            return {"stat" : hotellinglawleytrace,"approx" : hotelling,"df1" : df1, "df2" : df2}

        # Pillai statistics
        def Pillai_stat(rho,N,p,q):
            minpq = min(p,q)
            pillaibartletttrace = PillaiBartlettTrace(rho,p,q)
            pillai, df1, df2 = np.zeros(minpq),np.zeros(minpq),np.zeros(minpq)
            for i in range(minpq):
                k = i
                df1[i]  = (p - k)*(q - k)
                df2[i]  = minpq*(N - 1 + minpq - p - q + 2*k) 
                pillai[i] = pillaibartletttrace[i] / df1[i] * df2[i] / ( minpq - pillaibartletttrace[i] )
            return {"stat" : pillaibartletttrace,"approx" : pillai,"df1" : df1, "df2" : df2}

        # Roy
        def p_Roy(rho,N,p,q):
            stat = rho[0]**2
            df1 = q
            df2 = N - 1 - q
            approx = stat/df1 * df2 /(1 - stat)
            p_value = 1 - sp.stats.f.cdf(approx,df1,df2)
            return {"id" : "Roy" , "stat" : stat,"approx" : approx,"df1" : df1, "df2" : df2,"p.value" : p_value}
        
        def p_asym(rho,N,p,q,tstat="Wilks"):
            minpq = min(p,q)
            if len(rho) != minpq:
                raise TypeError("Function p.asym: Improper length of vector containing the canonical correlations")
            
            if tstat not in ["Wilks","Hotelling","Pillai","Roy"]:
                raise ValueError("'tstat' should be one of 'Wilks', 'Hotelling', 'Pillai' or 'Roy'")

            if tstat == "Wilks":
                out = RaoF_stat(rho,N,p,q)
                text = "\nWilks' Lambda, using F-approximation (Rao's F):\n"
            elif tstat == "Hotelling":
                out = Hotelling_stat(rho,N,p,q)
                text = "\nHotelling-Lawley Trace, using F-approximation:\n"
            elif tstat == "Pillai":
                out = Pillai_stat(rho,N,p,q)
                text = "\nPillai-Bartlett Trace, using F-approximation:\n"
            elif tstat == "Roy":
                minpq = 1
                out = p_Roy(rho, N, p,q)
                text = "\nRoy's Largest Root, using F-approximation:\n"
            
            ### Extract data
            stat, approx, df1, df2 = out["stat"], out["approx"], out["df1"], out["df2"]

            if tstat != "Roy":
                p_value = np.zeros(minpq)
                for i in range(minpq):
                    p_value[i] = 1 - sp.stats.f.cdf(approx[i],df1[i], df2[i])
            else:
                p_value = 1 - sp.stats.f.cdf(approx,df1, df2)
            
            tab = np.c_[stat,approx,df1,df2,p_value]
            rn = [str(k+1) + " to " + str(minpq) + ": " for k in range(minpq)]
            tab = pd.DataFrame(tab,columns=["stat","approx","df1","df2","p.value"],index=rn)
            return {"text" : text,"stat" : tab}

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if vars is None and vars_with is None:
            raise ValueError("Either 'vars' or 'vars_with should be assigned")
        
        #### Check if vars is not None
        if vars is not None:
            if (isinstance(vars,int) or isinstance(vars,float)):
                vars_idx = [int(vars)]
            elif ((isinstance(vars,list) or isinstance(vars,tuple))  and len(vars)>=1):
                vars_idx = [int(x) for x in vars]

        # Check if vars_with is not None 
        if vars_with is not None:
            if (isinstance(vars_with,int) or isinstance(vars_with,float)):
                vars_with_idx = [int(vars_with)]
            elif ((isinstance(vars_with,list) or isinstance(vars_with,tuple))  and len(vars_with)>=1):
                vars_with_idx = [int(x) for x in vars_with]

        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Check if all columns are numerics
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise TypeError("All columns in X must be numeric")
        
        # Check if missing values
        X = recodecont(X)["Xcod"]
        
        # Store data
        Xtot = X.copy()
        
        if vars is None and vars_with is not None:
            X2 = X.iloc[:,vars_with_idx]
            X1 = X.drop(columns=X2.columns)
        elif vars is not None and vars_with is None:
            X1 = X.iloc[:,vars_idx]
            X2 = X.drop(columns=X1.columns)
        else:
            X1 = X.iloc[:,vars_idx]
            X2 = X.iloc[:,vars_with_idx]
        
        n_rows, n_colsx = X1.shape
        n_colsy = X2.shape[1]
        
        # Rank remark
        if n_colsx > n_rows:
            raise TypeError("The number of rows must be less than or equal to the number of columns in X")
        if n_colsy > n_rows:
            raise TypeError("The number of rows must be less than or equal to the number of columns in Y")
        if n_colsx > n_colsy: 
            raise TypeError("The number of columns in X must be less than or equal to the number of columns in Y")
        
        # Set number of componenets
        if n_components is None:
            ncp = n_colsx
        elif not isinstance(n_components, int):
            raise TypeError("'n_components' must be an integer")
        elif n_components <= 0:
            raise TypeError("'n_components' msut be a positive integer")
        else:
            ncp = min(n_components, n_colsx)
        
        self.call_ = {"Xtot" : Xtot,
                      "X" : X1,
                      "Y" : X2,
                      "n_components" : ncp}

        # Informations
        infos = pd.DataFrame({"Name" : ["Variables VAR", "Variables WITH", "Observations"],
                              "Value" : [X1.shape[1],X2.shape[1],X1.shape[0]]})
        
        # Statistics
        xstats, ystats = X1.describe().T, X2.describe().T
        stats = pd.concat((xstats,ystats),axis=0)

        self.statistics_ = {"infos" : infos, "stats" : stats}

        ######################################### Covariance Matrix ############################################
        # Unbiaised covariance
        xcov, ycov = X1.cov(ddof=0), X2.cov(ddof=0)
        xycov = pd.DataFrame(np.cov(X1,X2,rowvar=False,ddof=0)[:X1.shape[1],X1.shape[1]:],index=X1.columns,columns=X2.columns)

        # Biaised Covariance Matrix
        xcovb, ycovb = (n_rows/(n_rows - 1))*xcov, (n_rows/(n_rows - 1))*ycov
        xycovb = (n_rows/(n_rows - 1))*xycov
        self.cov_ = {"xcov" : xcov, "ycov" : ycov, "xycov" : xycov, "xcovb" : xcovb, "ycovb" : ycovb,"xycovb": xycovb}

        ########################################## Pearson Correlation ########################################
        xcorr, ycorr = X1.corr(method="pearson"), X2.corr(method="pearson")
        xycorr = pd.DataFrame(np.corrcoef(X1,X2,rowvar=False)[:X1.shape[1],X1.shape[1]:],index = X1.columns, columns=X2.columns)
        self.corr_ = {"xcorr" : xcorr, "ycorr" : ycorr, "xycorr" : xycorr}

        #########
        # Squared canonical correlation and standardized canonical coefficients of Y
        mat1  = np.dot(np.dot(np.dot(np.linalg.inv(ycorr) ,xycorr.T),np.linalg.inv(xcorr)),xycorr)
        eigen_value1, eigen_vector1 = np.linalg.eig(mat1)
        squared_cancorr = eigen_value1[:ncp]
        ystandardized_cancoef = eigen_vector1[:,:ncp]

        # Squared canonical correlation and raws canonical coefficients of X
        mat2 = np.dot(np.dot(np.dot(np.linalg.inv(xcorr),xycorr),np.linalg.inv(ycorr)),xycorr.T)
        _, eigen_vector2 = np.linalg.eig(mat2)
        xstandardized_cancoef = eigen_vector2[:,:n_components]
        self.standardized_cancoef_ =  {"x_standardized" : xstandardized_cancoef, "y_standardized" : ystandardized_cancoef}

        ##### Apply the standardize
        xmeans, ymeans = X1.mean(axis=0).values.reshape(1,-1), X2.mean(axis=0).values.reshape(1,-1)
        if standardize:
            xstd, ystd = X1.std(axis=0,ddof=0).values.reshape(1,-1), X2.std(axis=0,ddof=0).values.reshape(1,-1)
        else:
            xstd, ystd = 1.0, 1.0
        
        Z1, Z2 = (X1 - xmeans)/xstd, (X2 - ymeans)/ystd

        T = np.dot(Z2.T, Z2)
        H = np.dot(np.dot(np.dot(np.dot(Z2.T,Z1),np.linalg.inv(np.dot(Z1.T,Z1))),Z1.T),Z2)
        E = T - H
        eigen_value3, eigen_vector3 = np.linalg.eig(np.dot(np.linalg.inv(E),H))
        eigen_values = eigen_value3[:n_components]
        eigen_vectors = eigen_vector3[:,:n_components]

        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)
        # store in 
        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns=["eigenvalue","difference","proportion","cumulative"],index = ["Dim."+str(x+1) for x in range(eig.shape[0])])

        ##########
        # Apply global function - Canonical correlation analysis function
        cancorr = rcc(X=X1,Y=X2,standardize=standardize)
        self.can_corr_ = pd.DataFrame(np.c_[cancorr["cor"],squared_cancorr],columns=["canonical correlation","squared canonical correlation"])

        ##### Canonical coefficients
        self.coef_ = {"xcoef" : cancorr["xcoef"],"ycoef" : cancorr["ycoef"]}

        ##### Separate into Individuals and variables scores
        self.ind_ = {k: v for k, v in cancorr["scores"].items() if k in ["xscores","yscores"]}
        self.var_ = {k: v for k, v in cancorr["scores"].items() if k not in ["xscores","yscores"]}

        #####################################  canonical statistics tests
        wilks = p_asym(cancorr["cor"],n_rows,n_colsx,n_colsy,tstat="Wilks")
        hotelling = p_asym(cancorr["cor"],n_rows,n_colsx,n_colsy,tstat="Hotelling")
        pillai = p_asym(cancorr["cor"],n_rows,n_colsx,n_colsy,tstat="Pillai")
        roy = p_asym(cancorr["cor"],n_rows,n_colsx,n_colsy,tstat="Roy")
        self.tstat_ = {"wilks" : wilks,"hotelling" : hotelling,"pillai" : pillai,"roy" : roy}
        
        self.model_ = "cca"