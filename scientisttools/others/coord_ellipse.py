# -*- coding: utf-8 -*-
import numpy as np
from pandas import concat, DataFrame
import scipy.stats as st

def coord_ellipse(X,level_conf=0.95,npoints=100,bary=False):
    def ellipse(X,g,bary,level_conf,npoints):
        quali = X.columns.tolist()[0]
        dat = X.query(f"{quali} == @g").iloc[:,1:3]
        mu, covar = np.mean(dat,axis=0), dat.cov(ddof=1)
        if bary:
            covar = covar.div(dat.shape[0])
        r = covar.iloc[0,1]
        sigma = np.sqrt([covar.iloc[0,0],covar.iloc[1,1]])
        if sigma[0] > 0:
            r = r/sigma[0]
        if sigma[1] > 0:
            r = r/sigma[1]
        r = min(max(r,-1),1)
        d, a, t = np.arccos(r), np.linspace(0,2*np.pi,npoints), np.sqrt(st.chi2.ppf(level_conf,2))
        val1, val2= t*sigma[0]*np.cos(a + (d/2)) + mu.iloc[0], t*sigma[1]*np.cos(a + (d/2)) + mu.iloc[1]
        res, values = DataFrame({f"{quali}" : [g]*npoints}), DataFrame(np.c_[val1,val2],columns=dat.columns.tolist())
        return concat((res,values),axis=1)
    
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    u = np.sort(np.unique(X.iloc[:,0]))
    res = concat((ellipse(X,g,bary,level_conf,npoints) for g in u),axis=0,ignore_index=True)
    return res