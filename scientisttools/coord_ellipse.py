# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as st

def coord_ellipse(X,bary,level,npoints):
    def ellipse(X,g,bary,level,npoints):
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
        d, a = np.arccos(r), np.linspace(0,2*np.pi,npoints)
        t = np.sqrt(st.chi2.ppf(level,2))
        val1 = t*sigma[0]*np.cos(a + (d/2)) + mu.iloc[0]
        val2 = t*sigma[1]*np.cos(a + (d/2)) + mu.iloc[1]
        res = pd.DataFrame({f"{quali}" : [g]*npoints})
        values = pd.DataFrame(np.c_[val1,val2],columns=dat.columns.tolist())
        res = pd.concat((res,values),axis=1)
        return res
    u = np.sort(np.unique(X.iloc[:,0]))
    res = pd.concat((ellipse(X,g,bary,level,npoints) for g in u),axis=0,ignore_index=True)
    return res