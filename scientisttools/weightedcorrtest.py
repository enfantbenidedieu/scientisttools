
import numpy as np
import scipy as sp
from .weightedcorrcoef import weightedcorrcoef

def weightedcorrtest(x,y,weights):
    """
    
    
    
    """
    statistic = weightedcorrcoef(x=x,y=y,w=weights)[0,1]
    t_stat = statistic*np.sqrt(((len(x)-2)/(1- statistic**2)))
    dof = len(x) - 2
    pvalue = 2*(1 - sp.stats.t.cdf(np.abs(t_stat),dof))
    return {"statistic" : statistic,"dof" : dof ,"pvalue" : pvalue}
