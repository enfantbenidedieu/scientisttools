# -*- coding: utf-8 -*-
from pandas import concat
from sklearn.utils.validation import check_is_fitted

#intern functions
from ..functions.method_desc import method_desc, attr_desc
from ..functions.utils import is_namedtuple

def sprintf(obj, 
            **kwargs):
    """
    Print the analysis results
    
    Parameters
    ----------
    obj : class
        A fitted model.

    **kwargs :
        Additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

    Returns
    -------
    NoneType
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if is_namedtuple(obj):
        name, obj_attr = obj.__class__.__name__.replace("Result",""), list(obj._asdict().keys())
    else:
        check_is_fitted(obj)
        name, obj_attr = obj.__class__.__name__, [s for s in dir(obj) if s[0].isalpha() and s.endswith("_")]
    #et name
    res = concat((attr_desc(x) for x in obj_attr),axis=0,ignore_index=True).to_markdown(tablefmt="simple",index=False,**kwargs)
    
    text  = """
**Results of the {} ({})**\n
*The results are available in the following objects:\n
{}
""".format(method_desc(name=name),name,res)
    print(text)