# -*- coding: utf-8 -*-
import plotnine as pn

def text_label(texttype,**kwargs):
    """
    Function to choose between ``geom_text`` and ``geom_label``
    -----------------------------------------------------------

    Parameters
    ----------
    text_type : {"text", "label"}, default = "text"

    **kwargs : geom parameters

    return
    ------

    """
    if texttype not in ["text","label"]:
        raise TypeError("'texttype' should be one of 'text', 'label'")
    
    if texttype == "text":
        return pn.geom_text(**kwargs)
    elif texttype == "label":
        return pn.geom_label(**kwargs)