# -*- coding: utf-8 -*-
import plotnine3d as pn3d

def text3d_label(texttype,**kwargs):
    """
    Function to choose between `geom_text_3d` and `geom_label_3d`
    ------------------------------------------------------------

    Parameters
    ----------
    text_type : {"text", "label"}, default = "text"

    **kwargs : geom parameters

    return
    ------

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if texttype not in ["text","label"]:
        raise TypeError("'texttype' should be one of 'text', 'label'")
    
    if texttype == "text":
        return pn3d.geom_text_3d(**kwargs)
    elif texttype == "label":
        return pn3d.geom_label_3d(**kwargs)