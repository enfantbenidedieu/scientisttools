# -*- coding: utf-8 -*-
def get_mds(self) -> dict:
    """
    Extract the results for Multidimension Scaling - (MDS & CMDSCALE)
    -----------------------------------------------------------------

    self : an object of class MDS, CMDSCALE

    Returns
    -------
    a dictionary of informations

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ not in ["mds","cmdscale"]:
        raise ValueError("'self' must be an object of class MDS, CMDSCALE")
    return self.result_