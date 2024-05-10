# Canonical Correspondence Analysis (CCPA)
# https://uw.pressbooks.pub/appliedmultivariatestatistics/chapter/ca-dca-and-cca/
# https://en.wikipedia.org/wiki/Canonical_correspondence_analysis
# https://www.xlstat.com/fr/solutions/fonctionnalites/analyse-canonique-des-correspondances-acc
# https://gist.github.com/perrygeo/7572735
# https://rdrr.io/rforge/ade4/man/cca.html
# https://rdrr.io/cran/MultBiplotR/man/CCA.html
# https://search.r-project.org/CRAN/refmans/MultBiplotR/html/CCA.html


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin


class CCPA(BaseEstimator,TransformerMixin):
    """
    Canonical Correspondence Analysis (CCPA)
    ----------------------------------------


    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass