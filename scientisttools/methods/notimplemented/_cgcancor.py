# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, trace, linalg, real, insert, diff,nan, cumsum, c_, diag, sum,sqrt
from pandas import DataFrame, Series, concat, CategoricalDtype
from itertools import chain, repeat
from collections import OrderedDict, namedtuple
from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin

from ..functions.preprocessing import preprocessing
from ..functions.get_sup_label import get_sup_label
from ..functions.statistics import wmean, wstd, wcorr, func_groupby
from ..functions.func_eta2 import func_eta2
from ..functions.cov2corr import cov2corr
from ..functions.utils import check_is_bool, is_all_numeric_dtype, is_all_object_or_category_dtype
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix
from ..others._splitgroup import splitgroup, RVstats

class cGCANCOR(BaseEstimator,TransformerMixin):
    """
    Carroll's Generalized Canonical Correlation Analysis (GCANCORR)

    Performs generalized canonical correlation analysis (GCANCOR) in the sense of Carroll's
    
    
    """