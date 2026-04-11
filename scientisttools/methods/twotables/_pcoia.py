

from sklearn.base import BaseEstimator, TransformerMixin

class PCOIA(BaseEstimator,TransformerMixin):
    """
    Procustean CO-inertia Analysis (PCOIA)

    Performns procustean co-inertia analysis (PCOA) in the sense of <https://sdray.github.io/publication/dray-2003-d/>_

    References
    ----------
    
    
    """
    def __init__(
            self, ncp=5, group=None, type_group=None, name_group=None, row_w=None, col_w=None, ind_sup=None,  tol = 1e-7
    ):
        self.ncp = ncp
        self.group = group
        self.type_group = type_group
        self.name_group = name_group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.tol = tol
    
    def fit(self,X,y=None):
        """
        Fit the model to ``X``

        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns)
            Training data, where ``n_rows`` in the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored

        Returns
        -------
        self : object
            Returns the instance itself
        """