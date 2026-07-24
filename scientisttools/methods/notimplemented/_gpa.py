
from sklearn.base import BaseEstimator, TransformerMixin

class GPA(BaseEstimator,TransformerMixin):
    """
    Generalized Procustes Analysis (GPA)

    Performs Generalised Procrustes Analysis (GPA)

    Parameters
    ----------


    See
    see https://www.sensorysociety.org/knowledge/sspwiki/Pages/Generalized%20Procrustes%20Analysis.aspx
    see https://www.xlstat.com/solutions/features/generalized-procrustes-analysis-gpa

    """
    def __init__(
            self,group=None,group_type=None, name_group=None, tol=10**(-10), max_iter=200
    ):
        self.group = group
        self.group_type = group_type
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,X,y=None):
        pass