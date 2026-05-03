

from sklearn.base import BaseEstimator, TransformerMixin

class GMCOA(BaseEstimator,TransformerMixin):
    """
    Generalized Multiple CO-inertia Analysis (GMCOA)

    Performns Generalized Multiple CO-inertia Analysis (GMCOA)
    
    Parameters
    ----------
    
    """
    def __init__(
            self, ncp=5
    ):
        self.ncp = ncp

    def fit(self,X,y=None):
        """
        
        """
        