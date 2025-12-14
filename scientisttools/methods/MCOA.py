

from sklearn.base import BaseEstimator, TransformerMixin

class MCOA(BaseEstimator,TransformerMixin):
    """
    Multiple CO-inertia Analysis (MCOA)
    -----------------------------------

    
    
    """
    def __init__(self,
                 n_components = 5):
        
        self.n_components = n_components


    def fit(self,X,y=None):
        pass