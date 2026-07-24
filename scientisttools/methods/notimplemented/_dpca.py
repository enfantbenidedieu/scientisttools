

from sklearn.base import BaseEstimator,TransformerMixin

class DPCA(BaseEstimator,TransformerMixin):
    """
    Double Principal Component Analysis (DPCA)

    https://www.researchgate.net/publication/353203033_Dual_PCA

    https://www.jmlr.org/papers/volume19/17-436/17-436.pdf
    
    """