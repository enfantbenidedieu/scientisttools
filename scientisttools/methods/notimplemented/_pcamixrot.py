

class PCAmixrot:
    """
    Varimax rotation in Principal Component Analysis of Mixed Data (PCAmixrot)
    
    """
    def __init__(
            self, ncp=2, normalize=True, max_iter=1000, tol=1e-5
    ):
        self.ncp = ncp
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self,obj):
        return NotImplementedError("Not yet implemented")