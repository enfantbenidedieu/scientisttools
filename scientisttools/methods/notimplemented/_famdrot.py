

class FAMDrot:
    """
    Varimax rotation in Factor Analaysis of Mixed Data (FAMDrot)
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