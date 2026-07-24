
from sklearn.base import BaseEstimator,TransformerMixin

class mbPLS(BaseEstimator,TransformerMixin):
    """
    Multiblock Partial Least Squares (mbPLS)

    Performs multiblock partial least squares (mbPLS)

    see https://cran.r-project.org/web/packages/ade4/ade4.pdf
    
    """
    def __init__(
            self,ncp=5, group=None, group_type=None, name_group=None, num_group_target=None, num_group_sup=None, weight_type="lambda1"
    ):
        self.ncp = ncp
        self.group = group
        self.group_type = group_type
        self.name_group = name_group
        self.num_group_target = num_group_target
        self.num_group_sup = num_group_sup
        self.weight_type = weight_type

    def fit(self,X,y=None):
        pass
        