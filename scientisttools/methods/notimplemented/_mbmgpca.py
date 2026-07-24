
from sklearn.base import BaseEstimator, TransformerMixin

class mbmgPCA(BaseEstimator,TransformerMixin):
    """
    Multi-block and Multi-group Principal Component Analysis (mbmgPCA)
    
    
    
    
    
    
    
    """
    def __init__(
        self, ncp = 5, row_group = None, name_row_group = None, col_group = None, name_col_group = None, num_row_group_sup = None, num_col_group_sup = None
    ):
        self.ncp = ncp
        self.row_group = row_group
        self.name_row_group = name_row_group
        self.col_group = col_group
        self.name_col_group = name_col_group
        self.num_row_group_sup = num_row_group_sup
        self.num_col_group_sup = num_col_group_sup

    def fit(self,X,y=None):
        pass