
class QuadraticDiscriminantAnalysis:
    def __init__(self,features_columns, target_columns,priors=None):
        self.features_columns = features_columns
        self.target_columns = target_columns
        self.priors_ = priors
    
    def fit(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")