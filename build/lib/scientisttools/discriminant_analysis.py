# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

##########################################################################################
#                       CANONICAL DISCRIMINANT ANALYSIS
##########################################################################################

class CDA(BaseEstimator,TransformerMixin):
    """Canonical Discriminant Analysis
    
    """
    def __init__(self,feature_columns=None,target_columns=None,priors=None,method ="FR"):
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.priors = priors
        self.method = method

    def fit(self,X,y):
        # Compute
        self.feature_ = X
        self.target_ = y

        if self.feature_columns is None:
            raise NotImplementedError("Error : This method is not implemented yet.")

        # Initialize 
        self.eig_ = None
        self.eigen_vectors = None
        self.total_variance = None
        self.between_variance = None
        self.within_variance = None

        if self.method == "FR":
            self._computed_fr(X,y)
        elif self.method == "GB":
            self._computed_gb(X,y)
        
        return self

    def _computed_fr(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def _computed_gb(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def _computed_stats(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def transform(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def predict(self,X):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def predict_proba(self,X):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def fit_transform(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def plot_boxplot(self,ax=None):
        if ax is None:
            ax =plt.gca()
        raise NotImplementedError("Error : This method is not implemented yet.")
    

#####################################################################################
#           LINEAR DISCRIMINANT ANALYSOS (LDA)
#####################################################################################


class LDA(BaseEstimator,TransformerMixin):
    """Linear Discriminant Analysis
    
    """
    def __init__(self,feature_columns,target_columns):
        self.feature_columns = feature_columns
        self.target_columns = target_columns
    
    def fit(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

######################################################################################
#           QUADRATIC DISCRIMINANT ANALYSIS (QDA)
#####################################################################################

class QDA(BaseEstimator,TransformerMixin):
    """Quadratic Discriminant Analysis
    
    """
    def __init__(self,features_columns, target_columns,priors=None):
        self.features_columns = features_columns
        self.target_columns = target_columns
        self.priors_ = priors
    
    def fit(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")

#####################################################################################
#           LOCAL FISHER DISCRIMINANT ANALYSIS (LFDA)
######################################################################################
 
class LFDA(BaseEstimator,TransformerMixin):
    """Local Fisher Discriminant Analysis
    
    """
    def __init__(self,feature_columns,target_columns):
        self.feature_columns = feature_columns
        self.target_columns = target_columns

    def fit(self,X,y):
        raise NotImplementedError("Error : This method is not implemented yet.")