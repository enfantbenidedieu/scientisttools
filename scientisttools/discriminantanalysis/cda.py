# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CanonicalDiscriminantAnalysis:
    def __init__(self,feature_columns=None,target_columns=None,priors=None,method ="FR"):
        self.feature_columns_ = feature_columns
        self.target_columns_ = target_columns
        self.priors_ = priors
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

    def fit_transform(self,X,y)

    def plot_boxplot(self,ax=None):
        if ax is None:
            ax =plt.gca()
        raise NotImplementedError("Error : This method is not implemented yet.")
