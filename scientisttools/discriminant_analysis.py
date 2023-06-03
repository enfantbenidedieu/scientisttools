# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
import pandas as pd
from plydata import *
from functools import reduce
from scientisttools.utils import eta2,paste
from scipy.spatial.distance import pdist,squareform
from statsmodels.multivariate.manova import MANOVA
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

##########################################################################################
#                       CANONICAL DISCRIMINANT ANALYSIS
##########################################################################################

class CANDISC(BaseEstimator,TransformerMixin):
    """Canonical Discriminant Analysis

    Performs a canonical discriminant analysis, computes squared 
    Mahalanobis distances between class means, and performs both 
    univariate and multivariate one-way analyses of variance

    Parameters
    ----------
    n_components : 

    target : 

    row_labels :

    row_labels : array of strings or None, default = None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
     
    col_labels : array of strings or None, default = None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.


    Return
    ------
    
    """
    def __init__(self,n_components=None,target=list[str],row_labels=None,features_labels=None):
        self.n_components = n_components
        self.target = target
        self.row_labels = row_labels
        self.features_labels = features_labels

    def fit(self,X,y=None):
        """Fit the Canonical Discriminant Analysis model

        Parameters
        ----------
        X : DataFrame,
            Training Data
        
        Returns:
        --------
        self : object
            Fitted estimator
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        self.target_ = self.target
        if self.target_ is None:
            raise ValueError("Error :'target' must be assigned.")

        self._computed_stats(X=X)
        
        return self
        
    def _mahalanobis_distances(self,X,y):
        """
        Compute the Mahalanobis squared distance

        Parameters
        ----------
        X : pd.DataFrame.
            The
        
        y : pd.Series

        
        """

        # Matrice de covariance intra - classe utilisée par Mahalanobis
        W = (self.n_samples_/(self.n_samples_ - self.n_classes_))*X

        # Invesion
        invW = pd.DataFrame(np.linalg.inv(W),index=W.index,columns=W.columns)

        disto = pd.DataFrame(np.zeros((self.n_classes_,self.n_classes_)),index=self.classes_,columns=self.classes_)
        for i in np.arange(0,self.n_classes_-1):
            for j in np.arange(i+1,self.n_classes_):
                # Ecart entre les 2 vecteurs moyennes
                ecart = y.iloc[i,:] - y.iloc[j,:]
                # Distance de Mahalanobis
                disto.iloc[i,j] = np.dot(np.dot(ecart,invW),np.transpose(ecart))
                disto.iloc[j,i] = disto.iloc[i,j]
        
        self.squared_mdist_ = disto
    
    @staticmethod
    def anova_table(aov):
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        return aov
    
    @staticmethod
    def univariate_test_statistics(x,res):
        """
        Compute univariate Test Statistics

        Parameters
        ----------
        x : float. 
            Total Standard Deviation
        res : OLSResults.
            Results class for for an OLS model.
        
        Return
        -------
        univariate test statistics

        """

        return np.array([x,res.rsquared,res.rsquared/(1-res.rsquared),res.fvalue,res.f_pvalue])
    
    def _global_performance(self,lw):
        """Compute Global statistic - Wilks' Lambda - Bartlett statistic and Rao

        Parameters:
        ----------
        lw : float
            Wilks lambda's value
        
        Returns:
        --------

        
        """

        ## Bartlett Test
        # Statistique B de Bartlett
        B = -(self.n_samples_ - 1 - ((self.n_features_ + self.n_classes_)/2))*np.log(lw)
        # Degré de liberté
        ddl = self.n_features_*(self.n_classes_ - 1)
        
        ## RAO test
        # ddl numérateur
        ddlnum = self.n_features_*(self.n_classes_ - 1)
        # Valeur intermédiaire pour calcul du ddl dénominateur
        temp = self.n_features_**2 + (self.n_classes_ - 1)**2 - 5
        temp = np.where(temp>0,np.sqrt(((self.n_features_**2)*((self.n_classes_ - 1)**2)-4)/temp),1)
        # ddl dénominateur
        ddldenom = (2*self.n_samples_ - self.n_features_ - self.n_classes_ - 2)/2*temp - (ddlnum - 2)/2
        # statistic de test
        frao = ((1-(lw**(1/temp)))/(lw**(1/temp)))*(ddldenom/ddlnum)
        # Resultat
        res = pd.DataFrame({"Stat" : ["Wilks' Lambda",f"Bartlett -- C({int(ddl)})",f"Rao -- F({int(ddlnum)},{int(ddldenom)})"],
                            "Value" : [lw,B,frao],
                            "p-value": [np.nan,1 - st.chi2.cdf(B,ddl),1 - st.f.cdf(frao,ddlnum,ddldenom)]})
        return res

    
    def _univariate_test(self,eigen):
        # Statistique de test
        if isinstance(eigen,float) or isinstance(eigen,int):
            q = 1
        else:
            q = len(eigen)
        LQ = np.prod([(1-i) for i in eigen])
        # r
        r = self.n_samples_ - 1 - (self.n_features_+self.n_classes_)/2
        # t
        if ((self.n_features_ - self.n_classes_ + q + 1)**2 + q**2 - 5) > 0 : 
            t = np.sqrt((q**2*(self.n_features_-self.n_classes_+q+1)**2-4)/((self.n_features_ - self.n_classes_ + q + 1)**2 + q**2 - 5)) 
        else: 
            t = 1
        # u
        u = (q*(self.n_features_-self.n_classes_+q+1)-2)/4
        # F de RAO
        FRAO = ((1 - LQ**(1/t))/(LQ**(1/t))) * ((r*t - 2*u)/(q*(self.n_features_ - self.n_classes_ + q + 1)))
        # ddl1 
        ddl1 = q*(self.n_features_ - self.n_classes_ + q + 1)
        # ddl2
        ddl2 = r*t - 2*u
        res_rao = pd.DataFrame({"statistic":FRAO,"DDL num." : ddl1, "DDL den." : ddl2,
                      "Pr>F": 1 - st.f.cdf(FRAO,ddl1,ddl2)},index=["test"])
        return res_rao
    
    @staticmethod
    def betweencorrcoef(g_k,z_k,name,lda,weights):
        def m(x, w):
            return np.average(x,weights=w)
        def cov(x, y, w):
            return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)
        def corr(x, y, w):
            return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))
        return corr(g_k[name], z_k[lda],weights)


    def _computed_stats(self,X):
        """
        
        
        """
        # Continuous variables
        x = X.drop(columns=self.target_)
        # Qualitative variables - target
        y = X[self.target_]

        # Features columns
        self.features_labels_ = self.features_labels
        if self.features_labels_ is None:
            self.features_labels_ = x.columns
        # Update x
        x = x[self.features_labels_]
        # New data
        X = pd.concat([x,y],axis=1)

        # Compute mean and standard deviation
        mean_std_var = x.agg(func = ["mean","std"])

        # categories
        self.classes_ = np.unique(y)

        # Number of rows and continuous variables
        self.n_samples_, self.n_features_ = x.shape

        # Number of groups
        self.n_classes_ = len(self.classes_)

        # Set row labels
        self.row_labels_ = self.row_labels
        if self.row_labels_ is None:
            self.row_labels = ["row."+str(i+1) for i in np.arange(0,self.n_samples_)]

        # Number of components
        self.n_components_ = self.n_components
        if ((self.n_components_ is None) or (self.n_components_ > min(self.n_classes_-1,self.n_samples_))):
            self.n_components_ = min(self.n_classes_-1,self.n_features_)

        # Compute univariate ANOVA
        univariate_test = pd.DataFrame(np.zeros((self.n_features_,5)),index=self.features_labels_,
                                       columns=["Std. Dev.","R-squared","Rsq/(1-Rsq)","F-statistic","Prob (F-statistic)"])
        univariate_anova = dict()
        for lab in self.features_labels_:
            model = smf.ols(formula="{}~C({})".format(lab,"+".join(self.target_)), data=X).fit()
            univariate_test.loc[lab,:] = self.univariate_test_statistics(mean_std_var.loc["std",lab],model)
            univariate_anova[lab] = self.anova_table(sm.stats.anova_lm(model, typ=2))

        # Compute MULTIVARIATE ANOVA - MANOVA Test
        manova = MANOVA.from_formula(formula="{}~{}".format(paste(self.features_labels_,collapse="+"),"+".join(self.target_)), data=X).mv_test(skip_intercept_test=True)

        # Tukey Honestly significant difference - univariate
        tukey_test = dict()
        for name in self.features_labels_:
            comp = mc.MultiComparison(x[name],y[self.target_[0]])
            post_hoc_res = comp.tukeyhsd()
            tukey_test[name] = post_hoc_res.summary()

        # Bonferroni correction
        bonf_test = dict()
        for name in self.features_labels_:
            comp = mc.MultiComparison(x[name],y[self.target_[0]])
            tbl, a1, a2 = comp.allpairtest(st.ttest_ind, method= "bonf")
            bonf_test[name] = tbl
        
        # Sidak Correction
        sidak_test = dict()
        for name in self.features_labels_:
            comp = mc.MultiComparison(x[name],y[self.target_[0]])
            tbl, a1, a2 = comp.allpairtest(st.ttest_ind, method= "sidak")
            sidak_test[name] = tbl

        # Summary information
        summary_infos = pd.DataFrame({
            "Total Sample Size" : self.n_samples_,
            "Variables" : self.n_features_,
            "Classes" : self.n_classes_,
            "DF Total" : self.n_samples_ - 1,
            "DF Within Classes" : self.n_samples_ - self.n_classes_,
            "DF Between Classes" : self.n_classes_-1
        },index=["value"]).T

         # Rapport de correlation - Correlation ration
        eta2_res = dict()
        for name in self.features_labels_:
            eta2_res[name] = eta2(y,x[name])
        eta2_res = pd.DataFrame(eta2_res).T

        # Number of eflemnt in each group
        I_k = y.value_counts(normalize=False)

        # Initial prior - proportion of each element
        p_k = y.value_counts(normalize=True)

        # Class level information
        class_level_information = pd.concat([I_k,p_k],axis=1,ignore_index=False)
        class_level_information.columns = ["Frequency","Proportion"]
        

        # Mean by group
        g_k = X.groupby(self.target_).mean()

        # Covariance totale
        V = x.cov(ddof=0)

        # Variance - Covariance par groupe
        V_k = X.groupby(self.target_).cov(ddof=1)

        # Matrice de variance covariance intra - classe
        W = list(map(lambda k : (I_k[k]-1)*V_k.loc[k],self.classes_))
        W = (1/self.n_samples_)*reduce(lambda i,j : i + j, W)

        # Matrice de Variance Covariance inter - classe, obtenue par différence
        B = V - W
        print(B)

        # Squared Mahalanobis distances between class means
        self._mahalanobis_distances(X=W,y=g_k)

        # First Matrix - French approach
        M1 = B.dot(np.linalg.inv(V)).T
        eigen1, _ = np.linalg.eig(M1)
        eigen_values1 = eigen1[:self.n_components_]

        # Second Matrix - Anglosaxonne approach
        M2 = B.dot(np.linalg.inv(W)).T
        eigen2, _ = np.linalg.eig(M2)
        eigen_values2 = eigen2[:self.n_components_]

        # Eigenvalue informations
        eigen_values = eigen_values2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # Correction approach
        xmean = mean_std_var.loc["mean",:]
        C = pd.concat(list(map(lambda k : np.sqrt(p_k.loc[k,])*(g_k.loc[k,]-xmean),self.classes_)),axis=1)
        C.columns = self.classes_
        
        M3 = np.dot(np.dot(C.T,np.linalg.inv(V)),C)
        eigen3, vector3 = np.linalg.eig(M3)
        # Reverse sort eigenvalues - Eigenvalues aren't in best order
        new_eigen = np.array(sorted(eigen3,reverse=True))
        idx = [list(eigen3).index(x) for x in new_eigen]

        # New eigen vectors
        eigen = new_eigen[:self.n_components_]
        vector = vector3[:,idx][:,:self.n_components_]

        # vecteur beta
        beta_l = np.dot(np.dot(np.linalg.inv(V),C),vector)

        # Coefficients 
        u_l = np.apply_along_axis(func1d=lambda x : x*np.sqrt((self.n_samples_-self.n_classes_)/(self.n_samples_*eigen*(1-eigen))),arr=beta_l,axis=1)
        # Intercept
        u_l0 = -np.dot(np.transpose(u_l),xmean)

        # Coordonnées des individus
        row_coord = np.apply_along_axis(arr=np.dot(x,u_l),func1d=lambda x : x + u_l0,axis=1)
        # Coord collumns
        self.dim_index_ = ["LD"+str(x+1) for x in np.arange(0,self.n_components_)]

        # Class Means on Canonical Variables
        gmean_coord = (pd.concat([pd.DataFrame(row_coord,columns=self.dim_index_,index=y.index),y],axis=1,ignore_index=False)
                         .groupby(self.target_)
                         .mean())

        # Coordonnées des centres de classes
        z_k = mapply(g_k.dot(u_l),lambda x : x + u_l0,axis=1,progressbar=False)
        z_k.columns = self.dim_index_

        # Distance entre barycentre
        disto = pd.DataFrame(squareform(pdist(z_k,metric="sqeuclidean")),columns=self.classes_,index=self.classes_)

        # Lambda de Wilks
        lw = np.linalg.det(W)/np.linalg.det(V)
        
        # global performance
        self.global_performance_ = self._global_performance(lw=lw)

        # Test sur un ensemble de facteurs
        res_rao = pd.DataFrame(np.zeros((self.n_components_,4)),columns=["statistic","DDL num.","DDL den.","Pr>F"]).astype("float")
        for i in np.arange(0,self.n_components_):
            res_rao.iloc[-i,:] = self._univariate_test(eigen=eigen_values1[-(i+1):])
        res_rao = res_rao.sort_index(ascending=False).reset_index(drop=True)

        self.likelihood_test_ = res_rao

        ##
        # Corrélation totale
        tcorr = np.transpose(np.corrcoef(x=row_coord,y=x,rowvar=False)[:self.n_components_,self.n_components_:])
        tcorr = pd.DataFrame(tcorr,columns=self.dim_index_,index=self.features_labels_)

        # Within correlation
        z1 = row_coord - gmean_coord.loc[y[self.target_[0]],:].values
        g_k_long = g_k.loc[y[self.target_[0]],:]
        z2 = pd.DataFrame(x.values - g_k_long.values,index=g_k_long.index,columns=self.features_labels_)
        wcorr = np.transpose(np.corrcoef(x=z1,y=z2,rowvar=False)[:self.n_components_,self.n_components_:])
        wcorr = pd.DataFrame(wcorr,columns=self.dim_index_,index=self.features_labels_)

        # Between correlation
        bcorr = pd.DataFrame(np.zeros((self.n_features_,self.n_components_)),index=self.features_labels_,columns=self.dim_index_)
        for name in x.columns:
            for name2 in bcorr.columns:
                bcorr.loc[name,name2]=self.betweencorrcoef(g_k,z_k,name,name2,p_k.values)
        

        # Fonction de classement
        # Coefficients de la fonction de décision (w_kj, k=1,...,K; j=1,...,J)
        u_l_df = pd.DataFrame(u_l,index=self.features_labels_,columns=self.dim_index_)
        S_omega_k = pd.DataFrame(map(lambda k :
                            pd.DataFrame(map(lambda l : u_l_df.loc[:,l]*z_k.loc[k,l],self.dim_index_)).sum(axis=0),self.classes_), index = self.classes_).T
        
        # Constante de la fonction de décision
        S_omega_k0 = pd.DataFrame(
                    map(lambda k : (np.log(p_k.loc[k,])+sum(u_l0.T*z_k.loc[k,:])-
                0.5*sum(z_k.loc[k,:]**2)),self.classes_),index = self.classes_,
                columns = ["intercept"]).T

         # Store all informations
        self.eig_ = np.array([eigen_values[:self.n_components_],
                              difference[:self.n_components_],
                              proportion[:self.n_components_],
                              cumulative[:self.n_components_]])
        
        self.eigen_vectors_ = vector
        
        # Coefficients
        self.coef_ = u_l
        # Intercept
        self.intercept_ = u_l0
        self.row_coord_ = row_coord

        # Fonction de classement - coefficient
        self.score_coef_ = np.array(S_omega_k)
        self.score_intercept_ = np.array(S_omega_k0)

        # Store all information
        self.gmean_ = g_k                                                # Mean in each group
        self.tcov_ = V                                                   # Total covariance
        self.gcov_ = V_k
        self.wcov_ = W
        self.bcov_ = V - W
        self.correlation_ratio_ = eta2_res
        self.summary_information_ = summary_infos
        self.class_level_information_ = class_level_information
        self.univariate_test_statistis_ = univariate_test
        self.gmean_coord_ = gmean_coord

        self.anova_ = univariate_anova
        self.manova_ = manova
        self.tukey_ = tukey_test
        self.bonferroni_correction_ = bonf_test
        self.sidak_ = sidak_test
        self.gdisto_ = disto
        self.gcenter_ = z_k
        self.priors_ = p_k

        # Correlation
        self.tcorr_ = tcorr
        self.wcorr_ = wcorr
        self.bcorr_ = bcorr

        # Data
        self.data_ = X

        self.model_ = "candisc"
    
    def decision_function(self,X):

        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. 

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features)
            DataFrame of samples (test vectors).

        Returns
        -------
        C : DataFrame of shape (n_samples_,) or (n_samples_, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is (n_samples_,), giving the
            log likelihood ratio of the positive class.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        scores = X.dot(self.score_coef_).add(self.score_intercept_,axis="columns")
        return scores


    def transform(self,X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            Input data
        
        Returns:
        --------
        X_new : DataFrame of shape (n_samples_, n_components_)
            Transformed data.
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        predict = np.apply_along_axis(arr=np.dot(X,self.coef_),func1d=lambda x : x + self.intercept_,axis=1)
        return pd.DataFrame(predict,index=X.index,columns=self.dim_index_)

    def predict(self,X):
        """Predict class labels for samples in X

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            The data matrix for which we want to get the predictions.
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples)
            Vectors containing the class labels for each sample
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        predict_proba = self.predict_proba(X)
        predict = np.unique(self.classes_)[np.argmax(predict_proba.values,axis=1)]
        predict = pd.DataFrame(predict,columns=["predict"],index=X.index)
        return predict


    def predict_proba(self,X):
        """Estimate probability

        Parameters
        ----------
        X : DataFrame of shape (n_samples_,n_features_)
            Input data.
        
        Returns:
        --------
        C : DataFrame of shape (n_samples_,n_classes_)
            Estimated probabilities.
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        # Decision
        scores = self.decision_function(X)
        # Probabilité d'appartenance - transformation 
        C = mapply(mapply(scores,lambda x : np.exp(x),axis=0,progressbar=False),lambda x : x/np.sum(x),axis=1,progressbar=False)
        C.columns = self.classes_
        return C

    def fit_transform(self,X):
        """Fit to data, then transform it

        Fits transformer to `x` and returns a transformed version of X.

        Parameters:
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            Input samples
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_features_)
            Transformed data.
        
        
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        self.fit(X)

        return pd.DataFrame(self.row_coord_,index=X.index,columns=self.dim_index_)
    
    def score(self,X,y,sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    

#####################################################################################
#           LINEAR DISCRIMINANT ANALYSOS (LDA)
#####################################################################################


class DISCRIM(BaseEstimator,TransformerMixin):
    """Linear Discriminant Analysis

     Develops a discriminant criterion to classify each observation into groups
    
    """
    def __init__(self,features_labels=None,target=None):
        self.features_labels = features_labels
        self.target = target
    
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