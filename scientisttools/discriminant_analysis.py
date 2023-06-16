# -*- coding: utf-8 -*-
import scipy.stats as st
import numpy as np
import pandas as pd
from plydata import *
from functools import reduce, partial
from scientisttools.utils import eta2,paste
from scientisttools.decomposition import MCA, FAMD
from scientisttools.extractfactor import get_eig,get_mca_mod,get_mca_ind
from scipy.spatial.distance import pdist,squareform
from statsmodels.multivariate.manova import MANOVA
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score
from functools import partial


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

    row_labels : array of strings or None, default = None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
     
    features_labels : array of strings or None, default = None
        - If features_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If features_labels is None : labels are automatically computed for
          each column.


    Return
    ------
    
    """
    def __init__(self,
                 n_components=None,
                 target=list[str],
                 row_labels=None,
                 features_labels=None,
                 priors = None):
        self.n_components = n_components
        self.target = target
        self.row_labels = row_labels
        self.features_labels = features_labels
        self.priors = priors

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
        res_rao = pd.DataFrame({"statistic":FRAO,"DDL num." : ddl1, "DDL den." : ddl2,"Pr>F": 1 - st.f.cdf(FRAO,ddl1,ddl2)},index=["test"])
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
        if self.priors is None:
            p_k = y.value_counts(normalize=True)
        else:
            p_k = pd.Series(self.priors,index=self.classes_)

        # Class level information
        class_level_information = pd.concat([I_k,p_k],axis=1,ignore_index=False)
        class_level_information.columns = ["n(k)","p(k)"]
        

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

# Démarche probabiliste
class LINEARDISC(BaseEstimator,TransformerMixin):
    """Linear Discriminant Analysis

     Develops a discriminant criterion to classify each observation into groups

     Parameters:
     ----------

    distribution : {'multinomiale','homoscedastique'}
    priors : array-like of shape (n_classes,), default = None
        The class prior probabilities. By default, the class proportions are inferred from training data.
    
        
    Returns
    ------
    coef_ : DataFrame of shape (n_features,n_classes_)

    intercept_ : DataFrame of shape (1, n_classes)
    
    """
    def __init__(self,
                 features_labels=None,
                 target=list[str],
                 distribution = "homoscedastik",
                 row_labels = None,
                 priors = None):
        self.features_labels = features_labels
        self.target = target
        self.distribution = distribution
        self.row_labels = row_labels
        self.priors = priors
    
    def fit(self,X,y=None):
        """Fit the Linear Discriminant Analysis model

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
        
        self.data_ = X
        
        self.target_ = self.target
        if self.target_ is None:
            raise ValueError("Error :'target' must be assigned.")

        self._computed_stats(X=X)
        
        return self
    
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


        # Effectif par classe
        I_k = y.value_counts(normalize=False)

        # Initial prior - proportion of each element
        if self.priors is None:
            p_k = y.value_counts(normalize=True)
        else:
            p_k = pd.Series(self.priors,index=self.classes_)

        # Class level information
        class_level_information = pd.concat([I_k,p_k],axis=1,ignore_index=False)
        class_level_information.columns = ["n(k)","p(k)"]
        

        # Mean by group
        g_k = X.groupby(self.target_).mean()

        # Covariance totale
        V = x.cov(ddof=1)

        # Variance - Covariance par groupe - Matrices de covariance conditionnelles
        V_k = X.groupby(self.target_).cov(ddof=1)

        # Matrice de variance covariance intra - classe (corrivé en tenant compte du nombre de catégorie)
        W = list(map(lambda k : (I_k[k]-1)*V_k.loc[k],self.classes_))
        W = (1/(self.n_samples_-self.n_classes_))*reduce(lambda i,j : i + j, W)

        # Matrice de Variance Covariance inter - classe, obtenue par différence
        B = V - W
        
        # global performance
        self.global_performance_ = self._global_performance(V=V,W=W)

        # F - Exclusion - Statistical Evaluation
        self.statistical_evaluation_ = self._f_exclusion(V=V,W=W)

         # Squared Mahalanobis distances between class means
        self._mahalanobis_distances(X=W,y=g_k)

        if self.distribution == "homoscedastik":
            self._homoscedastik(W=W,gk=g_k,pk=p_k)

        # Store all information                                       # Mean in each group
        self.correlation_ratio_ = eta2_res
        self.summary_information_ = summary_infos
        self.class_level_information_ = class_level_information
        self.univariate_test_statistis_ = univariate_test
        self.anova_ = univariate_anova
        self.manova_ = manova
        self.tukey_ = tukey_test
        self.bonferroni_correction_ = bonf_test
        self.sidak_ = sidak_test
        self.summary_information_ = summary_infos
        self.priors_ = p_k
        self.tcov_ = V
        self.bcov_ = B
        self.wcov_ = W
        self.gcov_ = V_k
        self.mean_ = mean_std_var.loc["mean",:]
        self.std_ = mean_std_var.loc["std",:]
        self.gmean_ = g_k

        # Distance aux centres de classes
        self.generalized_distance(X=x)


        self.model_ = "lda"

    def _homoscedastik(self,W,gk,pk):
        ##########################################################################################
        #           Fonctions de classement linaire
        ############################################################################################

        # Inverse de la matrice de variance - covariance intra - class
        invW = np.linalg.inv(W)

        # Calcul des coeffcients des variabes - features
        coef = gk.dot(invW).rename_axis(None).T
        coef.index = self.features_labels_

        # Constantes
        u = np.log(pk)
        b = gk.dot(invW)
        b.columns = self.features_labels_
        b = (1/2)*b.dot(gk.T)

        intercept = pd.DataFrame(dict({ k : u.loc[k,]-b.loc[k,k] for k in self.classes_}),index=["Intercept"])

        self.coef_ = coef
        self.intercept_ = intercept

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
        W= X

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
    
    def generalized_distance(self,X):
        """Compute Generalized Distance
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        gen_dist = pd.DataFrame(columns=self.classes_,index=X.index).astype("float")
        for g in self.classes_:
            ecart =  X.sub(self.gmean_.loc[g].values,axis="columns")
            Y = np.dot(np.dot(ecart,np.linalg.inv(self.wcov_)),ecart.T)
            gen_dist.loc[:,g] = np.diag(Y) - 2*np.log(self.priors_.loc[g,])
        
        self.gen_dist_ = gen_dist
        
        return gen_dist

    def _global_performance(self,V,W):
        """Compute Global statistic - Wilks' Lambda - Bartlett statistic and Rao

        Parameters:
        ----------
        V : 
        
        Returns:
        --------

        
        """

        # Wilks' Lambda
        biased_V = ((self.n_samples_ - 1)/self.n_samples_)*V
        biased_W = ((self.n_samples_ - self.n_classes_)/self.n_samples_)*W
        
        # Lambda de Wilks
        lw = np.linalg.det(biased_W)/np.linalg.det(biased_V)

        ## Bartlett Test
        # Statistique B de Bartlett
        LB = -(self.n_samples_ - 1 - ((self.n_features_ + self.n_classes_)/2))*np.log(lw)
        # Degré de liberté
        ddl = self.n_features_*(self.n_classes_ - 1)
        
        ## RAO test
        # Valeur de A
        A = self.n_samples_ - self.n_classes_ - (1/2)*(self.n_features_ - self.n_classes_ + 2)
        
        # Valeur de B
        B = self.n_features_**2 + (self.n_classes_ - 1)**2 - 5
        if B > 0 :
            B = np.sqrt(((self.n_features_**2)*((self.n_classes_ - 1)**2)-4)/(B))
        else:
            B = 1

        # Valeur de C
        C = (1/2)*(self.n_features_*(self.n_classes_ - 1)-2)
        # statistic de test
        frao = ((1-(lw**(1/B)))/(lw**(1/B)))*((A*B-C)/(self.n_features_*(self.n_classes_ - 1)))

        # ddl numérateur
        ddlnum = self.n_features_*(self.n_classes_ - 1)
        # ddl dénominateur
        ddldenom = A*B-C
        
        # Resultat
        res = pd.DataFrame({"Stat" : ["Wilks' Lambda",f"Bartlett -- C({int(ddl)})",f"Rao -- F({int(ddlnum)},{int(ddldenom)})"],
                            "Value" : [lw,LB,frao],
                            "p-value": [np.nan,1 - st.chi2.cdf(LB,ddl),1 - st.f.cdf(frao,ddlnum,ddldenom)]})
        return res
    
    def _f_exclusion(self,V,W):

        """
        
        
        """

        # Wilks' Lambda
        biased_V = ((self.n_samples_ - 1)/self.n_samples_)*V
        biased_W = ((self.n_samples_ - self.n_classes_)/self.n_samples_)*W

        # Lambda de Wilks
        lw = np.linalg.det(biased_W)/np.linalg.det(biased_V)

        def fexclusion(j,W,V,n,K,lw):
            J = W.shape[1]
            # degrés de liberté - numérateur
            ddlsuppnum = K - 1
            # degrés de liberté dénominateur
            ddlsuppden = n - K - J + 1
            # Matrices intermédiaires numérateur
            tempnum = W.copy().values
            # Supprimer la référence de la variable à traiter
            tempnum = np.delete(tempnum, j, axis = 0)
            tempnum = np.delete(tempnum, j, axis = 1)
            # Même chose pour le numérateur
            tempden = V.values
            tempden = np.delete(tempden, j, axis = 0)
            tempden = np.delete(tempden, j, axis = 1)
            # Lambda de Wilk's sans la variable
            lwVar = np.linalg.det(tempnum)/np.linalg.det(tempden)
            # FValue
            fvalue = ddlsuppden/ddlsuppnum * (lwVar/lw-1)
            # Récupération des résultats
            return np.array([lwVar,lw/lwVar,fvalue,1 - st.f.cdf(fvalue, ddlsuppnum, ddlsuppden)])
        
        # Degré de liberté du numérateur
        ddl1 = self.n_classes_ - 1
        # Degré de liberté du dénominateur
        ddl2 = self.n_samples_ - self.n_classes_ - self.n_features_ +1 
        fextract = partial(fexclusion,W=biased_W,V=biased_V,n=self.n_samples_,K=self.n_classes_,lw=lw)
        res_contrib = pd.DataFrame(np.array(list(map(lambda j : fextract(j=j),np.arange(self.n_features_)))),
                                   columns=["Wilks L.","Partial L.",f"F{(ddl1,ddl2)}","p-value"],
                                   index= self.features_labels_)
            
        return res_contrib


    
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
        
        if self.distribution == "multinomiale":
            scores = pd.DataFrame(columns=self.classes_,index=X.index).astype("float")
            for g in self.classes_:
                ecart =  X.sub(self.gmean_.loc[g].values,axis="columns")
                Y = np.dot(np.dot(ecart,np.linalg.inv(self.gcov_.loc[g])),ecart.T)
                scores.loc[:,g] = np.log(self.priors_.loc[g,]) - (1/2)*np.log(np.linalg.det(self.gcov_.loc[g,:]))-(1/2)*np.diag(Y)
        elif self.distribution == "homoscedastik":
            scores = X.dot(self.coef_).add(self.intercept_.values,axis="columns")
            scores.index = X.index
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
        
        if self.distribution == "homoscedastik":
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





#######################################################################################################
#           Discriminant Qualitatives (DISQUAL)
#######################################################################################################

class DISQUAL(BaseEstimator,TransformerMixin):

    """Discriminant Qualitatives

    Performs discriminant analysis for categorical variables using multiple correspondence analysis (MCA)

    Parameters:
    ----------
    n_components : 
    target :
    features_labels :
    row_labels :
    priors : 

    Returns:
    -------
    eig_ :
    mod_ :
    row_ :
    lda_model_ :
    mca_model_ :
    statistical_evaluation_ : 
    projection_function_ : 
    lda_coef_ :
    lda_intercept_ :
    coef_ :
    intercept_ :



    
    
    """


    def __init__(self,
                 n_components = None,
                 target = list[str],
                 features_labels=None,
                 row_labels = None,
                 priors=None):
        
        self.n_components = n_components
        self.target = target
        self.features_labels = features_labels
        self.row_labels = row_labels
        self.priors = priors
    

    def fit(self,X,y=None):
        """Fit the Linear Discriminant Analysis with categories variables

        Parameters:
        -----------
        X : DataFrame of shape (n_samples, n_features+1)
            Training data
        
        y : None

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

        self._compute_stats(X=X)
        
        return self
    
    def _compute_stats(self,X):
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

        # categories
        self.classes_ = np.unique(y)

        # Number of observations
        self.n_samples_, self.n_features_ = x.shape

        # Number of groups
        self.n_classes_ = len(self.classes_)

        # Set row labels
        self.row_labels_ = self.row_labels
        if self.row_labels_ is None:
            self.row_labels = ["row."+str(i+1) for i in np.arange(0,self.n_samples_)]

        # Analyse des correspondances multiples (MCA)
        mca = MCA(n_components=self.n_components,row_labels=self.row_labels_,var_labels=self.features_labels_,
                  mod_labels=None,matrix_type="completed",benzecri=False,greenacre=False,
                  row_sup_labels=None,quali_sup_labels=None,quanti_sup_labels=None,graph=False,
                  figsize=(20,8)).fit(x)
        
        # Stockage des résultats de l'ACM
        mod = get_mca_mod(mca)
        row = get_mca_ind(mca)

        # Fonction de projection
        fproj = mapply(mod["coord"],lambda x : x/(self.n_features_*np.sqrt(mca.eig_[0])),axis=1,progressbar=False)

        # Données pour l'Analyse Discriminante Linéaire
        row_coord = row["coord"]
        row_coord.columns = list(["Z"+str(x+1) for x in np.arange(0,mca.n_components_)])
        new_X = pd.concat([y,row_coord],axis=1)

        # Analyse Discriminante Linéaire
        lda = LINEARDISC(target=self.target_,
                         distribution="homoscedastik",
                         features_labels=row_coord.columns,
                         row_labels=self.row_labels_,
                         priors=self.priors).fit(new_X)
        
        # LDA coefficients and intercepts
        lda_coef = lda.coef_
        lda_intercept = lda.intercept_ 

        # Coefficient du DISCQUAL
        coef = pd.DataFrame(np.dot(fproj,lda_coef),index=fproj.index,columns=lda_coef.columns)
        
        # Sortie de l'ACM
        self.projection_function_ = fproj
        self.n_components_ = mca.n_components_

        # Sortie de l'ADL
        self.lda_coef_ = lda_coef
        self.lda_intercept_ = lda_intercept
        self.lda_features_labels_ = list(["Z"+str(x+1) for x in np.arange(0,mca.n_components_)])
        
        # Informations du DISCQUAL
        self.statistical_evaluation_ = lda.statistical_evaluation_
        self.coef_ = coef
        self.intercept_ = lda_intercept

        # Stockage des deux modèles
        self.mca_model_ = mca
        self.lda_model_ = lda

        self.model_ = "disqual"
    
    def fit_transform(self,X):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and returns a transformed version of `X`.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features+1)
            Input samples.

        Returns
        -------
        X_new :  DataFrame of shape (n_samples, n_features_new)
            Transformed array.
        """

        self.fit(X)

        return self.mca_model_.row_coord_
    
    def transform(self,X):
        """Project data to maximize class separation

        Parameters:
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        X_new : DataFrame of shape (n_samples, n_components_)
        
        """

        coord = self.mca_model_.transform(X)
        coord = pd.DataFrame(coord[:,:self.n_components_],index=X.index,columns=self.lda_features_labels_)

        return coord
    
    def predict(self,X):
        """Predict class labels for samples in X

        Parameters:
        -----------
        X : DataFrame of shape (n_samples, n_features)
            The dataframe for which we want to get the predictions
        
        Returns:
        --------
        y_pred : DtaFrame of shape (n_samples, 1)
            DataFrame containing the class labels for each sample.
        
        """

        return self.lda_model_.predict(self.transform(X))
    
    def predict_proba(self,X):
        """Estimate probability

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data
        
        Returns:
        -------
        C : DataFrame of shape (n_samples, n_classes)
            Estimate probabilities
        
        """

        return self.lda_model_.predict_proba(self.transform(X))
    
    def score(self, X, y, sample_weight=None):

        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
        
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

##########################################################################################
#           Linear Discriminant Analysis under Continuous and Categorical variables (MIXDISC)
###########################################################################################



class MIXDISC(BaseEstimator,TransformerMixin):
    """Discriminant Analysis under Continuous and Categorical variables (MIXDISC)

    Performs linear discriminant analysis with both continuous and catogericals variables

    Parameters:
    -----------
    n_components
    target :
    quanti_features_labels :
    quali_features_labels:
    row_labels :
    prioirs:
    grid_search : 

    
    
    """




    def __init__(self,target=list[str],features_labels=None):

        self.target = target
        self.features_labels = features_labels
    

    def fit(self,X,y=None):
        pass



#######################################################################################################
#               Stepwise Discriminant Analysis (STEPDISC) - Discriminant Analysis Procedure
#######################################################################################################


class STEPDISC(BaseEstimator,TransformerMixin):
    """Stepwise Discriminant Analysis

    Performs a stepwise discriminant analysis to select a subset of the quantitative variables for use
    in discriminating among the classes. It can be used for forward selection, backward elimination, or
    stepwise selection.

    Parameters
    ----------
    level : float, default = 0.01
    method : {'forward','backward','stepwise'}, default = 'forward'
    alpha : the alpha level. The default alpha = 0.01.
    lambda_init : Initial Wilks Lambda/ Default = None
    verbose : 

    
    
    """
    

    def __init__(self,
                 method="forward",
                 alpha=0.01,
                 lambda_init = None,
                 model_train = False,
                 verbose = True):
        
        self.method = method
        self.alpha = alpha
        self.lambda_init = lambda_init
        self.model_train = model_train
        self.verbose = verbose
    

    def fit(self,clf):
        """Fit

        Parameter
        ---------
        clf : an instance of class LINEARDISC or CANDISC
        
        
        """

        if clf.model_ not in ["candisc","lda"]:
            raise ValueError("Error : 'clf' must be and instance of class 'LINEARDISC' or 'CANDISC'.")
        
        self._compute_stats(clf)
        
        return self
    
    def _compute_forward(self,clf):
        raise NotImplementedError("Error : This method is not implemented yet.")
    
    def _compute_backward(self,clf):
        """Backward Elimination

        Parameters:
        -----------
        clf : an instance of class LINEARDISC or CANDISC
        
        Return
        ------
        resultat : DataFrame
        
        """
        def fexclusion(j,W,V,n,K,lw):
            J = W.shape[1]
            # degrés de liberté - numérateur
            ddlsuppnum = K - 1
            # degrés de liberté dénominateur
            ddlsuppden = n - K - J + 1
            # Matrices intermédiaires numérateur
            tempnum = W.copy().values
            # Supprimer la référence de la variable à traiter
            tempnum = np.delete(tempnum, j, axis = 0)
            tempnum = np.delete(tempnum, j, axis = 1)
            # Même chose pour le numérateur
            tempden = V.values
            tempden = np.delete(tempden, j, axis = 0)
            tempden = np.delete(tempden, j, axis = 1)
            
            # Lambda de Wilk's sans la variable
            lwVar = np.linalg.det(tempnum)/np.linalg.det(tempden)
            # FValue
            fvalue = ddlsuppden/ddlsuppnum * (lwVar/lw-1)
            # Récupération des résultats
            return np.array([lwVar,lw/lwVar,fvalue,st.f.sf(fvalue, ddlsuppnum, ddlsuppden)])
        
        # Matrix V et W utilisées
        biased_V = ((clf.n_samples_ - 1)/clf.n_samples_)*clf.tcov_
        biased_W = ((clf.n_samples_ - clf.n_classes_)/clf.n_samples_)*clf.wcov_

        # Lambda de Wilks - Initialisation de la valeur du Lambda de Wilks
        lambdaInit = 0.0
        if self.lambda_init is None:
            lambdaInit = np.linalg.det(biased_W)/np.linalg.det(biased_V)
        else:
            lambdaInit = self.lambda_init
        
        # Liste de variables pour le test
        listInit = clf.features_labels_

        # Sauvegarde des résultats
        Result = pd.DataFrame(columns=["Wilks L.","Partial L.","F","p-value"]).astype("float")
        listvarDel = list()

        # 
        while True:
            # Résultat d'un étape
            fextract = partial(fexclusion,W=biased_W,V=biased_V,n=clf.n_samples_,K=clf.n_classes_,lw=lambdaInit)
            res = pd.DataFrame(np.array(list(map(lambda j : fextract(j=j),np.arange(biased_W.shape[1])))),
                               columns=["Wilks L.","Partial L.","F","p-value"])
            res.index = listInit

            # Affichage : verbose == True
            if self.verbose:
                print(res)
                print()

            # Extraction de la ligne qui maximise la p-value
            id = np.argmax(res.iloc[:,3])
            
            if res.iloc[id,3] > self.alpha:
                # Nom de la variable à rétirer
                listvarDel.append(listInit[id])
                # Rajouter la ligne de résultats
                Result = pd.concat([Result,res.iloc[id,:].to_frame().T],axis=0)
                # Rétirer
                del listInit[id]
                #listInit.pop(id)

                if len(listInit) == 0:
                    break
                else:
                    # Retirer les lignes et les clonnes des matrices
                    biased_W = np.delete(biased_W.values,id,axis=0)
                    biased_W = np.delete(biased_W,id,axis=1)
                    biased_W = pd.DataFrame(biased_W,index=listInit,columns = listInit)

                    # Retirer les lignes et les clonnes des matrices
                    biased_V = np.delete(biased_V.values,id,axis=0)
                    biased_V = np.delete(biased_V,id,axis=1)
                    biased_V = pd.DataFrame(biased_V,index=listInit,columns = listInit)
                    # Mise à jour de Lambda Init
                    lambdaInit = res.iloc[id,0]
            else:
                break
        
        # Sauvegarde des résultats
        resultat = pd.DataFrame(Result,index=listvarDel,columns=["Wilks L.","Partial L.","F","p-value"])
        return resultat
        
    def _compute_stepwise(self,clf):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def _compute_stats(self,clf):
        """
        
        
        """

        if self.method == "backward":
            overall_remove = self._compute_backward(clf=clf)
        elif self.method == "forward":
            overall_remove = self._compute_forward(clf=clf)
        elif self.method == "stepwise":
            overall_remove = self._compute_stepwise(clf=clf)

        features_remove = list(overall_remove.index)

        # Entraînement d'un modèle
        if self.model_train:
            # New features
            new_features = [x for x in clf.features_labels_ if x not in set(features_remove)]
            if clf.model_ == "lda":
                model = LINEARDISC(features_labels = new_features,target=clf.target_,distribution="homoscedastik",
                                   row_labels=clf.row_labels_).fit(clf.data_)
                self.train_model_ = model
            elif clf.model_ == "candisc":
                model = CANDISC(features_labels=new_features,target=clf.target_,row_labels=clf.row_labels_)
                self.train_model_ = model
            

        
        self.overall_remove_ = overall_remove
        self.features_remove = features_remove
    
######################################################################################
#           QUADRATIC DISCRIMINANT ANALYSIS (QDA)
#####################################################################################

class QUADISC(BaseEstimator,TransformerMixin):
    """Quadratic Discriminant Analysis
    
    """
    def __init__(self,features_columns, target_columns,priors=None):
        self.features_columns = features_columns
        self.target_columns = target_columns
        self.priors_ = priors
    
    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not implemented yet.")


#######################################################################################################
#           Mixture Discriminant Analysis (MIXTUREDISC)
#######################################################################################################


class MIXTUREDISC(BaseEstimator, TransformerMixin):
    """Mixture Discriminant Analysis (MIXTUREDISC)

    Performs mixture discriminant analysis




    Note
    ----
    The Linear Discriminant Analysis classifier assumes that each class comes from a single normal (or Gaussian)
    distribuition. This is too restrictive. 
    For Mixture Discriminant Analysis, there are classes, and each class is assumed to be a Gaussian mixture of 
    subclasses, where each data point has a probability of belonging to each class. Equality of covariance matrix,
    among classes, is still assumed.
    
    
    """




    def __init__(self):


        pass

    def fit(self,X):
        raise NotImplementedError("Error : This method is not yet implemented.")
    

#####################################################################################
#           LOCAL FISHER DISCRIMINANT ANALYSIS (LFDA)
######################################################################################
 
 # Démarche géométrique
 # https://plainenglish.io/blog/fischers-linear-discriminant-analysis-in-python-from-scratch-bbe480497504
 # https://stackoverflow.com/questions/62610782/fishers-linear-discriminant-in-python
 # https://goelhardik.github.io/2016/10/04/fishers-lda/
 # https://www.freecodecamp.org/news/an-illustrative-introduction-to-fishers-linear-discriminant-9484efee15ac/
 #https://github.com/prathmachowksey/Fisher-Linear-Discriminant-Analysis
class LOCALFISHERDISC(BaseEstimator,TransformerMixin):
    """Local Fisher Discriminant Analysis
    
    """
    def __init__(self,feature_columns,target_columns):
        self.feature_columns = feature_columns
        self.target_columns = target_columns

    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not implemented yet.")
    


###############################################################################################################
#               Flexible Discriminant Analysis (FDA)
###############################################################################################################


class FLEXIBLEDISC(BaseEstimator,TransformerMixin):
    """Flexible Discriminant Analysis



    Note:
    ----
    Flexible Discriminant Analyis is a flexible extension of Lienar Discriminant Analysis that uses non - linear
    combinations of predictors as splines. Flexible Discriminant Analysis is useful to model multivariate non - 
    normality or non - linear relationships among variables within each group, allowing for a more accurate classi-
    fication.
    
    """
    def __init__(self,feature_columns,target_columns):
        self.feature_columns = feature_columns
        self.target_columns = target_columns

    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not implemented yet.")


###############################################################################################################
#               Regularized Discriminant Analysis (RDA)
###############################################################################################################


class FLEXIBLEDISC(BaseEstimator,TransformerMixin):
    """Regularized Discriminant Analysis



    Note:
    ----
    RDA builds a classification rule by regularizing the group covariance matrices (Friedman 1989) allowing a 
    more robust model against multicollinearity in the data. This might be very useful for a large multivariate 
    data set containing highly correlated predictors.
    
    Regularized discriminant analysis is a kind of a trade-off between LDA and QDA. Recall that, in LDA we assume 
    equality of covariance matrix for all of the classes. QDA assumes different covariance matrices for all the 
    classes. Regularized discriminant analysis is an intermediate between LDA and QDA.

    RDA shrinks the separate covariances of QDA toward a common covariance as in LDA. This improves the estimate of 
    the covariance matrices in situations where the number of predictors is larger than the number of samples in 
    the training data, potentially leading to an improvement of the model accuracy.

    link : https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752
    
    
    """
    def __init__(self,feature_columns,target_columns):
        self.feature_columns = feature_columns
        self.target_columns = target_columns

    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not implemented yet.")



    
    

