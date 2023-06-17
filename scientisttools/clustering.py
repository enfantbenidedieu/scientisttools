
import numpy as np
import pandas as pd
from plydata import *
import scipy.stats as st
import matplotlib.pyplot as plt
from scientisttools.utils import eta2
from mapply.mapply import mapply
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from scientisttools.pyplot import plotHCPC



class HCPC(BaseEstimator,TransformerMixin):
    """Hierarchical Clustering on Principal Components

    Compute hierarchical clustering on principal components
    
    
    """



    def __init__(self,
                 n_clusters=None,
                 metric="euclidean",
                 method="ward",
                 min_clusters = 2,
                 max_clusters = 8,
                 max_iter = 300,
                 init="k-means++",
                 random_state = None,
                 graph=True,
                 figsize=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.graph = graph
        self.figsize=figsize


    def fit(self,res,):

        if res.model_ not in ["pca","mca"]:
            raise ValueError("Error : 'res' must be an objet of class 'PCA','MCA'.")

        self._compute_global_stats(res=res)
        
        if res.model_ == "pca":
            self._compute_stats_pca(res=res)
        elif res.model_ == "mca":
            self._compute_stats_mca(res=res)

        if self.graph:
            fig, axe = plt.subplots(figsize=self.figsize)
            plotHCPC(self,repel=True,ax=axe)

        
        return self
    

    def _compute_global_stats(self,res):

         # Linkage matrix
        self.method_ = self.method
        if self.method_ is None:
            self.method_ = "ward"
        
        self.metric_ = self.metric
        if self.metric_ is None:
            self.metric_ = "euclidean"

        self.n_clusters_ = self.n_clusters
        if self.n_clusters_ is None:
           kmeans = KMeans(init=self.init,max_iter=self.max_iter,random_state=self.random_state)
           visualizer = KElbowVisualizer(kmeans, k=(self.min_clusters,self.max_clusters),metric='distortion',
                                         timings=False,locate_elbow=True,show=False).fit(res.row_coord_)
           self.n_clusters_ = visualizer.elbow_value_

        link_mat = hierarchy.linkage(res.row_coord_,method=self.method_,metric = self.metric_,optimal_ordering=False)

        if res.model_ == "mca":
            active_data = res.original_data_
        else:
            active_data = res.active_data_
        
        labels = active_data.index
        
        # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_mat,n_clusters=self.n_clusters_)+1).reshape(-1, )
        cutree = list(["cluster_"+str(x) for x in cutree])

        # Class information
        cluster = pd.DataFrame(cutree, index = labels,columns = ["cluster"])

        ## Description des cluster
        row_coord = pd.DataFrame(res.row_coord_,index=labels,columns=res.dim_index_)
        coord_classe = pd.concat([row_coord, cluster], axis=1)
        cluster_infos = coord_classe.groupby("cluster").size().to_frame("n(k)")
        cluster_infos["p(k)"] = cluster_infos["n(k)"]/np.sum(cluster_infos["n(k)"])

        cluster_centers = coord_classe.groupby("cluster").mean()

        # Store First informations
        self.cluster_ = cluster
        self.cluster_infos_ = cluster_infos
        self.cluster_labels_ = list(["cluster_"+str(x+1) for x in np.arange(self.n_clusters_)])
        self.cluster_centers_ = cluster_centers

        # Paarongons
        parangons = pd.DataFrame(columns=["parangons","distance"])
        disto_far = dict()
        disto_near = dict()
        for k in np.unique(cluster):
            group = coord_classe[coord_classe["cluster"] == k].drop(columns=["cluster"])
            disto = mapply(group.sub(cluster_centers.loc[k,:],axis="columns"),lambda x : np.sum(x**2),axis=1,progressbar=False).to_frame("distance").reset_index()
            disto_near[k] = disto.sort_values(by="distance")
            disto_far[k] = disto.sort_values(by="distance",ascending=False)
            # Identification du parangon
            id = np.argmin(disto.iloc[:,1])
            parangon = pd.DataFrame({"parangons" : disto.iloc[id,0],"distance" : disto.iloc[id,1]},index=[k])
            parangons = pd.concat([parangons,parangon],axis=0)
        
        #Rapport de corrélation entre les axes et les cluster
        desc_axes = self._compute_quantitative(X=row_coord)

        # Informations globales
        self.linkage_matrix_ = link_mat
        self.parangons_ = parangons
        # Individuals closest to their cluster's center
        self.disto_near_ = pd.concat(disto_near,axis=0)
        # Individuals the farest from other clusters' center
        self.disto_far_ = pd.concat(disto_far,axis=0)
        self.desc_axes_gmean_ = desc_axes[0]
        self.desc_axes_correlation_ratio_ = desc_axes[1]
        self.desc_axes_infos_ = pd.concat(desc_axes[2],axis=0)
        self.row_labels_ = res.row_labels_
        self.row_coord_ = res.row_coord_
        self.n_components_ = res.n_components_
        self.dim_index_ = res.dim_index_
        self.data_cluster_ = pd.concat([active_data,cluster],axis=1)
        self.eig_ = res.eig_

        # Modèle
        self.model_ = "hcpc"

    def _compute_stats_pca(self,res):
        """Compute statistique for Principal Components Analysis 

        Parameters
        ----------
        res : An instance of class PCA

        Returns
        -------

        
        
        
        """

        # Reconstitution des données
        data = res.data_

        # Remove supplementrary row
        if res.row_sup_labels_ is not None:
            data = data.drop(index=res.row_sup_labels_)

        # Extraction des données actives
        active_data = res.active_data_

        # Compute statistic for active data
        res1 = self._compute_quantitative(X=active_data)
        
        self.gmean_ = res1[0]
        self.correlation_ratio_ =res1[1]
        self.desc_var_quanti_ = pd.concat(res1[2],axis=0)

        if res.quanti_sup_labels_ is not None:
            quanti_sup_data = data[res.quanti_sup_labels_]
            quanti_sup_res = self._compute_quantitative(X=quanti_sup_data)
            self.gmean_quanti_sup_ = quanti_sup_res[0]
            self.correlation_ratio_quanti_sup_ = quanti_sup_res[1]
            self.desc_var_quanti_sup_ = pd.concat(quanti_sup_res[2],axis=0)
        
        if res.quali_sup_labels_ is not None:
            quali_sup_data = data[res.quali_sup_labels_]
            quali_sup_res = self._compute_qualitative(X=quali_sup_data)
            #self.gmean_quanti_sup_ = quali_sup_res[0]
            #self.correlation_ratio_quanti_sup_ = quanti_sup_res[1]
            self.desc_var_quali_sup_ = quali_sup_res

            #self.desc_var_quali_ = self._compute_qualitative(X=data[res.quali_sup_labels_])
        
    def _compute_quantitative(self,X):
        """
        
        
        """

        
        # Dimension du tableau
        n_rows, n_cols = X.shape

        # Moyenne et écart - type globale
        means = X.mean(axis=0)
        std = X.std(axis=0,ddof=0)

        # Données quantitatives - concatenation
        df = pd.concat([X,self.cluster_],axis=1)

        # Moyenne conditionnelle par groupe
        gmean = df.groupby('cluster').mean().T

        # Correlation ratio
        correlation_ratio = dict()
        for name in X.columns:
            correlation_ratio[name] = eta2(self.cluster_["cluster"],X[name])
        correlation_ratio = pd.DataFrame(correlation_ratio).T.sort_values(by=['correlation ratio'],ascending=False)

        # Ecart - type conditionnelle conditionnelles
        gstd = df.groupby("cluster").std(ddof=0).T

        # Effectifs par cluster
        n_k = df.groupby("cluster").size()

        # valeur-test
        v_test = mapply(gmean,lambda x :np.sqrt(n_rows-1)*(x-means.values)/std.values, axis=0,progressbar=False)
        v_test = pd.concat(((v_test.loc[:,i]/np.sqrt((n_rows-n_k.loc[i,])/n_k.loc[i,])).to_frame(i) for i in list(n_k.index)),axis=1)

        # Calcul des probabilités associées aux valeurs test
        vtest_prob = mapply(v_test,lambda x : 2*(1-st.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False)

        # Arrange all result
        
        quanti = dict()
        for i,name in enumerate(self.cluster_labels_):
            df =pd.concat([v_test.iloc[:,i],vtest_prob.iloc[:,i],gmean.iloc[:,i],means,gstd.iloc[:,i],std],axis=1)
            df.columns = ["vtest","pvalue","mean in category","overall mean","sd in categorie","overall sd"]
            df = df.sort_values(by=['vtest'],ascending=False)
            df["significant"] =np.where(df["pvalue"]<0.001,"***",np.where(df["pvalue"]<0.05,"**",np.where(df["pvalue"]<0.1,"*"," ")))
            quanti[f"cluster_{i+1}"] = df
        
        # Add columns
        gmean.columns = self.cluster_labels_

        
        return gmean,correlation_ratio,quanti
    
    def _compute_qualitative(self,X):
        """
        
        
        """

        # Test de chi-2
        df = pd.concat([X,self.cluster_],axis=1)

        # Chi2 squared test - Tschuprow's T
        chi2_test = loglikelihood_test = pd.DataFrame(columns=["statistic","df","pvalue"],index=X.columns).astype("float")
        cramers_v = tschuprow_t = pearson= pd.DataFrame(columns=["value"],index=X.columns).astype("float")
        for cols in X.columns:
            lb = LabelEncoder().fit_transform(X[cols])
            tab = pd.crosstab(lb,self.cluster_.values.ravel())

            # Chi2 - test
            chi2 = st.chi2_contingency(tab,correction=False)
            chi2_test.loc[cols,:] = np.array([chi2.statistic,chi2.dof,chi2.pvalue])

            # log-likelihood test
            loglikelihood = st.chi2_contingency(tab,lambda_="log-likelihood")
            loglikelihood_test.loc[cols,:] = np.array([loglikelihood.statistic,loglikelihood.dof,loglikelihood.pvalue])

            # Cramer's V
            cramers_v.loc[cols,:] = st.contingency.association(tab,method="cramer")

            # Tschuprow T statistic
            tschuprow_t.loc[cols,:] = st.contingency.association(tab,method="tschuprow")

            # Pearson
            pearson.loc[cols,:] = st.contingency.association(tab,method="pearson")
        
        quali_test = dict({"chi2" : chi2_test,"log-likelihood-test":loglikelihood_test,"cramer's V":cramers_v,
                           "tschuprow's T":tschuprow_t,"pearson":pearson})
        
        return quali_test
    
    def _compute_stats_mca(self,res):
        """
        
        
        """


         # Reconstitution des données
        data = res.data_

        # Remove supplementrary row
        if res.row_sup_labels_ is not None:
            data = data.drop(index=res.row_sup_labels_)

        # Extraction des données actives
        active_data = res.original_data_

        # Compute statistic for active data
        res1 = self._compute_qualitative(X=active_data)

        if res.quanti_sup_labels_ is not None:
            quanti_sup_data = data[res.quanti_sup_labels_]
            quanti_sup_res = self._compute_quantitative(X=quanti_sup_data)
            self.gmean_quanti_sup_ = quanti_sup_res[0]
            self.correlation_ratio_quanti_sup_ = quanti_sup_res[1]
            self.desc_var_quanti_sup_ = pd.concat(quanti_sup_res[2],axis=0)
        
        if res.quali_sup_labels_ is not None:
            quali_sup_data = data[res.quali_sup_labels_]
            quali_sup_res = self._compute_qualitative(X=quali_sup_data)
            #self.gmean_quanti_sup_ = quali_sup_res[0]
            #self.correlation_ratio_quanti_sup_ = quanti_sup_res[1]
            self.desc_var_quali_sup_ = quali_sup_res
        
        dummies = pd.concat((pd.get_dummies(active_data[cols],prefix=cols,prefix_sep='=') for cols in active_data.columns),axis=1)
        dummies_stats = dummies.agg(func=[np.sum,np.mean]).T
        dummies_stats.columns = ["n(s)","p(s)"]

        # Listing MOD/CLASS
        dummies_classe = pd.concat([dummies,self.cluster_],axis=1)
        mod_class = dummies_classe.groupby("cluster").mean().T

        class_mod = dummies_classe.groupby("cluster").sum().T
        class_mod = class_mod.div(dummies_stats["n(s)"].values,axis="index").mul(100)

        var_category = dict()
        for i,name in enumerate(self.cluster_labels_):
            df =pd.concat([class_mod.iloc[:,i],mod_class.iloc[:,i],dummies_stats["n(s)"]],axis=1)
            df.columns = ["Class/Mod","Mod/Class","Global"]
            var_category[f"cluster_{i+1}"] = df


        self.desc_var_quali_ = res1
        self.var_quali_infos_ = dummies_stats
        self.desc_var_category_ = pd.concat(var_category,axis=0)



class VARCLUST(BaseEstimator,TransformerMixin):
    """Variables Clusstering
    
    
    
    """



    def __init__(self,
                 nb_clusters=None,
                 matrix_type = "completed",
                 metric = "dice",
                 col_labels = None,
                 row_labels = None,
                 ):
        self.nb_clusters = nb_clusters
        self.matrix_type = matrix_type
        self.metric = metric
        self.row_labels = row_labels
        self.col_labels = col_labels



    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not yet implemented.")






        

        
        




    