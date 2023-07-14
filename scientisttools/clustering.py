
import numpy as np
import pandas as pd
from plydata import *
import scipy.stats as st
from scientisttools.utils import eta2
from mapply.mapply import mapply
from scipy.cluster import hierarchy
from scientistmetrics import scientistmetrics
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.base import BaseEstimator, TransformerMixin
from scientisttools.utils import from_dummies

##################################################################################################################3
#           Hierachical Clustering Analysis on Principal Components (HCPC)
###################################################################################################################

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
                 parallelize = False):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.parallelize = parallelize

    def fit(self,res):
        """
        
        
        """

        if res.model_ not in ["pca","mca"]:
            raise ValueError("Error : 'res' must be an objet of class 'PCA','MCA'.")
        
        # Set parallelize 
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        self._compute_global_stats(res=res)
        
        if res.model_ == "pca":
            self._compute_stats_pca(res=res)
        elif res.model_ == "mca":
            self._compute_stats_mca(res=res)
        
        return self
    

    def _compute_global_stats(self,res):
        """
        
        
        """

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

        # Linkage matrix
        link_mat = hierarchy.linkage(res.row_coord_,method=self.method_,metric = self.metric_,optimal_ordering=False)

        # Order
        order = hierarchy.leaves_list(link_mat)

        if res.model_ == "mca":
            active_data = res.original_data_
        else:
            active_data = res.active_data_
        
        # Save labels
        labels = res.row_labels_
        
        # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_mat,n_clusters=self.n_clusters_)+1).reshape(-1, )
        #cutree = pd.read_excel("d:/Bureau/PythonProject/packages/scientisttools/data/mca_hcpc.xlsx")["clust"].values
        cutree = list(["cluster_"+str(x) for x in cutree])

        # Class information
        cluster = pd.DataFrame(cutree, index = labels,columns = ["cluster"])

        ## Description des cluster
        row_coord = pd.DataFrame(res.row_coord_,index=labels,columns=res.dim_index_)
        coord_classe = pd.concat([row_coord, cluster], axis=1)
        cluster_infos = coord_classe.groupby("cluster").size().to_frame("n(k)")
        cluster_infos["p(k)"] = cluster_infos["n(k)"]/np.sum(cluster_infos["n(k)"])

        # Mean by cluster
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
            disto = mapply(group.sub(cluster_centers.loc[k,:],axis="columns"),lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_).to_frame("distance").reset_index(drop=False)
            # Identification du parangon
            id = np.argmin(disto.iloc[:,1])
            parangon = pd.DataFrame({"parangons" : disto.iloc[id,0],"distance" : disto.iloc[id,1]},index=[k])
            parangons = pd.concat([parangons,parangon],axis=0)
            # Extraction des distances
            disto_near[k] = disto.sort_values(by="distance").reset_index(drop=True)
            disto_far[k] = disto.sort_values(by="distance",ascending=False).reset_index(drop=True)
        
        #Rapport de corrélation entre les axes et les cluster
        desc_axes = self._compute_quantitative(X=row_coord)

        # Informations globales
        self.linkage_matrix_ = link_mat
        self.order_ = order
        self.labels_ = labels
        self.parangons_ = parangons
        # Individuals closest to their cluster's center
        self.disto_near_ = pd.concat(disto_near,axis=0)
        # Individuals the farest from other clusters' center
        self.disto_far_ = pd.concat(disto_far,axis=0)
        self.desc_axes_gmean_ = desc_axes[0]
        self.desc_axes_correlation_ratio_ = desc_axes[1]
        self.desc_axes_infos_ = pd.concat(desc_axes[2],axis=0)
        self.data_cluster_ = pd.concat([active_data,cluster],axis=1)

        ### Agglomerative result
        self.distances_ = link_mat[:,2]
        self.children_ = link_mat[:,:2].astype(int)

        # Save the input model
        self.factor_model_ = res

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
        self.correlation_ratio_ = res1[1]
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
        v_test = mapply(gmean,lambda x :np.sqrt(n_rows-1)*(x-means.values)/std.values, axis=0,progressbar=False,n_workers=self.n_workers_)
        v_test = pd.concat(((v_test.loc[:,i]/np.sqrt((n_rows-n_k.loc[i,])/n_k.loc[i,])).to_frame(i) for i in list(n_k.index)),axis=1)

        # Calcul des probabilités associées aux valeurs test
        vtest_prob = mapply(v_test,lambda x : 2*(1-st.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False,n_workers=self.n_workers_)

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
        """Perform qualitative
        
        """

        # concatenate
        df = pd.concat([X,self.cluster_],axis=1)

        # Convert all str columns to category columns
        df = mapply(df,lambda x: x.astype("category") if x.dtype == "O" else x,axis=0,progressbar=False,n_workers=self.n_clusters_)

        # Chi2 squared test - Tschuprow's T
        chi2_test = loglikelihood_test = pd.DataFrame(columns=["statistic","df","pvalue"],index=X.columns).astype("float")
        cramers_v = tschuprow_t = pearson= pd.DataFrame(columns=["value"],index=X.columns).astype("float")
        for cols in X.columns:
            # Crosstab
            tab = pd.crosstab(df[cols],df["cluster"])
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
        
        quali_test = dict({"chi2" : chi2_test,"gtest":loglikelihood_test,"cramer":cramers_v,"tschuprow":tschuprow_t,"pearson":pearson})
        
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
        mod_class = dummies_classe.groupby("cluster").mean().T.mul(100)

        class_mod = dummies_classe.groupby("cluster").sum().T
        class_mod = class_mod.div(dummies_stats["n(s)"].values,axis="index").mul(100)

        var_category = dict()
        for i,name in enumerate(self.cluster_labels_):
            df =pd.concat([class_mod.iloc[:,i],mod_class.iloc[:,i],dummies_stats["p(s)"].mul(100)],axis=1)
            df.columns = ["Class/Mod","Mod/Class","Global"]
            var_category[f"cluster_{i+1}"] = df

        self.desc_var_quali_ = res1
        self.var_quali_infos_ = dummies_stats
        self.desc_var_category_ = pd.concat(var_category,axis=0)


#########################################################################################################################
#       Clustering of Variables
###########################################################################################################################
class VARCLUS(BaseEstimator,TransformerMixin):
    """Clustering of variables
    
    
    
    """
    def __init__(self,
                 nb_clusters=None,
                 matrix_type = "completed",
                 metric = "dice",
                 col_labels = None,
                 row_labels = None):
        self.nb_clusters = nb_clusters
        self.matrix_type = matrix_type
        self.metric = metric
        self.row_labels = row_labels
        self.col_labels = col_labels

    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not yet implemented.")
    
##################################################################################################################################
#           Hierarchical Clustering Analysis of Continuous Variables (VARHCA)
##################################################################################################################################

class VARHCA(BaseEstimator,TransformerMixin):
    """Hierarchical Clustering Analysis of Continuous Variables

    Parameters
    ----------
    n_clusters :
    var_labels :
    var_sup_labels :
    min_clusters :
    max_clusters :
    matrix_type : {"completed","correlation"}
    metric :
    method :
    max_iter :
    init :
    random_state :
    """

    def __init__(self,
                 n_clusters=None,
                 var_labels = None,
                 var_sup_labels = None,
                 min_clusters = 2,
                 max_clusters = 5,
                 matrix_type = "completed",
                 metric="euclidean",
                 method="ward",
                 max_iter = 300,
                 init="k-means++",
                 random_state = None,
                 parallelize=False):
        self.n_clusters = n_clusters
        self.var_labels = var_labels
        self.var_sup_labels = var_sup_labels
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.matrix_type = matrix_type
        self.metric = metric
        self.method = method
        self.max_iter =max_iter
        self.init = init
        self.random_state = random_state
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """ Fit
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Extract supplementary columns
        self.var_sup_labels_ = self.var_sup_labels
        if self.var_sup_labels_ is not None:
            if self.matrix_type == "completed":
                Xsup = X[self.var_sup_labels_]
                active_data = X.drop(columns=self.var_sup_labels_)
            elif self.matrix_type == "correlation":
                Xsup = X[self.var_sup_labels_].drop(index=self.var_sup_labels_)
                active_data = X.drop(columns=self.var_sup_labels_).drop(index=self.var_sup_labels_)
        else:
            active_data = X

        self.data_ = X
        self.active_data_ = active_data
        
        # Compute global stat
        self._compute_stats(active_data)
        # 
        if self.var_sup_labels_ is not None:
            self.data_sup_ = Xsup
            self.corr_mean_square_ = self.transform(Xsup)

        # Compute supplementary statistique

        return self
        
        # From covariance to correlation
        # https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    
    def _compute_stats(self,X):
        """Compute global statistiques
        
        """
        # Set parallelize option
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

         # Compute correlation matrix
        if self.matrix_type == "completed":
            corr = X.corr(method="pearson")
        elif self.matrix_type == "correlation":
            corr = X

        # Variable labels
        self.var_labels_ = self.var_labels
        if self.var_labels_ is None:
            self.var_labels_ = corr.index
        
        # Number of variables
        self.n_vars_ = len(self.var_labels_)

         # Linkage matrix
        self.method_ = self.method
        if self.method_ is None:
            self.method_ = "ward"
        
        self.metric_ = self.metric
        if self.metric_ is None:
            self.metric_ = "euclidean"

        # Compute dissimilary matrix : sqrt(1 - x**2)
        D = mapply(corr,lambda x : np.sqrt(1 - x**2),axis=0,progressbar=False,n_workers=self.n_workers_)

        # Vectorize
        VD = squareform(D)

        # Compute number of clusters
        self.n_clusters_ = self.n_clusters
        if self.n_clusters_ is None:
           kmeans = KMeans(init=self.init,max_iter=self.max_iter,random_state=self.random_state)
           visualizer = KElbowVisualizer(kmeans, k=(self.min_clusters,self.max_clusters),metric='distortion',
                                         timings=False,locate_elbow=True,show=False).fit(D)
           self.n_clusters_ = visualizer.elbow_value_

        # Linkage Matrix
        link_mat = hierarchy.linkage(VD,method=self.method_,metric = self.metric_,optimal_ordering=False)

        # Order
        order = hierarchy.leaves_list(link_mat)

         # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_mat,n_clusters=self.n_clusters_)+1).reshape(-1, )
        cutree = list(["cluster_"+str(x) for x in cutree])

        # Class information
        cluster = pd.DataFrame(cutree, index = self.var_labels_,columns = ["cluster"])

        # Store First informations
        self.cluster_ = cluster
        self.cluster_labels_ = list(["cluster_"+str(x+1) for x in np.arange(self.n_clusters_)])
        self.covariance_matrix_ = corr
        self.linkage_matrix_ = link_mat

        ### Agglomerative result
        self.distances_ = link_mat[:,2]
        self.children_ = link_mat[:,:2].astype(int)
        self.order_ = order

        # Model name
        self.model_ = "varhca"
    
    def transform(self,X,y=None):
        """
        
        
        """
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        if self.matrix_type == "completed":
            corr_with = pd.DataFrame(np.corrcoef(self.active_data_,X,rowvar=False)[:self.n_vars_,self.n_vars_:],index = self.var_labels_,columns=X.columns)
        elif self.matrix_type == "correlation":
            corr_with = X
        
        # Concatenation
        corr_class = pd.concat([corr_with,self.cluster_],axis=1)
        #moyenne des carrés des corrélations avec les groupes
        corr_mean_square = mapply(corr_class.groupby("cluster"),lambda x : np.mean(x**2,axis=0),progressbar=False,n_workers=self.n_workers_)
        return corr_mean_square

    
    def fit_transform(self,X,y=None):
        """
        
        """

        self.fit(X)
        return self.linkage_matrix_

###################################################################################################################################
#       Hierarchical Clustering Analysis of Categorical Variables (CATVARHCA)
###################################################################################################################################

class CATVARHCA(BaseEstimator,TransformerMixin):
    """Hierarchical Clustering Analysis of categorical variables

    Parameters
    ----------
    n_clusters:
    var_labels :
    var_sup_labels :
    mod_labels :
    mod_sup_labels :
    min_clusters:
    max_clusters:
    diss_metric : {"cramer","dice","bothpos"}
    """
    def __init__(self,
                 n_clusters=None,
                 var_labels = None,
                 mod_labels = None,
                 sup_labels = None,
                 min_clusters = 2,
                 max_clusters = 5,
                 diss_metric = "cramer",
                 matrix_type = "completed",
                 metric="euclidean",
                 method="ward",
                 max_iter = 300,
                 init="k-means++",
                 random_state = None,
                 parallelize=False):
        self.n_clusters = n_clusters
        self.var_labels = var_labels
        self.mod_labels = mod_labels
        self.sup_labels = sup_labels
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.diss_metric = diss_metric
        self.matrix_type = matrix_type
        self.metric = metric
        self.method = method
        self.max_iter =max_iter
        self.init = init
        self.random_state = random_state
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """
        
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Extract supplementary columns
        self.sup_labels_ = self.sup_labels
        if self.sup_labels_ is not None:
            Xsup = X[self.sup_labels_]
            active_data = X.drop(columns=self.sup_labels_)
        else:
            active_data = X

        self.data_ = X
        self.active_data_ = active_data

        self._compute_stat(active_data)

        return self
    
    def _diss_cramer(self,X):
        """Compute Dissimilary Matrix using Cramer's V statistic
        
        """
        if self.matrix_type == "completed":
            M = X
        elif self.matrix_type == "disjonctif" :
            M = from_dummies(X,sep="_")
        
        # Store matrix
        self.dummies_matrix_ = pd.concat((pd.get_dummies(M[cols],prefix=cols,prefix_sep='_',drop_first=False) for cols in M.columns),axis=1)
        self.original_data_ = M

        # Convert you str columns to Category columns
        M = mapply(M,lambda x: x.astype("category") if x.dtype == "O" else x,axis=0,progressbar=False,n_workers=self.n_workers_)

        # Compute dissimilarity matrix using cramer's V
        D = mapply(scientistmetrics(X=M,method="cramer"),lambda x : 1 - x, axis=0,progressbar=False,n_workers=self.n_workers_)
        return D
    
    @staticmethod
    def funSqDice(col1,col2):
        return 0.5*np.sum((col1-col2)**2)
    
    @staticmethod
    def funbothpos(col1,col2):
        return 1 - (1/len(col1))*np.sum(col1*col2)
    
    def _diss_modality(self,X):
        """Compute Distance matrix using Dice index
        
        """
        if self.matrix_type == "completed":
            M =  pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_',drop_first=False) for cols in (X.columns if self.var_labels is None else self.var_labels)),axis=1)
        elif self.matrix_type == "disjonctif": 
            M = X
        
        self.dummies_matrix_ = M
        self.original_data_ = from_dummies(M,sep="_")

        # Compute Dissimilarity Matrix
        D = pd.DataFrame(index=M.columns,columns=M.columns).astype("float")
        for row in M.columns:
            for col in M.columns:
                if self.diss_metric == "dice":
                    D.loc[row,col] = np.sqrt(self.funSqDice(M[row].values,M[col].values))
                elif self.diss_metric == "bothpos":
                    D.loc[row,col] = self.funbothpos(M[row].values,M[col].values)
        if self.diss_metric == "bothpos":
            np.fill_diagonal(D.values,0)
        return D

    def _compute_stat(self,X):
        """
        
        
        """
        # Parallel option
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1
        
        # Compute Dissimilarity Matrix
        if self.diss_metric == "cramer":
            D = self._diss_cramer(X)
        elif self.diss_metric in ["dice","bothpos"]:
            D = self._diss_modality(X)
        
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
                                         timings=False,locate_elbow=True,show=False).fit(D)
           self.n_clusters_ = visualizer.elbow_value_

        # Linkage matrix
        link_mat = hierarchy.linkage(squareform(D),method=self.method_,metric = self.metric_,optimal_ordering=False)

        # Order
        order = hierarchy.leaves_list(link_mat)
        
        # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_mat,n_clusters=self.n_clusters_)+1).reshape(-1, )
        cutree = list(["cluster_"+str(x) for x in cutree])

        # Class information
        cluster = pd.DataFrame(cutree, index=D.columns,columns = ["cluster"])

        cluster_infos = cluster.groupby("cluster").size().to_frame("n(k)")
        cluster_infos["p(k)"] = cluster_infos["n(k)"]/np.sum(cluster_infos["n(k)"])

        # Store First informations
        self.cluster_ = cluster
        self.cluster_labels_ = list(["cluster_"+str(x+1) for x in np.arange(self.n_clusters_)])
        self.linkage_matrix_ = link_mat

        ### Agglomerative result
        self.distances_ = link_mat[:,2]
        self.children_ = link_mat[:,:2].astype(int)
        self.order_ = order
        self.labels_ = D.columns
        self.diss_matrix_ = D
        self.cluster_infos_ = cluster_infos

        # Model name
        self.model_ = "catvarhca"
    
    def transform(self,X):
        """
        
        
        """
         # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")


        if self.diss_metric == "cramer":
            if self.matrix_type == "completed":
                M = X
            elif self.matrix_type == "disjonctif" :
                M = from_dummies(X,sep="_")
            # V de cramer 
            D = pd.DataFrame(index=self.labels_,columns=M.columns).astype("float")
            for row in self.original_data_.columns:
                for col in M.columns:
                    tab = pd.crosstab(self.original_data_[row],M[col])
                    D.loc[row,col] = st.contingency.association(tab,method="cramer") 
        elif self.diss_metric in ["dice","bothpos"]:
            if self.matrix_type == "completed":
                M =  pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_',drop_first=False) for cols in X.columns),axis=1)
            elif self.matrix_type == "disjonctif": 
                M = X
            
            # Compute Dissimilarity Matrix
            D = pd.DataFrame(index=self.dummies_matrix_.columns,columns=M.columns).astype("float")
            for row in self.dummies_matrix_.columns:
                for col in M.columns:
                    if self.diss_metric == "dice":
                        D.loc[row,col] = self.funSqDice(self.dummies_matrix_[row].values,M[col].values)
                    elif self.diss_metric == "bothpos":
                        D.loc[row,col] = self.funbothpos(self.dummies_matrix_[row].values,M[col].values)
        # 
        if self.method in ["ward","average"]:
            corr_sup = pd.concat([D,self.cluster_],axis=1).groupby("cluster").mean()
        elif self.method == "single":
            corr_sup = pd.concat([D,self.cluster_],axis=1).groupby("cluster").min()
        elif self.method == "complete":
            corr_sup = pd.concat([D,self.cluster_],axis=1).groupby("cluster").max()
        
        return corr_sup

##############################################################################################################
#       Hierachical Clustering Analysis on Principal Components of Variables (VARHCPC)
##############################################################################################################
        
class VARHCPC(BaseEstimator,TransformerMixin):
    """Variables Hierachical Clustering on Principal Components

    Performs Hierarchical Clustering on variables using principal components

    Parameters:
    ------------



    Returns:
    -------
    
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
                 parallelize=False):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.parallelize = parallelize

    def fit(self,res):
        """
        
        """
        if res.model_ not in ["pca","mca"]:
            raise ValueError("Error : 'res' must be an objet of class 'PCA','MCA'.")
        
        self._compute_global_stats(res)

        return self
    
    @staticmethod
    def weighted_average(val_col_name, wt_col_name):
        def inner(group):
            return (group[val_col_name] * group[wt_col_name]).sum() / group[wt_col_name].sum()
        inner.__name__ = 'weighted_averages'
        return inner
    
    def _compute_global_stats(self,res):
        """
        
        
        """
        # Set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Extract principal components
        if res.model_ == "pca":
            X = res.col_coord_
            labels = res.col_labels_
        elif res.model_ == "mca":
            X = res.mod_coord_
            labels = res.mod_labels_

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
                                         timings=False,locate_elbow=True,show=False).fit(X)
           self.n_clusters_ = visualizer.elbow_value_
        
        # Linkage matrix
        link_mat = hierarchy.linkage(X,method=self.method_,metric = self.metric_,optimal_ordering=False)

        # Order
        order = hierarchy.leaves_list(link_mat)

        # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_mat,n_clusters=self.n_clusters_)+1).reshape(-1, )
        cutree = list(["cluster_"+str(x) for x in cutree])

        # Class information
        cluster = pd.DataFrame(cutree, index = labels,columns = ["cluster"])

        # Cluster example
        cluster_infos = cluster.groupby("cluster").size().to_frame("n(k)")
        cluster_infos["p(k)"] = cluster_infos["n(k)"]/np.sum(cluster_infos["n(k)"])

        # Store First informations
        self.cluster_ = cluster
        self.cluster_infos_ = cluster_infos
        self.cluster_labels_ = list(["cluster_"+str(x+1) for x in np.arange(self.n_clusters_)])
        
        ## Description des cluster par 
        coord = pd.DataFrame(X,index=labels,columns=res.dim_index_)
        if res.model_ == "pca":
            coord_classe = pd.concat([coord, cluster], axis=1)
            cluster_centers = coord_classe.groupby("cluster").mean()
        elif res.model_ == "mca":
            weight = pd.DataFrame(res.mod_infos_[:,1]*res.n_vars_,columns=["weight"],index=labels)
            coord_classe = pd.concat([weight,coord,cluster], axis=1)
            cluster_centers = pd.concat((mapply(coord_classe.groupby("cluster"), 
                                     self.weighted_average(col,"weight"),axis=0,n_workers=self.n_workers_,progressbar=False).to_frame(col) for col in res.dim_index_),axis=1)
        
        # Centre des clusters
        self.cluster_centers_ = cluster_centers
        self.labels_ = labels

        ### Agglomerative result
        self.linkage_matrix_ = link_mat
        self.distances_ = link_mat[:,2]
        self.children_ = link_mat[:,:2].astype(int)
        self.order_ = order
        self.factor_model_ = res

        # Modèle
        self.model_ = "varhcpc"
    
    def fit_transform(self,X):
        """
        
        """
        self.fit(X)
        return self.linkage_matrix_

###############################################################################################################################
#                   Variables Kmeans (VARKMEANS)
###############################################################################################################################

class VARKMEANS(BaseEstimator,TransformerMixin):
    """
    
    
    
    """
    def __init__(self,
                 n_clusters=None):
        self.n_clusters = n_clusters
    
    def fit(self,X,y=None):
        raise NotImplementedError("Error : This method is not yet implemented.")







        
        




    