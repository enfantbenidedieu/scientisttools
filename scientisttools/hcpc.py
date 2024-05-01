# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp

from mapply.mapply import mapply
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from sklearn.base import BaseEstimator, TransformerMixin


from .eta2 import eta2
from .revaluate_cat_variable import revaluate_cat_variable


class HCPC(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering on Principal Components (HCPC)
    ------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs an agglomerative hierarchical clustering on results from a factor analysis.
    Results include paragons, description of the clusters.

    Parameters
    ----------
    
    model : an object of class PCA, MCA, FAMD

    n_clusters : an integer.  If a (positive) integer, the tree is cut with nb.cluters clusters.
                if None, the tree is automatically cut
    
    min_cluster : an integer. The least possible number of clusters suggested.

    max_cluster : an integer. The higher possible number of clusters suggested; by default the minimum between 10 and the number of individuals divided by 2.

    metric : The metric used to built the tree, default = "euclidean"

    method : The method used to built the tree, default = "ward"

    proba : The probability used to select axes and variables, default = 0.05

    n_paragons : An integer. The number of edited paragons.

    order : A boolean. If True, clusters are ordered following their center coordinate on the first axis.

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    call_ : A list or parameters and internal objects.

    cluster_ : a dictionary with clusters informations 

    data_clust_ :  The original data with a supplementary column called clust containing the partition.

    desc_var_ : The description of the classes by the variables.

    desc_axes_ : The description of the classes by the factors (axes)

    desc_ind_ : The paragons (para) and the more typical individuals of each cluster

    model_ : string. The model fitted = 'hcpc'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Escofier B, Pagès J (2008), Analyses Factorielles Simples et Multiples.4ed, Dunod
    """
    def __init__(self,
                 model,
                 n_clusters=3,
                 min_cluster = 3,
                 max_cluster = None,
                 metric="euclidean",
                 method="ward",
                 proba = 0.05,
                 n_paragons = 5,
                 order = True,
                 parallelize = False):
        
        # Check if model 
        if model.model_ not in ["pca","mca","famd"]:
            raise TypeError("'model' must be an objet of class 'PCA','MCA','FAMD'")
        
        # Set parallelize 
        if parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Automatic cut tree
        def auto_cut_tree(model,min_clust,max_clust,metric,method,order,weights=None):
            if order:
                data = pd.concat((model.ind_["coord"],model.call_["X"],model.call_["ind_weights"]),axis=1)
                if weights is not None:
                    weights = weights[::-1]
                data = data.sort_values(by=data.columns.tolist()[0],ascending=True)
                model.ind_["coord"] = data.iloc[:,:model.ind_["coord"].shape[1]]
                model.call_["X"] = data.iloc[:,(model.ind_["coord"].shape[1]+1):(data.shape[1]-1)]
                model.call_["ind_weights"] = data.iloc[:,-1]
            
            # Extract
            X = model.ind_["coord"]
            # Dissimilarity matrix
            do = pdist(X,metric=metric)**2
            # Set weights
            if weights is None:
                weights = np.ones(X.shape[0])
            # Effec
            eff = np.zeros(shape=(len(weights),len(weights)))
            for i in range(len(weights)):
                for j in range(len(weights)):
                    eff[i,j] = (weights[i]*weights[j])/(sum(weights))/(weights[i]+weights[j])
            dissi = do*eff[np.triu_indices(eff.shape[0], k = 1)]
            # Agglometrive clustering
            link_matrix = hierarchy.linkage(dissi,metric=metric,method=method)
            inertia_gain = link_matrix[:,2][::-1]
            intra = np.cumsum(inertia_gain[::-1])[::-1]
            quot = intra[(min_clust-1):max_clust]/intra[(min_clust-2):(max_clust-1)]
            nb_clust = (np.argmin(quot)+1) + min_clust - 1
            return nb_clust

        ################# Quantitatives variables descriptions
        def quanti_var_desc(X,cluster,n_workers,proba):
            # Dimension du tableau
            n  = X.shape[0]
            # Moyenne et écart - type globale
            means, std = X.mean(axis=0), X.std(axis=0,ddof=0)

            # Concatenate with original data
            data = pd.concat([X,cluster],axis=1)

            # Moyenne conditionnelle - variance conditionnelle and effectif
            gmean, gstd,nk = data.groupby('clust').mean().T, data.groupby("clust").std(ddof=0).T, data.groupby('clust').size()

            # valeur-test
            v_test = mapply(gmean,lambda x :np.sqrt(n - 1)*(x-means.values)/std.values, axis=0,progressbar=False,n_workers=n_workers)
            v_test = pd.concat(((v_test.loc[:,k]/np.sqrt((n - nk.loc[k])/nk.loc[k])).to_frame(k) for k in nk.index.tolist()),axis=1)
            # Calcul des probabilités associées aux valeurs test
            vtest_prob = mapply(v_test,lambda x : 2*(1-sp.stats.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False,n_workers=n_workers)
    
            # Arrange all result
            quanti = {}
            for k  in nk.index.tolist():
                df = pd.concat((v_test.loc[:,k],gmean.loc[:,k],means,gstd.loc[:,k],std,vtest_prob.loc[:,k]),axis=1)
                df.columns = ["vtest","Mean in category","Overall mean","sd in categorie","Overall sd","pvalue"]
                quanti[str(k)] = df.sort_values(by=['vtest'],ascending=False).query("pvalue < @proba")
            
            # Correlation ratio
            corr_eta2 = (pd.concat((pd.DataFrame(eta2(cluster,X[col],digits=10),index=[col]) for col in X.columns.tolist()),axis=0)
                            .sort_values(by="Eta2",ascending=False)[["Eta2","pvalue"]]
                            .query("pvalue < @proba"))
            return corr_eta2,quanti
        
        # Qualitatives variables description
        def quali_var_desc(X,cluster,proba):
            ######## Tableau Disjonctif complex
            dummies = pd.concat((pd.get_dummies(X[col],prefix=col,prefix_sep='=') for col in X.columns),axis=1)            
            dummies_stats = dummies.agg(func=[np.sum,np.mean]).T
            dummies_stats.columns = ["n(s)","p(s)"]

            # chi2 & Valeur - test
            chi2_test = pd.DataFrame(columns=["statistic","dof","pvalue"]).astype("float")
            v_test = pd.DataFrame()
            for col in X.columns.tolist():
                # Crosstab
                tab = pd.crosstab(X[col],cluster)
                tab.index = [col+"="+x for x in tab.index.tolist()]

                # Chi2 test
                chi = sp.stats.chi2_contingency(tab,correction=False)
                row_chi2 = pd.DataFrame({"statistic" : chi.statistic,"dof" : chi.dof,"pvalue" : chi.pvalue},index=[col])
                chi2_test = pd.concat((chi2_test,row_chi2),axis=0)

                # Valeur - test
                nj, nk, n = tab.sum(axis=1), tab.sum(axis=0), tab.sum().sum()
                for j in tab.index.tolist():
                    for k in tab.columns.tolist():
                        pi = (nj.loc[j]*nk.loc[k])/n
                        num, den = tab.loc[j,k] - pi, ((n-nk.loc[k])/(n-1))*(1-nj.loc[j]/n)*pi
                        tab.loc[j,k] = num/np.sqrt(den)
                v_test = pd.concat((v_test,tab),axis=0) 
            
            # Filter using probability
            chi2_test = chi2_test.query("pvalue < @proba").sort_values(by="pvalue")
            # vtest probabilities
            vtest_prob = mapply(v_test,lambda x : 2*(1-sp.stats.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False,n_workers=n_workers)

            # Listing MOD/CLASS
            dummies_classe = pd.concat([dummies,cluster],axis=1)
            mod_class = dummies_classe.groupby("clust").mean().T.mul(100)

            # class/Mod
            class_mod = dummies_classe.groupby("clust").sum().T
            class_mod = class_mod.div(dummies_stats["n(s)"].values,axis="index").mul(100)

            var_category = {}
            for i in np.unique(cluster):
                df = pd.concat((class_mod.loc[:,i],mod_class.loc[:,i],dummies_stats["p(s)"].mul(100),vtest_prob.loc[:,i],v_test.loc[:,i]),axis=1)
                df.columns = ["Class/Mod","Mod/Class","Global","pvalue","vtest"]
                var_category[str(i)] = df.sort_values(by=['vtest'],ascending=False).query("pvalue < @proba")
            
            return chi2_test,var_category

        ####################################################################################################################
        #   
        ####################################################################################################################
        
        # Set linkage method
        if method is None:
            method = "ward"
        
        # Set linkage metric
        if metric is None:
            metric = "euclidean"
        
        # Set max cluster
        if max_cluster is None:
            max_cluster = min(10,round(model.ind_["coord"].shape[0]/2))
        else:
            max_cluster = min(max_cluster,model.ind_["coord"].shape[0]-1)
        
        # Set number of clusters
        if n_clusters is None:
            n_clusters = auto_cut_tree(model=model,min_clust=min_cluster,max_clust=max_cluster,method=method,metric=metric,order=order,
                                       weights=np.ones(model.ind_["coord"].shape[0]))
        elif not isinstance(n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")

        # Agglomerative clustering
        link_matrix = hierarchy.linkage(model.ind_["coord"],method=method,metric=metric)
        # cut the hierarchical tree
        cutree = (hierarchy.cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, )
        cluster = pd.Series([str(x) for x in cutree],index =  model.ind_["coord"].index.tolist(),name = "clust")

        ##### Store data clust
        data_clust = model.call_["Xtot"]
        # Drop the supplementary individuals
        if model.ind_sup is not None:
            if isinstance(model.ind_sup,int):
                ind_sup = [ind_sup]
            elif isinstance(model.ind_sup,list) or isinstance(model.ind_sup,tuple):
                ind_sup = [x for x in model.ind_sup]
            data_clust = data_clust.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in ind_sup])
        data_clust = pd.concat((data_clust,cluster),axis=1)
        self.data_clust_ = data_clust

        # Tree elements
        tree = {"order":order,
                "linkage" : link_matrix,
                "height":link_matrix[:,2],
                "method":method,
                "metric" : metric,
                "merge":link_matrix[:,:2],
                "n_obs":link_matrix[:,3],
                "data": model.ind_["coord"],
                "n_clusters" : n_clusters}

        self.call_ = {"model" : model,"X" : data_clust,"tree" : tree}

        ############################################################################################################
        ## Description des cluster
        ################################################################################################################
        # Concatenate individuals coordinates with classe
        coord_classe = pd.concat([model.ind_["coord"], cluster], axis=1)
        # Count by cluster
        cluster_count = coord_classe.groupby("clust").size()
        cluster_count.name = "effectif"

        # Coordinates by cluster
        cluster_coord = coord_classe.groupby("clust").mean()

        # Value - test by cluster
        axes_mean =  model.ind_["coord"].mean(axis=0)
        axes_std = model.ind_["coord"].std(axis=0,ddof=0)
        cluster_vtest = mapply(cluster_coord,lambda x :np.sqrt(cluster.shape[0]-1)*(x-axes_mean.values)/axes_std.values,axis=1,progressbar=False,n_workers=n_workers)
        cluster_vtest = pd.concat(((cluster_vtest.loc[i,:]/np.sqrt((cluster.shape[0]-cluster_count.loc[i])/cluster_count.loc[i])).to_frame(i).T for i in cluster_count.index.tolist()),axis=0)
        
        # Store cluster informations
        self.cluster_ = {"cluster" : cluster,"coord" : cluster_coord , "vtest" : cluster_vtest, "effectif" : cluster_count}

        #################################################################################################
        ## Axis description
        #################################################################################################
        axes_desc = quanti_var_desc(X=model.ind_["coord"],cluster=cluster,n_workers=n_workers,proba=proba)
        dim_clust = pd.concat((model.ind_["coord"],cluster),axis=1)

        axes_call = {"X" : dim_clust,"proba" : proba,"num_var" : dim_clust.shape[1]}
        self.desc_axes_ = {"quanti_var" : axes_desc[0],"quanti" : axes_desc[1], "call" : axes_call}
        
        ############################################################################################################"
        #   Individuals description
        ############################################################################################################
        paragons = {}
        disto_far = {}
        for k in np.unique(cluster):
            group = coord_classe.query("clust == @k").drop(columns=["clust"])
            disto = mapply(group.sub(cluster_coord.loc[k,:],axis="columns"),lambda x : x**2,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            disto.name = "distance"
            paragons[f"Cluster : {k}"] = disto.sort_values(ascending=True).iloc[:n_paragons]
            disto_far[f"Cluster : {k}"] = disto.sort_values(ascending=False).iloc[:n_paragons]
        
        self.desc_ind_ = {"para" : paragons, "dist" : disto_far}

        ###############################################################################################################
        #  Principal Component Analysis (PCA)
        ###############################################################################################################
        ##########################################
        data_call = {"X" : data_clust, "proba" : proba, "num_var" : data_clust.shape[1]}

        data = model.call_["X"]
        # Principal Component Analysis (PCA)
        if model.model_ == "pca":
            ####### Distance to origin
            cluster_var = pd.concat((model.call_["Z"],cluster),axis=1).groupby("clust").mean()
            cluster_dist2 = mapply(cluster_var,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=1)
            cluster_dist2.name = "dist"
            self.cluster_["dist"] =  np.sqrt(cluster_dist2)
            # Add supplementary quantitatives variables
            if model.quanti_sup is not None:
                if (isinstance(model.quanti_sup,int) or isinstance(model.quanti_sup,float)):
                    quanti_sup = [int(model.quanti_sup)]
                elif ((isinstance(model.quanti_sup,list) or isinstance(model.quanti_sup,tuple))  and len(model.quanti_sup)>=1):
                    quanti_sup = [int(x) for x in model.quanti_sup]
                
                X_quanti_sup = model.call_["Xtot"].iloc[:,quanti_sup]
                if model.ind_sup is not None:
                    X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in model.ind_sup])
                
                ###### Transform to float
                X_quanti_sup = X_quanti_sup.astype("float")
                data = pd.concat((data,X_quanti_sup),axis=1)
            
            ######### Description of quantitatives variables
            quanti_desc = quanti_var_desc(X=data,cluster=cluster,n_workers=n_workers,proba=proba)
            self.desc_var_ = {"quanti_var" : quanti_desc[0],"quanti" : quanti_desc[1]}

            if model.quali_sup is not None:
                if (isinstance(model.quali_sup,int) or isinstance(model.quali_sup,float)):
                    quali_sup = [int(model.quali_sup)]
                elif ((isinstance(model.quali_sup,list) or isinstance(model.quali_sup,tuple))  and len(model.quali_sup)>=1):
                    quali_sup = [int(x) for x in model.quali_sup]
                
                X_quali_sup = model.call_["Xtot"].iloc[:,quali_sup]
                if model.ind_sup is not None:
                    X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in model.ind_sup])
                
                ###### Transform to object
                X_quali_sup = X_quali_sup.astype("object")

                # Description of qualitatives variables
                quali_desc = quali_var_desc(X=X_quali_sup,cluster=cluster,proba=proba)
                self.desc_var_ = {**self.desc_var_, **{"test_chi2" : quali_desc[0],"category" : quali_desc[1]}}

            # Add data call
            self.desc_var_["call"] = data_call
        
        elif model.model_ == "mca": # Multiple Correspondence Analysis (MCA)
            # Add supplementary categoricals variables
            if model.quali_sup is not None:
                if (isinstance(model.quali_sup,int) or isinstance(model.quali_sup,float)):
                    quali_sup = [int(model.quali_sup)]
                elif ((isinstance(model.quali_sup,list) or isinstance(model.quali_sup,tuple))  and len(model.quali_sup)>=1):
                    quali_sup = [int(x) for x in model.quali_sup]
                
                X_quali_sup = model.call_["Xtot"].iloc[:,quali_sup]
                if model.ind_sup is not None:
                    X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in model.ind_sup])
                
                # Transform to object
                X_quali_sup = X_quali_sup.astype("object")
                data = pd.concat((data,X_quali_sup),axis=1)
                
            # Description of qualitatives variables
            quali_desc = quali_var_desc(X=data,cluster=cluster,proba=proba)
            self.desc_var_ = {"test_chi2" : quali_desc[0],"category" : quali_desc[1]}

            ################ Add supplementary quantitatives variables
            # Add supplementary quantitatives variables
            if model.quanti_sup is not None:
                if (isinstance(model.quanti_sup,int) or isinstance(model.quanti_sup,float)):
                    quanti_sup = [int(model.quanti_sup)]
                elif ((isinstance(model.quanti_sup,list) or isinstance(model.quanti_sup,tuple))  and len(model.quanti_sup)>=1):
                    quanti_sup = [int(x) for x in model.quanti_sup]
                
                X_quanti_sup = model.call_["Xtot"].iloc[:,quanti_sup]
                if model.ind_sup is not None:
                    X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in model.ind_sup])
                
                ###### Transform to float
                X_quanti_sup = X_quanti_sup.astype("float")
                
                # Description of quantitative variables
                quanti_desc = quanti_var_desc(X=X_quanti_sup,cluster=cluster,n_workers=n_workers,proba=proba)
                self.desc_var_ = {**self.desc_var_ , **{"quanti_var" : quanti_desc[0],"quanti" : quanti_desc[1]}}
                
            # Add data call
            self.desc_var_["call"] = data_call
        # Factor Analysis of Mixed Data (FAMD)
        elif model.model_ == "famd":
            ##### Split data in to 
            ########################################################################################
            # Select quantitatives variables
            X_quanti = data.select_dtypes(exclude=["object","category"])
            # Add supplementary quantitatives variables
            if model.quanti_sup is not None:
                if (isinstance(model.quanti_sup,int) or isinstance(model.quanti_sup,float)):
                    quanti_sup = [int(model.quanti_sup)]
                elif ((isinstance(model.quanti_sup,list) or isinstance(model.quanti_sup,tuple))  and len(model.quanti_sup)>=1):
                    quanti_sup = [int(x) for x in model.quanti_sup]
                
                X_quanti_sup = model.call_["Xtot"].iloc[:,quanti_sup]
                if model.ind_sup is not None:
                    X_quanti_sup = X_quanti_sup.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in model.ind_sup])
                
                ###### Transform to float
                X_quanti_sup = X_quanti_sup.astype("float")
                X_quanti = pd.concat((X_quanti,X_quanti_sup),axis=1)
            
            # Description of quantitatives variables
            quanti_desc = quanti_var_desc(X=X_quanti,cluster=cluster,n_workers=n_workers,proba=proba)

            ########################################################################################
            # Select categoricals variables
            X_quali = data.select_dtypes(include=["object","category"])
            # Add supplementary qualitatiave data
            if model.quali_sup is not None:
                if (isinstance(model.quali_sup,int) or isinstance(model.quali_sup,float)):
                    quali_sup = [int(model.quali_sup)]
                elif ((isinstance(model.quali_sup,list) or isinstance(model.quali_sup,tuple))  and len(model.quali_sup)>=1):
                    quali_sup = [int(x) for x in model.quali_sup]
                
                X_quali_sup = model.call_["Xtot"].iloc[:,quali_sup]
                if model.ind_sup is not None:
                    X_quali_sup = X_quali_sup.drop(index=[name for i, name in enumerate(model.call_["Xtot"].index.tolist()) if i in model.ind_sup])
                
                # Transform to object
                X_quali_sup = X_quali_sup.astype("object")
                X_quali = pd.concat((X_quali,X_quali_sup),axis=1)
                
            # Description of qualitatives variables
            quali_desc = quali_var_desc(X=X_quali,cluster=cluster,proba=proba)
            self.desc_var_ = {**self.desc_var_, **{"test_chi2" : quali_desc[0],"category" : quali_desc[1], "call" : data_call}}
        # Modèle
        self.model_ = "hcpc"
