
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as st
from mapply.mapply import mapply
from scientistmetrics import scientistmetrics
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.base import BaseEstimator, TransformerMixin
from scientisttools.utils import from_dummies, eta2,revaluate_cat_variable
from scientisttools.extractfactor import dimdesc
import fastcluster

##################################################################################################################3
#           Hierachical Clustering Analysis on Principal Components (HCPC)
###################################################################################################################

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

    n_clusters : an integer



    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
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
            raise ValueError("'model' must be an objet of class 'PCA','MCA','FAMD'")
        
        # Set parallelize 
        if parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Automatic cut tree
        def auto_cut_tree(model,min_clust,max_clust,metric,method,order,weights=None):
            """
            
            """
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
            link_matrix = fastcluster.linkage(dissi,metric=metric,method=method)
            inertia_gain = link_matrix[:,2][::-1]
            intra = np.cumsum(inertia_gain[::-1])[::-1]
            quot = intra[(min_clust-1):max_clust]/intra[(min_clust-2):(max_clust-1)]
            nb_clust = (np.argmin(quot)+1) + min_clust - 1
            return nb_clust
        
        ###### Correlation ratio
        def quanti_eta2(data,categories,proba):
            """
            Global correlatio ratio
            -----------------------

            Parameters
            ----------
            
            
            """
            res = (pd.concat((pd.DataFrame(eta2(categories,data[col],digits=10),index=[col]) for col in data.columns.tolist()),axis=0)
                            .rename(columns={"correlation ratio" : "Eta2"})
                            .sort_values(by="Eta2",ascending=False)[["Eta2","pvalue"]]
                            .query("pvalue < @proba"))
            return res
        
        # Chi-square test of independence of variables in a contingency table.
        def chi2_test(data,categories,proba):
            """
            Chi2 test of independence of variables in a contingency table
            --------------------------------------------------------------

            Parameters
            ---------
            data : pandas dataframe of shape (n_rows, n_cols) containing categoricals variables

            categories :  pandas series

            Return
            ------

            """
            chi2_test = pd.DataFrame(columns=["statistic","dof","pvalue"]).astype("float")
            for col in data.columns.tolist():
                tab = pd.crosstab(data[col],categories)
                chi = st.chi2_contingency(tab,correction=False)
                row_chi2 = pd.DataFrame({"statistic" : chi.statistic,"dof" : chi.dof,"pvalue" : chi.pvalue},index=[col])
                chi2_test = pd.concat((chi2_test,row_chi2),axis=0)
            
            # Filter using proba
            chi2_test = chi2_test.query("pvalue < @proba")
            return chi2_test


        ################# Quantitatives
        def quanti_var_desc(X,cluster,n_workers,proba):
            """
            Quantitative variable description
            ---------------------------------

            Parameters
            ----------
            X : pandas dataframe of shape (n_row, n_colsw)

            cluster : pandas series of shape (n_rows,)

            Return
            -------
            """
            # Dimension du tableau
            n_rows  = X.shape[0]
            # Moyenne et écart - type globale
            overall_means = X.mean(axis=0)
            overall_std = X.std(axis=0,ddof=0)
            # Concatenate with original data
            data = pd.concat([X,cluster],axis=1)
            # Moyenne conditionnelle par groupe
            gmean = data.groupby('clust').mean().T
            # Ecart - type conditionnelle conditionnelles
            gstd = data.groupby("clust").std(ddof=0).T
            # Effectifs par cluster
            effectif = data.groupby('clust').size()
            # valeur-test
            v_test = mapply(gmean,lambda x :np.sqrt(n_rows-1)*(x-overall_means.values)/overall_std.values, axis=0,progressbar=False,n_workers=n_workers)
            v_test = pd.concat(((v_test.loc[:,i]/np.sqrt((n_rows-effectif.loc[i])/effectif.loc[i])).to_frame(i) for i in effectif.index.tolist()),axis=1)
            # Calcul des probabilités associées aux valeurs test
            vtest_prob = mapply(v_test,lambda x : 2*(1-st.norm(0,1).cdf(np.abs(x))),axis=0,progressbar=False,n_workers=n_workers)
    
            # Arrange all result
            quanti = {}
            for i  in effectif.index.tolist():
                df = pd.concat([v_test.loc[:,i],vtest_prob.loc[:,i],gmean.loc[:,i],overall_means,gstd.loc[:,i],overall_std],axis=1)
                df.columns = ["vtest","pvalue","mean in category","overall mean","sd in categorie","overall sd"]
                quanti[str(i)] = df.sort_values(by=['vtest'],ascending=False).query("pvalue < @proba")

            return quanti
        
        def quali_var_desc(X,cluster):
            """
            Categorical variables description
            ---------------------------------

            Parameters
            ----------
            X : pandas datafrme of shape (n_rows, n_columns)
            
            
            """
            ###### Revaluate category
            X = revaluate_cat_variable(X)

            ######## Tableau Disjonctif complex
            dummies = pd.concat((pd.get_dummies(X[col],prefix=col,prefix_sep='=') for col in X.columns),axis=1)            
            dummies_stats = dummies.agg(func=[np.sum,np.mean]).T
            dummies_stats.columns = ["n(s)","p(s)"]

            # Listing MOD/CLASS
            dummies_classe = pd.concat([dummies,cluster],axis=1)
            mod_class = dummies_classe.groupby("clust").mean().T.mul(100)

            # class/Mod
            class_mod = dummies_classe.groupby("clust").sum().T
            class_mod = class_mod.div(dummies_stats["n(s)"].values,axis="index").mul(100)

            var_category = {}
            for i in np.unique(cluster):
                df = pd.concat([class_mod.loc[:,i],mod_class.loc[:,i],dummies_stats["p(s)"].mul(100)],axis=1)
                df.columns = ["Class/Mod","Mod/Class","Global"]
                var_category[str(i)] = df
            
            return var_category

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
        link_matrix = fastcluster.linkage(model.ind_["coord"],method=method,metric=metric)
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
                "height":link_matrix[:,2],
                "method":method,
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
        ####################################################################################################
        axes_quanti_var =  quanti_eta2(data=model.ind_["coord"],categories=cluster,proba=proba)
        axes_quanti = quanti_var_desc(X=model.ind_["coord"],cluster=cluster,n_workers=n_workers,proba=proba)
        dim_clust = pd.concat((model.ind_["coord"],cluster),axis=1)
        axes_call = {"X" : dim_clust,"proba" : proba,"num_var" : dim_clust.shape[1]}
        self.desc_axes_ = {"quanti_var" : axes_quanti_var,"quanti" : axes_quanti, "call" : axes_call}
        
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
            
            ######### Correlation ratio between original data and cluster
            quanti_var = quanti_eta2(data=data,categories=cluster,proba=proba)
            quanti = quanti_var_desc(X=data,cluster=cluster,n_workers=n_workers,proba=proba)

            # Store result
            self.desc_var_ = {"quanti_var" : quanti_var,"quanti" : quanti}

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
                # chi2 test between categorical variables and cluster
                test_chi2 = chi2_test(data=X_quali_sup,categories=cluster,proba=proba)
                if test_chi2.shape[0]>0:
                    self.desc_var_["test_chi2"] = test_chi2
                category = quali_var_desc(X=X_quali_sup,cluster=cluster)
                self.desc_var_["category"] = category
            # Add data call
            self.desc_var_["call"] = data_call
        # Multiple Correspondence Analysis (MCA)
        elif model.model_ == "mca":
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
                
            # chi2  test between categorical variable and 
            test_chi2 = chi2_test(data=data,categories=cluster,proba=proba)
            category = quali_var_desc(X=data,cluster=cluster)

            # Store result
            self.desc_var_ = {"test_chi2" : test_chi2,"category" : category}

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
                # Correlation ratio 2
                quanti_var = quanti_eta2(data=X_quanti_sup,categories=cluster,proba=proba)
                if quanti_var.shape[0]>0:
                    self.desc_var_["quanti_var"] = quanti_var
                quanti = quanti_var_desc(X=X_quanti_sup,cluster=cluster,n_workers=n_workers,proba=proba)
                self.desc_var_["quanti"] = quanti
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
            
            ######### Correlation ratio between original data and cluster
            quanti_var = quanti_eta2(data=X_quanti,categories=cluster,proba=proba)
            quanti = quanti_var_desc(X=X_quanti,cluster=cluster,n_workers=n_workers,proba=proba)

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
                data = pd.concat((data,X_quali_sup),axis=1)
                
            # chi2  test between categorical variable and 
            test_chi2 = chi2_test(data=data,categories=cluster,proba=proba)
            category = quali_var_desc(X=data,cluster=cluster)

            # Store result
            self.desc_var_ = {"quanti_var" : quanti_var,"quanti" : quanti, "call" : data_call}

        # Modèle
        self.model_ = "hcpc"


##################################################################################################################################
#           Hierarchical Clustering Analysis of Continuous Variables (VARHCA)
##################################################################################################################################

class VARHCA(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis of Continuous Variables (VARHCA)
    -----------------------------------------------------------------

    Description
    -----------



    Parameters
    ----------
    n_clusters : nmber of clusters, default = 3
    
    """

    def __init__(self,
                 n_clusters=3,
                 var_sup = None,
                 min_clusters = 2,
                 max_clusters = 5,
                 matrix_type = "completed",
                 metric = "euclidean",
                 method = "ward",
                 max_iter = 300,
                 random_state = None,
                 parallelize=False):
        self.n_clusters = n_clusters
        self.var_sup = var_sup
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.matrix_type = matrix_type
        self.metric = metric
        self.method = method
        self.max_iter =max_iter
        self.random_state = random_state
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        X : pandas/polars DataFrame of float, shape (n_rows, n_columns) or (n_columns, n_columns)

        y : None
            y is ignored

        Returns:
        --------
        self : object
                Returns the instance itself
        """

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Check of matrix type is one of 'completed' or 'correlation'
        if self.matrix_type not in ["completed","correlation"]:
            raise TypeError("'matrix_type' should be one of 'completed', 'correlation'")

        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        if not all_num:
            raise TypeError("All columns must be numeric")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        #  Check if supplementary variables
        if self.var_sup is not None:
            if (isinstance(self.var_sup,int) or isinstance(self.var_sup,float)):
                var_sup = [int(self.var_sup)]
            elif ((isinstance(self.var_sup,list) or isinstance(self.var_sup,tuple))  and len(self.var_sup)>=1):
                var_sup = [int(x) for x in self.var_sup]
            
        ####################################### Save the base in a new variables
        # Store data
        Xtot = X

        ####################################### Drop supplementary variables columns ########################################
        if self.var_sup is not None:
            if self.matrix_type == "completed":
                X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in var_sup])
            elif self.matrix_type == "correlation":
                X = (X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in var_sup])
                      .drop(index=[name for i, name in enumerate(Xtot.index.tolist()) if i in var_sup]))

        ##################################### Compute Pearson correlation matrix ##############################################
        if self.matrix_type == "completed":
            corr_matrix = X.corr(method="pearson")
        elif self.matrix_type == "correlation":
            corr_matrix = X

        # Linkage matrix
        if self.method is None:
            method = "ward"
        else:
            method = self.method
        
        ########################## metrics
        if self.metric is None:
            metric = "euclidean"
        else:
            metric = self.metric
        
        ################## Check numbers of clusters
        if self.n_clusters is None:
            n_clusters = 3
        elif not isinstance(self.n_clusters,int):
            raise TypeError("'n_clusters' must be an integer")
        else:
            n_clusters = self.n_clusters
        
        # Compute dissimilary matrix : sqrt(1 - x**2)
        D = mapply(corr_matrix,lambda x : np.sqrt(1 - x**2),axis=0,progressbar=False,n_workers=n_workers)

        # Linkage Matrix with vectorize dissimilarity matrix
        link_matrix = fastcluster.linkage(squareform(D),method=method,metric = metric)

         # Coupure de l'arbre
        cutree = (hierarchy.cut_tree(link_matrix,n_clusters=n_clusters)+1).reshape(-1, )
        cutree = [str(x) for x in cutree]

        # Class information
        cluster = pd.Series(cutree, index = corr_matrix.index.tolist(),name = "clust")

        # Tree elements
        tree = {"height":link_matrix[:,2],
                "method":method,
                "metric" : metric,
                "merge":link_matrix[:,:2],
                "n_obs":link_matrix[:,3],
                "data": corr_matrix,
                "n_clusters" : n_clusters}
        
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "tree" : tree}

        ################################### Informations abouts clusters
        data_clust = pd.concat((corr_matrix,cluster),axis=1)
        # Count by cluster
        cluster_count = data_clust.groupby("clust").size()
        cluster_count.name = "effectif"

        # Store cluster informations
        self.cluster_ = {"cluster" : cluster,"data_clust" : data_clust ,"effectif" : cluster_count}

        # Model name
        self.model_ = "varhca"

        return self
        
        # From covariance to correlation
        # https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    
    def transform(self,X,y=None):
        """
        
        
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Test if X is a DataFrame
        if isinstance(X,pd.Series):
            X = X.to_frame()
        elif not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns.tolist())
        if not all_num:
            raise TypeError("All columns must be numeric")

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        if self.matrix_type == "completed":
            corr_with = pd.DataFrame(np.corrcoef(self.call_["X"],X,rowvar=False)[:self.call_["X"].shape[1],self.call_["X"].shape[1]:],
                                     index = self.call_["X"].columns.tolist(),columns=X.columns.tolist())
        elif self.matrix_type == "correlation":
            corr_with = X
        
        # Concatenation
        data_clust = pd.concat([corr_with,self.cluster_["cluster"]],axis=1)
        #moyenne des carrés des corrélations avec les groupes
        corr_mean_square = mapply(data_clust.groupby("clust"),lambda x : np.mean(x**2,axis=0),progressbar=False,n_workers=n_workers)
        return corr_mean_square
    
    def fit_transform(self,X,y=None):
        """
        
        """
        self.fit(X)
        return self.cluster_["cluster"]

###################################################################################################################################
#       Hierarchical Clustering Analysis of Categorical Variables (CATVARHCA)
###################################################################################################################################

class CATVARHCA(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis of Categorical Variables (VATVARHCA)
    ---------------------------------------------------------------------

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