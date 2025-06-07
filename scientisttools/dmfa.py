# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl

from mapply.mapply import mapply
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

# Intern functions
from .pca import PCA
from .namedtuplemerge import namedtuplemerge

class DMFA(BaseEstimator,TransformerMixin):
    def __init__(self,
                 num_fact=None,
                 standardize = True,
                 n_components = 5,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize=False):
        self.num_fact = num_fact
        self.standardize = standardize
        self.n_components = n_components
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set index name as None
        X.index.name = None

        # set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Check if num_fact is assigned
        if self.num_fact is None:
            raise ValueError("num_fact must be assigned.")

        # Set num fact label
        if isinstance(self.num_fact,int):
            num_fact = self.num_fact
            num_fact_label = X.columns[num_fact]
        elif isinstance(self.num_fact,str):
            num_fact_label = self.num_fact
            num_fact = X.columns.tolist().index(num_fact_label)
        else:
            raise TypeError("'num_fact' must be either a string or an integer.")

        # Drop level if ndim greater than 1 and reset columns name
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Checks if categorical/object and convert to categorical
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns:
                X[col] = pd.Categorical(X[col],categories=sorted(np.unique(X[col])),ordered=True)
                
        # Check if supplementary categorical variables
        if self.quali_sup is not None:
            if isinstance(self.quali_sup,(int,float)):
                quali_sup = [int(self.quali_sup)]
            elif isinstance(self.quali_sup,(list,tuple)) and len(self.quali_sup)>=1:
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup].tolist()
        else:
            quali_sup_label = None

        #  Check if supplementary quantitative variables
        if self.quanti_sup is not None:
            if isinstance(self.quanti_sup,(int,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif isinstance(self.quanti_sup,(list,tuple)) and len(self.quanti_sup)>=1:
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup].tolist()
        else:
            quanti_sup_label = None
        
        # Make a copy of the data
        Xtot = X.copy()

        # Extract group name
        group_name = np.unique(Xtot.iloc[:,self.num_fact]).tolist()
    
        # check if group name is an integer
        if all(isinstance(x, (int, float)) for x in group_name):
            group_name = ["Gr".format(x+1) for x in group_name]
            Xtot.loc[:,num_fact_label] = Xtot.map(zip(np.unique(Xtot.iloc[:,self.num_fact]).tolist(),group_name))

        # group index
        group_index = OrderedDict()
        for g in group_name:
            group_index[g] = np.where(Xtot.loc[:,num_fact_label]==g)[0]

        # Standardize data
        Cov, X_c, Z = OrderedDict(), OrderedDict(), pd.DataFrame().astype(float)

        for g in group_name:
            if self.quali_sup is not None:
                X_g = X[X.loc[:,num_fact_label] == g].drop(columns=quali_sup_label+[num_fact_label])
            else:
                X_g = X[X.loc[:,num_fact_label] == g].drop(columns=num_fact_label)

            # compute average
            center_g = np.mean(X_g,axis=0)
            center_g.name = "center"

            # compute standard deviation
            if self.standardize:
                scale_g = np.std(X_g,axis=0,ddof=0)
            else:
                scale_g = pd.Series(np.repeat(1,X_g.shape[1]),index=X_g.columns.tolist())
            scale_g.name = "scale"

            # Standardize the data
            Z_g = mapply(X_g,lambda x : (x - center_g.values)/scale_g.values,axis=1,progressbar=False,n_workers=n_workers)

            # Compute covariance or correlation
            if not self.standardize:
                Cov_g = Z_g.cov(ddof=0)
            else:
                Cov_g = Z_g.corr(method="pearson")

            # Concatenate
            Z = pd.concat((Z,Z_g),axis=0)
            # Add all elements
            X_c[g] = namedtuple(g,["X","Z","center","scale"])(X_g,Z_g,center_g,scale_g)
            Cov[g] = Cov_g

        #Store standardize data
        self.Xc_ = namedtuple("X_c",X_c.keys())(*X_c.values())

        #store covariance
        self.Cov_ = namedtuple("Cov",Cov.keys())(*Cov.values())

        # Add group variable
        Z = pd.concat((Xtot.iloc[:,num_fact],Z),axis=1)

        # 
        if self.quali_sup is None:
            # Find supplementary quantitative variables index
            if self.quanti_sup is not None:
                index = [Z.columns.tolist().index(x) for x in quanti_sup_label]
            else:
                index = None
            res = PCA(n_components = self.n_components,quali_sup=0,quanti_sup=index,parallelize = self.parallelize).fit(Z)

            # Extract supplementary quantitative variables informations
            if self.quanti_sup is not None:
                self.quanti_sup_ = res.quanti_sup_
        else:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            # Concatenate
            X_quali_sup_new = pd.concat((mapply(Xtot[[x,num_fact_label]],lambda x: '_'.join(x),axis=1,progressbar=False,n_workers=n_workers) for x in quali_sup_label),axis=1)
            X_quali_sup_new.columns = [x+"_"+num_fact_label for x in quali_sup_label]
            # Concatenate
            X_quali_sup = pd.concat((X_quali_sup,X_quali_sup_new),axis=1)
            
            # Concatenate
            Z = pd.concat((Xtot.iloc[:,num_fact],X_quali_sup,Z.iloc[:,1:]),axis=1)
            # Find supplementary quantitatives columns index
            index = [Z.columns.tolist().index(x) for x in X_quali_sup.columns.tolist()]

            # Find supplementary quantitative variables index
            if self.quanti_sup is not None:
                index2 = [Z.columns.tolist().index(x) for x in quanti_sup_label]
            else:
                index2 = None

            # Update PCA with
            res = PCA(n_components = self.n_components,quali_sup=[0]+index,quanti_sup=index2,parallelize = self.parallelize).fit(Z)

            # Extract supplementary quantitative variables informations
            if self.quanti_sup is not None:
                self.quanti_sup_ = res.quanti_sup_

        # Store 
        self.call_ = namedtuplemerge("call",res.call_,namedtuple("call",'num_fact')(num_fact_label))

        # Extract elements
        self.eig_, self.svd_, self.ind_ , self.var_ = res.eig_, res.svd_, res.ind_, res.var_

        #------------------------------------------------------------------------
        ## Partiel factor coordinates
        #------------------------------------------------------------------------
        n_components = res.call_.n_components
        var_partiel, cor_dim_gr = OrderedDict(),OrderedDict()
        
        for g in group_name:
            cor_g = np.corrcoef(res.ind_.coord.iloc[group_index[g],:],X_c[g].Z,rowvar=False)[n_components:,:n_components]
            var_partiel[g] = pd.DataFrame(cor_g,index=X_c[g].Z.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
            cor_dim_gr[g] = res.ind_.coord.iloc[group_index[g],:].corr(method="pearson")

        #store
        self.var_partiel_ = namedtuple("var_partiel",var_partiel.keys())(*var_partiel.values())
        self.cor_dim_gr_ = namedtuple("cor_dim_gr",cor_dim_gr.keys())(*cor_dim_gr.values())

        #-------------------------------------------------------------------------------------
        ## Group informations : coordinates, cos2
        #-------------------------------------------------------------------------------------
        var_coord = res.var_.coord
        group_coord = pd.DataFrame(index=group_name,columns=var_coord.columns).astype(float)
        group_coord_n = pd.DataFrame(index=group_name,columns=var_coord.columns).astype(float)
        group_cos2 = pd.DataFrame(index=group_name,columns=var_coord.columns).astype(float)

        for i, g in enumerate(group_name):
            if self.quanti_sup is None: 
                Cov_g = Cov[g]
            else: 
                Cov_g = Cov[g].drop(index=quanti_sup_label,columns=quanti_sup_label)
            for j in range(n_components):
                group_coord.iloc[i,j] = np.sum(np.diag(np.outer(var_coord.iloc[:,j],np.dot(var_coord.iloc[:,j],Cov_g))))/res.eig_.iloc[j,0]

            # 
            eigen = np.real(np.linalg.eig(Cov_g)[0])
            group_coord_n.iloc[i,:] = group_coord.iloc[i,:].div(eigen[0])
            group_cos2.iloc[i,:] = group_coord.iloc[i,:].pow(2).div(np.sum(eigen**2))  

        #store all group informations
        self.group_ = namedtuple("group",["name","coord","coord_n","cos2"])(group_name,group_coord,group_coord_n,group_cos2)  
            
        #--------------------------------------------------------------------------------------
        ## Statistics for supplementary categorical variables
        #--------------------------------------------------------------------------------------
        if res.quali_sup_.coord.shape[0] > len(group_name):
            quali_sup_coord = res.quali_sup_.coord.iloc[len(group_name):,:]
            quali_sup_cos2 = res.quali_sup_.cos2.iloc[len(group_name):,:]
            quali_sup_vtest = res.quali_sup_.vtest.iloc[len(group_name):,:]
            quali_sup_sqdisto, quali_sup_eta2 = res.quali_sup_.dist, res.quali_sup_.eta2

            #Store all supplementary categorical variables informations
            self.quali_sup_ = namedtuple("quali_sup",["coord","cos2","vtest","dist","eta2"])(quali_sup_coord,quali_sup_cos2,quali_sup_vtest,quali_sup_sqdisto,quali_sup_eta2)

        self.model_ = "dmfa"

        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_columns)
            Training data, where `n_samples` is the number of samples and `n_columns` is the number of columns.
        
        `y` : None
            y is ignored.
        
        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.ind_.coord

def supvarDMFA(self,X_quanti_sup=None, X_quali_sup=None):


    res = {}
    return res