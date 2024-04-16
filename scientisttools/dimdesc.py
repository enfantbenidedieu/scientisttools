# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
from .eta2 import eta2


def dimdesc(self,axis=None,proba=0.05):
    """
    Dimension description
    ----------------------

    Description
    -----------
    This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    Parameters
    ----------
    self : an object of class PCA, CA, MCA, FAMD

    axis : int or list. default axis= 0

    proba : the significance threshold considered to characterized the dimension (by default 0.05)

    Return
    ------
    Returns a dictionary including

    quanti	: the description of the dimensions by the quantitative variables. The variables are sorted.
    
    quali	: the description of the dimensions by the categorical variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    """
    if self.model_ == "pca":
        # Active data
        data = self.call_["X"]
        row_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()

        # Add supplementary continuous variables
        if self.quanti_sup is not None:
            X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
            data = pd.concat([data,X_quanti_sup],axis=1)
    
        corrdim = {}
        for idx in row_coord.columns:
            # For continuous variables
            corDim = pd.DataFrame(columns=["correlation","pvalue"]).astype("float")
            for col in data.columns:
                if np.issubdtype(data[col].dtype, np.number): #(data[col].dtypes in ["float64","int64","float32","int32"]):
                    res = sp.stats.pearsonr(data[col],row_coord[idx])
                    row_RD = pd.DataFrame({"correlation" : res.statistic,"pvalue":res.pvalue},index = [col])
                    corDim = pd.concat([corDim,row_RD],axis=0)
            # Filter by pvalue
            corDim = corDim.query('pvalue < @proba').sort_values(by="correlation",ascending=False)

            # For categorical variables
            if self.quali_sup is not None:
                quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
                if self.ind_sup is not None:
                    quali_sup = quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
                
                corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','Eta2','F-stats','pvalue'])
                for col in quali_sup.columns.tolist():
                    row_RD = pd.DataFrame(eta2(quali_sup[col],row_coord[idx],digits=8),index=[col])
                    corqDim = pd.concat([corqDim,row_RD],axis=0)
                # Filter by pvalue
                corqDim = (corqDim.query('pvalue < @proba').sort_values(by="Eta2",ascending=False).rename(columns={"R2": "Eta2"}))
            
            if self.quali_sup is None:
                res = corDim
            else:
                if corqDim.shape[0] != 0 :
                    res = {"quanti":corDim,"quali":corqDim}
                else:
                    res = corDim
            
            corrdim[idx] = res
    elif self.model_ == "ca":
        # Extract row coordinates
        row_coord = self.row_["coord"]
        # Exctract columns coordinates
        col_coord = self.col_["coord"]

        # Add Supplementary row
        if self.row_sup is not None:
            row_sup_coord = self.row_sup_["coord"]
            row_coord = pd.concat([row_coord,row_sup_coord],axis=0)
        
        # Add supplmentary columns
        if self.col_sup is not None:
            col_coord_sup = self.col_sup_["coord"]
            col_coord = pd.concat([col_coord,col_coord_sup],axis=0)

        # Select axis
        if axis is not None:
            row_coord = row_coord.iloc[:,axis]
            col_coord = col_coord.iloc[:,axis]
            if isinstance(row_coord,pd.Series):
                row_coord = row_coord.to_frame()
                col_coord = col_coord.to_frame()
        
        corrdim = {}
        for idx in row_coord.columns:
            corrdim[idx] = {"row" : (row_coord[idx].to_frame().sort_values(by=idx,ascending=True).rename(columns={idx:"coord"})),
                            "col" : (col_coord[idx].to_frame().sort_values(by=idx,ascending=True).rename(columns={idx:"coord"}))}
    elif self.model_ == "mca":
        # Select data
        data = self.call_["X"]
        # Extract individuals coordinates
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()

        if self.quali_sup is not None:
            X_quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
                data = pd.concat([data,X_quali_sup],axis=1)

        corrdim = {}
        for idx in ind_coord.columns:
            # Pearson correlation test
            if self.quanti_sup is not None:
                X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
                if self.ind_sup is not None:
                    X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
                
                corDim = pd.DataFrame(columns=["statistic","pvalue"]).astype("float")
                for col in X_quanti_sup.columns.tolist():
                    if (X_quanti_sup[col].dtypes in ["float64","int64","float32","int32"]):
                        res = sp.stats.pearsonr(X_quanti_sup[col],ind_coord[idx])
                        row_RD = pd.DataFrame({"statistic" : res.statistic,"pvalue":res.pvalue},index = [col])
                        corDim = pd.concat([corDim,row_RD])
                # Filter by pvalue
                corDim = (corDim.query('pvalue < @proba').sort_values(by="statistic",ascending=False))

            # Correlation ratio (eta2)
            corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','Eta2','F-stats','pvalue']).astype("float")
            for col in data.columns.tolist():
                row_RD = pd.DataFrame(eta2(data[col],ind_coord[idx],digits=8),index=[col])
                corqDim = pd.concat([corqDim,row_RD])
            # Filter by pvalue
            corqDim = (corqDim.query('pvalue < @proba').sort_values(by="Eta2",ascending=False).rename(columns={"Eta2" : "R2"}))
        
            if self.quanti_sup is None:
                res = corqDim
            else:
                if corDim.shape[0] != 0 :
                    res = {"quali":corqDim,"quanti":corDim}
                else:
                    res = corqDim
            corrdim[idx] = res
    elif self.model_ == "famd":
        # Extract row coord
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()
        
        # Select continuous active data
        quanti_data = self.call_["X"][self.quanti_var_["coord"].index.tolist()]
        # Add supplementary continuous variables
        if self.quanti_sup is not None:
            X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_sup_["coord"].index.tolist()].astype("float")
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
            quanti_data = pd.concat([quanti_data,X_quanti_sup],axis=1)

        # Select categorical active variables
        quali_data = self.call_["X"].drop(columns=quanti_data.columns.tolist())
        # Add supplementary categorical variables
        if self.quali_sup is not None:
            X_quali_sup = self.call_["Xtot"].loc[:,self.quali_sup_["eta2"].index.tolist()].astype("object")
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=[name for name in self.call_["Xtot"].index.tolist() if name in self.ind_sup_["coord"].index.tolist()])
            quali_data = pd.concat([quali_data,X_quali_sup],axis=1)
        
        # Correlation between coninuous variable and axis
        corrdim = {}
        for idx in ind_coord.columns:
            # For continuous variables
            corDim = pd.DataFrame(columns=["correlation","pvalue"]).astype("float")
            for col in quanti_data.columns.tolist():
                if (quanti_data[col].dtypes in ["float64","int64","float32","int32"]):
                    res = sp.stats.pearsonr(quanti_data[col],ind_coord[idx])
                    row_RD = pd.DataFrame({"correlation" : res.statistic,"pvalue":res.pvalue},index = [col])
                    corDim = pd.concat([corDim,row_RD],axis=0)
            # Filter by pvalue
            corDim = (corDim.query('pvalue < @proba').sort_values(by="correlation",ascending=False))

            # For categorical variable    
            corqDim = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','Eta2','F-stats','pvalue'])
            for col in quali_data.columns.tolist():
                row_RD = pd.DataFrame(eta2(quali_data[col],ind_coord[idx],digits=8),index=[col])
                corqDim = pd.concat([corqDim,row_RD],axis=0)
            # Filter by pvalue
            corqDim = (corqDim.query('pvalue < @proba').sort_values(by="Eta2",ascending=False).rename(columns={"Eta2" : "R2"}))

            if corDim.shape[0] == 0 and corqDim.shape[0] != 0:
                res = corqDim
            elif corDim.shape[0] != 0 and corqDim.shape[0] == 0:
                res = corDim
            else:
                res = {"quanti":corDim,"quali":corqDim}
              
            corrdim[idx] = res

    return corrdim