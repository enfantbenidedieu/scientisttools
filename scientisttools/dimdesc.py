# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
from .eta2 import eta2
import statsmodels.formula.api as smf

from .splitmix import splitmix
from .recodecont import recodecont
from .weightedcorrtest import weightedcorrtest


def dimdesc(self,axis=None,proba=0.05):
    """
    Dimension description
    ----------------------

    Description
    -----------
    This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    Usage
    -----
    ```
    >>> dimdesc(self,axis=None,proba=0.05)
    ```

    Parameters
    ----------
    `self` : an object of class PCA, CA, MCA, SpecificMCA, FAMD, PCAMIX, MPCA, MFA, MFAQUAL, MFAMIX

    `axis` : an integer or a list/tuple specifying the axis (by default = None)

    `axis` : the significance threshold considered to characterized the dimension (by default 0.05)

    Returns
    -------
    dictionary of dataframes including :

    `quanti` :  the description of the dimensions by the quantitative variables. The variables are sorted.
    
    `quali`	: the description of the dimensions by the categorical variables
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    """

    # Description for continuous variables
    def contdesc(data,coord,proba=0.05):
        """
        Continuous variables description
        --------------------------------

        Description
        -----------
        Description continuous by quantitative variables

        Parameters
        ----------
        data : pandas dataframe of continuous variables of shape (n_rows, n_columns)

        coord : individuals coordinates of shape (n_rows, 1)

        proba : the significance threshold considered to characterized the category (by default 0.05)

        Return
        ------
        value : pandas dataframe of shape (n_columns, 2)

        Author(s)
        ---------
        Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
        """
        # Fill NA with mean
        data = recodecont(data)["Xcod"]

        # For continuous variables
        value = pd.DataFrame(index=data.columns,columns=["correlation","pvalue"]).astype("float")
        for col in data.columns:
            res = sp.stats.pearsonr(data[col],coord)
            value.loc[col,:] = [res.statistic,res.pvalue]
        value = value.query('pvalue < @proba').sort_values(by="correlation",ascending=False)
        return value
    
    # Categories description
    def catdesc(data,coord,proba=0.05):
        """
        Categories description
        ----------------------

        Description
        -----------
        Description of the categories of one factor by categorical variables

        Parameters
        ----------
        data : pandas dataframe of continuous variables of shape (n_rows, n_columns)

        coord : individuals coordinates of shape (n_rows, 1)

        proba : the significance threshold considered to characterized the category (by default 0.05)

        Return
        ------
        value : pandas dataframe

        category : pandas dataframe

        Author(s)
        ---------
        Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
        """
        # Correlation ratio
        value = pd.DataFrame(columns=['Sum. Intra','Sum. Inter','Eta2','F-stats','pvalue'])
        for col in data.columns:
            row_RD = pd.DataFrame(eta2(data[col],coord,digits=8),index=[col])
            value = pd.concat([value,row_RD],axis=0)
        value = (value.query('pvalue < @proba').sort_values(by="Eta2",ascending=False).rename(columns={"R2": "Eta2"}))

        # OLS regression
        dummies = pd.concat((pd.get_dummies(data[col],prefix=col,prefix_sep="=",dtype=int) for col in data.columns),axis=1)
        category = pd.DataFrame(index=dummies.columns,columns=["Estimate","pvalue"]).astype("float")
        for col in dummies.columns:
            df = pd.concat((coord,dummies[col]),axis=1)
            df.columns = ["y","x"]
            res = smf.ols(formula="y~C(x)", data=df).fit()
            category.loc[col,:] = [res.params.values[1],res.pvalues.values[1]]
        category = category.query("pvalue < @proba")
        return value, category

    if self.model_ == "pca":
        # Active data
        data = self.call_["X"]
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()

        # Add supplementary continuous variables
        if self.quanti_sup is not None:
            X_quanti_sup = self.call_["Xtot"].loc[:,self.call_["quanti_sup"]].astype("float")
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
            data = pd.concat([data,X_quanti_sup],axis=1)
    
        corrdim = {}
        for idx in ind_coord.columns:
            res = {}
            quanti = contdesc(data=data,coord=ind_coord[idx],proba=proba)
            if quanti.shape[0] != 0:
                res["quanti"] = quanti

            if self.quali_sup is not None:
                quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
                if self.ind_sup is not None:
                    quali_sup = quali_sup.drop(index=self.call_["ind_sup"])
                quali, category =  catdesc(data=quali_sup,coord=ind_coord[idx],proba=proba)
                if quali.shape[0] != 0 :
                    res["quali"] = quali
                if category.shape[0] != 0:
                    res["category"] = category
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
    elif self.model_ in ["mca","specificmca"]:
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
            X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
            data = pd.concat([data,X_quali_sup],axis=1)

        corrdim = {}
        for idx in ind_coord.columns:
            res =  {}
            quali, category = catdesc(data=data,coord=ind_coord[idx],proba=proba)
            if quali.shape[0] != 0:
                res["quali"] = quali
            if category.shape[0] != 0:
                res["category"] = category
        
            if self.quanti_sup is not None:
                X_quanti_sup = self.call_["Xtot"].loc[:,self.call_["quanti_sup"]].astype("float")
                if self.ind_sup is not None:
                    X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
                quanti = contdesc(data=X_quanti_sup,coord=ind_coord[idx],proba=proba)
                if quanti.shape[0]!=0:
                    res["quanti"] = quanti
            corrdim[idx] = res
    elif self.model_ in ["famd","pcamix","mpca"]:
        # Extract row coord
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()
            
        # Extract active data
        X = self.call_["X"]
        # Split data
        X_quanti = splitmix(X)["quanti"]
        X_quali = splitmix(X)["quali"]

        # Create quanti dataset
        if X_quanti is None:
            quanti_data = pd.DataFrame().astype("float")
        else:
            quanti_data = X_quanti
        
        # Create quali dataset
        if X_quali is None:
            quali_data = pd.DataFrame().astype("object")
        else:
            quali_data = X_quali

        # Add supplementary continuous variables
        if self.quanti_sup is not None:
            X_quanti_sup = self.call_["Xtot"].loc[:,self.call_["quanti_sup"]].astype("float")
            if self.ind_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
            quanti_data = pd.concat([quanti_data,X_quanti_sup],axis=1)

        # Add supplementary categorical variables
        if self.quali_sup is not None:
            X_quali_sup = self.call_["Xtot"].loc[:,self.call_["quali_sup"]].astype("object")
            if self.ind_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
            quali_data = pd.concat([quali_data,X_quali_sup],axis=1)
        
        # Correlation between coninuous variable and axis
        corrdim = {}
        for idx in ind_coord.columns:
            res =  {}
            if quanti_data.shape[1] > 0:
                quanti = contdesc(data=quanti_data,coord=ind_coord[idx],proba=proba)
                if quanti.shape[0] != 0:
                    res["quanti"] = quanti
            
            if quali_data.shape[1] > 0:
                quali, category = catdesc(data=quali_data,coord=ind_coord[idx],proba=proba)
                if quali.shape[0] != 0:
                    res["quali"] = quali
                if category.shape[0] != 0:
                    res["category"] = category

            corrdim[idx] = res
    elif self.model_ == "mfa":
        # Select data
        data = self.call_["X"]
        # Extract individuals coordinates
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()
    
        if hasattr(self,"quanti_var_sup_"):
            X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_var_sup_["coord"].index].astype("float")
            if hasattr(self,"ind_sup_"):
                X_quanti_sup = X_quanti_sup.drop(index=self.ind_sup_["coord"].index)
            data = pd.concat([data,X_quanti_sup],axis=1)
        
        corrdim = {}
        for idx in ind_coord.columns:
            res = {}
            quanti = contdesc(data=data,coord=ind_coord[idx],proba=proba)
            if quanti.shape[0] != 0:
                res["quanti"] = quanti

            ##### Check if qualitatives variables in Data set
            if hasattr(self,"quali_var_sup_"):
                X_quali_sup = self.call_["Xtot"].loc[:,self.quali_var_sup_["eta2"].index].astype("object")
                if hasattr(self,"ind_sup_"):
                    X_quali_sup = X_quali_sup.drop(index=self.ind_sup_["coord"].index)
                # For qualitatives variables
                quali, category = catdesc(data=X_quali_sup,coord=ind_coord[idx],proba=proba)
                if quali.shape[0]:
                    res["quali"] = quali
                if category.shape[0]:
                    res["category"] = category
            corrdim[idx] = res
    elif self.model_ == "mfaqual":
        # Select data
        data = self.call_["X"]
        # Extract individuals coordinates
        ind_coord = self.ind_["coord"]
        # Select axis
        if axis is not None:
            ind_coord = ind_coord.iloc[:,axis]
            if isinstance(ind_coord,pd.Series):
                ind_coord = ind_coord.to_frame()
        
        if hasattr(self,"quali_var_sup_"):
            X_quali_sup = self.call_["Xtot"].loc[:,self.quali_var_sup_["eta2"].index].astype("object")
            if hasattr(self,"ind_sup_"):
                X_quali_sup = X_quali_sup.drop(index=self.ind_sup_["coord"].index)
            data = pd.concat((data,X_quali_sup),axis=1)
        
        corrdim = {}
        for idx in ind_coord.columns:
            res = {}
            quali, category = catdesc(data=data,coord=ind_coord[idx],proba=proba)
            if quali.shape[0]:
                res["quali"] = quali
            if category.shape[0]:
                res["category"] = category
            
            if hasattr(self,"quanti_var_sup_"):
                X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_var_sup_["coord"].index].astype("float")
                if hasattr(self,"ind_sup_"):
                    X_quanti_sup = X_quanti_sup.drop(index=self.ind_sup_["coord"].index)
                quanti = contdesc(data=X_quanti_sup,coord=ind_coord[idx],proba=proba)
                if quanti.shape[0] != 0:
                    res["quanti"] = quanti
            corrdim[idx] = res
    elif self.model_ == "mfamix":
        quali_data = splitmix(self.call_["X"])["quali"]
        quanti_data = splitmix(self.call_["X"])["quanti"]

        ind_coord = self.ind_["coord"]

        # Add supplementary quantitative variables
        if hasattr(self,"quanti_var_sup_"):
            X_quanti_sup = self.call_["Xtot"].loc[:,self.quanti_var_sup_["coord"].index].astype("float")
            if hasattr(self,"ind_sup_"):
                X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
            quanti_data = pd.concat((quanti_data,X_quanti_sup),axis=1)
        
        # Add supplementary qualitatives variables
        if hasattr(self,"quali_var_sup_"):
            X_quali_sup = self.call_["Xtot"].loc[:,self.quali_var_sup_["eta2"].index].astype("object")
            if hasattr(self,"ind_sup_"):
                X_quali_sup = X_quali_sup.drop(index=self.call_["ind_sup"])
            quali_data = pd.concat((quali_data,X_quali_sup),axis=1)
        
        corrdim = {}
        for idx in ind_coord.columns:
            res = {}
            quanti = contdesc(data=quanti_data,coord=ind_coord[idx],proba=proba)
            if quanti.shape[0] != 0:
                res["quanti"] = quanti
            quali, category = catdesc(data=quali_data,coord=ind_coord[idx],proba=proba)
            if quali.shape[0]:
                res["quali"] = quali
            if category.shape[0]:
                res["category"] = category
            corrdim[idx] = res
    else:
        raise TypeError("Factor method not allowed")
    # elif self.model_ == "mfact":
    #     quanti_data = self.call_["X"]
    #     ind_coord = self.ind_["coord"]
    #     ind_weights = self.call_["ind_weights"]

    #     if hasattr(self,"freq_sup_"):
    #         X_quanti_sup = self.call_["Xtot"].loc[:,self.freq_sup_["coord"].index].astype("float")
    #         if hasattr(self,"ind_sup_"):
    #             X_quanti_sup = X_quanti_sup.drop(index=self.call_["ind_sup"])
    #         quanti_data = pd.concat((quanti_data,X_quanti_sup),axis=1)
        
    #     corrdim = {}
    #     for idx in ind_coord.columns:
    #         quanti = pd.DataFrame(index=quanti_data.columns,columns=["correlation","pvalue"]).astype("float")
    #         for col in quanti_data.columns:
    #             res = weightedcorrtest(x=quanti_data[col],y=ind_coord[idx],weights=ind_weights)
    #             quanti.loc[col,:] = [res["statistic"],res["pvalue"]]
    #         #quanti = contdesc(data=quanti_data,coord=ind_coord[idx],proba=proba)
    #         print(quanti)
    #         corrdim[idx] = quanti
    return corrdim