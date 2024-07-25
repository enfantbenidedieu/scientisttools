# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from mapply.mapply import mapply
from sklearn.base import BaseEstimator, TransformerMixin

from .svd_triplet import svd_triplet
from .weightedcorrcoef import weightedcorrcoef
from .function_eta2 import function_eta2
from .recodecont import recodecont
from .revaluate_cat_variable import revaluate_cat_variable

class CA(BaseEstimator,TransformerMixin):
    """
    Correspondence Analysis (CA)
    ----------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Correspondence Analysis (CA) including supplementary row and/or column points, supplementary quantitative variables and supplementary categorical variables.

    Usage
    -----
    ```python
    CA(n_components = 5, row_weights = None, row_sup = None, col_sup = None, quanti_sup = None, quali_sup = None, parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 5)

    `row_weights` : an optional row weights (by default, a list/tuple of 1 and each row has a weight equals to its margin); the weights are given only for the active rows

    `row_sup` : list/tuple indicating the indexes of the supplementary rows

    `col_sup` : list/tuple indicating the indexes of the supplementary columns

    `quanti_sup` : list/tuple indicating the indexes of the supplementary continuous variables

    `quali_sup` : list/tuple indicating the indexes of the categorical supplementary variables

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using pandas apply

    Attributes
    ----------
    `eig_`  : pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    `svd_` : dictionary of matrices containing all the results of the singular value decomposition

    `col_` : dictionary of pandas dataframe with all the results for the column variable (coordinates, square cosine, contributions, inertia)

    `row_` : dictionary of pandas dataframe with all the results for the row variable (coordinates, square cosine, contributions, inertia)

    `col_sup_` : dictionary of pandas dataframe containing all the results for the supplementary column points (coordinates, square cosine)

    `row_sup_` : dictionary of pandas dataframe containing all the results for the supplementary row points (coordinates, square cosine)

    `quanti_sup_` : if quanti_sup is not None, a dictionary of pandas dataframe containing the results for the supplementary continuous variables (coordinates, square cosine)

    `quali_sup_` : if quali.sup is not None, a dictionary of pandas dataframe with all the results for the supplementary categorical variables (coordinates of each categories of each variables, v.test which is a criterion with a Normal distribution, square correlation ratio)

    `summary_quanti_` : summary statistics for quantitative variables (actives and supplementary)

    `summary_quali_` : summary statistics for supplementary qualitative variables if quali_sup is not None

    `chi2_test_` : chi-squared test. If supplementary qualitative variables are greater than 2. 
    
    `call_` : dictionary with some statistics

    `model_` : string specifying the model fitted = 'ca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    Escofier B, Pagès J (2023), Analyses Factorielles Simples et Multiples. 5ed, Dunod

    Husson, F., Le, S. and Pages, J. (2009). Analyse de donnees avec R, Presses Universitaires de Rennes.

    Husson, F., Le, S. and Pages, J. (2010). Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.

    Lebart L., Piron M., & Morineau A. (2006). Statistique exploratoire multidimensionnelle. Dunod, Paris 4ed.

    Pagès J. (2013). Analyse factorielle multiple avec R : Pratique R. EDP sciences

    Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Université Lumière Lyon 2, Version 1.0

    See Also
    --------
    get_ca_row, get_ca_col, get_ca, summaryCA, dimdesc, predictCA, supvarCA, fviz_ca_row, fviz_ca_col, fviz_ca_biplot

    Examples
    --------
    ```python
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> from scientisttools import CA, summaryCA
    >>> res_ca = CA(row_sup=list(range(14,18)),col_sup=list(range(5,8)),parallelize=True)
    >>> res_ca.fit(children)
    >>> summaryCA(res_ca)
    ```
    """
    def __init__(self,
                 n_components = 5,
                 row_weights = None,
                 row_sup = None,
                 col_sup = None,
                 quanti_sup = None,
                 quali_sup = None,
                 parallelize = False):
        self.n_components = n_components
        self.row_weights = row_weights
        self.row_sup = row_sup
        self.col_sup = col_sup
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the model to X
        ------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_rows, n_columns)
            Training data, where `n_rows` in the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y` : None
            y is ignored.
        
        Returns
        -------
        `self` : object
            Returns the instance itself.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Drop level if ndim greater than 1 and reset columns names
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Checks if categoricals variables in X and transform to factor (category)
        is_quali = X.select_dtypes(include=["object","category"])
        if is_quali.shape[1]>0:
            for col in is_quali.columns:
                X[col] = X[col].astype("object")
        
        # Set supplementary rows labels
        if self.row_sup is not None:
            if (isinstance(self.row_sup,int) or isinstance(self.row_sup,float)):
                row_sup = [int(self.row_sup)]
            elif (isinstance(self.row_sup,list) or isinstance(self.row_sup,tuple)) and len(self.row_sup) >=1:
                row_sup = [int(x) for x in self.row_sup]
            row_sup_label = X.index[row_sup]
        else:
            row_sup_label = None
        
        ##############################################################################################
        # Set supplementary columns labels
        ##############################################################################################
        if self.col_sup is not None:
            if (isinstance(self.col_sup,int) or isinstance(self.col_sup,float)):
                col_sup = [int(self.col_sup)]
            elif (isinstance(self.col_sup,list) or isinstance(self.col_sup,tuple)) and len(self.col_sup) >=1:
                col_sup = [int(x) for x in self.col_sup]
            col_sup_label = X.columns[col_sup]
        else:
            col_sup_label = None
        
        ##############################################################################################
        # Set supplementary quantitatives variables labels
        ##############################################################################################
        if self.quanti_sup is not None:
            if (isinstance(self.quanti_sup,int) or isinstance(self.quanti_sup,float)):
                quanti_sup = [int(self.quanti_sup)]
            elif (isinstance(self.quanti_sup,list) or isinstance(self.quanti_sup,tuple)) and len(self.quanti_sup) >=1:
                quanti_sup = [int(x) for x in self.quanti_sup]
            quanti_sup_label = X.columns[quanti_sup]
        else:
            quanti_sup_label = None
        
        ##############################################################################################
        # Set supplementary qualitatives variables labels
        ##############################################################################################
        if self.quali_sup is not None:
            if (isinstance(self.quali_sup,int) or isinstance(self.quali_sup,float)):
                quali_sup = [int(self.quali_sup)]
            elif (isinstance(self.quali_sup,list) or isinstance(self.quali_sup,tuple)) and len(self.quali_sup) >=1:
                quali_sup = [int(x) for x in self.quali_sup]
            quali_sup_label = X.columns[quali_sup]
        else:
            quali_sup_label = None

        #####################################################################################################
        # Store data - Save the base in a variables
        #####################################################################################################
        Xtot = X.copy()
        
        ################################# Drop supplementary columns #############################################
        if self.col_sup is not None:
            X = X.drop(columns=col_sup_label)
        
        ################################# Drop supplementary quantitatives variables ###############################
        if self.quanti_sup is not None:
            X = X.drop(columns=quanti_sup_label)
        
         ################################# Drop supplementary qualitatives variables ###############################
        if self.quali_sup is not None:
            X = X.drop(columns=quali_sup_label)
        
        ################################## Drop supplementary rows ##################################################
        if self.row_sup is not None:
            # Extract supplementary rows
            X_row_sup = X.loc[row_sup_label,:]
            X = X.drop(index=row_sup_label)

        ################################## Start Compute Correspondence Analysis (CA) ###############################
        # Active data
        X = X.astype("int")

        ##### Set row weights
        if self.row_weights is None:
            row_weights = np.ones(X.shape[0])
        elif not isinstance(self.row_weights,list):
            raise TypeError("'row_weights' must be a list of row weight")
        elif len(self.row_weights) != X.shape[0]:
            raise TypeError(f"'row_weights' must be a list with length {X.shape[0]}.")
        
        # Set number of components
        if self.n_components is None:
            n_components = min(X.shape[0]-1,X.shape[1]-1)
        elif isinstance(self.n_components,float):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components <= 0:
            raise TypeError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,X.shape[0]-1,X.shape[1]-1)

        ####################################################################################################################
        ####### total
        total = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0).sum()

        ##### Table des frequences
        freq = mapply(X,lambda x : x*(row_weights/total),axis=0,progressbar=False,n_workers=n_workers)
        
        ####### Calcul des marges lignes et colones
        col_marge = freq.sum(axis=0)
        col_marge.name = "col_marge"
        row_marge = freq.sum(axis=1)
        row_marge.name = "row_marge"

        ###### Compute Matrix used in SVD
        Z = mapply(freq,lambda x : x/row_marge,axis=0,progressbar=False,n_workers=n_workers)
        Z = mapply(Z,lambda x : (x/col_marge)-1,axis=1,progressbar=False,n_workers=n_workers)

        ## Rows informations : Margin, Weight, square distance to origin, Inertia
        # Row square distance to origin
        row_dist2 = mapply(Z,lambda x : (x**2)*col_marge,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        # Row inertia
        row_inertia = row_marge*row_dist2
        # add all informations
        row_infos = np.c_[row_weights,row_marge,row_dist2,row_inertia]
        row_infos = pd.DataFrame(row_infos,columns=["Weight","Margin","Sq. Dist.","Inertia"],index=X.index.tolist())

        ## Columns informations : Margin, square disstance to origin and inertia
        # Columns square distance to origin 
        col_dist2 = mapply(Z,lambda x : (x**2)*row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        # Columns inertia
        col_inertia = col_marge*col_dist2
        # Add all informations
        col_infos = np.c_[col_marge,col_dist2,col_inertia]
        col_infos = pd.DataFrame(col_infos,columns=["Margin","Sq. Dist.","Inertia"],index=X.columns)

        ###### Store call informations
        self.call_ = {"X" : X,
                      "Xtot" : Xtot ,
                      "Z" : Z ,
                      "row_weights" : pd.Series(row_weights,index=X.index,name="weight"),
                      "col_marge" : col_marge,
                      "row_marge" : row_marge,
                      "n_components":n_components,
                      "row_sup" : row_sup_label,
                      "col_sup" : col_sup_label,
                      "quanti_sup" : quanti_sup_label,
                      "quali_sup" : quali_sup_label}
        
        # Generalized Singular Value Decomposition (GSVD)
        svd = svd_triplet(X=Z,row_weights=row_marge,col_weights=col_marge,n_components=n_components)
        self.svd_ = svd

        # Eigenvalues
        eigen_values = svd["vs"][:min(X.shape[0]-1,X.shape[1]-1)]**2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        eig = np.c_[eigen_values,difference,proportion,cumulative]
        self.eig_ = pd.DataFrame(eig,columns =["eigenvalue","difference","proportion","cumulative"],index=["Dim."+str(x+1) for x in range(eig.shape[0])])

        ## Row informations : coordinates, contributions and square cosine
        # Row coordinates
        row_coord = svd["U"].dot(np.diag(svd["vs"][:n_components]))
        row_coord = pd.DataFrame(row_coord,index=X.index.tolist(),columns=["Dim."+str(x+1) for x in range(n_components)])

        # Row contributions
        row_contrib = mapply(row_coord,lambda x: 100*(x**2)*row_marge,axis=0,progressbar=False,n_workers=n_workers)
        row_contrib = mapply(row_contrib,lambda x : x/eigen_values[:n_components], axis=1,progressbar=False,n_workers=n_workers)

        # Row square cosines (Cos2)
        row_cos2 = mapply(row_coord,lambda x: (x**2)/row_dist2.values,axis=0,progressbar=False,n_workers=n_workers)

        # Store all informations
        self.row_ = {"coord" : row_coord, "contrib" : row_contrib, "cos2" : row_cos2, "infos" : row_infos}

        ## Columns informations : coordinates, contributions and square cosinus
        #### Columns coordinates
        col_coord = svd["V"].dot(np.diag(svd["vs"][:n_components]))
        col_coord = pd.DataFrame(col_coord,index=X.columns.tolist(),columns=["Dim."+str(x+1) for x in range(n_components)])

        # Columns contributions
        col_contrib = mapply(col_coord,lambda x: 100*(x**2)*col_marge,axis=0,progressbar=False,n_workers=n_workers)
        col_contrib = mapply(col_contrib,lambda x : x/eigen_values[:n_components], axis=1,progressbar=False,n_workers=n_workers)
        
        # Columns square cosinus (Cos2)
        col_cos2 = mapply(col_coord,lambda x: (x**2)/col_dist2.values, axis = 0,progressbar=False,n_workers=n_workers)

        # Store all informations
        self.col_ = {"coord" : col_coord, "contrib" : col_contrib, "cos2" : col_cos2, "infos" : col_infos}
        
        ############################################################################################################
        #  Compute others indicators 
        #############################################################################################################
        # Weighted X with the row weight
        weighted_X = mapply(X,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers)

        # Compute chi - squared test
        statistic,pvalue,dof, expected_freq = sp.stats.chi2_contingency(weighted_X, lambda_=None,correction=False)

        # log - likelihood - tes (G - test)
        g_test_res = sp.stats.chi2_contingency(weighted_X, lambda_="log-likelihood")

        # Absolute residuals
        resid = weighted_X - expected_freq

        # Standardized resid
        standardized_resid = resid/np.sqrt(expected_freq)

        # Adjusted residuals
        adjusted_resid = mapply(standardized_resid,lambda x : x/np.sqrt(1-row_marge),axis=0,progressbar=False,n_workers=n_workers)
        adjusted_resid = mapply(adjusted_resid,lambda x : x/np.sqrt(1-col_marge),axis=1,progressbar=False,n_workers=n_workers)

        ##### Chi2 contribution
        chi2_contribution = mapply(standardized_resid,lambda x : 100*(x**2)/statistic,axis=0,progressbar=False,n_workers=n_workers)

        # Attraction repulsio,
        attraction_repulsion_index = weighted_X/expected_freq

        # Return indicators
        chi2_test = pd.DataFrame([statistic,dof,pvalue],columns=["value"],index=["statistic","dof","pvalue"])
        
        # log-likelihood test
        log_likelihood_test = pd.DataFrame([g_test_res[0],g_test_res[1]],columns=["value"],index=["statistic","pvalue"],)

        # Association test
        association = pd.DataFrame([sp.stats.contingency.association(X, method=name) for name in ["cramer","tschuprow","pearson"]],
                                   columns=["statistic"],index=["cramer","tschuprow","pearson"])
    
        # Inertia
        inertia = np.sum(row_inertia)
        kaiser_threshold = np.mean(eigen_values)
        kaiser_proportion_threshold = 100/min(X.shape[0]-1,X.shape[1]-1)
       
       # Others informations
        self.others_ = {"res" : resid,
                        "chi2" : chi2_test,
                        "g_test" : log_likelihood_test,
                        "adj_resid" : adjusted_resid,
                        "chi2_contrib" : chi2_contribution,
                        "std_resid" : standardized_resid,
                        "attraction" : attraction_repulsion_index,
                        "association" : association,
                        "inertia" : inertia,
                        "kaiser_threshold" : kaiser_threshold,
                        "kaiser_proportion_threshold" : kaiser_proportion_threshold}

        # Compute supplementary rows
        if self.row_sup is not None:
            #######
            X_row_sup = X_row_sup.astype("int")
            # Sum row
            row_sum = X_row_sup.sum(axis=1)

            # Standardize with the row sum
            X_row_sup = mapply(X_row_sup,lambda x : x/row_sum,axis=0,progressbar=False,n_workers=n_workers)

            # Row Supplementary coordinates
            row_sup_coord = X_row_sup.dot(svd["V"][:,:n_components])
            row_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
            row_sup_coord.index = X_row_sup.index.tolist()

            # Supplementary rows square distance to origin
            row_sup_dist2 = mapply(X_row_sup,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            row_sup_dist2.name = "Sq. Dist."

            # Supplementary rows square cosine
            row_sup_cos2 = mapply(row_sup_coord,lambda x : (x**2)/row_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Set informations
            self.row_sup_ = {"coord" : row_sup_coord,"cos2" : row_sup_cos2, "dist" : row_sup_dist2}

        # Compute supplementary columns
        if self.col_sup is not None:
            X_col_sup = Xtot.loc[:,col_sup_label]
            if self.row_sup is not None:
                X_col_sup = X_col_sup.drop(index=row_sup_label)
            
            # Transform to int
            X_col_sup = X_col_sup.astype("int")
            ### weighted with row weight
            X_col_sup = mapply(X_col_sup,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers)

            # Compute columns sum
            col_sum = X_col_sup.sum(axis=0)
            X_col_sup = mapply(X_col_sup,lambda x : x/col_sum,axis=1,progressbar=False,n_workers=n_workers)

            # Supplementary columns coordinates
            col_sup_coord = X_col_sup.T.dot(svd["U"][:,:n_components])
            col_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Supplementary columns square distance to origin
            col_sup_dist2 = mapply(X_col_sup,lambda x : ((x - row_marge)**2)/row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
            col_sup_dist2.name = "Sq. Dist."

            # Supplementary columns square cosine
            col_sup_cos2 = mapply(col_sup_coord,lambda x : (x**2)/col_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Store all informations
            self.col_sup_ = {"coord" : col_sup_coord,"cos2" : col_sup_cos2,"dist" : col_sup_dist2}
        
        #################################################################################################################################
        # Compute supplementary continues variables
        #################################################################################################################################

        if self.quanti_sup is not None:
            X_quanti_sup = Xtot.loc[:,quanti_sup_label]
            if self.row_sup is not None:
                X_quanti_sup = X_quanti_sup.drop(index=row_sup_label)
            
            ##### From frame to DataFrame
            if isinstance(X_quanti_sup,pd.Series):
                X_quanti_sup = X_quanti_sup.to_frame()
            
            ################ Transform to float
            X_quanti_sup = X_quanti_sup.astype("float")
            X_quanti_sup = recodecont(X_quanti_sup)

            ##################### Compute statistics
            summary_quanti_sup = X_quanti_sup.describe().T.reset_index().rename(columns={"index" : "variable"})
            summary_quanti_sup["count"] = summary_quanti_sup["count"].astype("int")
            self.summary_quanti_ = summary_quanti_sup
                
            ############# Compute average mean
            quanti_sup_coord = weightedcorrcoef(x=X_quanti_sup,y=row_coord,w=row_marge)[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
            quanti_sup_coord = pd.DataFrame(quanti_sup_coord,index=X_quanti_sup.columns.tolist(),columns=["Dim."+str(x+1) for x in range(quanti_sup_coord.shape[1])])
            
            #################### Compute cos2
            quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers)

            # Set all informations
            self.quanti_sup_ = {"coord" : quanti_sup_coord, "cos2" : quanti_sup_cos2}
        
        # Compute supplementary qualitatives informations
        if self.quali_sup is not None:
            X_quali_sup = Xtot.loc[:,quali_sup_label]
            if self.row_sup is not None:
                X_quali_sup = X_quali_sup.drop(index=row_sup_label)
            
            ############### From Frame to DataFrame
            if isinstance(X_quali_sup,pd.Series):
                X_quali_sup = X_quali_sup.to_frame()
            
             ########### Set all elements as objects
            X_quali_sup = X_quali_sup.astype("object")
            X_quali_sup = revaluate_cat_variable(X_quali_sup)

            # Compute statistiques summary_quali
            summary_quali = pd.DataFrame()
            for col in X.columns.tolist():
                eff = X[col].value_counts().to_frame("count").reset_index().rename(columns={"index" : "categorie"})
                eff.insert(0,"variable",col)
                summary_quali = pd.concat([summary_quali,eff],axis=0,ignore_index=True)
            summary_quali["count"] = summary_quali["count"].astype("int")
            self.summary_quali_ = summary_quali

            # Sum of columns by group
            quali_sup = pd.DataFrame().astype("float")
            for col in X_quali_sup.columns.tolist():
                data = pd.concat((X,X_quali_sup[col]),axis=1).groupby(by=col,as_index=True).sum()
                data.index.name = None
                quali_sup = pd.concat((quali_sup,data),axis=0)
    
            # Calculate sum by row
            quali_sum = quali_sup.sum(axis=1)
            # Devide by sum
            quali_sup = mapply(quali_sup,lambda x : x/quali_sum,axis=0,progressbar=False,n_workers=n_workers)

            # Supplementary Categories coordinates
            quali_sup_coord = quali_sup.dot(svd["V"][:,:n_components])
            quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

            # Supplementary Categories square distance to origin
            quali_sup_dist2 = mapply(quali_sup,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
            quali_sup_dist2.name="Sq. Dist."

            # Supplementary categories square cosine
            quali_sup_cos2 = mapply(quali_sup_coord,lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

            # Disjonctif table
            dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in X_quali_sup.columns),axis=1)
            # Compute : weighted count by categories
            n_k = mapply(dummies,lambda x : x*row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)*total

            # Supplementary categories value-test 
            if total > 1:
                coef = np.array([np.sqrt(n_k[i]*((total - 1)/(total - n_k[i]))) for i in range(len(n_k))])
            else:
                coef = np.sqrt(n_k)
            quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*coef,axis=0,progressbar=False,n_workers=n_workers)

            # Supplementary categories correlation ratio
            quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=row_coord.values,weights=row_marge,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)
            
            # Store all informations
            self.quali_sup_ = {"coord" : quali_sup_coord,"cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"eta2" : quali_sup_eta2,"dist" : quali_sup_dist2}
        
        self.model_ = "ca"

        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X
        ----------------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_rows, n_columns)
            Training data, where `n_rows` is the number of rows and `n_columns` is the number of columns.
            X is a contingency table containing absolute frequencies.

        `y` : None.
            y is ignored.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_rows, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.row_["coord"]

    def transform(self,X):
        """
        Apply the dimensionality reduction on X
        ---------------------------------------

        Description
        -----------
        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_rows, n_columns)
            New data, where `n_rows` is the number of row points and `n_columns` is the number of columns

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_rows, n_components)
            Projection of X in the principal components where `n_rows` is the number of rows and `n_components` is the number of the components.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # Checif if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        # Extract number of components
        n_components = self.call_["n_components"]

        # Set type to int
        X = X.astype("int")
        row_sum = X.sum(axis=1)
        coord = mapply(X,lambda x : x/row_sum,axis=0,progressbar=False,n_workers=n_workers).dot(self.svd_["V"][:,:n_components])
        coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
        coord.index = X.index.tolist()
        return coord

def predictCA(self,X):
    """
    Predict projection for new rows with Correspondence Analysis (CA)
    -----------------------------------------------------------------

    Description
    -----------
    Performs the coordinates, square cosinus and square distance to origin of new rows with Correspondence Analysis

    Usage
    -----
    ```python
    >>> predictCA(self,X)
    ```

    Parameters
    ----------
    `self` : an object of class CA

    `X` : pandas/polars dataframe in which to look for columns with which to predict. X must contain columns with the same names as the original data

    Return
    ------
    dictionary of dataframes including :

    `coord` : factor coordinates (scores) for supplementary rows

    `cos2` : square cosinus for supplementary rows

    `dist` : square distance to origin for supplementary rows

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> # Actifs elements
    >>> actif = children.iloc[:14,:5]
    >>> # Supplementary rows
    >>> row_sup = children.iloc[14:,:5]
    >>> # Correspondence Analysis (CA)
    >>> from scientisttools import CA
    >>> res_ca = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)))
    >>> res_ca.fit(children)
    >>> # Prediction on supplementary rows
    >>> from scientisttools import predictCA
    >>> predict = predictCA(res_ca, X=row_sup)
    ```
    """
    # Check if self is an object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()

    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # Set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    n_components = self.call_["n_components"]
    col_marge = self.call_["col_marge"]

    # Transform to int
    X = X.astype("int")
    # Sum row
    row_sum = X.sum(axis=1)

    # Correction with row sum
    X = mapply(X,lambda x : x/row_sum,axis=0,progressbar=False,n_workers=n_workers)

    # Supplementary coordinates
    coord = X.dot(self.svd_["V"][:,:n_components])
    coord.columns = ["Dim."+str(x+1) for x in range(n_components)]
    
    # Supplementary square distance to origin
    dist2 = mapply(X,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
    dist2.name = "Sq. Dist."
    
    # Supplementary square cosinus
    cos2 = mapply(coord,lambda x : (x**2)/dist2,axis=0,progressbar=False,n_workers=n_workers)

    # Store all informations
    res = {"coord" : coord, "cos2" : cos2, "dist" : dist2}
    return res

def supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None):
    """
    Supplementary columns/variables with Correspondence Analysis (CA)
    -----------------------------------------------------------------

    Description
    -----------
    Performns the coordinates, square cosinus and square distance to origin of supplementary columns/variables with Correspondence Analysis (CA)

    Usage
    -----
    ```python
    >>> supvarCA(self,X_col_sup=None,X_quanti_sup=None, X_quali_sup=None)   
    ```

    Parameters
    ----------
    `self` : an object of class CA

    `X_col_sup` : pandas/polars dataframe of supplementary columns

    `X_quanti_sup` : pandas/polars dataframe of supplementary quantitatives columns

    `X_quali_sup` : pandas/polars dataframe of supplementary qualitatives columns

    Returns
    -------
    dictionary including : 

    `col` : dictionary containing the results of the supplementary columns variables:
        * coord : factor coordinates (scores) of the supplementary columns
        * cos2 : square cosinus of the supplementary columns
        * dist : distance to origin of the supplementary columns

    `quanti` : dictionary containing the results of the supplementary quantitatives variables:
        * coord : factor coordinates (scores) of the supplementary quantitativaes variables
        * cos2 : square cosinus of the supplementary quantitatives variables
    
    `quali` : dictionary containing the results of the supplementary qualitatives/categories variables :
        * coord : factor coordinates (scores) of the supplementary categories
        * cos2 : square cosinus of the supplementary categories
        * vtest : value-test of the supplementary categories
        * dist : square distance to origin of the supplementary categories
        * eta2 : square correlation ratio of the supplementary qualitatives variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> # Add qualitatives variables
    >>> children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +["D"]*4
    >>> # Supplementary columns
    >>> X_col_sup = children.iloc[:14,5:8]
    >>> # Supplementary qualitatives variables
    >>> X_quali_sup = children.iloc[:14,8]
    >>> from scientisttools import CA
    >>> res_ca = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)),quali_sup=8)
    >>> res_ca.fit(children)
    >>> # Supplementary columns/variables projections
    >>> from scientisttools import supvarCA
    >>> supvar = supvarCA(res_ca,X_col_sup=X_col_sup,X_quanti_sup=X_col_sup,X_quali_sup=X_quali_sup)
    >>> # Extract supplementary columns results
    >>> supvarcol = supvar["col"]
    >>> # Extract supplementary quantitatives variables results
    >>> supvarquanti = supvar["quanti"]
    >>> # Extract supplementary qualitatives variables results
    >>> supvarquali = supvar["quali"]
    ```
    """
    # Check if self is and object of class CA
    if self.model_ != "ca":
        raise TypeError("'self' must be an object of class CA")
    
    # set parallelize
    if self.parallelize:
        n_workers = -1
    else:
        n_workers = 1
    
    # Extract elements
    row_weights = self.call_["row_weights"] # row weights
    n_components = self.call_["n_components"] # number of components
    row_marge = self.call_["row_marge"] # row marge
    row_coord = self.row_["coord"] # row coordinates
    
    ## For supplementary columns
    if X_col_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_col_sup,pl.DataFrame):
            X_col_sup = X_col_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_col_sup,pd.Series):
            X_col_sup = X_col_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_col_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_col_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to int
        X_col_sup = X_col_sup.astype("int")

        ### weighted with row weight
        X_col_sup = mapply(X_col_sup,lambda x : x*row_weights,axis=0,progressbar=False,n_workers=n_workers)
        # Compute columns sum
        col_sum = X_col_sup.sum(axis=0)
        X_col_sup = mapply(X_col_sup,lambda x : x/col_sum,axis=1,progressbar=False,n_workers=n_workers)

        # Supplementary columns coordinates
        col_sup_coord = X_col_sup.T.dot(self.svd_["U"][:,:n_components])
        col_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Supplementary distance to origin
        col_sup_dist2 = mapply(X_col_sup,lambda x : ((x - row_marge)**2)/row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        col_sup_dist2.name = "Sq. Dist."

        # Supplementary squared cosinus
        col_sup_cos2 = mapply(col_sup_coord,lambda x : (x**2)/col_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)
        
        # Store all informations
        col_sup = {"coord" : col_sup_coord,"cos2" : col_sup_cos2,"dist" : col_sup_dist2}
    else:
        col_sup = None

    ########################################################################################################################
    #                                          For supplementary quantitatives variables
    #########################################################################################################################

    if X_quanti_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_quanti_sup,pl.DataFrame):
            X_quanti_sup = X_quanti_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quanti_sup,pd.Series):
            X_quanti_sup = X_quanti_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quanti_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quanti_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to float
        X_quanti_sup = X_quanti_sup.astype("float")

        # Recode variables
        X_quanti_sup = recodecont(X_quanti_sup)["Xcod"]
     
        ############# Compute average mean
        quanti_sup_coord = weightedcorrcoef(x=X_quanti_sup,y=row_coord,w=row_marge)[:X_quanti_sup.shape[1],X_quanti_sup.shape[1]:]
        quanti_sup_coord = pd.DataFrame(quanti_sup_coord,index=X_quanti_sup.columns,columns=["Dim."+str(x+1) for x in range(n_components)])
        
        #################### Compute cos2
        quanti_sup_cos2 = mapply(quanti_sup_coord,lambda x : (x**2),axis=0,progressbar=False,n_workers=n_workers)

        # Set all informations
        quanti_sup = {"coord" : quanti_sup_coord, "cos2" : quanti_sup_cos2}
    else:
        quanti_sup = None
    
    ###########################################################################################################################
    #                                                   For supplementary qualitatives variables
    ###########################################################################################################################

    if X_quali_sup is not None:
        # check if X is an instance of polars dataframe
        if isinstance(X_quali_sup,pl.DataFrame):
            X_quali_sup = X_quali_sup.to_pandas()
        
        # If pandas series, transform to pandas dataframe
        if isinstance(X_quali_sup,pd.Series):
            X_quali_sup = X_quali_sup.to_frame()
        
        # Check if X is an instance of pd.DataFrame class
        if not isinstance(X_quali_sup,pd.DataFrame):
            raise TypeError(
            f"{type(X_quali_sup)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Transform to object
        X_quali_sup = X_quali_sup.astype("object")

        # Extract active data
        X = self.call_["X"]
        col_marge = self.call_["col_marge"]
        total = X.sum().sum()

        # Check if two columns have the same categories
        X_quali_sup = revaluate_cat_variable(X_quali_sup)

        # Sum of columns by group
        quali = pd.DataFrame().astype("float")
        for col in X_quali_sup.columns.tolist():
            data = pd.concat((X,X_quali_sup[col]),axis=1).groupby(by=col,as_index=True).sum()
            data.index.name = None
            quali = pd.concat((quali,data),axis=0)
        ############################################################################################
        # Calculate sum by row
        quali_sum = quali.sum(axis=1)
        # Devide by sum
        quali = mapply(quali,lambda x : x/quali_sum,axis=0,progressbar=False,n_workers=n_workers)

        # Categories coordinates
        quali_sup_coord = quali.dot(self.svd_["V"][:,:n_components])
        quali_sup_coord.columns = ["Dim."+str(x+1) for x in range(n_components)]

        # Categories dist2
        quali_sup_dist2 = mapply(quali,lambda x : ((x - col_marge.values)**2)/col_marge.values,axis=1,progressbar=False,n_workers=n_workers).sum(axis=1)
        quali_sup_dist2.name="Sq. Dist."

        # Sup Cos2
        quali_sup_cos2 = mapply(quali_sup_coord,lambda x : (x**2)/quali_sup_dist2,axis=0,progressbar=False,n_workers=n_workers)

        # Disjonctif table
        dummies = pd.concat((pd.get_dummies(X_quali_sup[col],dtype=int) for col in X_quali_sup.columns),axis=1)
        # Compute : weighted count by categories
        n_k = mapply(dummies,lambda x : x*row_marge,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)*total

        ######## Weighted of coordinates to have 
        if total > 1:
            coef = np.array([np.sqrt(n_k[i]*((total - 1)/(total - n_k[i]))) for i in range(len(n_k))])
        else:
            coef = np.sqrt(n_k)
        # Value - test
        quali_sup_vtest = mapply(quali_sup_coord,lambda x : x*coef,axis=0,progressbar=False,n_workers=n_workers)

        # Correlation ratio
        quali_sup_eta2 = pd.concat((function_eta2(X=X_quali_sup,lab=col,x=row_coord.values,weights=row_marge,n_workers=n_workers) for col in X_quali_sup.columns),axis=0)
        
        # Set all informations
        quali_sup = {"coord" : quali_sup_coord,"cos2" : quali_sup_cos2,"vtest" : quali_sup_vtest,"eta2" : quali_sup_eta2,"dist" : quali_sup_dist2}
    else:
        quali_sup = None
    
    # Store all informations
    res = {"col" : col_sup, "quanti" : quanti_sup, "quali" : quali_sup}
    return res