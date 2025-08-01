# -*- coding: utf-8 -*-
from scipy.stats import chi2, contingency, chi2_contingency
from pandas import DataFrame, api, concat, crosstab
from collections import OrderedDict, namedtuple

def association(X:DataFrame,alpha=0.05):
    """
    Assocition between nominal variables
    ------------------------------------

    Description
    -----------
    Compute the degree of association between two nominales variables

    Usage
    -----
    ```
    >>> association(X,0.05)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of nominal variables

    `alpha`: a numeric value indicating the percent point

    Return
    ------
    a namedtuple of pandas DataFrame containing all the results for the association including:

    `chi2`: Pearson's chi-squared test
    
    `gtest`: log-likelihood ratio (i.e the "G-test")
    
    `association`: degree of association between two nominal variables ("cramer", "tschuprow", "pearson")

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    References
    ----------
    * "Contingency table", https://en.wikipedia.org/wiki/Contingency_table
    * "Pearson's chi-squared test", https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    * "Tschuprow's T", https://en.wikipedia.org/wiki/Tschuprow's_T
    * "Cramer's V", https://en.wikipedia.org/wiki/Cramer's_V
    * "Nominal Association: Phi and Cramer's V", http://www.people.vcu.edu/~pdattalo/702SuppRead/MeasAssoc/NominalAssoc.html
    * Gingrich, Paul, "Association Between Variables", http://uregina.ca/~gingrich/ch11a.pdf

    Examples
    --------
    ```
    >>> #load poison dataset
    >>> poison = load_poison()
    >>> from scientisttools import association
    >>> res = association(poison.iloc[:,2:],0.05)
    >>> res.association.head() #association
    >>> res.chi2.head() #pearson's chi-squared test
    >>> res.gtest.head() #log-likelihood ratio (G-test)
    ```
    """
    #check if X is an instance of pandas DataFrame
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #check if all columns are categoricals
    all_cat = all(api.types.is_string_dtype(X[q]) for q in X.columns)
    if not all_cat:
        raise TypeError("All columns in `X` must be categoricals")

    #conclusion on test
    def test_conclusion(pvalue,alpha=0.05):
        if pvalue <= alpha:
            return "Dependent (reject H0)"
        else:
            return "Independent (H0 holds true)"

    #Chi2 statistic test
    chi2_test = DataFrame(columns=["variable1","variable2","statistic","dof","value","pvalue","conclusion"])
    association, g_test = DataFrame(columns=["variable1","variable2","cramer","tschuprow","pearson"]), chi2_test.copy()
    idx = 0
    for i in range(X.shape[1]-1):
        for j in range(i+1,X.shape[1]):
            #pearson chi-squared test
            statistic, pvalue, dof, _ = chi2_contingency(crosstab(X.iloc[:,i],X.iloc[:,j]),lambda_=None,correction=False)
            row_chi2 = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],statistic=statistic,dof=dof,value=chi2.ppf(1-alpha,dof),pvalue=pvalue,conclusion=test_conclusion(pvalue,alpha)),index=[idx])
            chi2_test = concat((chi2_test,row_chi2),axis=0,ignore_index=True)
            #log-likelihood test (G-test)
            g_stat, g_pvalue, g_dof = chi2_contingency(crosstab(X.iloc[:,i],X.iloc[:,j]), lambda_="log-likelihood")[:3]
            row_gtest = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],statistic=g_stat,dof=g_dof,value=chi2.ppf(1-alpha,g_dof),pvalue=g_pvalue,conclusion=test_conclusion(g_pvalue,alpha)),index=[idx])
            g_test = concat((g_test,row_gtest),axis=0,ignore_index=True)
            #others association tests (cramer, tschuprow, pearson)
            asso_test = [contingency.association(crosstab(X.iloc[:,i],X.iloc[:,j]),method=k,correction=False) for k in ["cramer","tschuprow","pearson"]]
            row_asso = DataFrame(OrderedDict(variable1=X.columns[i],variable2=X.columns[j],cramer=asso_test[0],tschuprow=asso_test[1],pearson=asso_test[2]),index=[idx])
            association = concat((association,row_asso),axis=0,ignore_index=True)
            idx += 1
    #transform to int
    chi2_test["dof"], g_test["dof"] = chi2_test["dof"].astype("int"), g_test["dof"].astype("int")
    #convert to namedtuple
    return namedtuple("association",["association","chi2","gtest"])(association,chi2_test,g_test)