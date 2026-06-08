# -*- coding: utf-8 -*-
from pandas import DataFrame

def method_desc(
        name
):
    """
    Method description

    Parameters
    ----------
    obj: class
        A fitted model.

    Returns
    -------
    long description of the model.
    """
    match name:
        #one table methods
        case "CA": return "Correspondence Analysis"
        case "FA": return "Factor Analysis"
        case "FAMD": return "Factor Analysis of Mixed Data"
        case "FAMDrot": return "Varimax rotation in Factor Analysis of Mixed Data"
        case "FArot": return "Varimax rotation in Factor Analysis"
        case "MCA": return "Multiple Correspondence Analysis"
        case "MCArot": return "Varimax rotation in Multiple Correspondence Analysis"
        case "MPCA": return "Mixed Principal Component Analysis"
        case "PCA": return "Principal Component Analysis"
        case "PCAmix": return "Principal Component Analysis of Mixed Data"
        case "PCAmixrot": return "Varimax rotation in Principal Component Analysis of Mixed Data"
        case "PCArot": return "Varimax rotation in Principal Component Analysis"
        case "PCoA": return "Principal Coordinates Analysis"
        #two tables methods
        case "BWCA": return "Between-Class/Within-Class Analysis"
        case "CANCORR": return "Canonical Correlation Analysis"
        case "CCA": return "Canonical Correspondence Analysis"
        case "COIA": return "CO-inertia Analysis"
        case "PCAiv": return "Principal Component Analysis with (orthogonal) instrumental variables"
        case "PCOA": return "Procustean CO-inertia Analysis"
        case "Procrustes": return "Procrustean Analysis"
        #multi tables methods
        case "BGC": return "Between Group Comparison"
        case "cGCANCORR": return "Carroll's Generalized Canonical Correlation Analysis"
        case "DCCSWA": return "Dual Common Component and Specific Weights Analysis"
        case "DISTATIS": return "Analyse of Multiple Distance Matrices"
        case "DGPA": return "Dual Generalized Procustes Analysis"
        case "DMCOA": return "Dual Multiple CO-inertia Analysis"
        case "DMFA": return "Dual Multiple Factor Analysis"
        case "DSTATIS": return "Dual STATIS"
        case "FCPCA": return "Flury's Common Principal Component Analysis"
        case "GCCA": return "Generalized Canonical Correspondence Analysis"
        case "GPA": return "Generalized Procustes Analysis"
        case "hGCANCORR": return "Carroll's Generalized Canonical Correlation Analysis"
        case "ICA": return "Internal Correspondence Analysis"
        case "mbmgPCA": return "multiblock and multigroup Principal Component Analysis"
        case "mbPCA": return "multiblock Principal Component Analysis"
        case "mbPCAiv": return "multiblock Principal Component Analysis with (Orthogonal) Instrumental Variables"
        case "mbPLS": return "multiblock Partial Least Squares"
        case "MCOIA": return "Multiple CO-inertia Analysis"
        case "MFA": return "Multiple Factor Analysis"
        case "MFADT": return "Multiple Factor Analysis for Distance Tables"
        case "mgPCA": return "multigroup Principal Component Analysis"
        case "PMFA": return "Procuste Multiple Factor Analysis"
        case "STATIS": return "Structuration de Tableaux A Trois Indices de la Statistique"
        #others functions
       

def attr_desc(
        attr
):
    """
    Attributes description

    Parameters
    ----------
    attr : str
        The model attribute name.

    Returns
    -------
    desc : DataFrame of shape (1,2)
        Attribute long description
    """
    match attr:
        case "call_": desc = "summary called parameters"
        case "cancoef_": desc = "results for canonical coefficients"
        case "cancorr_": desc = "results for canonical correlation"
        case "coef_": desc = "Coefficients"
        case "col_": desc = "results for columns"
        case "col_group_": desc = "results for columns groups"
        case "col_sup_": desc = "results for supplementary columns"
        case "col_sup_group_": desc = "results for supplementary columns groups"
        case "corr_": desc = "results for correlations"
        case "cov_": desc = "results for covariances"
        case "dim_": desc = "results for dimensions"
        case "eig_": desc = "eigenvalues"
        case "explained_variance_": desc = "explained variance"
        case "evd_": desc = "results for eigen values decomposition"
        case "freq_": desc = "results for the frequencies"
        case "freq_sup_": desc = "results for the supplementary frequencies"
        case "group_": desc = "results for the groups"
        case "group_sup_": desc = "results for the supplementary groups"
        case "ind_": desc = "results for the individuals"
        case "ind_sup_": desc = "results for the supplementary individuals"
        case "inertia_": desc = "results for the inertia"
        case "iv_": desc = "results for the instrumental variables"
        case "lambd_": desc = "eigenvalues (computed on the separate analyses)"
        case "levels_": desc = "results for the levels"
        case "levels_sup_": desc =  "results for the supplementary levels"
        case "manova_": desc =  "results for multivariate analysis of variance"
        case "partial_": desc = "results for the partial variables"
        case "partial_axes_": desc = "results for the partial axes"
        case "quali_var_": desc = "results for the qualitative variables"
        case "quali_var_sup_": desc =  "results for the supplementary qualitative variables"
        case "quanti_var_": desc = "results for the quantitative variables"
        case "quanti_var_sup_": desc = "results for the supplementary quantitative variables"
        case "ratio_": desc = "inertia (ratio) percentage"
        case "rotmat_": desc = "rotation matrix"
        case "row_": desc = "results for the rows"
        case "row_group_": desc = "results for the rows groups"
        case "row_sup_": desc = "results for the supplementary rows"
        case "row_sup_group_": desc = "results for the supplementary rows groups"
        case "separate_analyses_": desc = "results for the separate analyses"
        case "sscp_": desc = "results for sum of squared cross product"
        case "svd_": desc = "results for generalized singular values decomposition"
        case "var_": desc = "results for variables"
        case "var_partiel_": desc = "partial coordinate of the variables for each group"
    return DataFrame([[f".{attr}", desc]],columns=["name","description"])