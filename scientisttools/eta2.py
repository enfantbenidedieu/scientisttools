# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

def eta2(categories,value,digits=4):
    """
    Calcul du rapport de corréltion eta carré
    -----------------------------------------

    Description
    -----------
    Cette fonction calcule le rapport de corrélation eta carré qui est une mesure d'association importante entre une variable quantitative et une variable qualitative.

    Parameters
    ----------
    categories : un facteur associé à la variable qualitative

    value : un vecteur associé à la variable quantitatives

    digits : int, default=3. Number of decimal printed

    Return
    ------
    a dictionary of numeric elements

    Sum. Intra : la somme des carrés intra

    Sum. Inter : La somme des carrés inter

    Correlation ratio : La valeur du rapport de corrélation empirique

    F-stats : La statistique de test F de Fisher
    
    pvalue : la probabilité critique

    References
    ----------
    F. Bertrand, M. Maumy-Bertrand, Initiation à la Statistique avec R, Dunod, 4ème édition, 2023.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    see also https://stackoverflow.com/questions/52083501/how-to-compute-correlation-ratio-or-eta-in-python
    """
    K = len(np.unique(categories, return_inverse=True)[0])
    n = value.shape[0]
    
    cat = np.unique(categories, return_inverse=True)[1]
    values = np.array(value)
    
    scintra = 0.0
    scinter = 0.0
    for i in np.unique(cat):
        subgroup = values[np.argwhere(cat == i).flatten()]
        scintra += np.sum((subgroup-np.mean(subgroup))**2)
        scinter += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    eta2 = scinter/(scinter+scintra)
    f_stat = (scinter/(K-1))/(scintra/(n-K))
    pvalue = np.round(sp.stats.f.sf(f_stat, K-1, n-K),digits)
    return {'Sum. Intra':round(scintra,digits),'Sum. Inter':round(scinter,digits),'Eta2': round(eta2,digits),'F-stats': round(f_stat,digits),'pvalue': pvalue}