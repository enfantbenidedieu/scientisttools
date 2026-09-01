# -*- coding: utf-8 -*-
from numpy import sqrt, insert, diff, nan, cumsum, c_, diag, linalg, real
from pandas import Series, DataFrame

# intern functions
from .cov2corr import cov2corr

def RVstats(model, 
            tol=1e-7):
    """
    RV Statistics

    Parameters
    ----------
    model : dict
        A dictionary containing separate principal component analysis (sPCA).

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Returns
    -------
    result : dict
        An object with the following keys:

        traceRV : DataFrame of shape (n_groups, n_groups)
            The trace RV between groups.
        RV : DataFrame of shape (n_groups, n_groups)
            The RV coefficient between groups.
        eig : DataFrame of shape (rank_rv, 4)
            The eigenvalue of RV matrix, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.
        coord : DataFrame of shape (n_groups, rank_rv)
            The coordinates of the groups.
        infos : DataFrame of shape (n_groups, 3)
            Additionals informations (weight, inertia and percentage of inertia) of the groups.
    """
    # unique classe
    uq_classe = list(model.keys())

    # trace RV between groups
    traceRV = DataFrame(index=uq_classe,columns=uq_classe).astype("float")
    for g1 in uq_classe:
        for g2 in uq_classe:
            traceRV.loc[g1,g2] = sum(diag(model[g1].call_.Vb.dot(model[g2].call_.Vb)))

    # RV coefficients
    RV = cov2corr(X=traceRV)

    # eigen decomposition of RV (=singular value decomposition of hermittian)
    svd = linalg.svd(RV,hermitian=True)
    # maximum number of components
    rank = sum(svd[1]/svd[1][0] > tol)

    # eigen values and eigen vectors
    eigvals, eigvects = real(svd[1][:rank]), real(svd[0])
    eigvects[:,0] = abs(eigvects[:,0])

    # RV eigen values informations
    eigdiff, eigprop = insert(-diff(eigvals),len(eigvals)-1,nan), 100*eigvals/sum(eigvals)
    #convert to DataFrame
    eig = DataFrame(c_[eigvals,eigdiff,eigprop,cumsum(eigprop)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],
                    index = [f"Dim{x+1}" for x in range(rank)])

    # group weights
    group_w = Series(eigvects[:,0],index=uq_classe,name="weight")
    # inertia of each group
    group_inertia = Series([sum(diag(model[g].call_.Vb)) for g in uq_classe],index=uq_classe,name="Inertia")
    # percentage of group inertia
    group_inertia_pct = 100*group_inertia/sum(group_inertia)
    # convert to DataFrame
    group_infos = DataFrame(c_[group_w,group_inertia,group_inertia_pct],columns=["Weight","Inertia","Inertia (%)"],index=uq_classe)
    # group coordinates
    group_coord = DataFrame(eigvects*sqrt(eigvals),index=uq_classe,columns=[f"Dim{x+1}" for x in range(rank)])
    # convert to dictionary
    res = {"traceRV":traceRV, "RV":RV, "eig":eig, "coord":group_coord, "infos":group_infos}
    return res