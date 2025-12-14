# -*- coding: utf-8 -*-
from numpy import log, diag, eye, full, zeros, linalg, ones, exp, repeat, sqrt, squeeze, dot, c_, cumsum
from pandas import DataFrame
from collections import namedtuple, OrderedDict
from typing import NamedTuple

def rotate_factors(loadings,
                   method = "varimax",
                   normalize = True,
                   power = 4,
                   kappa = None,
                   gamma = 0,
                   delta = 0.01,
                   max_iter = 1000,
                   tol = 1e-5):
    """
    Rotation function
    -----------------

    Description
    -----------
    Performs rotations on an unrotated factor loading Dataframe

    Parameters
    ----------
    `loadings`: pandas DataFrame with `p` rows and `k < p` columns
        The loading table.
    
    `method` : str, optional. Defaults to 'varimax'.
        The factor rotation method. Options include:
            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation) 
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)
            (h) geomin_obl (oblique rotation)
            (i) geomin_ort (orthogonal rotation)

    normalize : bool or None, optional. Defaults to `True`
        Whether to perform Kaiser normalization and de-normalization prior to and following rotation. Used for 'varimax' and 'promax' rotations.
        If ``None``, default for 'promax' is ``False``, and default for 'varimax' is ``True``.

    power : int, optional. Defaults to 4.
        The exponent to which to raise the promax loadings (minus 1).
        Numbers should generally range from 2 to 4.
        
    kappa : float, optional. Defaults to None.
        The kappa value for the 'equamax' objective. Ignored if the method is not 'equamax'.
        
    gamma : int, optional. Defaults to 0.
        The gamma level for the 'oblimin' objective. Ignored if the method is not 'oblimin'.
        
    delta : float, optional. Defaults to 0.01.
        The delta level for 'geomin' objectives. Ignored if the method is not 'geomin_*'.
        
    max_iter : int, optional. Defaults to 1000.
        The maximum number of iterations. Used for 'varimax' and 'oblique' rotations.
        
    tol : float, optional. Defaults to 1e-5.
        The convergence threshold. Used for 'varimax' and 'oblique' rotations.
        
    Returns
    -------
    `loadings`: a pandas DataFrame of the rotated loadings
        The loadings matrix.

    `rotmat`: numpy array of the rotation matrix.

    `phi`: pandas DataFrame indicating the factor correlations matrix.
    
    Notes
    -----
    Most of the rotations in this class are ported from R's ``GPARotation``
    package.

    References
    ----------
    [1] https://cran.r-project.org/web/packages/GPArotation/index.html
    [2] https://github.com/mvds314/factor_rotation

    Examples
    --------
    >>> import pandas as pd
    """
    #results as namedtuple
    def results(typename,grad,criterion):
        return namedtuple(typename,["grad","criterion"])(grad,criterion)
    
    def varimax(loadings:DataFrame,normalize=True,max_iter=1000,tol=1e-5):
        """
        Varimax Rotation
        ----------------

        Description
        -----------
        Perform varimax (orthogonal) rotation, with optional Kaiser normalization
        
        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.

        `normalize`: logical.  Should Kaiser normalization be performed? If so the rows of loadings are re-scaled to unit length before rotation, and scaled back afterwards.

        `max_iter`: int, optional. Defaults to 1000.
            The maximum number of iterations. Used for 'varimax' and 'oblique' rotations.

        `tol`: numeric. The tolerance for stopping: the relative change in the sum of singular values.

        Returns
        -------
        `loadings`: pandas DataFrame of the rotated loadings
            The loadings matrix.

        `rotmat`: numpy array of the rotation matrix.

        References
        ----------
        see https://stat.ethz.ch/R-manual/R-devel/library/stats/html/varimax.html
        """
        #make a copy of loadings
        X = loadings.copy()
        #shape of loadings
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # normalize the loadings matrix using sqrt of the sum of squares (Kaiser)
        if normalize:
            sc = X.copy().apply(lambda x: sqrt(sum(x**2)),axis=1)
            X = (X.T / sc).T

        #initialize the rotation matrix to N x N identity matrix
        rotmat = eye(n_cols)
        d = 0
        for _ in range(max_iter):
            old_d = d
            #take inner product of loading matrix and rotation matrix
            z = X.dot(rotmat)
            #transform data for singular value decomposition using updated formula : B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
            B = X.T.dot(z.pow(3) - z.dot(diag(squeeze(repeat(1, n_rows).dot(z.pow(2))))) / n_rows)
            #perform SVD on the transformed matrix
            U, S, V = linalg.svd(B)
            #take inner product of U and V, and sum of S
            rotmat = U.dot(V)
            d = sum(S)
            # check convergence
            if d < old_d * (1 + tol):
                break

        #take inner product of loading matrix and rotation matrix
        X =  X.dot(rotmat)
        # de-normalize the data
        if normalize:
            X = X.T.mul(sc)
        else:
            X = X.T
        #convert loadings matrix to data frame
        loadings = X.T.copy()
        loadings.columns = ["Dim."+ str(x+1) for x in range(n_cols)]
        return namedtuple("varimax",["loadings","rotmat"])(loadings,rotmat)

    #promax rotation
    def promax(loadings:DataFrame,power=4):
        """
        Promax rotation
        ---------------

        Description
        -----------
        Perform promax (oblique) rotation, with optional Kaiser normalization.

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.

        `power`: The power used the target. Valuies of 2 to 4 are recommended.

        Returns
        -------
        `loadings`: pandas DataFrame of the rotated loadings
            The loadings matrix.

        `rotmat`: numpy array of the rotation matrix.

        `phi`: pandas DataFrame indicating the factor correlations matrix.

        References
        ----------
        see https://stat.ethz.ch/R-manual/R-devel/library/stats/html/varimax.html
        """
        #make a copy of loadings
        X = loadings.copy()
        _ , n_cols = X.shape
        columns = ["Dim."+str(x+1) for x in range(n_cols)]
        if n_cols < 2:
            return X

        #first get varimax rotation
        X, rotmat = varimax(loadings=X)
        Y = X.mul(abs(X)**(power-1))
    
        #fit linear regression model and extract coefficients
        coef = linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    
        #calculate diagonal of inverse square
        try:
            diag_inv = diag(linalg.inv(coef.T.dot(coef)))
        except linalg.LinAlgError:
            diag_inv = diag(linalg.pinv(coef.T.dot(coef)))
        #transform and calculate inner products
        coef = coef.dot(diag(sqrt(diag_inv)))
        z = X.dot(coef)
        #update rotation matrix
        rotmat = rotmat.dot(coef)
        #r scores
        coef_inv = linalg.inv(coef)
        phi = DataFrame(coef_inv.dot(coef_inv.T),index=columns,columns=columns)

        # convert loadings matrix to data frame
        loadings = z.copy()
        loadings.columns = columns
        return namedtuple("promax",["loadings","rotmat","phi"])(loadings,rotmat,phi)

    #oblimax rotation ojective function
    def oblimax_obj(loadings:DataFrame) -> NamedTuple:
        """
        The Oblimax function objective
        ------------------------------

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.

        Returns
        -------
        a namedtuple containing the following fields:

        `grad`: pandas DataFrame containing the gradients

        `criterion` : float, containing the criterion for the objective.
        """
        gradient = - ((loadings.pow(3).mul(4) / loadings.pow(4).sum().sum()) - (loadings.mul(4)/ loadings.pow(2).sum().sum()))
        criterion = log(loadings.pow(4).sum().sum()) - 2 * log(loadings.pow(2).sum().sum())
        return results("oblimax",gradient,criterion)
    
    #quartimax rotation objective function
    def quartimax_obj(loadings:DataFrame) -> NamedTuple:
        """
        Quartimax function objective
        ----------------------------

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.

        Returns
        -------
        a namedtuple containing the following fields:

        `grad`: pandas DataFrame containing the gradients

        `criterion`: float, containing the criterion for the objective.
        """
        gradient = - loadings.pow(3)
        criterion = - sum(diag(loadings.pow(2).T.dot(loadings.pow(2)))) / 4
        return results("quartimax",gradient,criterion)

    #oblimin rotation objective function
    def oblimin_obj(loadings:DataFrame, gamma = 0) -> NamedTuple:
        """
        The Oblimin function objective
        ------------------------------

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.

        `gamma`: int, optional. Defaults to 0.
            The gamma level for the 'oblimin' objective.
        
        Returns
        -------
        a namedtuple containing the following fields:

        `grad`: pandas DataFrame containing the gradients

        `criterion`: float, containing the criterion for the objective.
        """
        X = loadings.pow(2).dot(eye(loadings.shape[1]) != 1)
        if gamma != 0:
            n_rows = loadings.shape[0]
            X = diag(full(1, n_rows)) - zeros((n_rows, n_rows)).dot(X)
        gradient, criterion = loadings.mul(X), loadings.pow(2).mul(X).sum().sum()/4
        return results("oblimin",gradient,criterion)
    
    #quartimin rotation objective function
    def quartimin_obj(loadings:DataFrame) -> NamedTuple: 
        """
        The Quartimin function objective
        --------------------------------

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.
        
        Returns
        -------
        a namedtuple containing the following fields:

        `grad`: pandas DataFrame containing the gradients

        `criterion`: float, containing the criterion for the objective.
        """
        X = loadings.pow(2).dot(eye(loadings.shape[1]) != 1)
        gradient, criterion = loadings.mul(X), loadings.pow(2).mul(X).sum().sum()/4
        return results("quartimin",gradient,criterion)

    #equamax rotation objective function
    def equamax_obj(loadings:DataFrame,kappa = 0) -> NamedTuple:
        """
        The Equamax function objective
        ------------------------------

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.
        
        `kappa` : float, optional. Defaults to 0.
            The kappa value for the 'equamax' objective
        
        Returns
        -------
        a namedtuple containing the following fields:

        `grad`: pandas DataFrame containing the gradients

        `criterion`: float, containing the criterion for the objective.
        """
        n_rows, n_cols = loadings.shape
        N, M = ones(n_cols) - eye(n_cols), ones(n_rows) - eye(n_rows)

        loadings_squared = loadings.pow(2)
        f1, f2 = (1 - kappa)*sum(diag(loadings_squared.T.dot(loadings_squared.dot(N))))/4, kappa*sum(diag(loadings_squared.T.dot(M.dot(loadings_squared))))/4
        
        gradient = loadings.mul(loadings_squared.dot(N)).mul(1-kappa) + loadings.mul(M.dot(loadings_squared)).mul(kappa)
        criterion = f1 + f2
        return results("qquamax",gradient,criterion)

    #geomin oblique rotation transformation
    def geomin_obj(loadings:DataFrame, delta = 0.01) -> NamedTuple:
        """
        The Geomin function objective
        -----------------------------

        Parameters
        ----------
        `loadings`: pandas DataFrame with `p` rows and `k < p` columns
            The loading table.
        
        `delta`: float, optional. Defaults to 0.01.
            The delta level for 'geomin' objectives.

        Returns
        -------
        a namedtuple containing the following fields:

        `grad`: pandas DataFrame containing the gradients

        `criterion`: float, containing the criterion for the objective.
        """
        n_rows, n_cols = loadings.shape
        loadings2 = loadings.pow(2).add(delta)

        pro = exp(log(loadings2).sum(1) / n_cols)
        rep = repeat(pro, n_cols).values.reshape(n_rows, n_cols)

        gradient, criterion = (2 / n_cols) * loadings.div(loadings2) * rep, sum(pro)
        return results("geomin",gradient,criterion)

    #oblique rotation transformation
    def oblique(loadings:DataFrame,method:str,gamma=0,delta=0.01,max_iter=1000,tol=1e-5):
        """
        Perform oblique rotations, except 'promax'.
        ------------------------------------------
        
        Description
        -----------
        A generic function for performing all oblique rotations, except for promax, which is implemented separately.

        Parameters
        ----------
        loadings : array-like
            The loading matrix
        method : str
            The obligue rotation method to use.

        Returns
        -------
        `loadings`: pandas DataFrame of the rotated loadings
            The loadings matrix.

        `rotmat`: numpy array of the rotation matrix.

        `phi`: pandas DataFrame indicating the factor correlations matrix. 
        """
        # initialize the rotation matrix
        _, n_cols = loadings.shape
        rotmat = eye(n_cols)

        # default alpha level
        alpha = 1
        rotmat_inv = linalg.inv(rotmat)
        new_loadings = loadings.dot(rotmat_inv.T)
        
        if method == "oblimin":
            obj = oblimin_obj(loadings=new_loadings,gamma=gamma)
        elif method == "quartimin":
            obj = quartimin_obj(loadings=new_loadings)
        elif method == "geomin_obl":
            obj = geomin_obj(loadings=new_loadings,delta=delta)

        gradient, criterion = - new_loadings.T.dot(obj.grad.dot(rotmat_inv)).T, obj.criterion

        if method == "oblimin":
            obj_t = oblimin_obj(loadings=new_loadings,gamma=gamma)
        elif method == "quartimin":
            obj_t = quartimin_obj(loadings=new_loadings)
        elif method == "geomin_obl":
            obj_t = geomin_obj(loadings=new_loadings,delta=delta)

        # main iteration loop, up to `max_iter`, calculate the gradient
        for _ in range(max_iter + 1):
            gradient_new = gradient.sub(rotmat.dot(diag(ones(gradient.shape[0]).dot(gradient.mul(rotmat)))))
            s = sqrt(sum(diag(gradient_new.T.dot(gradient_new))))

            if s < tol:
                break

            alpha = 2 * alpha

            # calculate the Hessian of the objective function
            for _ in range(11):
                X = rotmat - gradient_new.mul(alpha)
                v = 1 / sqrt(ones(X.shape[0]).dot(X**2))
                new_rotmat = X.dot(diag(v))
                
                new_loadings = loadings.dot(linalg.inv(new_rotmat).T)

                if method == "oblimin":
                    obj_t = oblimin_obj(loadings=new_loadings,gamma=gamma)
                elif method == "quartimin":
                    obj_t = quartimin_obj(loadings=new_loadings)
                elif method == "geomin_obl":
                    obj_t = geomin_obj(loadings=new_loadings,delta=delta)
            
                improvement = criterion - obj_t.criterion

                if improvement > 0.5 * s**2 * alpha:
                    break

                alpha = alpha / 2

            rotmat = new_rotmat
            criterion = obj_t.criterion
            gradient = - new_loadings.T.dot(obj_t.grad).dot(linalg.inv(new_rotmat)).T

        #set columns
        columns = ["Dim."+str(x+1) for x in range(n_cols)]
        #calculate phi
        phi = DataFrame(rotmat.T.dot(rotmat),index=columns,columns=columns)
        
        #convert loadings matrix to data frame
        loadings = new_loadings.copy()
        loadings.columns = columns
        return namedtuple("oblique",["loadings","rotmat","phi"])(loadings, rotmat.values, phi)

    #orthogonal rotation transformation
    def orthogonal(loadings:DataFrame,method:str,kappa=0,delta=0.01,max_iter=1000,tol=1e-5):
        """
        Perform orthogonal rotations, except 'varimax'
        ---------------------------------------------

        Description
        -----------
        A generic function for performing all orthogonal rotations, except for varimax, which is implemented separately.

        Parameters
        ----------
        loadings : :obj:`numpy.ndarray`
            The loading matrix
        method : str
            The orthogonal rotation method to use.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`
            The loadings matrix
        rotmat : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The rotation matrix
        """
        arr = loadings.copy()

        # initialize the rotation matrix
        _, n_cols = arr.shape
        rotmat = eye(n_cols)

        # default alpha level
        alpha = 1
        new_loadings = arr.dot(rotmat)

        if method == "oblimax":
            obj = oblimax_obj(loadings=new_loadings)
        elif method == "quartimax":
            obj = quartimax_obj(loadings=new_loadings)
        elif method == "equamax":
            obj = equamax_obj(loadings=new_loadings,kappa=kappa)
        elif method == "geomin_ort":
            obj = geomin_obj(loadings=new_loadings,delta=delta)

        gradient, criterion = arr.T.dot(obj.grad), obj.criterion

        if method == "oblimax":
            obj_t = oblimax_obj(loadings=new_loadings)
        elif method == "quartimax":
            obj_t = quartimax_obj(loadings=new_loadings)
        elif method == "equamax":
            obj_t = equamax_obj(loadings=new_loadings,kappa=kappa)
        elif method == "geomin_ort":
            obj_t = geomin_obj(loadings=new_loadings,delta=delta)

        # main iteration loop, up to `max_iter`, calculate the gradient
        for _ in range(max_iter + 1):
            M = rotmat.T.dot(gradient)
            S = (M + M.T) / 2
            gradient_new = gradient - rotmat.dot(S)
            s = sqrt(sum(diag(gradient_new.T.dot(gradient_new))))
            if s < tol:
                break

            alpha = 2 * alpha

            # calculate the Hessian of the objective function
            for _ in range(11):
                X = rotmat- alpha * gradient_new
                U, _, V = linalg.svd(X)
                new_rotmat = U.dot(V)
                new_loadings = arr.dot(new_rotmat)
                
                if method == "oblimax":
                    obj_t = oblimax_obj(loadings=new_loadings)
                elif method == "quartimax":
                    obj_t = quartimax_obj(loadings=new_loadings)
                elif method == "equamax":
                    obj_t = equamax_obj(loadings=new_loadings,kappa=kappa)
                elif method == "geomin_ort":
                    obj_t = geomin_obj(loadings=new_loadings,delta=delta)

                if obj_t.criterion < (criterion - 0.5 * s**2 * alpha):
                    break

                alpha = alpha / 2

            rotmat, criterion, gradient = new_rotmat, obj_t.criterion, arr.T.dot(obj_t.grad)

        # convert loadings matrix to data frame
        loadings = new_loadings.copy()
        loadings.columns = ["Dim."+str(x+1) for x in range(n_cols)]
        return namedtuple("orthogonal",["loadings","rotmat"])(loadings, rotmat)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #rotate factors
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set possible rotations
    orthogonal_rotations, oblique_rotations = ["varimax", "oblimax", "quartimax", "equamax", "geomin_ort"],  ["promax", "oblimin", "quartimin", "geomin_obl"]
    possible_rotations = orthogonal_rotations + oblique_rotations

    #check if method in possible rotations
    if method not in possible_rotations:
        raise ValueError("The value for `method` must be one of the following: {}.".format(", ".join(possible_rotations)))
    
    if kappa is None:
        kappa = 1/loadings.shape[0]

    #apply rotation transformation
    phi = None
    if method == "varimax":
        rot = varimax(loadings=loadings,normalize=normalize,max_iter=max_iter,tol=tol)
    elif method == "promax":
        rot = promax(loadings=loadings,power=power)
    elif method in oblique_rotations:
        rot = oblique(loadings=loadings,method=method,gamma=gamma,delta=delta,max_iter=max_iter,tol=tol)
    elif method in orthogonal_rotations:
        rot = orthogonal(loadings=loadings,method=method,kappa=kappa,delta=delta,max_iter=max_iter,tol=tol)

    #extract result
    if method in orthogonal_rotations:
        new_loadings, rotmat = rot.loadings, rot.rotmat
    elif method in oblique_rotations:
        new_loadings, rotmat, phi = rot.loadings, rot.rotmat, rot.phi

    #convert to ordered dictionary
    rotate = OrderedDict(loadings=new_loadings,rotmat=rotmat,phi=phi)
    #convert to namedtuple
    return namedtuple("rotation",rotate.keys())(*rotate.values())