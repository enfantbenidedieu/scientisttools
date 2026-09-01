
from numpy import ones, linalg, diag, sqrt



def procustemfa(X,Y,ortho=False,trans=False,magnify=False):
    """
    Procustean Multiple Factor Analysis
    
    
    
    """
    # dimensions of X and Y
    nx_samples, nx_cols = X.shape
    ny_samples, ny_cols = Y.shape
    # 
    if nx_samples != ny_samples:
        raise TypeError("Dimensions of X and Y must match")
    
    if ortho:
        if trans:
            x_mean = ((ones(nx_cols)/nx_cols).reshape(1, -1)).dot(X)
            y_mean = ((ones(ny_cols)/ny_cols).reshape(1, -1)).dot(Y)
            # centered
            zx, zy = X - x_mean, Y - y_mean
            # cross prod
            z = zy.T.dot(zx)
        else:
            x_mean = X.copy()
            z = Y.T.dot(X)
        # singular value decomposition
        svd = linalg.svd(z)
        # matrix of rotation
        rot = svd[2].T.dot(svd[0])

        if magnify:
            den = x_mean.sum()
            if not isinstance(den,float):
                den = den.sum()
            beta = sum(svd[1])/den
        else:
            beta = 1

        # 

    else:
        a = linalg.inv(X).dot(Y)
        b = sqrt(diag(linalg.inv(a.T.dot(a))))
        # rotation matrix
        rot = a
        

