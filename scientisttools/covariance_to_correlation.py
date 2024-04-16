# From covariance to correlation
# https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
import numpy as np

def covariance_to_correlation(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation