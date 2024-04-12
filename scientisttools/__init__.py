# -*- coding: utf-8 -*-


# Principal Components Analysis
from scientisttools.pca import PCA
from scientisttools.get_pca import get_pca_ind, get_pca_var, get_pca, summaryPCA
from scientisttools.fviz_pca import fviz_pca_ind, fviz_pca_var, fviz_pca

# Correspondence Analysis (CA)
from scientisttools.ca import CA
from scientisttools.get_ca import get_ca_row, get_ca_col, get_ca
from scientisttools.fviz_ca import fviz_ca_row, fviz_ca_col, fviz_ca_biplot, fviz_ca

# Multiple Correspondence Analysis (MCA)
from scientisttools.mca import MCA
from scientisttools.get_mca import get_mca_ind, get_mca_var, get_mca



from scientisttools.eigenvalue import get_eig,get_eigenvalue,fviz_eig,fviz_screeplot
from scientisttools.dimdesc import dimdesc
from scientisttools.reconst import reconst


from scientisttools.fviz_contrib import fviz_contrib
from scientisttools.fviz_corrplot import fviz_corrplot
from scientisttools.fviz_corrcircle import fviz_corrcircle



from scientisttools.version import __version__

__name__ = "scientisttools"
__author__ = 'Duverier DJIFACK ZEBAZE'
__email__ = 'duverierdjifack@gmail.com'