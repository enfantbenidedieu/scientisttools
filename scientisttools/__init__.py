# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .pcadash import PCADash
# -*- coding: utf-8 -*-

from .ca import CA
from .efa import EFA
from .mca import MCA
from .pca import PCA
from .famd import FAMD
from .wpca import WPCA
from .partialpca import PartialPCA

from .cda import CanonicalDiscriminantAnalysis
from .lda import LinearDiscriminantAnalysis
from .qda import QuadraticDiscriminantAnalysis
from .lfda import LocalFisherDiscriminantAnalysis

from .get_efa import get_efa,get_efa_ind,get_efa_var
from .get_ca import get_ca,get_ca_col,get_ca_row
from .get_dist import get_dist
from .get_eig import get_eig, get_eigenvalue
from .get_hclust import get_hclust
from .get_mca import get_mca, get_mca_ind, get_mca_mod,get_mca_var
from .get_mds import get_mds
from .get_melt import get_melt
from .get_pca import get_pca, get_pca_ind,get_pca_var
from .get_ppca import get_ppca,get_ppca_ind,get_ppca_var
from .get_famd import get_famd,get_famd_ind,get_famd_col,get_famd_mod,get_famd_var

from .ploteig import plot_eigenvalues
from .plotshepard import plot_shepard
from .plotpca import plotPCA
from .plotppca import plotPPCA
from .plotca import plotCA
from .plotcmds import plotCMDS
from .plotcontrib import plot_contrib
from .plotcorrcircle import plot_correlation_circle
from .plotcosines import plot_cosines
from .plotefa import plotEFA
from .plotfamd import plotFAMD
from .plotmca import plotMCA
from .plotmds import plotMDS

# -*- coding: utf-8 -*-

from .classicmds import CMDSCALE
from .smacof import SMACOF
from .mds import MDS

from .fviz import fviz
from .factor_summary import factor_summary
from .fviz_dist import fviz_dist
from .fviz_eig import fviz_eig,fviz_eigenvalue, fviz_screeplot
from .fviz_ca import fviz_ca_row, fviz_ca_col
from .fviz_mca import fviz_mca_ind, fviz_mca_col, fviz_mca_var
from .fviz_pca import fviz_pca_ind, fviz_pca_var

from .ggcorrplot import *

from .summarypca import summaryPCA
from .summaryppca import summaryPPCA
from .summaryca import summaryCA
from .summaryefa import summaryEFA
from .summarymca import summaryMCA
from .summaryfamd import summaryFAMD

from .base import *

__version__ = "0.0.2"