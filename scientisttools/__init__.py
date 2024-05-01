# -*- coding: utf-8 -*-
from __future__ import annotations

# Principal Components Analysis
from .pca import PCA
from .get_pca import get_pca_ind, get_pca_var, get_pca, summaryPCA
from .fviz_pca import fviz_pca_ind, fviz_pca_var, fviz_pca

# Correspondence Analysis (CA)
from .ca import CA
from .get_ca import get_ca_row, get_ca_col, get_ca, summaryCA
from .fviz_ca import fviz_ca_row, fviz_ca_col, fviz_ca_biplot, fviz_ca

# Multiple Correspondence Analysis (MCA)
from .mca import MCA
from .get_mca import get_mca_ind, get_mca_var, get_mca, summaryMCA
from .fviz_mca import fviz_mca_ind, fviz_mca_mod, fviz_mca_var, fviz_mca

# Factor Analysis of Mixed Data (FAMD)
from .famd import FAMD
from .get_famd import get_famd_ind, get_famd_var, get_famd, summaryFAMD
from .fviz_famd import fviz_famd_ind, fviz_famd_col, fviz_famd_mod, fviz_famd_var, fviz_famd

# Partial PCA
from .partialpca import PartialPCA
from .get_partialpca import get_partialpca_ind, get_partialpca_var, get_partialpca, summaryPartialPCA
from .fviz_partialpca import fviz_partialpca_ind, fviz_partialpca_var, fviz_partialpca

# Exploratory Factor Analysis (EFA)
from .efa import EFA
from .get_efa import get_efa_ind, get_efa_var, get_efa, summaryEFA
from .fviz_efa import fviz_efa_ind, fviz_efa_var, fviz_efa

# Multiple Factor Analysis (MFA)
from .mfa import MFA
from .get_mfa import get_mfa_ind, get_mfa_var, get_mfa_partial_axes, get_mfa, summaryMFA
from .fviz_mfa import fviz_mfa_ind, fviz_mfa_var, fviz_mfa_axes,fviz_mfa_group

# Multiple Factor Analysis for qualitative/categorical variables (MFAQUAL)
from .mfaqual import MFAQUAL
from .get_mfa import summaryMFAQUAL
from .fviz_mfa import fviz_mfa_mod

# Multiple Factor Analysis of Mixed Data (MFAMIX)
from .mfamix import MFAMIX
from .get_mfa import summaryMFAMIX

# Multiple Factor Analysis of contingence table
from .mfact import MFACT
from .fviz_mfa import fviz_mfa_freq
from .get_mfa import summaryMFACT

# Multidimensional Scaling
from .smacof import SMACOF
from .mds import MDS
from .get_mds import get_mds
from .fviz_mds import fviz_mds
from .fviz_shepard import fviz_shepard

from .cmdscale import CMDSCALE
from .fviz_cmdscale import fviz_cmdscale

########
from .eigenvalue import get_eig,get_eigenvalue,fviz_eig,fviz_screeplot
from .dimdesc import dimdesc
from .reconst import reconst

from .fviz_contrib import fviz_contrib
from .fviz_cos2 import fviz_cos2
from .fviz_corrplot import fviz_corrplot
from .fviz_corrcircle import fviz_corrcircle

# HCPC
from .hcpc import HCPC
from .fviz_hcpc import plot_dendrogram, fviz_hcpc_cluster

# VARHCA
from .varhca import VARHCA
from .catvarhca import CATVARHCA
from .varhcpc import VARHCPC
from .fviz_hcpc import fviz_varhcpc_cluster

__version__ = '0.1.5'
__name__ = "scientisttools"
__author__ = 'Duverier DJIFACK ZEBAZE'
__email__ = 'duverierdjifack@gmail.com'