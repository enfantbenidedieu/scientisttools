# -*- coding: utf-8 -*-
from __future__ import annotations

#----------------------------------------------------------------------------------------------------------------------------------------
## Methods
#----------------------------------------------------------------------------------------------------------------------------------------
#generalized factor analysis (GFA)
from scientisttools.methods.PCA import PCA, predictPCA, supvarPCA
from scientisttools.methods.PartialPCA import PartialPCA, predictPartialPCA, supvarPartialPCA
from scientisttools.methods.EFA import EFA
from scientisttools.methods.CA import CA, predictCA, supvarCA
from scientisttools.methods.MCA import MCA, predictMCA, supvarMCA
#from .specificmca import SpecificMCA, predictSpecificMCA, supvarSpecificMCA
#from .famd import FAMD, predictFAMD, supvarFAMD
#from .pcamix import PCAMIX, predictPCAMIX, supvarPCAMIX
#from .mpca import MPCA, predictMPCA, supvarMPCA
#from .mfa import MFA, predictMFA, supvarMFA
#from .mfaqual import MFAQUAL
#from .mfamix import MFAMIX
#from .mfact import MFACT, predictMFACT, supvarMFACT
#from .dmfa import DMFA
#from .cca import CCA

#multidimensional 
#from .smacof import SMACOF
#from .mds import MDS
#from .cmdscale import CMDSCALE

#clustering analysing
#from .hcpc import HCPC
#from .varhca import VARHCA
#from .catvarhca import CATVARHCA
#from .varhcpc import VARHCPC



#----------------------------------------------------------------------------------------------------------------------------------------
## Extraction
#----------------------------------------------------------------------------------------------------------------------------------------
from scientisttools.extraction.get_eig import get_eig,get_eigenvalue
from scientisttools.extraction.get_pca import get_pca_ind, get_pca_var, get_pca, summaryPCA
from scientisttools.extraction.get_partialpca import get_partialpca_ind, get_partialpca_var, get_partialpca, summaryPartialPCA
from scientisttools.extraction.get_efa import get_efa_ind, get_efa_var, get_efa, summaryEFA
from scientisttools.extraction.get_ca import get_ca_row, get_ca_col, get_ca, summaryCA
from scientisttools.extraction.get_mca import get_mca_ind, get_mca_var, get_mca, summaryMCA
#from .get_famd import get_famd_ind, get_famd_var, get_famd, summaryFAMD
#from .get_pcamix import get_pcamix_ind, get_pcamix_var, get_pcamix, summaryPCAMIX
#from .get_mpca import get_mpca_ind, get_mpca_var, get_mpca, summaryMPCA
#from .get_mfa import get_mfa_ind, get_mfa_var, get_mfa_partial_axes, get_mfa, summaryMFA
#from .get_dmfa import get_dmfa_ind,get_dmfa_var,get_dmfa,summaryDMFA

#from .get_mds import get_mds


#----------------------------------------------------------------------------------------------------------------------------------------
## Visualization
#----------------------------------------------------------------------------------------------------------------------------------------
#from scientisttools.plot.fviz_pca import fviz_pca_ind, fviz_pca_var, fviz_pca_biplot,fviz_pca
#from scientisttools.plot.fviz_partialpca import fviz_partialpca_ind, fviz_partialpca_var, fviz_partialpca_biplot,fviz_partialpca
#from scientisttools.plot.fviz_efa import fviz_efa_ind, fviz_efa_var, fviz_efa_biplot, fviz_efa
#from scientisttools.plot.fviz_ca import fviz_ca_row, fviz_ca_col, fviz_ca_biplot, fviz_ca
#from scientisttools.plot.fviz_mca import fviz_mca_ind, fviz_mca_mod, fviz_mca_var, fviz_mca_biplot, fviz_mca
#from .fviz_famd import fviz_famd_ind, fviz_famd_col, fviz_famd_mod, fviz_famd_var, fviz_famd
#from .fviz_pcamix import fviz_pcamix_ind, fviz_pcamix_col, fviz_pcamix_mod, fviz_pcamix_var, fviz_pcamix
#from .fviz_mpca import fviz_mpca_ind, fviz_mpca_col, fviz_mpca_mod, fviz_mpca_var, fviz_mpca
#from .fviz_mfa import fviz_mfa_ind, fviz_mfa_var, fviz_mfa_axes,fviz_mfa_group, fviz_mfa
#from .fviz_mfa import fviz_mfa_mod
#from .fviz_mfa import fviz_mfa_freq
#from .fviz_dmfa import fviz_dmfa_ind, fviz_dmfa_var, fviz_dmfa_group, fviz_dmfa_quali_sup
#from .fviz_cca import fviz_cca_ind, fviz_cca_var, fviz_cca_scatterplot, fviz_cca

#
#from .fviz_cmdscale import fviz_cmdscale
#from .fviz_mds import fviz_mds
#from .fviz_shepard import fviz_shepard

#clustering analysis
#from .fviz_hcpc import plot_dendrogram, fviz_hcpc_cluster
#from .fviz_hcpc import fviz_varhcpc_cluster



#others functions of visualization
from scientisttools.plot.fviz_eig import fviz_eig,fviz_screeplot
from scientisttools.plot.fviz_contrib import fviz_contrib
from scientisttools.plot.fviz_cos2 import fviz_cos2
from scientisttools.plot.fviz_corrplot import fviz_corrplot
from scientisttools.plot.fviz_corrcircle import fviz_corrcircle


#----------------------------------------------------------------------------------------------------------------------------------------
## Others functions
#----------------------------------------------------------------------------------------------------------------------------------------
# Others functions
#from .auto_cut_tree import auto_cut_tree
#from .catdesc import catdesc
#from .coeffLg import coeffLg
#from .coeffRV import coeffRV
#from .conditional_average import conditional_average
#from .contdesc import contdesc
from scientisttools.others.coord_ellipse import coord_ellipse
#from .covariance_to_correlation import covariance_to_correlation
from scientisttools.others.dimdesc import dimdesc
#from .eta2 import eta2
#from .function_eta2 import function_eta2
#from .function_lg import function_lg
#from .kmo import kmo_index
#from .namedtuplemerge import namedtuplemerge
#from .quali_var_desc import quali_var_desc
#from .quanti_var_desc import quanti_var_desc
#from .recodecat import recodecat
#from .recodecont import recodecont
#from .recodevar import recodevar
#from .recodevarfamd import recodevarfamd
from scientisttools.others.reconst import reconst
#from .revaluate_cat_variable import revaluate_cat_variable
#from .sim_dist import sim_dist
from scientisttools.others.splitmix import splitmix
#from .svd_triplet import svd_triplet
from scientisttools.others.wpearsonr import wpearsonr
#from .weightedcorrtest import weightedcorrtest


## Load all datasets
from .datasets import (
    load_autos,
    load_autos2,
    load_autosmds,
    load_body,
    load_burgundywines,
    load_cars2006,
    load_carsacpm,
    load_children,
    load_congressvotingrecords,
    load_decathlon,
    load_decathlon2,
    load_femmetravail,
    load_gironde,
    load_housetasks,
    load_jobrate,
    load_lifecyclesavings,
    load_madagascar,
    load_mortality,
    load_mushroom,
    load_music,
    load_poison,
    load_protein,
    load_qtevie,
    load_racescanines,
    load_tea,
    load_temperature,
    load_tennis,
    load_usarrests,
    load_wine,
    load_womenwork
)

__version__ = '0.1.6'
__name__ = "scientisttools"
__author__ = 'Duverier DJIFACK ZEBAZE'
__email__ = 'djifacklab@gmail.com'