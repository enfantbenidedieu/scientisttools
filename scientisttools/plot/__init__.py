# -*- coding: utf-8 -*-
from __future__ import annotations

from ._fviz_ca import fviz_ca, fviz_ca_biplot, fviz_ca_col, fviz_ca_row
from ._fviz_cancorr import fviz_cancorr, fviz_cancorr_ind, fviz_cancorr_scatter, fviz_cancorr_var
from ._fviz_cluster import fviz_cluster
from ._fviz_contrib import fviz_contrib
from ._fviz_corcircle import fviz_corcircle
from ._fviz_corrplot import fviz_corrplot
from ._fviz_cos2 import fviz_cos2
from ._fviz_dend import fviz_dend
from ._fviz_dend2 import fviz_dend2
from ._fviz_dmfa import fviz_dmfa, fviz_dmfa_ind, fviz_dmfa_var
from ._fviz_eig import fviz_eig, fviz_screeplot
from ._fviz_ellipses import fviz_ellipses
from ._fviz_fa import fviz_fa, fviz_fa_biplot, fviz_fa_ind, fviz_fa_var
from ._fviz_mca import fviz_mca, fviz_mca_biplot, fviz_mca_ind, fviz_mca_var
from ._fviz_mfa import fviz_mfa, fviz_mfa_ind, fviz_mfa_var
from ._fviz_mix import fviz_mix, fviz_mix_ind, fviz_mix_var
from ._fviz_pca import fviz_pca, fviz_pca_biplot, fviz_pca_ind, fviz_pca_var
from ._fviz_pcoa import fviz_pcoa, fviz_pcoa_ind, fviz_pcoa_shepard
from ._fviz import add_arrow, add_scatter, fviz_arrow, fviz_circle, fviz_scatter, set_axis

__all__ = [
    "add_arrow",
    "add_scatter",
    "fviz_arrow",
    "fviz_ca",
    "fviz_ca_biplot",
    "fviz_ca_col",
    "fviz_ca_row",
    "fviz_cancorr",
    "fviz_cancorr_ind",
    "fviz_cancorr_scatter",
    "fviz_cancorr_var",
    "fviz_circle",
    "fviz_cluster",
    "fviz_contrib",
    "fviz_corcircle",
    "fviz_corrplot",
    "fviz_cos2",
    "fviz_dend",
    "fviz_dend2",
    "fviz_dmfa",
    "fviz_dmfa_ind",
    "fviz_dmfa_var",
    "fviz_eig",
    "fviz_ellipses",
    "fviz_fa",
    "fviz_fa_biplot",
    "fviz_fa_ind",
    "fviz_fa_var",
    "fviz_mca",
    "fviz_mca_biplot",
    "fviz_mca_ind",
    "fviz_mca_var",
    "fviz_mfa",
    "fviz_mfa_ind",
    "fviz_mfa_var",
    "fviz_mix",
    "fviz_mix_ind",
    "fviz_mix_var",
    "fviz_pca",
    "fviz_pca_biplot",
    "fviz_pca_ind",
    "fviz_pca_var",
    "fviz_pcoa",
    "fviz_pcoa_ind",
    "fviz_pcoa_shepard",
    "fviz_scatter",
    "fviz_screeplot",
    "set_axis"
]