# -*- coding: utf-8 -*-
from __future__ import annotations

from ._ca import CA, statsCA
from ._fa import FA, statsFA
from ._famd import FAMD
from ._farot import FArot
from ._mca import MCA, statsMCA
from ._mpca import MPCA
from ._pca import PCA, statsPCA
from ._pcamix import PCAmix
from ._pcarot import PCArot, statsPCArot
from ._pcoa import PCoA 

__all__ = [
    "CA", "statsCA",
    "FA", "statsFA",
    "FAMD",
    "FArot",
    "MCA", "statsMCA",
    "MPCA",
    "PCA", "statsPCA",
    "PCAmix",
    "PCArot", "statsPCArot",
    "PCoA"
]