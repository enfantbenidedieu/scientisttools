# -*- coding: utf-8 -*-
from __future__ import annotations

from ._ca import CA
from ._fa import FA
from ._famd import FAMD
from ._farot import FArot
from ._mca import MCA
from ._mpca import MPCA
from ._pca import PCA
from ._pcamix import PCAmix
from ._pcarot import PCArot
from ._pcoa import PCoA 

__all__ = [
    "CA",
    "FA",
    "FAMD",
    "FArot",
    "MCA",
    "MPCA",
    "PCA",
    "PCAmix",
    "PCArot",
    "PCoA"
]