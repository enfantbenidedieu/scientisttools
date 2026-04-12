# -*- coding: utf-8 -*-
from __future__ import annotations

from ._bwca import BWCA
from ._cancorr import CANCORR
from ._cca import CCA
from ._coia import COIA
from ._pcaiv import PCAiv
from ._procrustes import Procrustes

__all__ = [
    "BWCA",
    "CANCORR",
    "CCA",
    "COIA",
    "PCAiv",
    "Procrustes"
]