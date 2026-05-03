# -*- coding: utf-8 -*-
from __future__ import annotations

from ._bgc import BGC
from ._dccswa import DCCSWA
from ._dgpa import DGPA
from ._distatis import DISTATIS
from ._dmcoia import DMCOIA
from ._dmfa import DMFA
from ._dstatis import DSTATIS
from ._fcpca import FCPCA
from ._ica import ICA
from ._mcoia import MCOIA
from ._mfa import MFA
from ._mgpca import mgPCA
from ._statis import STATIS

__all__ = [
    "BGC",
    "DCCSWA",
    "DGPA",
    "DISTATIS",
    "DMCOIA",
    "DMFA",
    "DSTATIS",
    "FCPCA",
    "ICA",
    "MCOIA",
    "MFA",
    "mgPCA",
    "STATIS"
]