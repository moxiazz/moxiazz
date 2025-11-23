"""
Chaotic Diffusion Framework
A Cold Diffusion-based framework for modeling and generating chaotic sequences.
"""

__version__ = "0.1.0"

from .core.chaotic_systems import LogisticMap, LorenzSystem, HenonMap
from .models.cold_diffusion import ChaoticColdDiffusion
from .models.degradation import ChaoticDegradation

__all__ = [
    "LogisticMap",
    "LorenzSystem", 
    "HenonMap",
    "ChaoticColdDiffusion",
    "ChaoticDegradation",
]
