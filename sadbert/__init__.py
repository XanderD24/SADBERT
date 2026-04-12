"""
SADBERT — Stereotype-content Analysis with DistilBERT
======================================================

A Python package for identifying stereotype content dimensions
(warmth, competence, morality, …) in natural language text and
classifying their valence (positive / neutral / negative).

Quick start
-----------
>>> import sadbert
>>> sadbert.get_stereotype_content("She is a warm and caring nurse.")

Or use the class directly for more control:

>>> from sadbert import SADBERT
>>> model = SADBERT(device="cuda", batch_size=64)
>>> model.get_stereotype_content(["honest", "lazy", "brilliant"])
"""

from .core import (
    SADBERT,
    get_stereotype_content,
    ALL_CATS,
    MAJOR_CATS,
    MINOR_CATS,
)

__version__ = "0.1.0"
__author__  = "Xander Deanhardt"
__email__   = "xanderd24@uchicago.edu"
__license__ = "MIT"

__all__ = [
    "SADBERT",
    "get_stereotype_content",
    "ALL_CATS",
    "MAJOR_CATS",
    "MINOR_CATS",
]
