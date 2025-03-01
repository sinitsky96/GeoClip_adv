"""
Adversarial attacks module.
""" 

from .pgd import PGD
from .universal import UPGD
from .patch_attack import PatchAttack

__all__ = ['PGD', 'UPGD', 'PatchAttack']