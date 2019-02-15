# -*- coding: utf-8 -*-

"""Top-level package for Regularization parameter estimation."""

__author__ = """Rien Lagerwerf"""
__email__ = 'm.j.lagerwerf@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.
from .reg_param import hello_world
from .problem_class import problem_definition_class
from .method_class import method_class
from .interp_class import recon_class, interp_class, recon_class_noref, interp_class_noref
from .interp_2param_class import recon_2param_class, interp_2param_class

# Denk ff na over welke ik echt nodig heb
from .support_functions import *


