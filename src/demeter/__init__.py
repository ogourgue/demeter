"""Demeter: A Python package for multiscale biogeomorphic modeling."""

__version__ = "2.3.0"

# Core modules - always available
from .cellular_automaton import CellularAutomaton
from . import cellular_automaton_functions

# Coupling modules - always available
from . import ca2tel
from . import tel2ca

# Optional Telemac support
try:
    from . import telemac
    HAS_TELEMAC = True
except ImportError:
    HAS_TELEMAC = False
    telemac = None

__all__ = [
    'CellularAutomaton',
    'cellular_automaton_functions',
    'ca2tel',
    'tel2ca',
    'telemac',
    'HAS_TELEMAC',
]
