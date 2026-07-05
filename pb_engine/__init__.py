"""
pb_engine — Simulador estadistico avanzado de Powerball.

ADVERTENCIA: Powerball es aleatorio e independiente. Este paquete CLASIFICA
combinaciones bajo criterios estadisticos; NO predice el sorteo ni aumenta la
probabilidad real de ganar (1 entre 292.201.338 para toda combinacion).
"""
__version__ = "1.0.0"

from .config import Settings, ScoreWeights, RULES  # noqa: F401
