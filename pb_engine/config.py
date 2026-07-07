"""Configuracion validada con Pydantic + logging (Loguru) + consola (Rich)."""
from __future__ import annotations

import sys
from pydantic import BaseModel, Field, field_validator, model_validator
from loguru import logger
from rich.console import Console

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configura Loguru con un formato limpio. Se invoca EXPLICITAMENTE desde
    los entrypoints (cli/api/app); NO en el import, para no pisar el logging de
    una aplicacion anfitriona que monte este paquete."""
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


class Rules(BaseModel):
    """Reglas de la matriz vigente de Powerball (desde 2015-10-07)."""
    white_min: int = 1
    white_max: int = 69
    n_white: int = 5
    pb_min: int = 1
    pb_max: int = 26
    era_start: str = "2015-10-07"

    @property
    def white_combos(self) -> int:
        from math import comb
        return comb(self.white_max - self.white_min + 1, self.n_white)

    @property
    def jackpot_odds(self) -> int:
        return self.white_combos * (self.pb_max - self.pb_min + 1)


class ScoreWeights(BaseModel):
    """Pesos del Statistical Confidence Score. Deben sumar 1.0."""
    frecuencia: float = 0.20
    reciente: float = 0.15
    bayes: float = 0.15
    montecarlo: float = 0.10
    bootstrap: float = 0.10
    entropia: float = 0.10
    diversidad: float = 0.10
    popularidad: float = 0.10

    @model_validator(mode="after")
    def _sum_to_one(self):
        s = sum(self.model_dump().values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Los pesos deben sumar 1.0 (suman {s:.4f})")
        return self

    def as_array(self):
        import numpy as np
        return np.array([self.frecuencia, self.reciente, self.bayes, self.montecarlo,
                         self.bootstrap, self.entropia, self.diversidad, self.popularidad])


class Settings(BaseModel):
    """Configuracion global del pipeline."""
    path_2016: str = "powerball_results_20160601_to_20260601.xlsx"
    path_2010: str = "powerball_resultados.xlsx"
    mc_iters: int = 10_000_000
    boot_reps: int = 2000
    recent_k: int = 10
    half_life: int = 120
    prior_strength: float = 69.0
    seed: int = 20260705
    n_plays: int = 1
    # Variedad: si True, elige al azar entre las mejores combinaciones (todas de
    # score casi identico y misma probabilidad real) -> jugadas distintas por
    # corrida. Si False, es determinista y devuelve siempre el maximo global.
    variety: bool = False
    top_pool: int = 500            # tamano del pool de mejores para la variedad
    pick_seed: int | None = None   # semilla de la eleccion (None -> usa `seed`)
    # Modo grupo/cobertura: elige N jugadas que ENTRE TODAS cubran la mayor
    # variedad de numeros (minima repeticion) -> ideal para una bolsa/syndicate.
    coverage: bool = False
    coverage_pool: int = 20000     # pool de mejores (por score) sobre el que cubrir
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    rules: Rules = Field(default_factory=Rules)

    @field_validator("mc_iters")
    @classmethod
    def _positive(cls, v):
        if v < 100_000:
            raise ValueError("mc_iters demasiado bajo para estabilidad Monte Carlo")
        return v


RULES = Rules()
