"""API REST con FastAPI. Ejecutar:  uvicorn pb_engine.api:app --reload

Endpoints:
  GET /health          -> estado
  GET /rules           -> reglas y probabilidades exactas
  GET /analysis        -> frecuencias, pruebas estadisticas, eras
  GET /play?n=1        -> jugada(s) sugerida(s) con SCS (NO prediccion)
"""
from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI, Query
from pydantic import BaseModel

from .config import Settings
from . import pipeline

app = FastAPI(title="Powerball Simulator API",
              description="Analisis estadistico de Powerball. NO predice el sorteo "
                          "ni aumenta la probabilidad real de ganar.",
              version="1.0.0")


class Play(BaseModel):
    jugada: int
    blancas: str
    powerball: int
    scs: float
    aviso: str = ("El SCS mide calidad estadistica, NO probabilidad de ganar. "
                  "Toda combinacion tiene la misma probabilidad: 1 entre 292.201.338.")


@lru_cache(maxsize=1)
def _cached_result(n: int):
    return pipeline.run(Settings(n_plays=n))


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": "Powerball es aleatorio; esto no predice."}


@app.get("/rules")
def rules():
    r = Settings().rules
    return {"white": [r.white_min, r.white_max], "powerball": [r.pb_min, r.pb_max],
            "white_combos": r.white_combos, "jackpot_odds_1_in": r.jackpot_odds}


@app.get("/analysis")
def analysis():
    res = _cached_result(1)
    a = res.analysis
    t = res.tests
    return {
        "sorteos_totales": res.validation["draws_total"],
        "era_actual_sorteos": a["n_draws"],
        "calientes": {int(k): int(v) for k, v in a["hot"].items()},
        "frios": {int(k): int(v) for k, v in a["cold"].items()},
        "chi2_blancas_p": round(t["chi2_white"][1], 4),
        "chi2_powerball_p": round(t["chi2_pb"][1], 4),
        "runs_test_p": round(t["runs_test"][3], 4),
        "entropia_pct_maximo": round(100 * t["entropy_white"][2], 2),
        "conclusion": "Compatible con aleatoriedad pura (sin poder predictivo).",
    }


@app.get("/play", response_model=list[Play])
def play(n: int = Query(1, ge=1, le=20)):
    res = _cached_result(n)
    return [Play(**row) for row in res.plays.to_dict("records")]
