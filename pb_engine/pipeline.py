"""Orquestador: integra Polars/DuckDB + SciPy + Numba + PyMC + DEAP + Optuna +
scikit-learn + Plotly, reutilizando la logica validada de `powerball_advanced`.

Devuelve un objeto con: validacion, pruebas estadisticas, modelo ensemble,
resultados de optimizadores, la(s) jugada(s) final(es) y figuras Plotly.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

import powerball_simulator as base
import powerball_advanced as adv
from . import data_io, engines
from .config import Settings


@dataclass
class Result:
    settings: Settings
    validation: dict
    eras: pd.DataFrame
    analysis: dict
    tests: dict
    model: adv.EnsembleModel
    exhaustive: tuple          # (C, S, meta)
    optimizers: dict
    pymc: dict
    optuna: dict
    plays: pd.DataFrame


def run(settings: Settings | None = None) -> Result:
    s = settings or Settings()
    r = s.rules

    # 1) Carga + validacion (Polars + DuckDB)
    df, rep = data_io.load(s.path_2016, s.path_2010, r)
    eras = data_io.era_summary(df)
    cur = data_io.current_era(df, r)

    # 2) Analisis base (frecuencias, pares, distribuciones) + pruebas (SciPy)
    a = base.analyze(cur)
    tests = adv.statistical_tests(cur, a)

    # 3) Modelo ensemble; se sustituye el Monte Carlo por el motor NUMBA (10M+)
    logger.info("Construyendo ensemble (bootstrap + bayes)...")
    model = adv.EnsembleModel(cur, a, mc_iters=200_000, boot_reps=s.boot_reps,
                              recent_k=s.recent_k, seed=s.seed)
    bw, _ = base.weights_bayesian(a, prior_strength=s.prior_strength)
    model.sum_density = engines.numba_monte_carlo(bw, s.mc_iters, r.n_white, s.seed)
    model.mc_iters = s.mc_iters
    # Aplica los pesos configurados (Pydantic-validados)
    model.WEIGHTS = s.weights.model_dump()

    # 4) Barrido exhaustivo con el Statistical Confidence Score
    C, S, meta = adv.exhaustive_scs(model, a)

    # 5) Optimizadores (validacion cruzada del optimo) — mismo objetivo RESTRINGIDO
    #    (penalizan patrones populares, igual que el barrido exhaustivo).
    logger.info("Validacion cruzada: DEAP GA + Simulated Annealing + Pareto...")

    def constrained_score(combo):
        C1 = np.sort(np.array(combo, ndmin=2).astype(np.int16))
        base_s = model.score_batch(C1)[0]
        if adv.undesirable_mask(C1, meta["sum_band"])[0]:
            return base_s - 100.0        # penalizacion fuerte por patron popular
        return base_s

    ga_combo, ga_s = engines.deap_ga(constrained_score, r)
    sa_combo, sa_s = adv.simulated_annealing(model, score_fn=constrained_score)
    front, knee = adv.pareto_front(model, C)
    optimizers = dict(
        exhaustive=(tuple(int(x) for x in C[0]), float(S[0])),
        deap_ga=(ga_combo, ga_s),
        annealing=(sa_combo, sa_s),
        pareto=dict(n_front=int(len(front)), knee=tuple(int(x) for x in C[knee])),
    )

    # 6) Inferencia bayesiana MCMC con PyMC (Powerball)
    try:
        pm_res = engines.pymc_bayesian_pb(a["pb_freq"].values.astype(int), seed=s.seed)
    except Exception as e:              # PyMC puede fallar por backend; se reporta
        logger.warning(f"PyMC no disponible en este entorno: {e}")
        pm_res = {"error": str(e)}

    # 7) Optuna: tuning de pesos separando 'popular' vs 'tipica' (clasificacion)
    logger.info("Optuna: tuning de pesos (separacion popular vs tipica)...")
    rng = np.random.default_rng(s.seed)
    sum_band = meta["sum_band"]
    # muestras 'tipicas' = alto SCS; 'populares' = patrones a evitar
    typ_C = C[:2000]
    pop_C = _popular_samples(rng, r, 2000)
    typ_feat = engines.sklearn_scale(model.feature_batch(typ_C))
    pop_feat = engines.sklearn_scale(model.feature_batch(pop_C))
    weighted = lambda feats, w: feats @ w
    best_w, gap = engines.optuna_tune_weights(weighted, pop_feat, typ_feat, seed=s.seed)
    optuna_res = dict(best_weights=best_w, separation_gap=gap)

    # 8) Seleccion final (diversa)
    chosen = []
    for k in range(len(C)):
        combo = tuple(int(x) for x in C[k])
        if all(len(set(combo) & set(c)) <= 2 for c, _ in chosen):
            chosen.append((combo, float(S[k])))
        if len(chosen) == s.n_plays:
            break
    plays = pd.DataFrame([
        dict(jugada=i, blancas=" ".join(f"{x:02d}" for x in c),
             powerball=model.best_pb, scs=round(sc, 2))
        for i, (c, sc) in enumerate(chosen, 1)
    ])

    return Result(s, rep, eras, a, tests, model, (C, S, meta),
                  optimizers, pm_res, optuna_res, plays)


def _popular_samples(rng, rules, n):
    """Genera combinaciones 'populares' (sesgos de jugadores) para el objetivo Optuna."""
    out = []
    for _ in range(n):
        kind = rng.integers(0, 3)
        if kind == 0:                      # cumpleanos: todo <=31
            c = np.sort(rng.choice(np.arange(1, 32), rules.n_white, replace=False))
        elif kind == 1:                    # secuencia consecutiva
            start = rng.integers(1, rules.white_max - rules.n_white)
            c = np.arange(start, start + rules.n_white)
        else:                              # numeros redondos / misma decena
            base_d = rng.integers(0, 6) * 10
            c = np.sort(rng.choice(np.arange(base_d + 1, base_d + 10),
                                   rules.n_white, replace=False))
        out.append(c)
    return np.array(out, dtype=np.int16)
