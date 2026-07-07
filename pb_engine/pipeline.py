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

    # 3) Modelo ensemble. La densidad de sumas (criterio Monte Carlo) se calcula
    #    con el motor NUMBA (10M+) y se INYECTA al modelo -> no se recalcula el MC
    #    interno. half_life y prior_strength se propagan de forma coherente.
    logger.info("Construyendo ensemble (Numba MC + bootstrap + bayes)...")
    bw, _ = base.weights_bayesian(a, prior_strength=s.prior_strength)
    sum_density = engines.numba_monte_carlo(bw, s.mc_iters, r.n_white, s.seed)
    model = adv.EnsembleModel(cur, a, boot_reps=s.boot_reps, recent_k=s.recent_k,
                              seed=s.seed, half_life=s.half_life,
                              prior_strength=s.prior_strength, sum_density=sum_density)
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
    # Escalado MinMax CONJUNTO (un solo scaler sobre la union): asi las features
    # siguen siendo comparables entre 'tipicas' y 'populares' y la brecha que
    # Optuna maximiza es real, no un artefacto de escalar cada grupo por separado.
    raw_typ = model.feature_batch(typ_C)
    raw_pop = model.feature_batch(pop_C)
    both = engines.sklearn_scale(np.vstack([raw_typ, raw_pop]))
    typ_feat, pop_feat = both[:len(raw_typ)], both[len(raw_typ):]
    weighted = lambda feats, w: feats @ w
    best_w, gap = engines.optuna_tune_weights(weighted, pop_feat, typ_feat, seed=s.seed)
    optuna_res = dict(best_weights=best_w, separation_gap=gap)

    # 8) Seleccion final (diversa): <=2 blancas compartidas; relaja a <=3 si no
    #    hay suficientes, para no devolver menos jugadas de las pedidas.
    #
    #    - Modo determinista (variety=False): recorre C en orden de score -> el
    #      maximo global; SIEMPRE devuelve la misma jugada (los datos no cambian).
    #    - Modo variedad (variety=True): baraja el pool de las `top_pool` mejores
    #      (todas de score casi identico y misma probabilidad real) con una
    #      semilla que cambia por corrida -> jugadas distintas cada vez.
    pick_rng = np.random.default_rng(s.pick_seed if s.pick_seed is not None else s.seed)
    if s.variety:
        pool_n = min(len(C), s.top_pool)
        order = list(pick_rng.permutation(pool_n))
    else:
        order = list(range(len(C)))

    def _select(max_overlap):
        out = []
        for k in order:
            combo = tuple(int(x) for x in C[k])
            if all(len(set(combo) & set(c)) <= max_overlap for c, _ in out):
                out.append((combo, float(S[k])))
            if len(out) == s.n_plays:
                break
        return out

    def _select_coverage():
        """Seleccion greedy de MAXIMA COBERTURA: cada jugada suma los numeros
        menos usados, con un TOPE duro de repeticiones por numero para forzar el
        reparto (todas siguen siendo no-populares y de score alto)."""
        from collections import Counter
        used = Counter()
        picked, out = set(), []
        P = min(s.coverage_pool, len(C))
        # tope inicial ~ 3 para 14 jugadas; se relaja solo si el pool no alcanza
        cap = max(2, (s.n_plays * r.n_white) // (r.white_max - r.white_min + 1) + 2)
        for _ in range(s.n_plays):
            best_key, best_i = None, None
            while best_i is None:
                for i in range(P):
                    if i in picked:
                        continue
                    if any(used[int(x)] >= cap for x in C[i]):
                        continue          # respeta el tope de repeticiones
                    gain = sum(1.0 / (1.0 + used[int(x)]) for x in C[i])
                    key = (gain, -i)      # empate -> mayor score (indice menor)
                    if best_key is None or key > best_key:
                        best_key, best_i = key, i
                if best_i is None:
                    cap += 1              # nadie cumple el tope -> relajar
                    if cap > s.n_plays:
                        break
            if best_i is None:
                break
            picked.add(best_i)
            for x in C[best_i]:
                used[int(x)] += 1
            out.append((tuple(int(x) for x in C[best_i]), float(S[best_i])))
        return out

    if s.coverage:
        chosen = _select_coverage()
    else:
        chosen = _select(2)
        if len(chosen) < s.n_plays:
            chosen = _select(3)
    if len(chosen) < s.n_plays:
        logger.warning(f"Solo {len(chosen)} jugadas de {s.n_plays} pedidas.")

    # Powerball (equiprobable). Cobertura: PBs DISTINTOS repartidos en 1-26.
    # Variedad: al azar entre los mas frecuentes. Determinista: top por jugada.
    top_pbs = (np.argsort(a["pb_freq"].values)[::-1] + r.pb_min).astype(int)
    if s.coverage:
        spread = np.unique(np.linspace(r.pb_min, r.pb_max, len(chosen)).round().astype(int))
        while len(spread) < len(chosen):     # completar si hubo colisiones
            spread = np.union1d(spread, [x for x in top_pbs if x not in spread][:1])
        pbs = [int(x) for x in spread[:len(chosen)]]
    elif s.variety:
        pbs = [int(x) for x in pick_rng.choice(top_pbs[:10], size=len(chosen))]
    else:
        pbs = [int(top_pbs[(i) % len(top_pbs)]) for i in range(len(chosen))]
    plays = pd.DataFrame([
        dict(jugada=i, blancas=" ".join(f"{x:02d}" for x in c),
             powerball=pbs[i - 1], scs=round(sc, 2))
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
        elif kind == 1:                    # secuencia consecutiva (incluye 65-69)
            start = rng.integers(1, rules.white_max - rules.n_white + 2)
            c = np.arange(start, start + rules.n_white)
        else:                              # numeros redondos / misma decena
            base_d = rng.integers(0, 6) * 10
            c = np.sort(rng.choice(np.arange(base_d + 1, base_d + 10),
                                   rules.n_white, replace=False))
        out.append(c)
    return np.array(out, dtype=np.int16)
