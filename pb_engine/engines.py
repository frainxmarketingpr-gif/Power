"""Motores de calculo: Numba (Monte Carlo), PyMC (MCMC bayesiano),
DEAP (algoritmo genetico), Optuna (tuning de pesos), scikit-learn (escalado).

Cada motor esta integrado de forma real (no decorativa). Ninguno predice el
sorteo; solo alimentan el ranking multicriterio."""
from __future__ import annotations

import numpy as np
from numba import njit
from loguru import logger


# ===========================================================================
# NUMBA — Monte Carlo acelerado (muestreo ponderado sin reemplazo)
# ===========================================================================
@njit(fastmath=True)
def _mc_sum_hist(white_w, iters, n_white, seed):
    """Simula `iters` sorteos de 5 blancas sin reemplazo segun pesos `white_w`
    y acumula el histograma de la SUMA. Compilado a maquina por Numba."""
    np.random.seed(seed)
    K = white_w.shape[0]
    hist = np.zeros(5 * K + 1, dtype=np.float64)
    w = np.empty(K, dtype=np.float64)
    for _ in range(iters):
        for k in range(K):
            w[k] = white_w[k]
        total = 0.0
        for k in range(K):
            total += w[k]
        s = 0
        for _pick in range(n_white):
            r = np.random.random() * total
            acc = 0.0
            idx = -1
            for k in range(K):
                if w[k] <= 0.0:          # bola ya extraida: no re-elegible
                    continue
                acc += w[k]
                if acc >= r:
                    idx = k
                    break
            if idx < 0:                  # deriva de flotantes: ultimo valido
                for k in range(K - 1, -1, -1):
                    if w[k] > 0.0:
                        idx = k
                        break
            s += idx + 1                 # numero = indice + 1
            total -= w[idx]
            w[idx] = 0.0
        hist[s] += 1.0
    return hist


def numba_monte_carlo(white_w: np.ndarray, iters: int, n_white: int = 5,
                      seed: int = 20260705) -> np.ndarray:
    """Densidad empirica (0..1) de la suma de 5 blancas tras `iters` sorteos."""
    logger.info(f"Monte Carlo (Numba): {iters:,} iteraciones...")
    if int((white_w > 0).sum()) < n_white:
        raise ValueError(f"Se requieren >= {n_white} pesos positivos para muestrear "
                         f"sin reemplazo; hay {int((white_w > 0).sum())}.")
    hist = _mc_sum_hist(white_w.astype(np.float64), int(iters), n_white, seed)
    if hist.max() > 0:
        hist /= hist.max()
    kern = np.array([0.25, 0.5, 0.25])
    hist = np.convolve(hist, kern, mode="same")
    if hist.max() > 0:
        hist /= hist.max()
    return hist


# ===========================================================================
# PyMC — Inferencia bayesiana / MCMC (Dirichlet-Multinomial del Powerball)
# ===========================================================================
def pymc_bayesian_pb(pb_counts: np.ndarray, draws: int = 400, tune: int = 400,
                     seed: int = 20260705) -> dict:
    """Posterior Dirichlet-Multinomial de la probabilidad de cada Powerball
    via NUTS (PyMC). Muestra que el intervalo creible contiene 1/26 (uniforme):
    evidencia bayesiana de que NO hay bolas 'favorecidas'."""
    import pymc as pm
    import arviz as az

    K = len(pb_counts)
    n = int(pb_counts.sum())
    logger.info(f"PyMC NUTS: posterior Dirichlet-Multinomial ({K} categorias)...")
    with pm.Model():
        p = pm.Dirichlet("p", a=np.ones(K))
        pm.Multinomial("obs", n=n, p=p, observed=pb_counts)
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                          random_seed=seed, progressbar=False,
                          compute_convergence_checks=False)
    post = idata.posterior["p"].values.reshape(-1, K)
    mean = post.mean(axis=0)
    hdi = az.hdi(idata, var_names=["p"], hdi_prob=0.94)["p"].values  # (K,2)
    uniform = 1.0 / K
    contains_uniform = np.array([lo <= uniform <= hi for lo, hi in hdi])
    return dict(mean=mean, hdi=hdi, uniform=uniform,
                pct_credible_contains_uniform=float(contains_uniform.mean()))


# ===========================================================================
# scikit-learn — escalado/normalizacion del feature-matrix del ranking
# ===========================================================================
def sklearn_scale(features: np.ndarray) -> np.ndarray:
    """MinMax a [0,1] por columna (robusto a escalas heterogeneas)."""
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit_transform(features)


# ===========================================================================
# DEAP — Algoritmo Genetico sobre combinaciones de 5 blancas
# ===========================================================================
def deap_ga(score_one, rules, generations: int = 60, pop_size: int = 300,
            seed: int = 7):
    """Optimiza el SCS con DEAP. `score_one(tuple)->float`. Devuelve (combo, score)."""
    import random
    from deap import base, creator, tools

    random.seed(seed)
    lo, hi, k = rules.white_min, rules.white_max, rules.n_white

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("combo", lambda: sorted(random.sample(range(lo, hi + 1), k)))
    tb.register("individual", tools.initIterate, creator.Individual, tb.combo)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (score_one(tuple(ind)),))

    def cx(a, b):
        genes = list(set(a) | set(b))          # union: siempre >= k elementos
        random.shuffle(genes)
        a[:] = sorted(genes[:k])
        random.shuffle(genes)
        b[:] = sorted(genes[:k])
        return a, b

    def mut(ind):
        i = random.randrange(k)
        new = random.randint(lo, hi)
        while new in ind:
            new = random.randint(lo, hi)
        ind[i] = new
        ind[:] = sorted(ind)
        return (ind,)

    tb.register("mate", cx)
    tb.register("mutate", mut)
    tb.register("select", tools.selTournament, tournsize=3)

    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    for _ in range(generations):
        offspring = list(map(tb.clone, tb.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                tb.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3:
                tb.mutate(m); del m.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)
        pop = tools.selBest(pop + offspring, pop_size)
    best = tools.selBest(pop, 1)[0]
    return tuple(best), float(best.fitness.values[0])


# ===========================================================================
# Optuna — tuning automatico de los 8 pesos del score
# ===========================================================================
def optuna_tune_weights(feature_fn, popular_feats, typical_feats, n_trials: int = 60,
                        seed: int = 20260705):
    """Ajusta los pesos para MAXIMIZAR la separacion del score entre combinaciones
    'populares' (a evitar) y 'tipicas/no-populares'. Es un objetivo de CLASIFICACION
    bien definido, NO de prediccion del sorteo.

    feature_fn(feats, w)->scores ; popular_feats/typical_feats: (m,8) ya escalados.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    keys = ["frecuencia", "reciente", "bayes", "montecarlo",
            "bootstrap", "entropia", "diversidad", "popularidad"]

    def objective(trial):
        raw = np.array([trial.suggest_float(k, 0.02, 0.40) for k in keys])
        w = raw / raw.sum()
        s_typ = feature_fn(typical_feats, w).mean()
        s_pop = feature_fn(popular_feats, w).mean()
        return s_typ - s_pop            # maximizar la brecha

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = np.array([study.best_params[k] for k in keys])
    best /= best.sum()
    return dict(zip(keys, best)), study.best_value
