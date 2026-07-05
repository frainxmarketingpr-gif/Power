#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 MODULO MATEMATICO AVANZADO — POWERBALL (capa ensemble multicriterio)
============================================================================
Se apoya en `powerball_simulator.py` (carga, limpieza, validacion, analisis)
y anade:

  * BATERIA DE PRUEBAS ESTADISTICAS (exploratorias / de honestidad):
      - Chi-cuadrado de bondad de ajuste a la uniforme (blancas y Powerball)
      - Kolmogorov-Smirnov sobre la distribucion de sumas
      - Runs Test de Wald-Wolfowitz (aleatoriedad de la serie)
      - Autocorrelacion de la serie de sumas
      - Entropia de Shannon de las frecuencias (vs. maxima teorica)
      - Cadena de Markov (transiciones aparecio/no-aparecio) — SOLO exploratoria

  * STATISTICAL CONFIDENCE SCORE (0-100), funcion compuesta de 8 criterios:
      20% Frecuencia historica     10% Bootstrap
      15% Frecuencia reciente      10% Entropia de la combinacion
      15% Inferencia bayesiana     10% Diversidad vs. sorteos recientes
      10% Monte Carlo (10M+)       10% Penalizacion por patrones populares

  * EVALUACION EXHAUSTIVA del espacio completo C(69,5)=11.238.513 combos
    (x26 PB = 292.201.338 boletos) -> optimo GLOBAL del subespacio recomendable
    (tras descartar patrones populares con el filtro canonico).

  * OPTIMIZACION (validacion cruzada del optimo): Algoritmo Genetico,
    Simulated Annealing y frente de Pareto multiobjetivo. Confirman que
    convergen a la misma region de alto score que el barrido exhaustivo.

ADVERTENCIA: nada de esto predice el sorteo. Si el sorteo es aleatorio e
independiente, NINGUNA combinacion tiene mayor probabilidad intrinseca de
salir. El score solo CLASIFICA combinaciones bajo criterios definidos.
============================================================================
"""
from __future__ import annotations
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

import powerball_simulator as base
from powerball_simulator import (WHITE_MIN, WHITE_MAX, PB_MIN, PB_MAX, N_WHITE,
                                 WHITE_COMBOS, JACKPOT_ODDS, MATCH5_ODDS)

WCOLS = [f"n{i}" for i in range(1, 6)]


# ===========================================================================
# A. BATERIA DE PRUEBAS ESTADISTICAS
# ===========================================================================
def _lilliefors_normal(x: np.ndarray, n_sim: int = 1000, seed: int = 20260705):
    """Test KS de normalidad con parametros estimados (Lilliefors). El p-valor
    se calibra por Monte Carlo: se re-muestrean normales del mismo tamano,
    se re-estiman media/sd y se compara el estadistico D. Sin dependencias
    externas y estadisticamente valido para parametros estimados."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=0)
    if sd == 0:
        return float("nan"), float("nan")
    z = (x - x.mean()) / sd
    d0 = stats.kstest(z, "norm").statistic
    rng = np.random.default_rng(seed)
    n = len(x)
    count = 0
    for _ in range(n_sim):
        s = rng.standard_normal(n)
        s = (s - s.mean()) / s.std(ddof=0)
        if stats.kstest(s, "norm").statistic >= d0:
            count += 1
    return float(d0), (count + 1) / (n_sim + 1)


def statistical_tests(df: pd.DataFrame, a: dict) -> dict:
    n = a["n_draws"]
    out = {}

    # --- Chi-cuadrado: uniformidad de las bolas blancas ---
    obs_w = a["white_freq"].values.astype(float)
    exp_w = np.full_like(obs_w, obs_w.sum() / len(obs_w))
    chi_w, p_w = stats.chisquare(obs_w, exp_w)
    out["chi2_white"] = (chi_w, p_w, len(obs_w) - 1)

    # --- Chi-cuadrado: uniformidad del Powerball ---
    obs_p = a["pb_freq"].values.astype(float)
    exp_p = np.full_like(obs_p, obs_p.sum() / len(obs_p))
    chi_p, p_p = stats.chisquare(obs_p, exp_p)
    out["chi2_pb"] = (chi_p, p_p, len(obs_p) - 1)

    # --- Kolmogorov-Smirnov (Lilliefors): sumas vs. Normal con parametros
    #     estimados de la propia muestra. El p-valor se calibra por Monte Carlo
    #     (la distribucion nula de D con parametros estimados NO es la de KS
    #     estandar; usarla directamente inflaria el p-valor).
    sums = a["sums"].astype(float)
    ks_d, ks_p = _lilliefors_normal(sums, n_sim=1000, seed=20260705)
    out["ks_sums"] = (ks_d, ks_p)

    # --- Runs Test (Wald-Wolfowitz) sobre la serie de sumas ---
    #     Se DESCARTAN los empates con la mediana (requisito del test).
    med = np.median(sums)
    signs = (sums[sums != med] > med).astype(int)
    runs = 1 + int((np.diff(signs) != 0).sum())
    n1 = int((signs == 1).sum()); n2 = int((signs == 0).sum())
    if n1 == 0 or n2 == 0:                       # serie degenerada
        out["runs_test"] = (runs, float("nan"), float("nan"), float("nan"))
    else:
        mu_r = 1 + 2 * n1 * n2 / (n1 + n2)
        var_r = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        z_runs = (runs - mu_r) / np.sqrt(var_r)
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
        out["runs_test"] = (runs, mu_r, z_runs, p_runs)

    # --- Autocorrelacion de las sumas (lags 1..5) ---
    x = sums - sums.mean()
    denom = np.sum(x * x)
    ac = [float(np.sum(x[:-k] * x[k:]) / denom) for k in range(1, 6)]
    out["autocorr_sums"] = ac

    # --- Entropia de Shannon de las frecuencias de blancas ---
    p = obs_w / obs_w.sum()
    pnz = p[p > 0]                               # evita 0*log2(0)=NaN
    H = -np.sum(pnz * np.log2(pnz))
    Hmax = np.log2(len(obs_w))
    out["entropy_white"] = (H, Hmax, H / Hmax)

    # --- Cadena de Markov exploratoria: P(sale en t | salio en t-1) ---
    arr = df[WCOLS].values
    appeared = np.zeros((len(arr), WHITE_MAX), dtype=bool)
    for i, row in enumerate(arr):
        for v in row:
            appeared[i, v - 1] = True
    prev = appeared[:-1]; cur = appeared[1:]
    p11 = cur[prev].mean()            # P(aparece | aparecio antes)
    p10 = cur[~prev].mean()           # P(aparece | no aparecio antes)
    base_rate = appeared.mean()
    out["markov"] = (p11, p10, base_rate)
    return out


# ===========================================================================
# B. PRECOMPUTO DE VECTORES PARA EL STATISTICAL CONFIDENCE SCORE
# ===========================================================================
class EnsembleModel:
    """Precalcula todos los vectores/superficies de los 8 criterios."""
    WEIGHTS = {
        "frecuencia":  0.20,
        "reciente":    0.15,
        "bayes":       0.15,
        "montecarlo":  0.10,
        "bootstrap":   0.10,
        "entropia":    0.10,
        "diversidad":  0.10,
        "popularidad": 0.10,
    }

    def __init__(self, df: pd.DataFrame, a: dict, mc_iters: int = 10_000_000,
                 boot_reps: int = 2000, recent_k: int = 10, seed: int = 20260705,
                 half_life: int = 120, prior_strength: float = 69.0,
                 sum_density: np.ndarray | None = None):
        self.a = a
        self.n = a["n_draws"]
        self.rng = np.random.default_rng(seed)
        K = WHITE_MAX - WHITE_MIN + 1

        # (1) Frecuencia historica normalizada
        wf = a["white_freq"].values.astype(float)
        self.freq_n = wf / wf.max()

        # (2) Frecuencia reciente (decaimiento exponencial configurable)
        rw, _ = base.weights_recency(df, half_life=half_life)
        self.recent_n = rw / rw.max()

        # (3) Inferencia bayesiana (posterior Dirichlet, prior configurable)
        bw, _ = base.weights_bayesian(a, prior_strength=prior_strength)
        self.bayes_n = bw / bw.max()

        # (4) Monte Carlo (10M+): densidad empirica de la SUMA bajo el modelo
        #     bayesiano. Se puede INYECTAR (p. ej. desde el motor Numba) para no
        #     recalcularla; si no, se estima aqui con `mc_iters`.
        self.mc_iters = mc_iters
        self.sum_density = (sum_density if sum_density is not None
                            else self._monte_carlo_sum_density(bw, mc_iters))

        # (5) Bootstrap: prob. por numero re-muestreando sorteos con reemplazo
        self.boot_n, self.boot_ci = self._bootstrap(df, boot_reps)

        # (6) Entropia -> se calcula por combinacion (decadas), no por numero.
        # (7) Diversidad vs. sorteos recientes
        recent_nums = set(df[WCOLS].values[-recent_k:].ravel().tolist())
        self.recent_mask = np.zeros(K + 1, dtype=bool)
        for v in recent_nums:
            self.recent_mask[v] = True
        self.recent_k = recent_k

        # (8) Popularidad: se calcula por combinacion (sesgos de jugadores).

        # Powerball "tipico" (cosmetico; el PB es equiprobable)
        self.best_pb = int(a["pb_freq"].idxmax())

    # --- Monte Carlo de 10M+ sorteos -> histograma de sumas ---
    def _monte_carlo_sum_density(self, white_w, iters):
        balls = np.arange(WHITE_MIN, WHITE_MAX + 1)
        logw = np.log(white_w)
        hist = np.zeros(5 * WHITE_MAX + 1, dtype=np.float64)
        batch = 500_000
        done = 0
        while done < iters:
            m = min(batch, iters - done)
            g = self.rng.gumbel(size=(m, len(balls)))
            idx = np.argpartition(logw + g, -N_WHITE, axis=1)[:, -N_WHITE:]
            s = balls[idx].sum(axis=1)
            np.add.at(hist, s, 1.0)
            done += m
        hist /= hist.max()
        # suavizado ligero
        k = np.array([0.25, 0.5, 0.25])
        hist = np.convolve(hist, k, mode="same")
        hist /= hist.max()
        return hist

    # --- Bootstrap de la frecuencia por numero ---
    def _bootstrap(self, df, reps):
        arr = df[WCOLS].values
        n = len(arr)
        # matriz sorteo x numero (conteo por sorteo)
        D = np.zeros((n, WHITE_MAX), dtype=np.int8)
        for i, row in enumerate(arr):
            for v in row:
                D[i, v - 1] += 1
        est = np.empty((reps, WHITE_MAX))
        for b in range(reps):
            samp = self.rng.integers(0, n, size=n)
            est[b] = D[samp].mean(axis=0)      # prob por numero en el resample
        mean = est.mean(axis=0)
        ci = np.percentile(est, [2.5, 97.5], axis=0)
        return mean / mean.max(), ci

    # --- Matriz de 8 features (0..1) por combinacion (para Optuna/sklearn) ---
    def feature_batch(self, C: np.ndarray) -> np.ndarray:
        idx = C - WHITE_MIN
        s_freq = self.freq_n[idx].mean(axis=1)
        s_rec = self.recent_n[idx].mean(axis=1)
        s_bay = self.bayes_n[idx].mean(axis=1)
        s_mc = self.sum_density[C.sum(axis=1)]
        s_boot = self.boot_n[idx].mean(axis=1)
        dec = C // 10
        H = np.zeros(len(C))
        for d in range(0, 7):
            cnt = (dec == d).sum(axis=1) / N_WHITE
            with np.errstate(divide="ignore", invalid="ignore"):
                H += np.where(cnt > 0, -cnt * np.log2(cnt), 0.0)
        s_ent = H / np.log2(N_WHITE)
        s_div = 1.0 - self.recent_mask[C].sum(axis=1) / N_WHITE
        frac_le31 = (C <= 31).sum(axis=1) / N_WHITE
        consec = (np.diff(C, axis=1) == 1).sum(axis=1) / (N_WHITE - 1)
        low_spread = np.clip((25 - (C[:, -1] - C[:, 0])) / 25.0, 0, 1)
        s_pop = 1.0 - np.clip(0.45 * frac_le31 + 0.30 * consec + 0.25 * low_spread, 0, 1)
        return np.stack([s_freq, s_rec, s_bay, s_mc, s_boot, s_ent, s_div, s_pop], axis=1)

    # --- Score compuesto para un LOTE de combinaciones (vectorizado) ---
    def score_batch(self, C: np.ndarray) -> np.ndarray:
        """C: (m,5) int, blancas ORDENADAS. Devuelve SCS 0-100 por fila."""
        idx = C - WHITE_MIN
        s_freq = self.freq_n[idx].mean(axis=1)
        s_rec = self.recent_n[idx].mean(axis=1)
        s_bay = self.bayes_n[idx].mean(axis=1)
        ssum = C.sum(axis=1)
        s_mc = self.sum_density[ssum]
        s_boot = self.boot_n[idx].mean(axis=1)

        # Entropia por combinacion (distribucion en 7 decenas)
        dec = C // 10
        H = np.zeros(len(C))
        for d in range(0, 7):
            cnt = (dec == d).sum(axis=1) / N_WHITE
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(cnt > 0, -cnt * np.log2(cnt), 0.0)
            H += term
        s_ent = H / np.log2(N_WHITE)          # 0..1

        # Diversidad vs. recientes
        recent_hits = self.recent_mask[C].sum(axis=1)
        s_div = 1.0 - recent_hits / N_WHITE

        # Penalizacion por patrones populares -> score = 1 - popularidad
        frac_le31 = (C <= 31).sum(axis=1) / N_WHITE
        consec = (np.diff(C, axis=1) == 1).sum(axis=1) / (N_WHITE - 1)
        spread = C[:, -1] - C[:, 0]
        low_spread = np.clip((25 - spread) / 25.0, 0, 1)
        pop = 0.45 * frac_le31 + 0.30 * consec + 0.25 * low_spread
        s_pop = 1.0 - np.clip(pop, 0, 1)

        W = self.WEIGHTS
        scs = (W["frecuencia"] * s_freq + W["reciente"] * s_rec +
               W["bayes"] * s_bay + W["montecarlo"] * s_mc +
               W["bootstrap"] * s_boot + W["entropia"] * s_ent +
               W["diversidad"] * s_div + W["popularidad"] * s_pop)
        return 100.0 * scs

    def score_one(self, combo) -> float:
        return float(self.score_batch(np.sort(np.array(combo, ndmin=2)))[0])


# ===========================================================================
# C. FILTRO DE PATRONES POPULARES / REDUNDANTES (vectorizado)
# ===========================================================================
def undesirable_mask(C: np.ndarray, sum_band: tuple[int, int]) -> np.ndarray:
    """Delega en la mascara canonica de powerball_simulator (fuente unica del
    filtro, ya con la correccion de '>=4 consecutivos' y cumpleanos)."""
    return base.undesirable_mask_vec(C, sum_band)


# ===========================================================================
# D. BARRIDO EXHAUSTIVO CON EL STATISTICAL CONFIDENCE SCORE
# ===========================================================================
def exhaustive_scs(model: EnsembleModel, a: dict, n_keep: int = 20000,
                   batch: int = 1_000_000):
    sums = a["sums"]
    sum_band = (int(np.percentile(sums, 10)), int(np.percentile(sums, 90)))
    gen = itertools.combinations(range(WHITE_MIN, WHITE_MAX + 1), N_WHITE)
    best_C = np.zeros((0, N_WHITE), dtype=np.int16)
    best_s = np.zeros(0)
    total = kept = 0
    while True:
        chunk = list(itertools.islice(gen, batch))
        if not chunk:
            break
        C = np.asarray(chunk, dtype=np.int16)
        total += len(C)
        good = ~undesirable_mask(C, sum_band)
        kept += int(good.sum())
        Cg = C[good]
        s = model.score_batch(Cg)
        best_C = np.vstack([best_C, Cg])
        best_s = np.concatenate([best_s, s])
        if len(best_s) > n_keep:
            top = np.argpartition(best_s, -n_keep)[-n_keep:]
            best_C, best_s = best_C[top], best_s[top]
    if len(best_s) == 0:
        raise ValueError("Ninguna combinacion supero los filtros; revisa "
                         "sum_band/datos de entrada.")
    order = np.argsort(best_s)[::-1]
    return best_C[order], best_s[order], dict(
        white_combos=total, tickets=total * (PB_MAX - PB_MIN + 1),
        kept=kept, sum_band=sum_band)


# ===========================================================================
# E. OPTIMIZADORES (validacion cruzada del optimo)
# ===========================================================================
def _rand_combo(rng):
    return np.sort(rng.choice(np.arange(WHITE_MIN, WHITE_MAX + 1), N_WHITE, replace=False))


def genetic_optimize(model, generations=120, pop=400, seed=7, score_fn=None):
    rng = np.random.default_rng(seed)
    sf = score_fn or (lambda P: model.score_batch(P))
    P = np.array([_rand_combo(rng) for _ in range(pop)], dtype=np.int16)
    best_c, best_f = None, -np.inf                  # elitismo: mejor global
    for _ in range(generations):
        fit = sf(P)
        b = int(np.argmax(fit))
        if fit[b] > best_f:
            best_c, best_f = P[b].copy(), float(fit[b])
        elite = P[np.argsort(fit)[::-1][:pop // 5]]
        children = [elite[0].copy()]                # el mejor pasa intacto
        while len(children) < pop:
            i, j = rng.integers(0, len(elite), 2)
            genes = np.unique(np.concatenate([elite[i], elite[j]]))
            if rng.random() < 0.3:  # mutacion
                genes = np.append(genes, rng.integers(WHITE_MIN, WHITE_MAX + 1))
            genes = np.unique(genes)
            if len(genes) < N_WHITE:
                continue
            child = np.sort(rng.choice(genes, N_WHITE, replace=False))
            children.append(child)
        P = np.array(children, dtype=np.int16)
    fit = sf(P)
    b = int(np.argmax(fit))
    if fit[b] > best_f:
        best_c, best_f = P[b].copy(), float(fit[b])
    return tuple(int(x) for x in best_c), best_f


def simulated_annealing(model, iters=20000, T0=5.0, seed=11, score_fn=None):
    rng = np.random.default_rng(seed)
    sf = score_fn or model.score_one
    cur = _rand_combo(rng)
    cur_s = sf(cur)
    best, best_s = cur, cur_s
    for t in range(iters):
        T = T0 * (1 - t / iters) + 1e-3
        nb = cur.copy()
        pos = rng.integers(0, N_WHITE)
        newv = rng.integers(WHITE_MIN, WHITE_MAX + 1)
        while newv in nb:
            newv = rng.integers(WHITE_MIN, WHITE_MAX + 1)
        nb[pos] = newv
        nb = np.sort(nb)
        s = sf(nb)
        if s > cur_s or rng.random() < np.exp((s - cur_s) / T):
            cur, cur_s = nb, s
            if s > best_s:
                best, best_s = nb, s
    return tuple(int(x) for x in best), float(best_s)


def pareto_front(model, C: np.ndarray, top=2000):
    """Frente de Pareto sobre 3 objetivos (maximizar): tipicidad, no-popularidad,
    diversidad. Devuelve indices no-dominados y el 'knee' (mayor SCS)."""
    C = C[:top]
    idx = C - WHITE_MIN
    typ = (model.freq_n[idx].mean(1) + model.bayes_n[idx].mean(1)) / 2
    frac_le31 = (C <= 31).sum(1) / N_WHITE
    consec = (np.diff(C, 1) == 1).sum(1) / (N_WHITE - 1)
    unpop = 1 - np.clip(0.6 * frac_le31 + 0.4 * consec, 0, 1)
    div = 1 - model.recent_mask[C].sum(1) / N_WHITE
    obj = np.stack([typ, unpop, div], axis=1)
    n = len(obj)
    dominated = np.zeros(n, bool)
    for i in range(n):
        if dominated[i]:
            continue
        # marca como dominados los puntos que i DOMINA (i es >= en todo y > en algo)
        dom = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        dominated[dom] = True
    front = np.where(~dominated)[0]
    scs = model.score_batch(C[front])
    knee = front[np.argmax(scs)]
    return front, knee


# ===========================================================================
# F. INFORME COMPLETO
# ===========================================================================
def run(path_2016, path_2010, mc_iters=10_000_000, n_plays=1):
    raw = base.load_raw(path_2016, path_2010)
    clean, rep = base.clean_and_validate(raw)
    cur = base.current_era(clean)
    a = base.analyze(cur)

    print("=" * 74)
    print(f" DATA: {rep['draws_total']} sorteos (2010-2026) | era actual: "
          f"{a['n_draws']} sorteos (5/69 + 1/26)")
    print("=" * 74)

    print("\n[A] BATERIA DE PRUEBAS ESTADISTICAS (aleatoriedad / uniformidad)")
    t = statistical_tests(cur, a)
    cw, pw, dfw = t["chi2_white"]
    cp, pp, dfp = t["chi2_pb"]
    print(f"  Chi2 blancas : X2={cw:8.2f} (gl={dfw})  p={pw:.3f}  -> "
          f"{'compatible con uniforme' if pw>0.05 else 'rechaza uniforme'}")
    print(f"  Chi2 PB      : X2={cp:8.2f} (gl={dfp})  p={pp:.3f}  -> "
          f"{'compatible con uniforme' if pp>0.05 else 'rechaza uniforme'}")
    ksd, ksp = t["ks_sums"]
    print(f"  KS sumas~N   : D={ksd:.3f}  p={ksp:.3f}")
    runs, mur, zr, pr = t["runs_test"]
    print(f"  Runs test    : rachas={runs} (esp={mur:.1f})  z={zr:+.2f}  p={pr:.3f}"
          f"  -> {'sin patron (aleatorio)' if pr>0.05 else 'posible patron'}")
    print(f"  Autocorr sum : lags1-5 = {[round(x,3) for x in t['autocorr_sums']]}")
    H, Hmax, Hr = t["entropy_white"]
    print(f"  Entropia     : H={H:.4f} / Hmax={Hmax:.4f} bits  ({100*Hr:.2f}% del maximo)")
    p11, p10, br = t["markov"]
    print(f"  Markov       : P(sale|salio)={p11:.3f}  P(sale|no salio)={p10:.3f}  "
          f"base={br:.3f}  -> {'sin memoria' if abs(p11-p10)<0.02 else 'revisar'}")

    print(f"\n[B] Construyendo modelo ensemble (Monte Carlo {mc_iters:,} iter, "
          f"bootstrap 2000)...")
    model = EnsembleModel(cur, a, mc_iters=mc_iters)

    print("[C] Barrido EXHAUSTIVO de C(69,5)=11.238.513 combinaciones...")
    C, S, meta = exhaustive_scs(model, a)
    print(f"    {meta['white_combos']:,} combos evaluados (x26 PB = "
          f"{meta['tickets']:,} boletos).")
    print(f"    {meta['kept']:,} pasan filtros ({100*meta['kept']/meta['white_combos']:.1f}%). "
          f"Banda suma P10-P90={meta['sum_band']}")

    # GA/SA optimizan el MISMO objetivo restringido que el barrido (penalizan
    # los patrones populares), para que la comparacion sea justa y converjan.
    def constrained(P):
        arr = np.asarray(P, dtype=np.int16)
        batched = arr.ndim > 1                       # discrimina por INPUT, no output
        C2 = np.sort(np.atleast_2d(arr), axis=1)
        s = model.score_batch(C2) - 100.0 * undesirable_mask(C2, meta["sum_band"])
        return s if batched else float(s[0])

    print("\n[D] Validacion cruzada (mismo objetivo filtrado; deben converger):")
    ga_combo, ga_s = genetic_optimize(model, score_fn=constrained)
    sa_combo, sa_s = simulated_annealing(model, score_fn=constrained)
    print(f"    Exhaustivo (optimo del subespacio recomendable): "
          f"{tuple(int(x) for x in C[0])}  SCS={S[0]:.2f}")
    print(f"    Algoritmo Genetico        : {ga_combo}  SCS={ga_s:.2f}")
    print(f"    Simulated Annealing       : {sa_combo}  SCS={sa_s:.2f}")
    front, knee = pareto_front(model, C)
    print(f"    Frente de Pareto          : {len(front)} soluciones no-dominadas; "
          f"knee = {tuple(int(x) for x in C[knee])}")

    # Seleccion final diversa
    chosen = []
    for k in range(len(C)):
        combo = tuple(int(x) for x in C[k])
        if all(len(set(combo) & set(c)) <= 2 for c, _ in chosen):
            chosen.append((combo, float(S[k])))
        if len(chosen) == n_plays:
            break

    print("\n" + "=" * 74)
    print(f" JUGADA{'S' if n_plays>1 else ''} FINAL{'ES' if n_plays>1 else ''} "
          f"(Statistical Confidence Score — NO probabilidad de ganar)")
    print("=" * 74)
    rows = []
    for i, (combo, sc) in enumerate(chosen, 1):
        print(f"  #{i}  {' '.join(f'{x:02d}' for x in combo)}  + PB {model.best_pb}"
              f"   SCS = {sc:.2f}/100")
        rows.append(dict(jugada=i, blancas=" ".join(f"{x:02d}" for x in combo),
                         powerball=model.best_pb, scs=round(sc, 2)))
    print(f"\n  Probabilidad real de jackpot: 1 entre {JACKPOT_ODDS:,} "
          f"(identica para TODA combinacion).")
    return pd.DataFrame(rows), model, a, t, meta, C, S


if __name__ == "__main__":
    import sys
    p2016 = sys.argv[1] if len(sys.argv) > 1 else "powerball_results_20160601_to_20260601.xlsx"
    p2010 = sys.argv[2] if len(sys.argv) > 2 else "powerball_resultados.xlsx"
    run(p2016, p2010)
