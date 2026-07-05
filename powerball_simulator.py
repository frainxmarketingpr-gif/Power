#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 SIMULADOR ESTADISTICO DE POWERBALL
============================================================================
Autor: analisis de ciencia de datos (uso educativo / de gestion de riesgo)

ADVERTENCIA MATEMATICA CENTRAL
------------------------------
Powerball es un sistema ALEATORIO sin memoria. Cada bola tiene, en cada
sorteo, la misma probabilidad que cualquier otra. NINGUN modelo de este
archivo predice el resultado ni aumenta de forma real y garantizada la
probabilidad de ganar el jackpot (1 entre 292.201.338).

El unico valor practico y honesto de este simulador es:
  1) Evitar combinaciones "populares" (cumpleanos, secuencias, patrones)
     que, si salieran, se repartirian entre muchisimos jugadores.
  2) Entender la incertidumbre real y la distribucion de los sorteos.

NO garantiza nada. NO recomienda gastar mas dinero.
============================================================================
"""

from __future__ import annotations
import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. PARAMETROS DE LAS REGLAS (ERA ACTUAL: desde 2015-10-07)
# ---------------------------------------------------------------------------
WHITE_MIN, WHITE_MAX = 1, 69      # 69 bolas blancas
PB_MIN, PB_MAX = 1, 26            # 26 bolas Powerball
N_WHITE = 5                       # se eligen 5 blancas
CURRENT_ERA_START = pd.Timestamp("2015-10-07")

# Cambios historicos de reglas (para separar la data por periodos)
RULE_ERAS = [
    ("Pre-2012",   None,                     pd.Timestamp("2012-01-15"), (1, 59), (1, 39)),
    ("2012-2015",  pd.Timestamp("2012-01-15"), pd.Timestamp("2015-10-07"), (1, 59), (1, 35)),
    ("Actual",     pd.Timestamp("2015-10-07"), None,                       (1, 69), (1, 26)),
]

# Probabilidades combinatorias exactas de la era actual
WHITE_COMBOS = 11_238_513         # C(69,5): nº de combinaciones de 5 blancas
PAIR_COMBOS  = 2_346              # C(69,2): nº de pares de blancas posibles
JACKPOT_ODDS = 292_201_338        # C(69,5) * 26  (jackpot: 5 blancas + PB)
MATCH5_ODDS  = 11_688_053         # premio "5 sin PB": C(69,5) * 26/25


# ---------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------------
def load_raw(path_2016: str, path_2010: str) -> pd.DataFrame:
    """Carga las dos fuentes oficiales, las normaliza y las fusiona.

    Fuente A: powerball.com (2016-2026).
    Fuente B: NY Open Data d6yy-54nr (2010-2026).
    Se usa la union; ante fechas duplicadas se conserva una sola fila
    (ambas fuentes coinciden salvo el orden de columnas).
    """
    a = pd.read_excel(path_2016, sheet_name="Datos")
    b = pd.read_excel(path_2010, sheet_name="Todos_desde_2010")

    fa = pd.DataFrame({"date": pd.to_datetime(a["date"])})
    for i, c in enumerate(["number_1", "number_2", "number_3", "number_4", "number_5"], 1):
        fa[f"n{i}"] = a[c].astype(int)
    fa["pb"] = a["powerball"].astype(int)
    fa["source"] = "powerball.com"

    fb = pd.DataFrame({"date": pd.to_datetime(b["fecha_sorteo"])})
    for i, c in enumerate(["bola_blanca_1", "bola_blanca_2", "bola_blanca_3",
                           "bola_blanca_4", "bola_blanca_5"], 1):
        fb[f"n{i}"] = b[c].astype(int)
    fb["pb"] = b["powerball"].astype(int)
    fb["source"] = "ny_open_data"

    # Antes de deduplicar: detectar discrepancias REALES entre fuentes para una
    # misma fecha (comparando las 5 blancas ORDENADAS + PB, para ignorar el orden
    # de columnas). Asi la deduplicacion no oculta conflictos de datos.
    wc = [f"n{i}" for i in range(1, 6)]
    merged = fa.merge(fb, on="date", suffixes=("_a", "_b"))
    if len(merged):
        wa = np.sort(merged[[f"n{i}_a" for i in range(1, 6)]].values, axis=1)
        wb = np.sort(merged[[f"n{i}_b" for i in range(1, 6)]].values, axis=1)
        disagree = (wa != wb).any(axis=1) | (merged["pb_a"].values != merged["pb_b"].values)
        conflicts = int(disagree.sum())
    else:
        conflicts = 0

    # La fuente B es la mas larga (2010+); la A solo aporta fechas ya presentes.
    # Unimos y quitamos duplicados por fecha (B tiene prioridad por cobertura).
    full = pd.concat([fb, fa], ignore_index=True)
    full = full.drop_duplicates(subset="date", keep="first")
    full = full.sort_values("date").reset_index(drop=True)
    full.attrs["source_overlaps"] = int(len(merged))
    full.attrs["source_conflicts"] = conflicts
    return full


def clean_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ordena las 5 blancas por fila, valida rangos por era y reporta errores."""
    wcols = [f"n{i}" for i in range(1, 6)]
    report = {}

    # Ordenar las blancas ascendentemente (canonico) -> corrige filas desordenadas
    df[wcols] = np.sort(df[wcols].values, axis=1)

    # Duplicados exactos de combinacion (deberian ser rarisimos y legitimos)
    combo_key = df[wcols + ["pb"]].astype(str).agg("-".join, axis=1)
    report["draws_total"] = len(df)
    report["date_duplicates"] = int(df.duplicated("date").sum())
    report["exact_combo_repeats"] = int(combo_key.duplicated().sum())
    # Discrepancias reales entre fuentes (calculadas antes de deduplicar)
    report["source_overlaps"] = df.attrs.get("source_overlaps", 0)
    report["source_conflicts"] = df.attrs.get("source_conflicts", 0)

    # Validacion de rango por era de reglas
    def era_of(d):
        for name, start, end, _, _ in RULE_ERAS:
            if (start is None or d >= start) and (end is None or d < end):
                return name
        return "?"

    df["era"] = df["date"].apply(era_of)
    bad = 0
    for name, start, end, (wlo, whi), (plo, phi) in RULE_ERAS:
        sub = df[df["era"] == name]
        if sub.empty:
            continue
        wbad = ((sub[wcols] < wlo) | (sub[wcols] > whi)).any(axis=1)
        pbad = (sub["pb"] < plo) | (sub["pb"] > phi)
        bad += int((wbad | pbad).sum())
    report["out_of_range_rows"] = bad
    report["nulls"] = int(df[wcols + ["pb"]].isnull().sum().sum())
    report["dup_whites_in_row"] = int((df[wcols].nunique(axis=1) < 5).sum())
    return df, report


def current_era(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo los sorteos de la matriz vigente (5/69 + 1/26)."""
    return df[df["date"] >= CURRENT_ERA_START].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. ANALISIS ESTADISTICO (sobre la era actual)
# ---------------------------------------------------------------------------
def analyze(df: pd.DataFrame) -> dict:
    wcols = [f"n{i}" for i in range(1, 6)]
    n_draws = len(df)
    if n_draws < 2:
        raise ValueError(
            f"Se requieren >=2 sorteos de la era actual para el analisis; "
            f"hay {n_draws}. Revisa las rutas de datos o el rango de fechas.")
    whites = df[wcols].values.ravel()

    # Frecuencia de cada blanca y del powerball
    white_freq = pd.Series(0, index=range(WHITE_MIN, WHITE_MAX + 1), dtype=int)
    white_freq = white_freq.add(pd.Series(Counter(whites)), fill_value=0).astype(int)
    pb_freq = pd.Series(0, index=range(PB_MIN, PB_MAX + 1), dtype=int)
    pb_freq = pb_freq.add(pd.Series(Counter(df["pb"].values)), fill_value=0).astype(int)

    # Recencia: ultimo sorteo (indice) en que aparecio cada numero
    last_seen = {n: None for n in range(WHITE_MIN, WHITE_MAX + 1)}
    for idx, row in df.reset_index(drop=True).iterrows():
        for n in row[wcols]:
            last_seen[n] = idx
    draws_since = {n: (n_draws - 1 - v) if v is not None else n_draws for n, v in last_seen.items()}

    # Calientes / frios (por frecuencia)
    hot = white_freq.sort_values(ascending=False).head(10)
    cold = white_freq.sort_values(ascending=True).head(10)

    # Pares y trios mas repetidos
    pair_c, trio_c = Counter(), Counter()
    for row in df[wcols].values:
        s = sorted(row)
        pair_c.update(itertools.combinations(s, 2))
        trio_c.update(itertools.combinations(s, 3))

    # Distribuciones por sorteo
    odd_count = (df[wcols].values % 2 == 1).sum(axis=1)      # nº impares (0..5)
    low_count = (df[wcols].values <= 34).sum(axis=1)          # nº "bajos" (1-34)
    sums = df[wcols].values.sum(axis=1)
    spread = df[wcols].values.max(axis=1) - df[wcols].values.min(axis=1)

    # Distancias entre numeros consecutivos ordenados
    gaps = np.diff(np.sort(df[wcols].values, axis=1), axis=1).ravel()

    # Repeticion entre sorteos consecutivos (cuantas blancas se repiten)
    arr = df[wcols].values
    carry = [len(set(arr[i]) & set(arr[i - 1])) for i in range(1, n_draws)]

    return dict(
        n_draws=n_draws, white_freq=white_freq, pb_freq=pb_freq,
        draws_since=pd.Series(draws_since), hot=hot, cold=cold,
        pairs=pair_c, trios=trio_c,
        odd_dist=pd.Series(Counter(odd_count)).sort_index(),
        low_dist=pd.Series(Counter(low_count)).sort_index(),
        sums=sums, spread=spread, gaps=gaps, carry=carry,
        sum_stats=(sums.mean(), sums.std(), np.percentile(sums, [5, 25, 50, 75, 95])),
    )


# ---------------------------------------------------------------------------
# 3. MODELOS DE PONDERACION
# ---------------------------------------------------------------------------
def weights_uniform() -> tuple[np.ndarray, np.ndarray]:
    w = np.ones(WHITE_MAX - WHITE_MIN + 1)
    p = np.ones(PB_MAX - PB_MIN + 1)
    return w / w.sum(), p / p.sum()


def weights_frequency(a: dict) -> tuple[np.ndarray, np.ndarray]:
    w = a["white_freq"].values.astype(float) + 1.0   # suavizado +1
    p = a["pb_freq"].values.astype(float) + 1.0
    return w / w.sum(), p / p.sum()


def weights_recency(df: pd.DataFrame, half_life: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Pondera cada aparicion por decaimiento exponencial: sorteos recientes pesan mas."""
    wcols = [f"n{i}" for i in range(1, 6)]
    n = len(df)
    decay = np.log(2) / half_life
    w = np.zeros(WHITE_MAX - WHITE_MIN + 1)
    p = np.zeros(PB_MAX - PB_MIN + 1)
    for i, row in df.reset_index(drop=True).iterrows():
        age = n - 1 - i
        wt = np.exp(-decay * age)
        for c in wcols:
            w[int(row[c]) - WHITE_MIN] += wt
        p[int(row["pb"]) - PB_MIN] += wt
    w += 1e-6
    p += 1e-6
    return w / w.sum(), p / p.sum()


def weights_bayesian(a: dict, prior_strength: float = 69.0) -> tuple[np.ndarray, np.ndarray]:
    """Posterior Dirichlet con prior uniforme. Muestra cuanto se ALEJA la data
    del azar puro: con prior fuerte, casi nada (esa es la leccion honesta)."""
    counts_w = a["white_freq"].values.astype(float)
    counts_p = a["pb_freq"].values.astype(float)
    kw = len(counts_w)
    kp = len(counts_p)
    post_w = counts_w + prior_strength / kw
    post_p = counts_p + prior_strength / kp
    return post_w / post_w.sum(), post_p / post_p.sum()


MODEL_BUILDERS = {
    "uniforme":   lambda df, a: weights_uniform(),
    "frecuencia": lambda df, a: weights_frequency(a),
    "recencia":   lambda df, a: weights_recency(df),
    "bayesiano":  lambda df, a: weights_bayesian(a),
}


# ---------------------------------------------------------------------------
# 4. SIMULACION MONTE CARLO
# ---------------------------------------------------------------------------
def simulate(white_w: np.ndarray, pb_w: np.ndarray, n: int, rng: np.random.Generator):
    """Genera n combinaciones (5 blancas sin reemplazo + 1 PB) segun los pesos."""
    whites = np.empty((n, N_WHITE), dtype=np.int16)
    balls = np.arange(WHITE_MIN, WHITE_MAX + 1)
    # Muestreo vectorizado por lotes (Gumbel-top-k para pesos no uniformes)
    logw = np.log(white_w)
    batch = 200_000
    for s in range(0, n, batch):
        e = min(s + batch, n)
        m = e - s
        g = rng.gumbel(size=(m, len(balls)))
        idx = np.argsort(logw + g, axis=1)[:, -N_WHITE:]
        picked = np.sort(balls[idx], axis=1)
        whites[s:e] = picked
    pbs = rng.choice(np.arange(PB_MIN, PB_MAX + 1), size=n, p=pb_w).astype(np.int16)
    return whites, pbs


# ---------------------------------------------------------------------------
# 5. FILTROS DE COMBINACIONES POCO RECOMENDABLES
# ---------------------------------------------------------------------------
def undesirable_mask_vec(C: np.ndarray, sum_band: tuple[int, int]) -> np.ndarray:
    """Mascara booleana (vectorizada) de combinaciones populares / poco
    recomendables. FUENTE UNICA del filtro: la usan is_undesirable,
    exhaustive_best y powerball_advanced.undesirable_mask.

    C: (m,5) int con las blancas ORDENADAS ascendentemente.
    (Filtrar no cambia la probabilidad de acertar; reduce el riesgo de
    compartir premio y descarta patrones populares.)"""
    C = np.asarray(C)
    diffs = np.diff(C, axis=1)                       # (m,4)
    s0, s4 = C[:, 0], C[:, -1]
    ssum = C.sum(axis=1)
    par = (C % 2 == 0).sum(axis=1)
    # a) >=4 numeros consecutivos = 3 diffs de 1 adyacentes (incluye 1-2-3-4-5)
    run4 = ((diffs[:, :-2] == 1) & (diffs[:, 1:-1] == 1) & (diffs[:, 2:] == 1)).any(axis=1)
    # b) progresion aritmetica exacta (todas las diffs iguales)
    arith = (np.ptp(diffs, axis=1) == 0)
    # c) todas bajas (<=34) o todas altas (>=35)
    lo_hi = (s4 <= 34) | (s0 >= 35)
    # d) todas pares o todas impares
    parity = (par == 0) | (par == 5)
    # e) sesgo de cumpleanos: 4+ numeros en 1-31 (dias del mes) -- no redundante
    birthday = (C <= 31).sum(axis=1) >= 4
    # f) suma fuera de la banda central historica (colas improbables)
    out_sum = (ssum < sum_band[0]) | (ssum > sum_band[1])
    # g) patron visual: mismo ultimo digito, o 4+ en una misma decena
    last_digit_same = (np.ptp(C % 10, axis=1) == 0)
    dec = C // 10
    dec_max = np.zeros(len(C), dtype=np.int8)
    for d in range(0, 7):
        dec_max = np.maximum(dec_max, (dec == d).sum(axis=1).astype(np.int8))
    visual = last_digit_same | (dec_max >= 4)
    return run4 | arith | lo_hi | parity | birthday | out_sum | visual


def is_undesirable(combo: np.ndarray, sum_band: tuple[int, int]) -> bool:
    """True si UNA combinacion cae en un patron popular / poco recomendable.
    Delega en la mascara vectorizada canonica."""
    C = np.sort(np.asarray(combo, dtype=np.int64).reshape(1, -1), axis=1)
    return bool(undesirable_mask_vec(C, sum_band)[0])


# ---------------------------------------------------------------------------
# 6. SCORE ESTADISTICO DE UNA COMBINACION
# ---------------------------------------------------------------------------
@dataclass
class Scorer:
    a: dict
    white_freq_n: np.ndarray = field(init=False)
    recency_n: np.ndarray = field(init=False)
    pair_sup: Counter = field(init=False)
    sum_mu: float = field(init=False)
    sum_sd: float = field(init=False)

    def __post_init__(self):
        wf = self.a["white_freq"].values.astype(float)
        self.white_freq_n = wf / wf.max()
        ds = self.a["draws_since"].reindex(range(WHITE_MIN, WHITE_MAX + 1)).values.astype(float)
        self.recency_n = 1.0 - (ds / ds.max())      # 1 = visto hace poco
        self.pair_sup = self.a["pairs"]
        self.sum_mu, self.sum_sd, _ = self.a["sum_stats"]

    def score(self, combo: np.ndarray) -> float:
        s = sorted(int(x) for x in combo)
        idx = [x - WHITE_MIN for x in s]
        f_score = self.white_freq_n[idx].mean()                    # 0..1
        r_score = self.recency_n[idx].mean()                       # 0..1
        pair_score = np.mean([self.pair_sup.get(p, 0) for p in itertools.combinations(s, 2)])
        pair_score = pair_score / (self.a["n_draws"] * 10 / PAIR_COMBOS + 1e-9)
        pair_score = min(pair_score / 5.0, 1.0)
        # Tipicidad de la suma (cercania al centro de la distribucion historica)
        z = abs(sum(s) - self.sum_mu) / self.sum_sd
        sum_score = float(np.exp(-0.5 * z * z))                    # 0..1
        # Balance par/impar y alto/bajo (2-3 o 3-2 es lo mas comun)
        par = sum(x % 2 == 0 for x in s)
        low = sum(x <= 34 for x in s)
        bal_score = (1.0 if par in (2, 3) else 0.5) * (1.0 if low in (2, 3) else 0.6)
        # Combinacion ponderada -> 0..100
        raw = (0.22 * f_score + 0.18 * r_score + 0.20 * pair_score +
               0.25 * sum_score + 0.15 * bal_score)
        return round(100 * raw, 2)

    def reason(self, combo: np.ndarray, pb: int) -> str:
        s = sorted(int(x) for x in combo)
        par = sum(x % 2 == 0 for x in s)
        low = sum(x <= 34 for x in s)
        tot = sum(s)
        return (f"par/impar {par}-{5-par}, bajo/alto {low}-{5-low}, "
                f"suma {tot} (centro hist. {self.sum_mu:.0f}), "
                f"rango {s[-1]-s[0]}; PB {pb}. Perfil estadisticamente tipico "
                f"y NO popular (evita cumpleanos/secuencias).")


# ---------------------------------------------------------------------------
# 6b. BARRIDO EXHAUSTIVO: LAS 292.201.338 COMBINACIONES POSIBLES
# ---------------------------------------------------------------------------
def exhaustive_best(df: pd.DataFrame, a: dict, n_plays: int = 1,
                    batch: int = 1_000_000):
    """Evalua TODO el espacio: C(69,5)=11.238.513 combos de blancas x 26 PB
    = 292.201.338 boletos. Cobertura total y determinista (sin azar).

    Puntua cada combinacion de blancas con el mismo criterio de tipicidad del
    Scorer, aplica los filtros de patrones populares y devuelve las mejores.
    """
    # Vectores de apoyo
    wf = a["white_freq"].values.astype(np.float64)
    white_freq_n = wf / wf.max()
    ds = a["draws_since"].reindex(range(WHITE_MIN, WHITE_MAX + 1)).values.astype(np.float64)
    recency_n = 1.0 - ds / ds.max()
    mu, sd, _ = a["sum_stats"]
    n_draws = a["n_draws"]

    # Matriz de soporte de pares 69x69
    K = WHITE_MAX - WHITE_MIN + 1
    PAIR = np.zeros((K, K), dtype=np.float64)
    for (i, j), c in a["pairs"].items():
        PAIR[i - 1, j - 1] = c
        PAIR[j - 1, i - 1] = c
    pair_norm = (n_draws * 10 / PAIR_COMBOS + 1e-9)

    sums = a["sums"]
    lo_band = int(np.percentile(sums, 10))
    hi_band = int(np.percentile(sums, 90))

    col_pairs = list(itertools.combinations(range(N_WHITE), 2))  # 10 pares de columnas
    col_i = np.array([p[0] for p in col_pairs])
    col_j = np.array([p[1] for p in col_pairs])

    total = 0
    kept_total = 0
    # Mantener top-N global
    best_combos = np.zeros((0, N_WHITE), dtype=np.int16)
    best_scores = np.zeros(0, dtype=np.float64)

    gen = itertools.combinations(range(WHITE_MIN, WHITE_MAX + 1), N_WHITE)
    while True:
        chunk = list(itertools.islice(gen, batch))
        if not chunk:
            break
        C = np.asarray(chunk, dtype=np.int16)          # (m,5) ya ordenado asc
        m = len(C)
        total += m
        idx = C - WHITE_MIN

        # ---- FILTROS (fuente unica canonica) ----
        good = ~undesirable_mask_vec(C, (lo_band, hi_band))
        kept_total += int(good.sum())
        if not good.any():
            continue

        ssum = C.sum(axis=1).astype(np.int32)
        par = (C % 2 == 0).sum(axis=1)
        low = (C <= 34).sum(axis=1)
        Cg = C[good]; idxg = idx[good]
        gg = ssum[good]; parg = par[good]; lowg = low[good]

        # ---- SCORE (identico al Scorer) ----
        f_score = white_freq_n[idxg].mean(axis=1)
        r_score = recency_n[idxg].mean(axis=1)
        pair_vals = PAIR[idxg[:, col_i], idxg[:, col_j]]      # (mg,10)
        pair_mean = pair_vals.mean(axis=1)
        pair_score = np.minimum((pair_mean / pair_norm) / 5.0, 1.0)
        z = (gg - mu) / sd
        sum_score = np.exp(-0.5 * z * z)
        bal = np.where(np.isin(parg, [2, 3]), 1.0, 0.5) * \
              np.where(np.isin(lowg, [2, 3]), 1.0, 0.6)
        raw = (0.22 * f_score + 0.18 * r_score + 0.20 * pair_score +
               0.25 * sum_score + 0.15 * bal)
        sc = 100.0 * raw

        # Fusionar con top-N global y recortar
        best_combos = np.vstack([best_combos, Cg])
        best_scores = np.concatenate([best_scores, sc])
        keep_n = max(n_plays * 4000, 4000)
        if len(best_scores) > keep_n:
            top = np.argpartition(best_scores, -keep_n)[-keep_n:]
            best_combos = best_combos[top]
            best_scores = best_scores[top]

    # Powerball: el PB es EQUIPROBABLE (cosmetico). Se toman los PB de mayor
    # frecuencia historica y se reparten entre jugadas para diversificar el eje
    # PB en lugar de repetir uno solo. Las 292M cubren los 26 PB por igual.
    pb_freq = a["pb_freq"].values.astype(np.float64)
    top_pbs = (np.argsort(pb_freq)[::-1] + PB_MIN).astype(int)

    order = np.argsort(best_scores)[::-1]
    scorer = Scorer(a)
    chosen = []
    for k in order:
        combo = tuple(int(x) for x in best_combos[k])
        if all(len(set(combo) & set(c)) <= 2 for c, _, _ in chosen):
            pb = int(top_pbs[len(chosen) % len(top_pbs)])
            chosen.append((combo, pb, round(float(best_scores[k]), 2)))
        if len(chosen) == n_plays:
            break

    rows = []
    for k, (combo, pb, sc) in enumerate(chosen, 1):
        rows.append({
            "jugada": k,
            "blancas": " ".join(f"{x:02d}" for x in combo),
            "powerball": pb,
            "score": sc,
            "razon": scorer.reason(np.array(combo), pb),
        })
    meta = dict(white_combos=total, tickets=total * (PB_MAX - PB_MIN + 1),
                kept=kept_total, sum_band=(lo_band, hi_band))
    return pd.DataFrame(rows), meta


# ---------------------------------------------------------------------------
# 7. GENERACION DE 20 JUGADAS SUGERIDAS
# ---------------------------------------------------------------------------
def generate_plays(df: pd.DataFrame, a: dict, n_sim: int = 1_000_000,
                   n_plays: int = 20, seed: int = 20260705) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # Pesos del ENSEMBLE: promedio de los 4 modelos de blancas y de PB
    ws, ps = [], []
    for name, build in MODEL_BUILDERS.items():
        w, p = build(df, a)
        ws.append(w); ps.append(p)
    white_w = np.mean(ws, axis=0); white_w /= white_w.sum()
    pb_w = np.mean(ps, axis=0); pb_w /= pb_w.sum()

    # Banda central de suma (percentiles 10-90 historicos)
    sums = a["sums"]
    sum_band = (int(np.percentile(sums, 10)), int(np.percentile(sums, 90)))

    # Simular 1M+ combinaciones con el ensemble
    whites, pbs = simulate(white_w, pb_w, n_sim, rng)

    # Filtrar poco recomendables
    keep = np.array([not is_undesirable(whites[i], sum_band) for i in range(n_sim)])
    whites_f, pbs_f = whites[keep], pbs[keep]

    # Puntuar y elegir 20 jugadas DIVERSAS de alto score
    scorer = Scorer(a)
    # Deduplicar combinaciones de blancas
    seen, cand = set(), []
    order = rng.permutation(len(whites_f))
    for i in order:
        key = tuple(whites_f[i])
        if key in seen:
            continue
        seen.add(key)
        cand.append((key, int(pbs_f[i])))
        if len(cand) >= 60_000:
            break
    scored = [(c, pb, scorer.score(np.array(c))) for c, pb in cand]
    scored.sort(key=lambda t: t[2], reverse=True)

    # Seleccion diversa: max solapamiento de 2 blancas entre jugadas elegidas
    chosen = []
    for combo, pb, sc in scored:
        if all(len(set(combo) & set(c)) <= 2 for c, _, _ in chosen):
            chosen.append((combo, pb, sc))
        if len(chosen) == n_plays:
            break

    rows = []
    for k, (combo, pb, sc) in enumerate(chosen, 1):
        rows.append({
            "jugada": k,
            "blancas": " ".join(f"{x:02d}" for x in combo),
            "powerball": pb,
            "score": sc,
            "razon": scorer.reason(np.array(combo), pb),
        })
    meta = dict(n_sim=n_sim, kept=int(keep.sum()), sum_band=sum_band,
                white_w=white_w, pb_w=pb_w)
    return pd.DataFrame(rows), meta


# ---------------------------------------------------------------------------
# 8. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def main(path_2016: str, path_2010: str, n_plays: int = 1):
    # Nota: el flujo principal usa el barrido EXHAUSTIVO (cobertura total), no
    # el muestreo Monte Carlo. generate_plays() queda como API alternativa.
    print(__doc__)
    raw = load_raw(path_2016, path_2010)
    clean, rep = clean_and_validate(raw)
    print("\n[VALIDACION DE DATOS]")
    for k, v in rep.items():
        print(f"  {k:24s}: {v}")

    print("\n[SORTEOS POR ERA DE REGLAS]")
    print(clean.groupby("era").agg(n=("pb", "size"),
                                   desde=("date", "min"),
                                   hasta=("date", "max")).to_string())

    cur = current_era(clean)
    a = analyze(cur)
    print(f"\n[ERA ACTUAL] {a['n_draws']} sorteos  "
          f"({cur['date'].min().date()} -> {cur['date'].max().date()})")

    print("\nTop 10 CALIENTES (mas frecuentes):")
    print(a["hot"].to_string())
    print("\nTop 10 FRIOS (menos frecuentes):")
    print(a["cold"].to_string())

    print("\n5 PARES mas repetidos:")
    for pair, c in a["pairs"].most_common(5):
        print(f"  {pair}: {c}")
    print("5 TRIOS mas repetidos:")
    for trio, c in a["trios"].most_common(5):
        print(f"  {trio}: {c}")

    print("\nDistribucion nº IMPARES por sorteo:\n", a["odd_dist"].to_string())
    print("\nDistribucion nº BAJOS (1-34) por sorteo:\n", a["low_dist"].to_string())
    mu, sd, pct = a["sum_stats"]
    print(f"\nSUMA de las 5 blancas: media={mu:.1f}  sd={sd:.1f}  "
          f"P5/25/50/75/95={pct}")
    print(f"Repeticion media entre sorteos consecutivos: "
          f"{np.mean(a['carry']):.3f} blancas")
    print(f"Powerball mas y menos frecuente: "
          f"{a['pb_freq'].idxmax()} ({a['pb_freq'].max()}) / "
          f"{a['pb_freq'].idxmin()} ({a['pb_freq'].min()})")

    # BARRIDO EXHAUSTIVO: evalua las C(69,5)=11.238.513 combinaciones de blancas
    # (x26 PB = 292.201.338 boletos). Cobertura total y determinista.
    plays, meta = exhaustive_best(cur, a, n_plays=n_plays)
    print(f"\n[BARRIDO EXHAUSTIVO] {meta['white_combos']:,} combinaciones de blancas "
          f"evaluadas (x26 PB = {meta['tickets']:,} boletos).")
    print(f"{meta['kept']:,} superaron los filtros "
          f"({100*meta['kept']/meta['white_combos']:.1f}%).")
    print(f"Banda de suma aceptada (P10-P90): {meta['sum_band']}")

    label = "JUGADA UNICA SUGERIDA" if n_plays == 1 else f"{n_plays} JUGADAS SUGERIDAS"
    print("\n" + "=" * 78)
    print(f" {label} (score = tipicidad estadistica, NO probabilidad)")
    print("=" * 78)
    with pd.option_context("display.max_colwidth", 100, "display.width", 200):
        print(plays[["jugada", "blancas", "powerball", "score", "razon"]].to_string(index=False))

    print("\n" + "=" * 78)
    print(" VERDAD MATEMATICA")
    print("=" * 78)
    print(f"""
  * Probabilidad de jackpot : 1 entre {JACKPOT_ODDS:,}
  * Probabilidad 5 blancas  : 1 entre {MATCH5_ODDS:,}
  * Cada combinacion (incluida 1-2-3-4-5 + 6) tiene EXACTAMENTE la misma
    probabilidad. Los "calientes/frios" son ruido estadistico esperable:
    con {a['n_draws']} sorteos, la frecuencia media por bola es
    ~{a['white_freq'].mean():.1f} con desviacion natural de +-{a['white_freq'].std():.1f}.
  * El sorteo NO tiene memoria: modelos de recencia/frecuencia describen el
    pasado pero NO predicen el futuro.
  * Valor real del simulador: elegir combinaciones NO populares para, EN EL
    RARO caso de ganar, no repartir el premio. No aumenta tu probabilidad de
    acertar y no justifica gastar mas dinero.
""")
    return clean, cur, a, plays, meta


if __name__ == "__main__":
    import sys
    p2016 = sys.argv[1] if len(sys.argv) > 1 else "powerball_results_20160601_to_20260601.xlsx"
    p2010 = sys.argv[2] if len(sys.argv) > 2 else "powerball_resultados.xlsx"
    main(p2016, p2010)
