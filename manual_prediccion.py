#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 MANUAL DE MODELADO Y PREDICCION — CODIGO FUNCIONAL
============================================================================
Implementaciones minimas, validadas y comentadas de los metodos del manual.
Cada funcion: valida entradas, maneja errores, separa train/test cuando aplica,
muestra resultados y NUNCA afirma certeza absoluta.

Premisa (HIPOTETICA, no demostrada): todo suceso es potencialmente modelable
si hay datos, variables, leyes, computo y un modelo adecuado. En la practica,
la precision SIEMPRE esta limitada por datos, supuestos, medicion y azar/caos.

Dependencias: numpy, scipy, scikit-learn.  Ejecutar:  python manual_prediccion.py
============================================================================
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RNG = np.random.default_rng(42)   # semilla fija -> reproducibilidad


# ---------------------------------------------------------------------------
# Utilidades de validacion y metricas
# ---------------------------------------------------------------------------
def validar_array(x, nombre="x", min_len=2):
    x = np.asarray(x, dtype=float)
    if x.size < min_len:
        raise ValueError(f"'{nombre}' necesita >= {min_len} valores (tiene {x.size}).")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"'{nombre}' contiene NaN o infinitos.")
    return x


def metricas(y_true, y_pred) -> dict:
    """MAE, RMSE y R^2 (error absoluto medio, raiz del error cuadratico, ajuste)."""
    y_true = validar_array(y_true, "y_true"); y_pred = validar_array(y_pred, "y_pred")
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def intervalo_confianza_media(x, conf=0.95):
    """IC de la media por t-Student. Devuelve (media, lo, hi). Supone datos ~ i.i.d."""
    x = validar_array(x, "x")
    n = x.size; m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - (1 - conf) / 2, df=n - 1)
    return float(m), float(m - t * se), float(m + t * se)


# ---------------------------------------------------------------------------
# 1. Regresion lineal   (y = b0 + b1 x + e)
# ---------------------------------------------------------------------------
def demo_regresion_lineal():
    n = 200
    x = np.linspace(0, 10, n).reshape(-1, 1)
    y = 2.5 * x.ravel() + 1.0 + RNG.normal(0, 2.0, n)     # datos de ejemplo
    Xtr, Xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=0)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    return {"pendiente": float(m.coef_[0]), "intercepto": float(m.intercept_),
            "test": metricas(yte, pred)}


# ---------------------------------------------------------------------------
# 2. Regresion logistica   P(y=1) = 1/(1+e^-(b0+b1 x))
# ---------------------------------------------------------------------------
def demo_regresion_logistica():
    n = 400
    x = RNG.normal(0, 1.5, n).reshape(-1, 1)
    p = 1 / (1 + np.exp(-(1.2 * x.ravel() - 0.3)))
    y = (RNG.random(n) < p).astype(int)                   # etiquetas 0/1
    Xtr, Xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=0)
    m = LogisticRegression().fit(Xtr, ytr)
    acc = m.score(Xte, yte)
    return {"coef": float(m.coef_[0][0]), "intercepto": float(m.intercept_[0]),
            "exactitud_test": float(acc)}


# ---------------------------------------------------------------------------
# 3. Inferencia bayesiana  (Beta-Binomial conjugado)
#    posterior de una proporcion p tras observar k exitos en n intentos
# ---------------------------------------------------------------------------
def demo_bayes(k=7, n=20, a_prior=1.0, b_prior=1.0):
    if not (0 <= k <= n) or n <= 0:
        raise ValueError("Se requiere 0 <= k <= n y n > 0.")
    a_post, b_post = a_prior + k, b_prior + (n - k)
    media = a_post / (a_post + b_post)
    lo, hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)   # intervalo creible 95%
    return {"posterior": f"Beta({a_post:.0f},{b_post:.0f})", "media": float(media),
            "IC95_creible": [float(lo), float(hi)]}


# ---------------------------------------------------------------------------
# 4. Monte Carlo  (estimar una probabilidad/integral con incertidumbre)
# ---------------------------------------------------------------------------
def demo_monte_carlo(n=200_000):
    if n < 1000:
        raise ValueError("Usa n >= 1000 para una estimacion estable.")
    pts = RNG.random((n, 2))
    dentro = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.0
    pi_est = 4 * dentro.mean()
    se = 4 * np.sqrt(dentro.mean() * (1 - dentro.mean()) / n)   # error estandar
    return {"pi_estimado": float(pi_est), "error_estandar": float(se),
            "IC95": [float(pi_est - 1.96 * se), float(pi_est + 1.96 * se)]}


# ---------------------------------------------------------------------------
# 5. Ecuaciones diferenciales  (EDO): decaimiento / crecimiento  dy/dt = r y
# ---------------------------------------------------------------------------
def demo_edo(r=-0.3, y0=100.0, t_max=10.0):
    t = np.linspace(0, t_max, 100)
    sol = odeint(lambda y, t: r * y, y0, t)
    return {"y_inicial": y0, "y_final": float(sol[-1, 0]),
            "analitico_final": float(y0 * np.exp(r * t_max))}


# ---------------------------------------------------------------------------
# 6. Modelo SIR (epidemias)   dS=-bSI/N, dI=bSI/N-gI, dR=gI
# ---------------------------------------------------------------------------
def demo_sir(beta=0.4, gamma=0.1, N=1000, I0=1, dias=160):
    def f(y, t):
        S, I, R = y
        return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]
    t = np.linspace(0, dias, dias)
    S, I, R = odeint(f, [N - I0, I0, 0], t).T
    return {"R0": beta / gamma, "pico_infectados": float(I.max()),
            "dia_pico": int(t[I.argmax()]), "recuperados_final": float(R[-1])}


# ---------------------------------------------------------------------------
# 7. Crecimiento logistico   dN/dt = r N (1 - N/K)
# ---------------------------------------------------------------------------
def demo_logistico(r=0.5, K=1000, N0=10, t_max=30):
    t = np.linspace(0, t_max, 200)
    N = odeint(lambda N, t: r * N * (1 - N / K), N0, t).ravel()
    return {"N_final": float(N[-1]), "capacidad_K": K,
            "t_mitad_aprox": float(t[np.argmin(np.abs(N - K / 2))])}


# ---------------------------------------------------------------------------
# 8. Ecuacion de Arrhenius   k = A e^(-Ea/RT)
# ---------------------------------------------------------------------------
def demo_arrhenius(A=1e13, Ea=75000.0, T=298.15):
    R = 8.314  # J/(mol K)
    if T <= 0:
        raise ValueError("T debe estar en Kelvin (> 0).")
    k = A * np.exp(-Ea / (R * T))
    return {"k_298K": float(k), "k_310K": float(A * np.exp(-Ea / (R * 310.0)))}


# ---------------------------------------------------------------------------
# 9. Cadena de Markov  (distribucion estacionaria)
# ---------------------------------------------------------------------------
def demo_markov():
    P = np.array([[0.7, 0.3], [0.4, 0.6]])          # matriz de transicion
    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Cada fila de P debe sumar 1.")
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pi / pi.sum()
    return {"estacionaria": [float(x) for x in pi]}


# ---------------------------------------------------------------------------
# 10. Filtro de Kalman 1D  (estimar posicion con ruido)
# ---------------------------------------------------------------------------
def demo_kalman(n=50, q=1e-3, r=0.1):
    verdad = np.cumsum(RNG.normal(0, 0.05, n))
    z = verdad + RNG.normal(0, np.sqrt(r), n)        # mediciones ruidosas
    x, P = 0.0, 1.0; est = []
    for zi in z:
        P += q                                       # prediccion
        Kk = P / (P + r)                             # ganancia de Kalman
        x = x + Kk * (zi - x)                        # correccion
        P = (1 - Kk) * P
        est.append(x)
    est = np.array(est)
    return {"RMSE_medicion": float(np.sqrt(np.mean((z - verdad) ** 2))),
            "RMSE_kalman": float(np.sqrt(np.mean((est - verdad) ** 2)))}


# ---------------------------------------------------------------------------
# 11. Serie temporal  (autorregresivo AR(1) con train/test)
# ---------------------------------------------------------------------------
def demo_serie_temporal(n=300):
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + RNG.normal(0, 1)
    corte = int(n * 0.8)
    tr, te = y[:corte], y[corte:]
    phi = np.dot(tr[:-1], tr[1:]) / np.dot(tr[:-1], tr[:-1])   # ajuste AR(1)
    pred = []; prev = tr[-1]
    for _ in te:
        prev = phi * prev; pred.append(prev)          # pronostico multi-paso
    return {"phi_estimado": float(phi), "test": metricas(te, pred),
            "nota": "El pronostico multi-paso pierde poder rapido; IC crece con el horizonte."}


# ---------------------------------------------------------------------------
# 12. Analisis de sensibilidad  (que entrada mueve mas la salida)
# ---------------------------------------------------------------------------
def demo_sensibilidad():
    # salida = f(a,b,c); perturbamos cada entrada +-10% y medimos el cambio
    base = dict(a=2.0, b=3.0, c=5.0)
    f = lambda p: p["a"] * p["b"] + p["c"] ** 2
    y0 = f(base); sens = {}
    for k in base:
        p = dict(base); p[k] *= 1.1
        sens[k] = abs(f(p) - y0) / y0                 # sensibilidad relativa
    return {"sensibilidad_relativa": {k: round(v, 4) for k, v in sens.items()}}


# ---------------------------------------------------------------------------
# Pseudocodigo universal COMPLETADO (punto 12 del manual)
# ---------------------------------------------------------------------------
def predecir_suceso(datos, ajustar_fn, predecir_fn):
    """Plantilla universal: valida -> limpia -> ajusta -> predice -> incertidumbre.
    ajustar_fn(X,y)->modelo ; predecir_fn(modelo,X)->y_pred."""
    X = np.asarray(datos["X"], dtype=float); y = np.asarray(datos["y"], dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y deben tener el mismo numero de filas.")
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)   # limpieza
    X, y = X[mask], y[mask]
    if len(y) < 10:
        return {"error": "No puede realizarse una prediccion confiable con los datos disponibles."}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    modelo = ajustar_fn(Xtr, ytr)
    pred = predecir_fn(modelo, Xte)
    m = metricas(yte, pred)
    resid = yte - pred
    inc = 1.96 * resid.std(ddof=1)                    # banda ~95% (aprox. Gaussiana)
    return {"metricas_test": m, "incertidumbre_+-": float(inc),
            "advertencia": "Prediccion probabilistica, no certeza. Valida fuera de muestra."}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demos = [
        ("Regresion lineal", demo_regresion_lineal),
        ("Regresion logistica", demo_regresion_logistica),
        ("Inferencia bayesiana (Beta-Binomial)", demo_bayes),
        ("Monte Carlo (estimar pi)", demo_monte_carlo),
        ("EDO (decaimiento dy/dt=ry)", demo_edo),
        ("Modelo SIR", demo_sir),
        ("Crecimiento logistico", demo_logistico),
        ("Arrhenius k=A e^(-Ea/RT)", demo_arrhenius),
        ("Cadena de Markov (estacionaria)", demo_markov),
        ("Filtro de Kalman 1D", demo_kalman),
        ("Serie temporal AR(1)", demo_serie_temporal),
        ("Analisis de sensibilidad", demo_sensibilidad),
    ]
    print("=" * 70)
    for nombre, fn in demos:
        try:
            print(f"\n[{nombre}]")
            for k, v in fn().items():
                print(f"   {k}: {v}")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}")

    # Demo del pseudocodigo universal + IC de la media
    print("\n[Pseudocodigo universal predecir_suceso()]")
    X = RNG.normal(0, 1, (100, 2)); y = X @ [1.5, -2.0] + RNG.normal(0, 0.5, 100)
    res = predecir_suceso({"X": X, "y": y},
                          lambda Xtr, ytr: LinearRegression().fit(Xtr, ytr),
                          lambda m, Xte: m.predict(Xte))
    for k, v in res.items():
        print(f"   {k}: {v}")
    print("\n[Intervalo de confianza de una media]")
    print("  ", intervalo_confianza_media(RNG.normal(50, 5, 40)))
    print("\n" + "=" * 70)
    print("ADVERTENCIA: toda salida es una estimacion con incertidumbre, NO una certeza.")
