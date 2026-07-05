# Simulador Estadístico Avanzado de Powerball — Informe

> **Verdad matemática, por delante de todo:** Powerball es un sistema **aleatorio
> e independiente, sin memoria**. Las pruebas de este informe lo confirman
> empíricamente. Ningún modelo predice el sorteo ni aumenta de forma real y
> garantizada la probabilidad de ganar. El *Statistical Confidence Score* **solo
> clasifica** combinaciones bajo criterios definidos; **no** predice el próximo
> resultado. No garantiza nada y no justifica gastar más dinero.

---

## 1. Resumen ejecutivo

- **Datos oficiales, validados entre sí:** `powerball.com` (2016–2026) + `NY Open
  Data d6yy-54nr` (2010–2026). Tras fusionar y limpiar: **1.952 sorteos**
  (2010-02-03 → 2026-05-30), **0 duplicados, 0 nulos, 0 fuera de rango**.
- **Segmentación por eras de reglas.** El modelado se limita a la matriz vigente
  **5/69 + 1/26 (desde 2015-10-07): 1.361 sorteos**.
- **Batería de pruebas estadísticas:** χ², Kolmogorov-Smirnov, Runs Test
  (Wald-Wolfowitz), autocorrelación, entropía de Shannon y cadena de Markov.
  **Todas son compatibles con aleatoriedad pura** (ver §4).
- **Modelo ensemble multicriterio** con un *Statistical Confidence Score* de
  **8 componentes ponderados** (§5).
- **Evaluación EXHAUSTIVA del espacio completo:** las **C(69,5) = 11.238.513**
  combinaciones de blancas × 26 Powerballs = **292.201.338 boletos**. Cobertura
  total, óptimo global garantizado (no muestreo). Monte Carlo adicional de
  **10.000.000** de iteraciones para el componente de simulación.
- **Validación cruzada:** Algoritmo Genético y Simulated Annealing convergen a la
  **misma** combinación óptima que el barrido exhaustivo. Frente de Pareto
  multiobjetivo calculado.
- A pedido tuyo: **1 sola jugada final**.

### 🎯 Jugada única (máxima cobertura: 1 entre 292.201.338)

| Blancas | Powerball | Statistical Confidence Score* |
|---|---|---|
| **03 · 23 · 36 · 53 · 63** | **04** | **≈ 89,7 / 100** |

\* Este puntaje mide **calidad estadística multicriterio** (frecuencia, recencia,
bayes, Monte Carlo, bootstrap, entropía, diversidad y no-popularidad), **no**
probabilidad de ganar. Esta combinación tiene **exactamente** la misma
probabilidad (1 entre 292.201.338) que cualquier otra, incluida `01-02-03-04-05`.

---

## 2. Metodología

1. **Ingesta** de las dos fuentes oficiales (`.xlsx`).
2. **Fusión** por unión + deduplicación por fecha (coinciden en los 1.286 sorteos
   solapados; la única diferencia era el orden de columnas).
3. **Limpieza:** ordenamiento canónico de las 5 blancas; verificación de nulos,
   duplicados de fecha/combinación y blancas repetidas.
4. **Validación de reglas por era** (rangos de blancas y Powerball por período).
5. **Segmentación**; análisis y modelado **solo sobre la era vigente**.
6. **Pruebas estadísticas** de aleatoriedad/uniformidad.
7. **Modelo ensemble** (8 componentes) + **Monte Carlo 10M** + **bootstrap 2000**.
8. **Barrido exhaustivo** de las 11.238.513 combinaciones con el score compuesto.
9. **Optimización** (GA, SA, Pareto) como validación cruzada del óptimo.
10. Código Python **reproducible** (semilla fija `20260705`).

---

## 3. Análisis de la data

### 3.1 Validación

| Métrica | Valor |
|---|---|
| Sorteos totales (2010–2026) | 1.952 |
| Duplicados por fecha | 0 |
| Combinaciones exactas repetidas | 0 |
| Filas fuera de rango (por era) | 0 |
| Valores nulos | 0 |
| Blancas repetidas dentro de una fila | 0 |

### 3.2 Eras de reglas (cambios históricos)

| Era | Sorteos | Desde | Hasta | Blancas | Powerball |
|---|---|---|---|---|---|
| Pre-2012 | 204 | 2010-02-03 | 2012-01-14 | 1–59 | 1–39 |
| 2012–2015 | 387 | 2012-01-18 | 2015-10-03 | 1–59 | 1–35 |
| **Actual** | **1.361** | **2015-10-07** | **2026-05-30** | **1–69** | **1–26** |

### 3.3 Frecuencias, calientes y fríos (era actual, 1.361 sorteos)

Frecuencia media por bola = **98,6 ± 11,0**. Esa dispersión es **ruido esperable
del azar**, no señal predictiva (lo confirma el χ² en §4).

- **Calientes (top):** 61 (122), 21 (121), 64 (119), 28 (118), 23 (116), 27 (116),
  33 (115), 32 (115), 36 (114), 63 (114).
- **Fríos (bottom):** 13 (73), 26 (78), 49 (79), 46 (81), 34 (83), 25 (86),
  48 (87), 55 (87), 14 (88), 65 (88).
- **Powerball:** más frecuente = **4** (64); menos = **16** (39).

### 3.4 Co-ocurrencia (pares / tríos)

| Pares top | Veces | | Tríos top | Veces |
|---|---|---|---|---|
| 52–64 | 15 | | 2–12–65 | 4 |
| 21–32 | 15 | | 7–15–36 | 4 |
| 37–44 | 15 | | 1–3–13 | 4 |
| 61–69 | 15 | | 12–20–21 | 4 |
| 51–61 | 14 | | 52–54–64 | 4 |

> 15 apariciones de un par en 1.361 sorteos está dentro de lo esperado por azar.
> No indica afinidad real. (Cuartetos: prácticamente ninguno se repite.)

### 3.5 Distribuciones estructurales

- **Par/impar:** domina **2 o 3 impares** (417 y 442 de 1.361). Todo-par/impar: 5,6 %.
- **Alto/bajo (1–34):** domina **2 o 3 bajas** (476 y 425).
- **Suma de las 5 blancas:** media **176,9 ± 43,0**; P5/25/50/75/95 = 104/149/178/206/248.
- **Distancia entre números:** los saltos entre blancas ordenadas se concentran
  en 1–15; combinaciones "apretadas" o progresiones exactas son minoría.
- **Repetición entre sorteos consecutivos:** media **0,35 blancas** (≈ azar).

---

## 4. Pruebas estadísticas de aleatoriedad (la evidencia honesta)

| Prueba | Estadístico | p-valor | Conclusión |
|---|---|---|---|
| **χ² uniformidad blancas** | X²=83,63 (gl 68) | **0,096** | Compatible con uniforme |
| **χ² uniformidad Powerball** | X²=23,15 (gl 25) | **0,569** | Compatible con uniforme |
| **Kolmogorov-Smirnov (sumas)** | D=0,020 | **0,622** | Sumas ≈ Normal |
| **Runs Test (Wald-Wolfowitz)** | z=+0,57 (692 rachas) | **0,569** | Sin patrón temporal |
| **Autocorrelación sumas (lags 1-5)** | ≤ 0,04 en valor absoluto | — | Sin dependencia serial |
| **Entropía de Shannon (blancas)** | 6,0997 / 6,1085 bits | — | **99,86 %** del máximo |
| **Cadena de Markov** | P(sale\|salió)=0,070 vs P(sale\|no)=0,073 | — | **Sin memoria** |

> **Lectura:** ninguna prueba detecta desviación significativa de la aleatoriedad.
> La entropía casi máxima y el Markov plano son la prueba matemática directa de
> que el histórico **no** contiene poder predictivo. Los "calientes/fríos"
> son fluctuación normal.

---

## 5. Modelo ensemble — Statistical Confidence Score

Cada combinación recibe un score compuesto (0–100) con estos **8 criterios**:

| Peso | Componente | Qué mide |
|---|---|---|
| 20 % | Frecuencia histórica | Frecuencia normalizada de sus 5 blancas |
| 15 % | Frecuencia reciente | Aparición con decaimiento exponencial (half-life 120) |
| 15 % | Inferencia bayesiana | Posterior Dirichlet (prior uniforme) por número |
| 10 % | Monte Carlo (10M) | Densidad empírica de su **suma** en 10.000.000 de sorteos simulados |
| 10 % | Bootstrap (2000) | Probabilidad por número re-muestreando sorteos con reemplazo |
| 10 % | Entropía | Dispersión de la combinación entre decenas (Shannon) |
| 10 % | Diversidad | 1 − solapamiento con los 10 sorteos más recientes |
| 10 % | Penalización popularidad | 1 − sesgo de jugadores (cumpleaños ≤31, consecutivos, poco rango) |

Técnicas de análisis integradas y su rol:

| Técnica | Rol en el sistema |
|---|---|
| Distribución hipergeométrica | Marco de probabilidad del sorteo (base combinatoria) |
| Monte Carlo (10M+) | Componente 4 + distribución de sumas/features |
| Inferencia bayesiana | Componente 3 |
| Bootstrap | Componente 5 (+ intervalos de confianza) |
| Frecuencias / Recencia / Gap | Componentes 1–2 y análisis exploratorio |
| Co-ocurrencia pares/tríos/cuartetos | Análisis exploratorio (§3.4) |
| Entropía de Shannon | Componente 6 + prueba de aleatoriedad |
| χ², KS, Runs, Autocorrelación, Markov | Pruebas de aleatoriedad (§4) |
| Par/impar, alto/bajo, decenas, sumas, distancias | Filtros + componentes de tipicidad |
| Rareza / popularidad esperada | Componente 8 + filtros |
| Ranking multicriterio / Ensemble | Score compuesto final |
| Algoritmo Genético, Simulated Annealing, Pareto | Validación cruzada del óptimo (§6) |

---

## 6. Barrido exhaustivo y validación cruzada

- **Espacio evaluado:** 11.238.513 combinaciones de blancas × 26 PB =
  **292.201.338 boletos**. Sin muestreo: **óptimo global garantizado**.
- **Filtros:** 8.447.438 combinaciones (75,2 %) superan los filtros de patrones
  populares/redundantes; banda de suma aceptada P10–P90 = **[120, 232]**.
- **Convergencia de optimizadores** (la mejor combinación coincide en los tres):

| Método | Combinación óptima | SCS |
|---|---|---|
| **Barrido exhaustivo** (óptimo global) | 03 23 36 53 63 | **≈ 89,7** |
| **Algoritmo Genético (DEAP)** | 03 23 36 53 63 | ≈ 89,7 |
| **Simulated Annealing** | 03 23 36 53 63 | ≈ 89,7 |
| Frente de Pareto (multiobjetivo) | 6 soluciones no-dominadas | — |

> El SCS es estable a ±0,1 según la semilla/iteraciones del Monte Carlo; la
> combinación óptima no cambia. Los tres optimizadores respetan el **mismo
> objetivo restringido** (penalizan patrones populares).

> Que tres algoritmos independientes lleguen al **mismo** óptimo confirma que la
> superficie del score está bien definida y que la selección **no** es arbitraria.

### Desglose de la jugada ganadora (03 23 36 53 63)

| Componente | Peso | Valor (0–1) |
|---|---|---|
| Frecuencia histórica | 0,20 | 0,916 |
| Frecuencia reciente | 0,15 | 0,687 |
| Inferencia bayesiana | 0,15 | 0,917 |
| Monte Carlo (suma) | 0,10 | 1,000 |
| Bootstrap | 0,10 | 0,916 |
| Entropía / Diversidad / No-popularidad | 0,30 | altos (spread amplio, 5 decenas, sin sesgo cumpleaños) |

Perfil: par/impar 3-2, bajo/alto 2-3, suma 178 (≈ centro histórico 177), rango 60,
una bola en cada franja de decenas → **estadísticamente típica y no-popular**.

---

## 7. Limitaciones

- **El sorteo no tiene memoria** (Markov y autocorrelación lo confirman).
  Frecuencia/recencia describen el pasado; no predicen el futuro.
- **Muestra pequeña frente al espacio:** 1.361 sorteos vs. 11,2 M de combinaciones.
- **Los filtros no cambian la probabilidad de acertar;** solo reducen el riesgo de
  **compartir** un eventual premio.
- **El SCS es una heurística de calidad multicriterio,** no una probabilidad de éxito.
- La data pre-2015 se excluye del modelado por el cambio de matriz de reglas.

---

## 8. Verdad matemática

- **Probabilidad del jackpot:** **1 entre 292.201.338** = C(69,5) × 26.
- **Acertar las 5 blancas** (sin PB): 1 entre 11.688.053.
- **Cada combinación es equiprobable.** No existe predicción real posible: las
  pruebas de §4 (χ², KS, Runs, entropía 99,86 %, Markov sin memoria) demuestran que
  el histórico es indistinguible del azar puro.
- **¿Para qué sirve el simulador?** Para lo único honesto: **evitar combinaciones
  populares** (cumpleaños, secuencias, patrones) que, de salir, se repartirían entre
  muchos ganadores; y para **entender el riesgo**. No adivina el sorteo, no mejora
  tu probabilidad de acertar y **no** es motivo para gastar más. Juega solo lo que
  puedas permitirte perder.

---

## 9. Stack de ingeniería (programas, librerías y motores)

El sistema integra un stack completo de ciencia de datos. Cada motor cumple una
función real (no decorativa):

| Componente | Rol en el sistema | Verificado |
|---|---|---|
| **Python 3.11+** (3.12 recomendado) | Lenguaje principal | ✅ |
| **NumPy / SciPy** | Cálculo numérico y pruebas estadísticas | ✅ |
| **Pandas** | Manejo de datos y tablas | ✅ |
| **Polars** | Lectura/fusión rápida de los xlsx | ✅ |
| **DuckDB** | Validación analítica por SQL | ✅ |
| **Numba** | Monte Carlo acelerado (JIT a máquina) | ✅ 10M iter |
| **PyMC** | Inferencia bayesiana / MCMC (NUTS) del Powerball | ✅ |
| **scikit-learn** | Normalización (MinMax) del feature-matrix | ✅ |
| **DEAP** | Algoritmo genético | ✅ converge |
| **Optuna** | Tuning automático de los 8 pesos | ✅ |
| **Plotly** | Dashboard interactivo (HTML autocontenido) | ✅ |
| **Pydantic** | Validación de configuración y pesos | ✅ |
| **FastAPI** | API REST (`/analysis`, `/play`) | ✅ |
| **Streamlit** | Interfaz visual | ✅ |
| **Rich / Loguru** | Consola elegante + logging | ✅ |
| **Pytest** | 10 pruebas unitarias | ✅ pasan |
| **Poetry / UV** | Gestión de dependencias (`pyproject.toml`) | ✅ |
| CuPy | Opcional, si hay GPU NVIDIA | — |

**Resultado empírico del motor PyMC (MCMC):** el **96 %** de los intervalos
creíbles (94 % HDI) de la probabilidad de cada Powerball **contienen 1/26**
(uniforme). Evidencia bayesiana directa de que no hay bolas "favorecidas".

### Estructura del proyecto

```
powerball_simulator.py     # base: carga, limpieza, análisis, barrido exhaustivo
powerball_advanced.py      # ensemble 8 componentes, pruebas, GA/SA/Pareto
pb_engine/                 # paquete integrado con todo el stack
  ├─ config.py             # Pydantic + Rich + Loguru
  ├─ data_io.py            # Polars + DuckDB
  ├─ engines.py            # Numba, PyMC, DEAP, Optuna, scikit-learn
  ├─ pipeline.py           # orquestador
  ├─ viz.py                # Plotly
  ├─ cli.py                # CLI (Rich)
  ├─ api.py                # FastAPI
  └─ app.py                # Streamlit
tests/test_engine.py       # pytest (10 pruebas)
pyproject.toml             # dependencias (UV/Poetry)
```

### Ejecución

```bash
pip install -r requirements.txt         # o: uv sync / poetry install
# CLI con salida Rich + dashboard Plotly:
python -m pb_engine.cli powerball_results_20160601_to_20260601.xlsx powerball_resultados.xlsx
# API REST:
uvicorn pb_engine.api:app --reload      # GET /play?n=1
# Interfaz visual:
streamlit run pb_engine/app.py
# Pruebas:
pytest -q
```

Todo es **reproducible** (semilla fija `20260705`). Para N jugadas diversas:
`Settings(n_plays=N)`.
