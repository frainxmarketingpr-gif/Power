# 🎱 Simulador Estadístico Avanzado de Powerball

> **Powerball es un sistema aleatorio e independiente.** Este proyecto **NO
> predice** el sorteo ni aumenta de forma real o garantizada la probabilidad de
> ganar (**1 entre 292.201.338** para *toda* combinación). Solo **clasifica**
> combinaciones bajo criterios estadísticos y ayuda a **evitar combinaciones
> populares** (para no repartir un eventual premio) y a **entender el riesgo**.
> No recomienda gastar más dinero.

Simulador construido sobre **data histórica oficial** (powerball.com 2016–2026 +
NY Open Data 2010–2026, **1.952 sorteos**, validadas entre sí). Analiza la era
vigente (matriz 5/69 + 1/26, **1.361 sorteos**), aplica una batería de pruebas
estadísticas, un ensemble multicriterio de 8 componentes y un **barrido
exhaustivo de las 11.238.513 combinaciones** (× 26 = 292.201.338 boletos).

## 📄 Informe completo

👉 **[INFORME_POWERBALL.md](./INFORME_POWERBALL.md)** — resumen ejecutivo,
metodología, análisis, modelos, pruebas, jugada final y la "Verdad matemática".

## 🎯 Jugada sugerida (óptimo global, validado por 3 optimizadores)

| Blancas | Powerball | Statistical Confidence Score |
|---|---|---|
| **03 · 23 · 36 · 53 · 63** | **04** | ≈ 89,7 / 100 |

*El SCS mide calidad estadística, **no** probabilidad de ganar.*

## 🚀 Uso rápido

```bash
pip install -r requirements.txt
python -m pb_engine.cli powerball_results_20160601_to_20260601.xlsx powerball_resultados.xlsx
uvicorn pb_engine.api:app --reload      # API REST
streamlit run pb_engine/app.py          # interfaz visual
pytest -q                               # pruebas
```

## 🧰 Stack

Python · NumPy · SciPy · Pandas · Polars · DuckDB · Numba · PyMC · scikit-learn ·
DEAP · Optuna · Plotly · Pydantic · FastAPI · Streamlit · Rich · Loguru · Pytest.

## ⚠️ Verdad matemática

Todas las pruebas (χ², Kolmogorov-Smirnov, Runs Test, autocorrelación, entropía
99,86 % del máximo, cadena de Markov sin memoria, y el 96 % de los intervalos
creíbles bayesianos conteniendo la uniforme) confirman que el histórico es
**indistinguible del azar puro**. Ninguna combinación tiene mayor probabilidad
intrínseca de salir. Juega solo lo que puedas permitirte perder.
