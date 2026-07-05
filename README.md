# 🎱 Simulador Estadístico Avanzado de Powerball

> **Powerball es un sistema aleatorio e independiente.** Este proyecto **NO
> predice** el sorteo ni aumenta de forma real o garantizada la probabilidad de
> ganar (**1 entre 292.201.338** para *toda* combinación). Solo **clasifica**
> combinaciones bajo criterios estadísticos y ayuda a **evitar combinaciones
> populares** (para no repartir un eventual premio) y a **entender el riesgo**.
> No recomienda gastar más dinero.

Simulador construido sobre **data histórica oficial** (powerball.com 2016–2026 +
NY Open Data 2010–2026 + actualizaciones, **1.967 sorteos** hasta **2026-07-04**,
validadas entre sí). Analiza la era vigente (matriz 5/69 + 1/26, **1.376 sorteos**),
aplica una batería de pruebas estadísticas, un ensemble multicriterio de 8
componentes y un **barrido exhaustivo de las 11.238.513 combinaciones**
(× 26 = 292.201.338 boletos).

> **Datos al día:** para sumar sorteos nuevos, añade filas a
> [`powerball_actualizaciones.csv`](./powerball_actualizaciones.csv)
> (columnas `date,weekday,n1..n5,pb,power_play`). Ambos cargadores lo fusionan
> automáticamente; no hace falta tocar los Excel.

## 📄 Informe completo

👉 **[INFORME_POWERBALL.md](./INFORME_POWERBALL.md)** — resumen ejecutivo,
metodología, análisis, modelos, pruebas, jugada final y la "Verdad matemática".

## 🎯 Jugada sugerida (se recalcula con cada actualización de datos)

Con los datos hasta 2026-07-04, el óptimo determinista es `18 · 23 · 32 · 52 · 64`
+ PB `14` (SCS ≈ 91). **Cambia cuando entran sorteos nuevos** — no es fija.

*El SCS mide calidad estadística, **no** probabilidad de ganar. Toda combinación
tiene la misma probabilidad: 1 entre 292.201.338.*

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
