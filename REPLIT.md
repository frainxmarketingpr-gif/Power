# Cómo correr el simulador en Replit

Sí, corre en Replit. El repo ya trae `main.py`, `.replit` (módulo
`python-3.11`) y `.streamlit/` configurados. Elige una de estas rutas.

## Opción A — Rápida (recomendada) — simplemente pulsa Run

1. **Create Repl → Import from GitHub** → pega la URL del repo.
   - Replit detecta el proyecto por `main.py` + `.replit` (módulo `python-3.11`).
2. Pulsa **Run**. El `.replit` ya hace `pip install -r requirements-lite.txt` y
   levanta **Streamlit** en el webview automáticamente.
   - Deja el Monte Carlo en **1.000.000** de iteraciones (ligero).

> La versión "lite" omite **PyMC** (compila en C y pesa mucho). El sistema lo
> detecta y sigue funcionando: solo se salta el módulo MCMC bayesiano.

Si prefieres el análisis por consola en vez de la web, en el *Shell*:
```bash
python main.py
```

## Opción B — Completa (incluye PyMC/MCMC)

Requiere un plan con más RAM/almacenamiento (PyMC + pytensor son pesados) y un
compilador C. En el *Shell*:

```bash
pip install -r requirements.txt          # o:  pip install -e ".[full]"
```

Si PyMC no compila en tu plan de Replit, **no pasa nada**: el pipeline reporta
"PyMC no disponible" y continúa con el resto del ensemble.

## Cambiar qué se ejecuta

Edita la línea `run` de `.replit` (o úsalo desde el Shell):

```bash
# Interfaz visual (por defecto):
streamlit run pb_engine/app.py --server.port 8080 --server.address 0.0.0.0

# API REST:
uvicorn pb_engine.api:app --host 0.0.0.0 --port 8080     # luego GET /play?n=1

# CLI con salida Rich + dashboard Plotly:
python -m pb_engine.cli powerball_results_20160601_to_20260601.xlsx powerball_resultados.xlsx

# Pruebas:
pytest -q
```

## Consejos de recursos en Replit

| Situación | Qué hacer |
|---|---|
| **Replit no reconoce el proyecto** | Asegúrate de que existan `main.py` y `.replit` en la raíz (ya incluidos). No debe haber `replit.nix` cuando `.replit` usa `modules` — se ignoran entre sí. Reimporta el repo si lo clonaste con la config vieja. |
| No aparece el botón Run | El `entrypoint = "main.py"` y `modules = ["python-3.11"]` del `.replit` lo habilitan; refresca la pestaña. |
| Free tier (RAM/almacenamiento justos) | `requirements-lite.txt`; Monte Carlo 1M |
| Instalación muy lenta | Instala primero `numpy scipy pandas`, luego el resto |
| PyMC no instala/compila | Ignóralo: el pipeline reporta "PyMC no disponible" y continúa |
| Se queda sin memoria | Baja `mc_iters` y usa `boot_reps=500` en `Settings` |

> **Recuerda:** nada de esto predice el Powerball. El sorteo es aleatorio; la
> herramienta solo clasifica combinaciones y evita patrones populares.
