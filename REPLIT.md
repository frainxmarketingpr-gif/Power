# Cómo correr el simulador en Replit

Sí, corre en Replit. El repo ya trae `.replit`, `replit.nix` y `.streamlit/`
configurados. Elige una de estas rutas según tu plan.

## Opción A — Rápida (recomendada, free tier)

1. **Create Repl → Import from GitHub** → pega la URL del repo.
2. En el panel *Shell*:
   ```bash
   pip install -r requirements-lite.txt
   ```
   *(La versión "lite" omite PyMC, que compila en C y pesa mucho. El sistema lo
   detecta y sigue funcionando: solo se salta el módulo MCMC bayesiano.)*
3. Pulsa **Run**. Se abre la interfaz **Streamlit** en el webview.
   - Deja el Monte Carlo en **1.000.000** de iteraciones (suficiente y ligero).

## Opción B — Completa (incluye PyMC/MCMC)

Requiere un plan con más RAM/almacenamiento (PyMC + pytensor son pesados).

```bash
pip install -r requirements.txt
```

`replit.nix` ya incluye `gcc`/`gfortran` para compilar pytensor. La primera
ejecución de PyMC tarda unos segundos en compilar.

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
| Free tier (RAM/almacenamiento justos) | `requirements-lite.txt`; Monte Carlo 1M |
| Instalación muy lenta | Instala primero `numpy scipy pandas`, luego el resto |
| PyMC no instala/compila | Ignóralo: el pipeline reporta "PyMC no disponible" y continúa |
| Se queda sin memoria | Baja `mc_iters` y usa `boot_reps=500` en `Settings` |

> **Recuerda:** nada de esto predice el Powerball. El sorteo es aleatorio; la
> herramienta solo clasifica combinaciones y evita patrones populares.
