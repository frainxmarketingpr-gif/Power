#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Punto de entrada del proyecto (lo que Replit reconoce como app Python).

- `python main.py`            -> analisis estadistico completo por consola.
- `streamlit run pb_engine/app.py` -> interfaz web (por defecto en Replit).
- `uvicorn pb_engine.api:app`      -> API REST.

Recordatorio: Powerball es aleatorio. Esto NO predice el sorteo ni aumenta la
probabilidad real de ganar (1 entre 292.201.338 para toda combinacion).
"""
import sys

from pb_engine.cli import main

if __name__ == "__main__":
    # Sin argumentos, usa los xlsx incluidos en la raiz del repo.
    if len(sys.argv) == 1:
        sys.argv += ["powerball_results_20160601_to_20260601.xlsx",
                     "powerball_resultados.xlsx"]
    main()
