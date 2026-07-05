"""Interfaz visual con Streamlit.  Ejecutar:  streamlit run pb_engine/app.py

Streamlit ejecuta este archivo como script suelto (no como paquete), por eso
se anade la raiz del repo a sys.path y se usan imports absolutos: asi funciona
tanto en local como en Replit."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go

from pb_engine.config import Settings, setup_logging
from pb_engine import pipeline

setup_logging()

st.set_page_config(page_title="Powerball — Simulador estadistico", layout="wide")

st.title("🎱 Simulador estadistico avanzado de Powerball")
st.warning("Powerball es **aleatorio e independiente**. Esta herramienta **no predice** "
           "el sorteo ni aumenta tu probabilidad real de ganar "
           "(1 entre 292.201.338 para toda combinacion). Solo clasifica combinaciones "
           "por criterios estadisticos y ayuda a **evitar combinaciones populares**.")

with st.sidebar:
    st.header("Configuracion")
    n_plays = st.slider("Numero de jugadas", 1, 20, 1)
    mc = st.select_slider("Iteraciones Monte Carlo",
                          [1_000_000, 5_000_000, 10_000_000], value=1_000_000)
    variety = st.toggle("Variedad (jugadas distintas cada vez)", value=True,
                        help="Elige al azar entre las mejores combinaciones "
                             "(todas de score casi identico y misma probabilidad "
                             "real). Desactivalo para el maximo global fijo.")
    run_btn = st.button("Ejecutar simulacion", type="primary")


if run_btn:
    import os
    # Semilla de eleccion nueva por corrida -> jugadas distintas en modo variedad
    pick_seed = int.from_bytes(os.urandom(4), "little") if variety else None
    res = pipeline.run(Settings(n_plays=n_plays, mc_iters=mc,
                                variety=variety, pick_seed=pick_seed))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Validacion de datos")
        st.json(res.validation)
        st.subheader("Pruebas de aleatoriedad")
        t = res.tests
        st.write({
            "Chi2 blancas (p)": round(t["chi2_white"][1], 3),
            "Chi2 Powerball (p)": round(t["chi2_pb"][1], 3),
            "Runs test (p)": round(t["runs_test"][3], 3),
            "Entropia (% del maximo)": round(100 * t["entropy_white"][2], 2),
        })
    with c2:
        st.subheader("Frecuencia de bolas blancas")
        wf = res.analysis["white_freq"]
        st.plotly_chart(go.Figure(go.Bar(x=list(wf.index), y=list(wf.values))),
                        use_container_width=True)

    st.subheader("Jugada(s) final(es)")
    st.dataframe(res.plays, use_container_width=True)
    st.caption("El SCS mide calidad estadistica, NO probabilidad de ganar.")
else:
    st.info("Configura los parametros y pulsa **Ejecutar simulacion**.")
