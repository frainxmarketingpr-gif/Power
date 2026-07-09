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
    coverage = st.toggle("Modo grupo (maxima cobertura)", value=False,
                         help="Para una bolsa/grupo: genera N jugadas que ENTRE "
                              "TODAS cubren la mayor variedad de numeros, con "
                              "minima repeticion. Reparte el riesgo (no cambia la "
                              "probabilidad de ganar).")
    run_btn = st.button("Ejecutar simulacion", type="primary")


if run_btn:
    import os
    # Semilla de eleccion nueva por corrida -> jugadas distintas en modo variedad
    pick_seed = int.from_bytes(os.urandom(4), "little") if variety else None
    res = pipeline.run(Settings(n_plays=n_plays, mc_iters=mc, variety=variety,
                                pick_seed=pick_seed, coverage=coverage))
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


# ---------------------------------------------------------------------------
# Registro del grupo (cuenta real: gasto vs premios, neto acumulado)
# ---------------------------------------------------------------------------
from pb_engine import registro

st.divider()
st.header("📒 Registro del grupo")
st.caption("La cuenta honesta: cuanto se gasta, cuanto se gana y el neto real "
           "acumulado sorteo a sorteo.")

with st.expander("➕ Anotar un sorteo"):
    with st.form("add_registro", clear_on_submit=True):
        cols = st.columns(4)
        f_fecha = cols[0].text_input("Fecha (YYYY-MM-DD)")
        f_n = cols[1].number_input("Boletos", 1, 500, 14)
        f_costo = cols[2].number_input("Gasto total $", 0.0, 100000.0, 42.0)
        f_premio = cols[3].number_input("Premio total $", 0.0, 1e9, 0.0)
        f_nota = st.text_input("Nota (sin comas)", "")
        if st.form_submit_button("Guardar en el registro"):
            if f_fecha.strip():
                registro.agregar(f_fecha.strip(), int(f_n), float(f_costo),
                                 float(f_premio), f_nota.replace(",", " "))
                st.success(f"Registrado {f_fecha}.")
            else:
                st.warning("Indica la fecha.")

led = registro.con_netos()
if led.empty:
    st.info("Aun no hay sorteos anotados. Usa **Anotar un sorteo** arriba.")
else:
    s = registro.resumen()
    m = st.columns(4)
    m[0].metric("Sorteos jugados", s["semanas"])
    m[1].metric("Gastado", f"${s['gastado_usd']:.0f}")
    m[2].metric("Ganado", f"${s['ganado_usd']:.0f}")
    m[3].metric("Neto acumulado", f"${s['neto_usd']:+.0f}",
                delta=f"{s['neto_promedio_semana']:+.1f}/sorteo")
    st.dataframe(led, use_container_width=True)
    if s["neto_usd"] < 0:
        st.caption("Neto negativo = lo normal en la loteria a la larga. "
                   "Es entretenimiento, no inversion.")
