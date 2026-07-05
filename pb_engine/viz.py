"""Visualizaciones con Plotly (se exportan a HTML autocontenido)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def dashboard(a: dict, model, best_combos, out_html: str = "powerball_dashboard.html"):
    """best_combos: array (m,5) del barrido exhaustivo YA calculado (Result.exhaustive[0]);
    se usa su mejor fila para el radar, sin re-ejecutar el barrido."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "polar"}]],
        subplot_titles=("Frecuencia de bolas blancas (1-69)",
                        "Frecuencia del Powerball (1-26)",
                        "Distribucion de la suma de 5 blancas",
                        "Componentes del score de la jugada final"),
    )

    wf = a["white_freq"]
    fig.add_bar(x=list(wf.index), y=list(wf.values), name="blancas",
                marker_color="#2b8cbe", row=1, col=1)
    mean = wf.mean()
    fig.add_hline(y=mean, line_dash="dash", line_color="grey", row=1, col=1)

    pf = a["pb_freq"]
    fig.add_bar(x=list(pf.index), y=list(pf.values), name="powerball",
                marker_color="#e34a33", row=1, col=2)

    sums = a["sums"]
    fig.add_histogram(x=sums, nbinsx=40, name="suma", marker_color="#31a354",
                      row=2, col=1)

    # Radar de componentes de la mejor combinacion (ya calculada, sin re-barrer)
    feats = model.feature_batch(np.asarray(best_combos)[:1])[0]
    labels = ["frec", "recien", "bayes", "MC", "boot", "entrop", "divers", "no-pop"]
    fig.add_scatterpolar(r=list(feats) + [feats[0]], theta=labels + [labels[0]],
                         fill="toself", name="mejor combo", row=2, col=2)

    fig.update_layout(height=820, showlegend=False,
                      title_text="Powerball — Dashboard estadistico "
                                 "(analisis, NO prediccion)")
    fig.write_html(out_html, include_plotlyjs="inline")
    return out_html
