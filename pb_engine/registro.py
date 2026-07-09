#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registro del grupo: lleva la cuenta REAL de cada sorteo jugado.

Anota por semana: nº de boletos, costo, premio ganado y calcula el neto y el
neto ACUMULADO. Sirve para ver, con numeros honestos, como le va al grupo mes a
mes — sin ilusiones (a la larga la loteria da neto negativo).

Uso CLI:
  python -m pb_engine.registro                      # muestra el resumen
  python -m pb_engine.registro add FECHA N COSTO PREMIO ["nota"]
      ej: python -m pb_engine.registro add 2026-07-08 14 28 7 "sorteo miercoles"

Tambien: calcular_premio_principal(tickets, fecha) computa el premio del sorteo
PRINCIPAL automaticamente contra la data historica (el Double Play es aparte).
"""
from __future__ import annotations

import os
import pandas as pd

LEDGER = "registro_grupo.csv"
COLS = ["fecha", "n_boletos", "costo_usd", "premio_usd", "nota"]


def _load() -> pd.DataFrame:
    if os.path.exists(LEDGER):
        df = pd.read_csv(LEDGER)
    else:
        df = pd.DataFrame(columns=COLS)
    # Forzar tipos numericos (robusto ante el backend de strings de pandas 3.0)
    for c in ["n_boletos", "costo_usd", "premio_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "nota" in df.columns:
        df["nota"] = df["nota"].astype(str)
    if "fecha" in df.columns:
        df["fecha"] = df["fecha"].astype(str)
    return df


def agregar(fecha: str, n_boletos: int, costo: float, premio: float, nota: str = "") -> pd.DataFrame:
    """Anade (o reemplaza) el registro de una fecha y guarda el CSV."""
    df = _load()
    df = df[df["fecha"] != str(fecha)]                     # evita duplicar la fecha
    fila = {"fecha": str(fecha), "n_boletos": int(n_boletos),
            "costo_usd": float(costo), "premio_usd": float(premio), "nota": nota}
    df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    df = df.sort_values("fecha").reset_index(drop=True)
    df.to_csv(LEDGER, index=False)
    return df


def con_netos(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Devuelve el registro con columnas neto y neto acumulado calculadas."""
    df = (df if df is not None else _load()).copy()
    if df.empty:
        return df
    df["neto_usd"] = df["premio_usd"] - df["costo_usd"]
    df["neto_acum_usd"] = df["neto_usd"].cumsum()
    return df


def resumen() -> dict:
    """Totales del grupo: gastado, ganado, neto, semanas, mejor premio."""
    df = con_netos()
    if df.empty:
        return {"semanas": 0}
    return {
        "semanas": int(len(df)),
        "boletos_total": int(df["n_boletos"].sum()),
        "gastado_usd": float(df["costo_usd"].sum()),
        "ganado_usd": float(df["premio_usd"].sum()),
        "neto_usd": float(df["neto_usd"].sum()),
        "mejor_premio_usd": float(df["premio_usd"].max()),
        "neto_promedio_semana": float(df["neto_usd"].mean()),
    }


# --- Calculo automatico del premio del sorteo PRINCIPAL (opcional) ----------
PRIZE = {(5, True): None, (5, False): 1_000_000, (4, True): 50_000, (4, False): 100,
         (3, True): 100, (3, False): 7, (2, True): 7, (1, True): 4, (0, True): 4}


def calcular_premio_principal(tickets, fecha, path_2016="powerball_results_20160601_to_20260601.xlsx",
                              path_2010="powerball_resultados.xlsx") -> dict:
    """Suma el premio del sorteo PRINCIPAL de `fecha` para una lista de boletos.
    tickets: lista de (whites:list[5], pb:int). El Double Play es juego aparte."""
    from . import data_io
    from .config import Rules
    from . import check
    df, _ = data_io.load(path_2016, path_2010, Rules())
    draw = check.lookup_draw(df, fecha)
    if draw is None:
        return {"error": f"No hay sorteo principal en {fecha}."}
    win = set(draw["blancas"]); wpb = draw["powerball"]
    total = 0; detalle = []
    for whites, pb in tickets:
        m = len(set(int(x) for x in whites) & win)
        pbm = int(pb) == wpb
        val = PRIZE.get((m, pbm), 0) or 0
        total += val
        detalle.append({"boleto": sorted(int(x) for x in whites), "pb": int(pb),
                        "aciertos": m, "acerto_pb": pbm, "premio_usd": val})
    return {"fecha": draw["date"], "sorteo": draw["blancas"], "pb": wpb,
            "premio_total_usd": total, "detalle": detalle}


def _print_resumen():
    df = con_netos()
    if df.empty:
        print("Registro vacio. Agrega con: python -m pb_engine.registro add FECHA N COSTO PREMIO")
        return
    print("REGISTRO DEL GRUPO\n" + "=" * 60)
    for _, r in df.iterrows():
        print(f"  {r['fecha']}  | {int(r['n_boletos']):>2} boletos | "
              f"gasto ${r['costo_usd']:>6.2f} | premio ${r['premio_usd']:>7.2f} | "
              f"neto ${r['neto_usd']:>+7.2f} | acum ${r['neto_acum_usd']:>+8.2f}")
    s = resumen()
    print("=" * 60)
    print(f"  {s['semanas']} sorteos | gastado ${s['gastado_usd']:.2f} | "
          f"ganado ${s['ganado_usd']:.2f} | NETO ${s['neto_usd']:+.2f}")
    print(f"  Mejor premio: ${s['mejor_premio_usd']:.2f} | "
          f"neto promedio/sorteo: ${s['neto_promedio_semana']:+.2f}")
    if s["neto_usd"] < 0:
        print("  (Neto negativo = lo normal en la loteria a la larga. Es entretenimiento.)")


def _main():
    import sys
    args = sys.argv[1:]
    if args and args[0] == "add":
        fecha, n, costo, premio = args[1], args[2], args[3], args[4]
        nota = args[5] if len(args) > 5 else ""
        agregar(fecha, int(n), float(costo), float(premio), nota)
        print(f"Registrado {fecha}.\n")
    _print_resumen()


if __name__ == "__main__":
    _main()
