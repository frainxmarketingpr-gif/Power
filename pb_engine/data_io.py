"""Carga y validacion de datos con Polars (velocidad) + DuckDB (SQL analitico).

Fuentes oficiales:
  A) powerball.com (2016-2026)
  B) NY Open Data d6yy-54nr (2010-2026)
Devuelve un DataFrame de pandas limpio y ordenado, mas un reporte de validacion.
"""
from __future__ import annotations

import duckdb
import polars as pl
import pandas as pd
from loguru import logger

from .config import Rules

WCOLS = [f"n{i}" for i in range(1, 6)]

RULE_ERAS = [
    ("Pre-2012",  None,        "2012-01-15", (1, 59), (1, 39)),
    ("2012-2015", "2012-01-15", "2015-10-07", (1, 59), (1, 35)),
    ("Actual",    "2015-10-07", None,         (1, 69), (1, 26)),
]


def _read_polars(path_2016: str, path_2010: str) -> pl.DataFrame:
    """Lee ambos xlsx con Polars, normaliza columnas y fusiona (union)."""
    a = pl.read_excel(path_2016, sheet_name="Datos")
    b = pl.read_excel(path_2010, sheet_name="Todos_desde_2010")

    a = a.select(
        pl.col("date").cast(pl.Date).alias("date"),
        *[pl.col(f"number_{i}").cast(pl.Int64).alias(f"n{i}") for i in range(1, 6)],
        pl.col("powerball").cast(pl.Int64).alias("pb"),
    ).with_columns(pl.lit("powerball.com").alias("source"))

    b = b.select(
        pl.col("fecha_sorteo").cast(pl.Date).alias("date"),
        *[pl.col(f"bola_blanca_{i}").cast(pl.Int64).alias(f"n{i}") for i in range(1, 6)],
        pl.col("powerball").cast(pl.Int64).alias("pb"),
    ).with_columns(pl.lit("ny_open_data").alias("source"))

    # B (2010+) tiene prioridad por cobertura; A solo aporta fechas ya presentes
    full = pl.concat([b, a]).unique(subset="date", keep="first").sort("date")
    return full


def load(path_2016: str, path_2010: str, rules: Rules | None = None):
    rules = rules or Rules()
    logger.info("Cargando fuentes con Polars...")
    pf = _read_polars(path_2016, path_2010)

    # Ordenar canonicamente las 5 blancas por fila
    import numpy as np
    arr = np.sort(pf.select(WCOLS).to_numpy(), axis=1)
    for i, c in enumerate(WCOLS):
        pf = pf.with_columns(pl.Series(c, arr[:, i]))

    df = pf.to_pandas()
    df["date"] = pd.to_datetime(df["date"])

    # --- Validacion analitica con DuckDB (SQL) ---
    logger.info("Validando con DuckDB (SQL)...")
    con = duckdb.connect()
    con.register("draws", df)
    rep = {}
    rep["draws_total"] = con.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
    rep["date_duplicates"] = con.execute(
        "SELECT COUNT(*)-COUNT(DISTINCT date) FROM draws").fetchone()[0]
    rep["nulls"] = con.execute(
        "SELECT SUM(CASE WHEN n1 IS NULL OR n2 IS NULL OR n3 IS NULL OR n4 IS NULL "
        "OR n5 IS NULL OR pb IS NULL THEN 1 ELSE 0 END) FROM draws").fetchone()[0] or 0
    rep["dup_whites_in_row"] = con.execute(
        "SELECT SUM(CASE WHEN len(list_distinct([n1,n2,n3,n4,n5]))<5 "
        "THEN 1 ELSE 0 END) FROM draws").fetchone()[0] or 0

    # Era por fila + validacion de rango por era
    def era_of(d):
        for name, start, end, _, _ in RULE_ERAS:
            if (start is None or str(d.date()) >= start) and (end is None or str(d.date()) < end):
                return name
        return "?"

    df["era"] = df["date"].apply(era_of)
    bad = 0
    for name, start, end, (wlo, whi), (plo, phi) in RULE_ERAS:
        sub = df[df["era"] == name]
        if sub.empty:
            continue
        wbad = ((sub[WCOLS] < wlo) | (sub[WCOLS] > whi)).any(axis=1)
        pbad = (sub["pb"] < plo) | (sub["pb"] > phi)
        bad += int((wbad | pbad).sum())
    rep["out_of_range_rows"] = bad
    con.close()
    logger.info(f"Datos OK: {rep['draws_total']} sorteos, "
                f"{rep['out_of_range_rows']} fuera de rango, {rep['nulls']} nulos.")
    return df, rep


def current_era(df: pd.DataFrame, rules: Rules | None = None) -> pd.DataFrame:
    rules = rules or Rules()
    return df[df["date"] >= pd.Timestamp(rules.era_start)].reset_index(drop=True)


def era_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por era via DuckDB."""
    con = duckdb.connect()
    con.register("draws", df)
    out = con.execute(
        "SELECT era, COUNT(*) AS n, MIN(date) AS desde, MAX(date) AS hasta "
        "FROM draws GROUP BY era ORDER BY desde").fetchdf()
    con.close()
    return out
