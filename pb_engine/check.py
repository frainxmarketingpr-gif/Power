"""Consulta por fecha y comparacion de un boleto contra el sorteo real.

Permite: (1) buscar los numeros ganadores de una fecha concreta en la data
historica, y (2) comparar una combinacion jugada contra ese sorteo real,
devolviendo aciertos y categoria de premio (segun la tabla oficial de la era
vigente 5/69 + 1/26).

ADVERTENCIA: comparar contra sorteos pasados es un BACKTEST didactico. No mejora
la probabilidad de aciertos futuros: Powerball es aleatorio y sin memoria.
"""
from __future__ import annotations

import pandas as pd

WCOLS = [f"n{i}" for i in range(1, 6)]

# Tabla oficial de premios de la era vigente (montos base, sin Power Play).
# Clave: (aciertos_blancas, acierta_powerball) -> (categoria, premio_usd)
PRIZE_TABLE = {
    (5, True):  ("Jackpot",            None),      # premio mayor (variable)
    (5, False): ("Match 5",            1_000_000),
    (4, True):  ("Match 4 + PB",       50_000),
    (4, False): ("Match 4",            100),
    (3, True):  ("Match 3 + PB",       100),
    (3, False): ("Match 3",            7),
    (2, True):  ("Match 2 + PB",       7),
    (1, True):  ("Match 1 + PB",       4),
    (0, True):  ("Solo Powerball",     4),
}


def _norm_date(date) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def lookup_draw(df: pd.DataFrame, date) -> dict | None:
    """Devuelve el sorteo de `date` (o None si no hubo sorteo ese dia).

    df: DataFrame historico con columnas date, n1..n5, pb (y opcional era).
    """
    d = _norm_date(date)
    row = df[df["date"].dt.normalize() == d]
    if row.empty:
        return None
    r = row.iloc[0]
    whites = sorted(int(r[c]) for c in WCOLS)
    return {
        "date": d.date().isoformat(),
        "weekday": d.day_name(),
        "blancas": whites,
        "powerball": int(r["pb"]),
        "era": r["era"] if "era" in row.columns else None,
    }


def prize_for(white_matches: int, pb_match: bool) -> tuple[str, int | None]:
    """Categoria y premio base para (aciertos_blancas, acierta_powerball)."""
    return PRIZE_TABLE.get((white_matches, pb_match), ("Sin premio", 0))


def check_ticket(df: pd.DataFrame, whites, pb: int, date) -> dict:
    """Compara un boleto (5 blancas + powerball) contra el sorteo real de `date`.

    Devuelve aciertos, si acerto el Powerball, categoria y premio base. Valida
    que el boleto sea legal (5 blancas distintas 1-69 y powerball 1-26).
    """
    whites = sorted(int(x) for x in whites)
    pb = int(pb)
    # Validacion de boleto legal (era vigente)
    errs = []
    if len(set(whites)) != 5:
        errs.append("Debes indicar 5 blancas distintas.")
    if any(not (1 <= x <= 69) for x in whites):
        errs.append("Las blancas deben estar en el rango 1-69.")
    if not (1 <= pb <= 26):
        errs.append("El Powerball debe estar en el rango 1-26.")
    if errs:
        return {"error": " ".join(errs)}

    draw = lookup_draw(df, date)
    if draw is None:
        return {"error": f"No hay sorteo registrado en {_norm_date(date).date()} "
                         f"(Powerball sortea lun/mie/sab). Revisa la fecha o la data."}

    # Solo la era vigente: antes de 2015-10-07 la matriz (y los premios) eran
    # distintos, asi que validar un boleto actual contra esos sorteos no procede.
    if draw.get("era") not in (None, "Actual"):
        return {"error": f"El sorteo de {draw['date']} pertenece a la era "
                         f"'{draw['era']}' (reglas y premios distintos). Solo se "
                         f"verifican boletos de la era vigente 5/69 + 1/26 "
                         f"(desde 2015-10-07)."}

    win_white = set(draw["blancas"])
    aciertos = sorted(set(whites) & win_white)
    pb_match = pb == draw["powerball"]
    categoria, premio = prize_for(len(aciertos), pb_match)

    return {
        "fecha": draw["date"],
        "tu_boleto": {"blancas": whites, "powerball": pb},
        "sorteo_real": {"blancas": draw["blancas"], "powerball": draw["powerball"]},
        "blancas_acertadas": aciertos,
        "n_aciertos_blancas": len(aciertos),
        "acerto_powerball": pb_match,
        "categoria": categoria,
        "premio_base_usd": premio,
        "nota": ("Backtest didactico contra un sorteo pasado. NO mejora la "
                 "probabilidad futura: el sorteo es aleatorio y sin memoria."),
    }


def compare_range(df: pd.DataFrame, whites, pb: int, start=None, end=None) -> dict:
    """Compara un boleto FIJO contra TODOS los sorteos historicos (o un rango).

    Util para ver cuantas veces esa combinacion habria premiado en el pasado
    (casi siempre 0 o premios minimos): ilustra lo improbable del jackpot.
    """
    whites_list = sorted(int(x) for x in whites)
    pb = int(pb)
    if (len(set(whites_list)) != 5 or any(not (1 <= x <= 69) for x in whites_list)
            or not (1 <= pb <= 26)):
        return {"error": "Boleto invalido: 5 blancas distintas 1-69 + Powerball 1-26."}
    whites_set = set(whites_list)
    sub = df
    if start is not None:
        sub = sub[sub["date"] >= _norm_date(start)]
    if end is not None:
        sub = sub[sub["date"] <= _norm_date(end)]

    tiers: dict[str, int] = {}
    best = None
    for _, r in sub.iterrows():
        win_white = {int(r[c]) for c in WCOLS}
        m = len(whites_set & win_white)
        pbm = pb == int(r["pb"])
        cat, premio = prize_for(m, pbm)
        if cat != "Sin premio":
            tiers[cat] = tiers.get(cat, 0) + 1
            if best is None or (m, pbm) > (best[1], best[2]):
                best = (r["date"].date().isoformat(), m, pbm, cat)
    return {
        "sorteos_evaluados": int(len(sub)),
        "veces_con_premio": int(sum(tiers.values())),
        "por_categoria": tiers,
        "mejor_resultado": (
            {"fecha": best[0], "aciertos_blancas": best[1],
             "acerto_pb": best[2], "categoria": best[3]} if best else None),
        "nota": ("En la inmensa mayoria de sorteos una combinacion fija NO premia. "
                 "Esto ilustra la magnitud real de la probabilidad, no una estrategia."),
    }


def _main():
    """CLI: python -m pb_engine.check FECHA [b1 b2 b3 b4 b5 PB]

    Sin boleto -> muestra los numeros ganadores de esa fecha.
    Con 5 blancas + PB -> compara el boleto contra el sorteo real de esa fecha.
    """
    import sys
    import json
    from .config import Settings
    from . import data_io

    args = sys.argv[1:]
    if not args:
        print("Uso: python -m pb_engine.check FECHA [b1 b2 b3 b4 b5 PB]")
        raise SystemExit(1)

    s = Settings()
    df, _ = data_io.load(s.path_2016, s.path_2010, s.rules)
    fecha = args[0]

    try:
        _norm_date(fecha)
    except (ValueError, TypeError):
        print(json.dumps({"error": f"Fecha invalida: {fecha!r} (usa YYYY-MM-DD)."},
                         ensure_ascii=False))
        raise SystemExit(1)

    if len(args) >= 7:
        whites = [int(x) for x in args[1:6]]
        pb = int(args[6])
        out = check_ticket(df, whites, pb, fecha)
    else:
        out = lookup_draw(df, fecha) or {"error": f"No hay sorteo en {fecha} (lun/mie/sab)."}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
