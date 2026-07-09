#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analizador de jugadas: revisa tus lineas contra un sorteo y muestra QUE
numeros ganadores cubriste, EN QUE lineas cayeron, y cual fue la mejor linea.

Tambien explica, con matematica, por que NO se puede "acomodar" los numeros para
que los ganadores caigan en la misma linea (sistema de rueda / wheeling): sin
conocer el resultado, ninguna colocacion aumenta la probabilidad.
"""
from __future__ import annotations
from math import comb

# Tabla de premios era vigente (sorteo principal)
PRIZE = {(5, True): "JACKPOT", (5, False): "$1,000,000", (4, True): "$50,000",
         (4, False): "$100", (3, True): "$100", (3, False): "$7",
         (2, True): "$7", (1, True): "$4", (0, True): "$4"}
PRIZE_VAL = {"JACKPOT": 0, "$1,000,000": 1_000_000, "$50,000": 50_000,
             "$100": 100, "$7": 7, "$4": 4, "-": 0}


def analizar(lineas, win_whites, win_pb):
    """lineas: lista de (nombre, whites:list[5], pb:int).
    Devuelve el analisis completo contra el sorteo (win_whites, win_pb)."""
    win = set(int(x) for x in win_whites)
    wpb = int(win_pb)
    filas, total = [], 0
    cobertura = {}          # numero ganador -> lineas donde aparece
    for nombre, whites, pb in lineas:
        ws = set(int(x) for x in whites)
        inter = sorted(ws & win)
        pbm = int(pb) == wpb
        cat = PRIZE.get((len(inter), pbm), "-")
        total += PRIZE_VAL.get(cat, 0)
        for n in inter:
            cobertura.setdefault(n, []).append(nombre)
        if pbm:
            cobertura.setdefault(f"PB{wpb}", []).append(nombre)
        filas.append(dict(linea=nombre, aciertos=inter, n=len(inter),
                          acerto_pb=pbm, premio=cat))
    cubiertos = sorted(n for n in win if n in cobertura)
    faltantes = sorted(n for n in win if n not in cobertura)
    mejor = max(filas, key=lambda f: (f["n"], f["acerto_pb"]))
    # ¿maximo de numeros ganadores juntos en UNA sola linea?
    max_en_una = max(f["n"] for f in filas)
    return dict(filas=filas, cubiertos=cubiertos, faltantes=faltantes,
                cobertura=cobertura, mejor=mejor, premio_total=total,
                max_ganadores_en_una_linea=max_en_una,
                pb_cubierto=(f"PB{wpb}" in cobertura))


def lineas_para_garantizar_5(pool_size):
    """Cuantas lineas hacen falta para GARANTIZAR que, si los 5 ganadores estan
    dentro de tu pool de `pool_size` numeros, los 5 caigan en una misma linea.
    = cubrir TODAS las combinaciones C(pool_size, 5)."""
    return comb(pool_size, 5)


def _print(res):
    print("ANALISIS DE JUGADAS vs SORTEO\n" + "=" * 60)
    for f in res["filas"]:
        pb = " +PB" if f["acerto_pb"] else ""
        print(f"  {f['linea']}: acierta {f['aciertos']} ({f['n']}){pb}  -> {f['premio']}")
    print("=" * 60)
    print(f"  Numeros ganadores que CUBRISTE: {res['cubiertos']}")
    print(f"  Numeros ganadores que te FALTARON (no estaban en ninguna linea): {res['faltantes']}")
    print(f"  Powerball: {'ACERTADO' if res['pb_cubierto'] else 'no'}")
    print(f"  Maximo de ganadores juntos en UNA linea: {res['max_ganadores_en_una_linea']}")
    print(f"  Premio total: ${res['premio_total']}")
