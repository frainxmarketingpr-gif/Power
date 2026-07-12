#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulador de Loto Cash PR (5 de 1-35 + Bolo 1-10).

Imita el ritual de las 5 pruebas pre-sorteo: corre 5 sorteos SIMULADOS y luego
entrega la jugada #6 CONSTRUIDA.

HONESTIDAD (leelo):
  - Los 5 sorteos son aleatorios uniformes. Van a salir todos distintos. Esa es
    la demostracion de que el azar manda: ninguno predice al siguiente.
  - La jugada #6 NO se deriva de los 5 anteriores. Eso seria mentira: las bolas
    no tienen memoria. La #6 se CONSTRUYE para no-compartir premio.
  - La #6 tiene EXACTAMENTE la misma probabilidad de ganar que cualquier combo:
    1 en 3,246,320. Lo que cambia es cuanta gente compartiria contigo si ganas.
"""
from __future__ import annotations
import os
import secrets
from math import comb

N_WHITE = 35          # blancas: 5 de 1..35
N_PICK = 5
N_BOLO = 10           # Bolo: 1 de 1..10
JACKPOT_ODDS = comb(N_WHITE, N_PICK) * N_BOLO   # 3,246,320

# --- Modelo de sesgos de jugador (documentado, NO data medida de PR) ---
# Cuanto MENOS marca la gente un numero, mas ORO es para no-compartir.
# Peso de "rareza": mas alto = menos popular = mejor para no compartir.
def rarity_weight(n: int) -> float:
    w = 1.0
    if n > 31:            # 32-35: nadie los marca (fuera de dias de mes)
        w += 2.5
    elif n > 25:          # 26-31: infrautilizados
        w += 1.2
    elif n <= 12:         # 1-12: dias/meses = muy jugados
        w -= 0.6
    if n in (7, 3, 11):   # "numeros de suerte"
        w -= 0.5
    return max(w, 0.15)

BOLO_RARITY = {1:0.5, 2:0.7, 3:0.4, 4:0.8, 5:0.6, 6:0.9, 7:0.4,
               8:1.3, 9:1.4, 10:1.5}   # 8,9,10 menos marcados


# --------------------------------------------------------------------------
def _rng_choice(pool, k, weights=None):
    """Muestreo sin reemplazo con secrets (CSPRNG). weights opcional."""
    pool = list(pool)
    out = []
    w = list(weights) if weights else [1.0] * len(pool)
    for _ in range(k):
        total = sum(w)
        r = secrets.randbelow(10**9) / 10**9 * total
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if r <= acc:
                out.append(pool.pop(i))
                w.pop(i)
                break
        else:
            out.append(pool.pop())
            w.pop()
    return out


def sorteo_aleatorio():
    """Un sorteo REAL simulado: uniforme, sin sesgo. Como la maquina de aire."""
    whites = sorted(_rng_choice(range(1, N_WHITE + 1), N_PICK))
    bolo = secrets.randbelow(N_BOLO) + 1
    return whites, bolo


# --- Receta estructural (para que la #6 se vea natural, no para ganar) ---
def cumple_estructura(whites) -> bool:
    s = sum(whites)
    pares = sum(1 for x in whites if x % 2 == 0)
    bajos = sum(1 for x in whites if x <= 17)
    decenas = len({(x - 1) // 10 for x in whites})
    rango = max(whites) - min(whites)
    consec = sum(1 for a, b in zip(whites, whites[1:]) if b - a == 1)
    return (74 <= s <= 118 and 2 <= pares <= 3 and 2 <= bajos <= 3
            and 3 <= decenas <= 4 and 17 <= rango <= 31 and consec <= 1)


def jugada_construida(intentos=20000):
    """La #6: no-popular + estructura realista + Bolo raro.
    Misma probabilidad de ganar; menor probabilidad de COMPARTIR."""
    w_rar = [rarity_weight(n) for n in range(1, N_WHITE + 1)]
    mejor, mejor_score = None, -1
    for _ in range(intentos):
        whites = sorted(_rng_choice(range(1, N_WHITE + 1), N_PICK, w_rar))
        if not cumple_estructura(whites):
            continue
        score = sum(rarity_weight(x) for x in whites)
        if score > mejor_score:
            mejor_score, mejor = score, whites
    bolo = _rng_choice(range(1, N_BOLO + 1), 1,
                       [BOLO_RARITY[i] for i in range(1, N_BOLO + 1)])[0]
    return mejor, bolo, mejor_score


def edge_no_compartir(whites, bolo) -> float:
    """Estimacion del multiplicador de valor por no-compartir (payout edge)."""
    rar = sum(rarity_weight(x) for x in whites) / N_PICK  # ~0.5..3
    rar += BOLO_RARITY[bolo] * 0.3
    # mapea rareza a multiplicador ~1.0 (popular) .. ~2.8 (muy raro)
    return round(1.0 + min(rar, 3.0) * 0.6, 2)


# --------------------------------------------------------------------------
def main():
    print("=" * 62)
    print(" SIMULADOR LOTO CASH PR  —  5 pruebas + jugada #6")
    print("=" * 62)
    print(f" Probabilidad del premio mayor: 1 en {JACKPOT_ODDS:,}")
    print(" (identica para TODA combinacion, incluida la #6)\n")

    print(" LAS 5 PRUEBAS (sorteos simulados, aleatorios):")
    print(" " + "-" * 58)
    previos = []
    for i in range(1, 6):
        w, b = sorteo_aleatorio()
        previos.append((w, b))
        print(f"   Prueba {i}:  {w}   Bolo {b}")
    # demostracion honesta: los 5 son distintos e independientes
    distintos = len({tuple(w) for w, _ in previos})
    print(f"\n   -> {distintos}/5 combinaciones distintas. Ninguna predice la otra.")
    print("      Las bolas no tienen memoria: esto lo prueba en vivo.\n")

    print(" JUGADA #6 (CONSTRUIDA, no derivada de las 5):")
    print(" " + "-" * 58)
    w6, b6, score = jugada_construida()
    edge = edge_no_compartir(w6, b6)
    print(f"   >>>  {w6}   Bolo {b6}  <<<")
    print(f"\n   Suma: {sum(w6)}   Pares: {sum(1 for x in w6 if x%2==0)}   "
          f"Bajos(<=17): {sum(1 for x in w6 if x<=17)}   "
          f"Rango: {max(w6)-min(w6)}")
    print(f"   Score de rareza: {score:.2f}  (mas alto = menos gente lo marca)")
    print(f"   Edge de NO-COMPARTIR estimado: {edge}x mas dinero para TI si ganas")
    print()
    print(" LO HONESTO:")
    print("   - Prob. de ganar de la #6 = 1 en 3,246,320. Igual que cualquiera.")
    print("   - La ventaja NO es ganar mas seguido; es COMPARTIR menos si ganas.")
    print("   - La #6 es independiente de las 5 pruebas. No hay prediccion.")
    print("=" * 62)


if __name__ == "__main__":
    main()
