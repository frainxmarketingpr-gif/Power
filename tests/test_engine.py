"""Pruebas unitarias (pytest) del motor estadistico."""
import itertools
from math import comb

import numpy as np
import pytest

import powerball_simulator as base
import powerball_advanced as adv
from pb_engine.config import ScoreWeights, Rules, Settings


# --------------------------- Combinatoria / reglas ---------------------------
def test_jackpot_odds_exact():
    r = Rules()
    assert r.white_combos == comb(69, 5) == 11_238_513
    assert r.jackpot_odds == 292_201_338


def test_match5_constant():
    # C(69,5) * 26/25 = 11.688.053,52 -> odds del premio "5 sin PB" (se trunca)
    assert int(comb(69, 5) * 26 / 25) == base.MATCH5_ODDS == 11_688_053


# --------------------------- Validacion de pesos -----------------------------
def test_weights_sum_to_one():
    w = ScoreWeights()
    assert abs(sum(w.model_dump().values()) - 1.0) < 1e-9


def test_weights_reject_bad_sum():
    with pytest.raises(Exception):
        ScoreWeights(frecuencia=0.9)


def test_settings_reject_low_mc():
    with pytest.raises(Exception):
        Settings(mc_iters=1000)


# --------------------------- Filtros de patrones -----------------------------
def test_filter_flags_popular_patterns():
    band = (120, 232)
    C = np.array([
        [1, 2, 3, 4, 5],        # secuencia consecutiva
        [1, 11, 21, 31, 41],    # progresion aritmetica
        [2, 4, 6, 8, 10],       # todo par + todo bajo
        [1, 5, 10, 20, 31],     # todo <=31 (cumpleanos)
    ])
    mask = adv.undesirable_mask(C, band)
    assert mask.all()


def test_filter_allows_balanced_combo():
    band = (120, 232)
    C = np.array([[3, 23, 36, 53, 63]])   # balanceada, suma 178
    assert not adv.undesirable_mask(C, band)[0]


# --------------------------- Numba Monte Carlo -------------------------------
def test_numba_mc_sum_density_shape_and_peak():
    from pb_engine import engines
    w = np.ones(69) / 69
    dens = engines.numba_monte_carlo(w, 300_000, 5, seed=1)
    assert dens.shape[0] == 5 * 69 + 1
    assert dens.max() == pytest.approx(1.0)
    # el pico debe estar cerca de la media teorica (5*35=175)
    assert 150 <= int(np.argmax(dens)) <= 200


# --------------------------- Determinismo del score --------------------------
@pytest.fixture(scope="module")
def model():
    p2016 = "/root/.claude/uploads/a4b780aa-d8bd-585a-acd0-c3b7c62fa30c/5fc9dfb0-powerball_results_20160601_to_20260601.xlsx"
    p2010 = "/root/.claude/uploads/a4b780aa-d8bd-585a-acd0-c3b7c62fa30c/81ca6cba-powerball_resultados.xlsx"
    raw = base.load_raw(p2016, p2010)
    clean, _ = base.clean_and_validate(raw)
    cur = base.current_era(clean)
    a = base.analyze(cur)
    return adv.EnsembleModel(cur, a, mc_iters=200_000, boot_reps=200), a


def test_score_deterministic(model):
    m, a = model
    C = np.array([[3, 23, 36, 53, 63]])
    s1 = m.score_batch(C)[0]
    s2 = m.score_batch(C)[0]
    assert s1 == s2
    assert 0 <= s1 <= 100


def test_features_in_unit_range(model):
    m, a = model
    C = np.array(list(itertools.islice(
        itertools.combinations(range(1, 70), 5), 5000)), dtype=np.int16)
    F = m.feature_batch(C)
    assert F.shape[1] == 8
    assert (F >= -1e-9).all() and (F <= 1 + 1e-9).all()
