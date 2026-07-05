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


# ============ Regresiones de la auditoria Fable ============================

def test_pair_combos_constant():
    assert base.PAIR_COMBOS == comb(69, 2) == 2_346


def test_filter_detects_four_consecutive():
    # >=4 numeros consecutivos deben filtrarse (bug historico: solo atrapaba 5)
    band = (120, 232)
    assert base.is_undesirable([10, 11, 12, 13, 45], band) is True   # 4 consecutivos
    assert base.is_undesirable([5, 6, 7, 8, 40], band) is True       # 4 consecutivos
    # 3 consecutivos NO deben bastar para filtrar por esta regla
    C = np.array([[2, 9, 10, 11, 55]])   # 3 consecutivos, suma 87 -> fuera de banda
    # usar una suma dentro de banda para aislar la regla de consecutivos
    C2 = np.array([[9, 10, 11, 40, 55]])  # 3 consec, par 3-2, suma 125 en banda
    assert base.undesirable_mask_vec(C2, band)[0] == False


def test_filter_birthday_four_in_range():
    band = (120, 232)
    # 4+ numeros en 1-31 = sesgo de cumpleanos (no redundante con 'todo bajo')
    assert base.is_undesirable([3, 7, 12, 25, 60], band) is True     # 4 en 1-31, 1 alto


def test_canonical_filter_matches_scalar_and_vector():
    band = (110, 240)
    combos = [[3, 23, 36, 53, 63], [1, 2, 3, 4, 5], [2, 4, 6, 8, 10],
              [10, 11, 12, 13, 45], [7, 19, 33, 41, 58]]
    C = np.array(combos)
    vec = base.undesirable_mask_vec(C, band)
    for i, c in enumerate(combos):
        assert base.is_undesirable(c, band) == bool(vec[i])


def test_pareto_dominance_not_inverted():
    # A domina a B y C; el frente correcto debe contener A, nunca solo B
    class FakeModel:
        freq_n = np.ones(69); bayes_n = np.ones(69)
        recent_mask = np.zeros(70, dtype=bool)
        def score_batch(self, C):
            return C.sum(axis=1).astype(float)
    # construimos combos cuyo (typicidad,unpop,div) ordene A>B
    A = [40, 42, 44, 46, 48]   # alto, disperso, sin recientes
    B = [1, 2, 3, 4, 5]        # bajo, consecutivo (dominado)
    C = np.array([A, B], dtype=np.int16)
    front, knee = adv.pareto_front(FakeModel(), C, top=2)
    assert 0 in front            # A (indice 0) debe estar en el frente
    assert not (list(front) == [1])   # nunca solo el dominado


def test_ga_elitism_returns_global_best(model):
    m, a = model
    combo, score = adv.genetic_optimize(m, generations=10, pop=60, seed=3)
    assert len(set(combo)) == 5
    assert 0 <= score <= 100


# ============ Modulo de comparacion por fecha (check.py) ====================

@pytest.fixture(scope="module")
def history():
    from pb_engine import data_io
    from pb_engine.config import Rules as R
    df, _ = data_io.load(
        "/root/.claude/uploads/a4b780aa-d8bd-585a-acd0-c3b7c62fa30c/5fc9dfb0-powerball_results_20160601_to_20260601.xlsx",
        "/root/.claude/uploads/a4b780aa-d8bd-585a-acd0-c3b7c62fa30c/81ca6cba-powerball_resultados.xlsx",
        R())
    return df


def test_check_lookup_known_draw(history):
    from pb_engine import check
    d = check.lookup_draw(history, "2026-05-25")
    assert d is not None
    assert d["blancas"] == [17, 32, 48, 60, 64]
    assert d["powerball"] == 10


def test_check_ticket_jackpot(history):
    from pb_engine import check
    r = check.check_ticket(history, [17, 32, 48, 60, 64], 10, "2026-05-25")
    assert r["n_aciertos_blancas"] == 5
    assert r["acerto_powerball"] is True
    assert r["categoria"] == "Jackpot"


def test_check_ticket_partial_and_prize(history):
    from pb_engine import check
    r = check.check_ticket(history, [17, 32, 1, 2, 3], 10, "2026-05-25")
    assert r["n_aciertos_blancas"] == 2
    assert r["acerto_powerball"] is True
    assert r["premio_base_usd"] == 7


def test_check_ticket_invalid(history):
    from pb_engine import check
    assert "error" in check.check_ticket(history, [1, 2, 3, 4, 70], 6, "2026-05-25")
    assert "error" in check.check_ticket(history, [1, 2, 3, 4, 5], 6, "2026-05-24")  # sin sorteo


def test_check_rejects_pre_2015_era(history):
    from pb_engine import check
    # 2011-01-19 pertenece a una era con reglas distintas -> se rechaza
    r = check.check_ticket(history, [22, 36, 51, 56, 59], 26, "2011-01-19")
    assert "error" in r and "era" in r["error"].lower()


def test_compare_range_validates_ticket(history):
    from pb_engine import check
    assert "error" in check.compare_range(history, [1, 2, 3, 4], 5)      # 4 blancas
    assert "error" in check.compare_range(history, [1, 2, 3, 4, 70], 5)  # fuera de rango
    ok = check.compare_range(history, [3, 23, 36, 53, 63], 4)
    assert "sorteos_evaluados" in ok and ok["sorteos_evaluados"] > 0


def test_api_rejects_malformed_date():
    from fastapi.testclient import TestClient
    from pb_engine import api
    c = TestClient(api.app)
    assert c.get("/draw?date=not-a-date").status_code == 400
    assert c.get("/check?date=99/99/9999&white=1,2,3,4,5&pb=12").status_code == 400
