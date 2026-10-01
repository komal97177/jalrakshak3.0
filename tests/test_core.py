import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core
from app.nlp import NLP, find_place, lang_of

B, _ = core.load_blocks(); M = core.FloodModel(B)


def st(rain=None, river="normal", month=8): return core.compute_state(B, M, None, None, rain=rain, river=river, month=month)


def test_model_loaded(): assert M.ready and M.bundle["metrics"]["loyo_auc"] > 0.8
def test_rain_raises_risk(): assert st(200, "high")["peak"] > st(0)["peak"]
def test_dry_season_green(): assert st(0, "normal", 2)["level"] == "GREEN"
def test_extreme_red(): assert st(200, "high")["level"] == "RED"
def test_alert_levels(): assert core.alert_level(0.1, 0) == "GREEN" and core.alert_level(0.5, 0) == "AMBER" and core.alert_level(0.1, 3) == "RED"
def test_exposure_from_census(): assert max(b["exposure"] for b in B) == 1.0 and sum(b["pop"] for b in B) > 2_000_000
def test_route_avoids_nothing_crashes():
    r = core.route(st(80)["blocks"], 25.39, 87.10, core.load_shelters()); assert r and r["path"] and r["distance_km"] >= 0
def test_dispatch_assigns_unique_units():
    d = core.dispatch(st(200, "high")["blocks"], [], core.load_units()); assert d and len({x["unit"] for x in d}) == len(d)
def test_hotspots_from_records(): assert len(core.hotspots(3)) == 3
def test_nlp():
    n = NLP(); assert n.ready
    assert n.read_report("Naugachhia me ghar me paani ghus gaya, naav chahiye")["flood_prob"] > 0.5
    assert n.read_report("lol my inbox is flooded")["flood_prob"] < 0.5
    assert find_place("सबौर में बाढ़") == "Sabour" and lang_of("kya Sabour safe hai") == "hi"
