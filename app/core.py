"""JalRakshak core logic (no web framework): data, flood-risk model, alerts, routing, dispatch, briefing."""
import csv, math, os, pickle, time
from pathlib import Path
import numpy as np
import joblib, requests
import networkx as nx
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"
DISTRICT = "Bhagalpur"
CENTER = (25.244, 86.972)
RIVER_REF_CMS = float(os.getenv("RIVER_REF_CMS", "50000"))  # discharge treated as index 1.0 (configurable)
RIVER_LEVELS = {"low": 0.15, "normal": 0.40, "high": 0.85}
# Rough Ganga centre-line through Bhagalpur district (lat, lon) - approximate, used as a model feature
GANGA = [(25.28, 86.65), (25.27, 86.74), (25.26, 86.85), (25.27, 86.95), (25.28, 87.03),
         (25.29, 87.12), (25.29, 87.22), (25.32, 87.30), (25.34, 87.42)]


def haversine(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


def dist_ganga_km(lat, lon):
    best, kx = 1e9, 111.0 * math.cos(math.radians(lat))
    for (a1, o1), (a2, o2) in zip(GANGA, GANGA[1:]):
        px, py, x1, y1, x2, y2 = lon * kx, lat * 111, o1 * kx, a1 * 111, o2 * kx, a2 * 111
        dx, dy = x2 - x1, y2 - y1
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


def features(lat, lon, month):
    m = 2 * math.pi * month / 12
    return [math.sin(m), math.cos(m), lat, lon, dist_ganga_km(lat, lon)]


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_blocks():
    blocks = _rows(DATA / "blocks.csv")
    cen_path = DATA / "census_blocks.csv"
    official = cen_path.exists()
    cen = {r["block_id"]: r for r in _rows(cen_path)} if official else {}
    seed = {r["name"]: r for r in _rows(DATA / "census_seed.csv")}
    for b in blocks:
        b["lat"], b["lon"], b["verified"] = float(b["lat"]), float(b["lon"]), int(b["verified"])
        if b["block_id"] in cen:
            r = cen[b["block_id"]]
            b["pop"], b["vuln_share"] = int(float(r["population"])), float(r["vuln_share"])
            b["households"] = int(float(r["households"]))
        else:
            b["pop"], b["vuln_share"], b["households"] = int(seed[b["name"]]["population"]), 0.0, None
    e = np.array([b["pop"] * (1 + b["vuln_share"]) for b in blocks], float)
    for b, v in zip(blocks, e / e.max()):
        b["exposure"] = round(float(v), 4)
    return blocks, ("Census of India 2011 PCA (official file)" if official else
                    "Census of India 2011 block population (seed); run scripts/prep_census.py for the full PCA")


def sigmoid(z): return 1 / (1 + math.exp(-z))
def logit(p): return math.log(p / (1 - p))


def risk_from(s, rain24, rain3, river):
    """Learned seasonal/spatial susceptibility s (from real flood records) adjusted by live drivers.
    The adjustment coefficients are hand-set (documented in the model card), not learned."""
    z = 0.6 * logit(min(max(s, 0.02), 0.98)) - 1.2
    z += min(4.0, max(-1.0, 0.03 * (rain24 - 30))) + 0.02 * min(rain3, 40) + 2.0 * (river - 0.4)
    return sigmoid(z)


def alert_level(peak, verified):
    if peak > 0.75 or verified >= 3: return "RED"
    if peak > 0.45 or verified >= 1: return "AMBER"
    return "GREEN"


class FloodModel:
    def __init__(self, blocks):
        self.blocks, self.bundle, self._cache = blocks, None, {}
        p = MODELS / "flood_rf.joblib"
        if p.exists(): self.bundle = joblib.load(p)

    @property
    def ready(self): return self.bundle is not None

    def susceptibility(self, month):
        if month not in self._cache:
            X = [features(b["lat"], b["lon"], month) for b in self.blocks]
            self._cache[month] = self.bundle["clf"].predict_proba(X)[:, 1]
            self._cache[month + 100] = self.bundle["reg"].predict(X)
        return self._cache[month], self._cache[month + 100]


# ---------------- live data (free, no key) ----------------
_cache = {}
def _cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl: return hit[1]
    try:
        val = fn()
    except Exception:
        val = hit[1] if hit else None
    _cache[key] = (time.time(), val)
    return val


def fetch_weather(blocks):
    def go():
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=12, params={
            "latitude": ",".join(str(b["lat"]) for b in blocks), "longitude": ",".join(str(b["lon"]) for b in blocks),
            "hourly": "precipitation", "past_days": 1, "forecast_days": 2, "timezone": "Asia/Kolkata"})
        r.raise_for_status()
        js = r.json(); js = js if isinstance(js, list) else [js]
        times = js[0]["hourly"]["time"]
        return {"times": times, "rain": [[x or 0.0 for x in j["hourly"]["precipitation"]] for j in js]}
    return _cached("wx", 900, go)


def fetch_river():
    def go():
        r = requests.get("https://flood-api.open-meteo.com/v1/flood", timeout=12, params={
            "latitude": 25.25, "longitude": 87.0, "daily": "river_discharge", "forecast_days": 3})
        r.raise_for_status()
        q = r.json()["daily"]["river_discharge"][0]
        return min(1.0, max(0.0, q / RIVER_REF_CMS))
    return _cached("river", 3600, go)


def _bell(total, n=72, start=24):
    w = np.exp(-0.5 * ((np.arange(24) - 8) / 5.0) ** 2); w /= w.sum()
    a = np.zeros(n); a[start:start + 24] = total * w
    return a


def compute_state(blocks, model, wx, river_live, reports_verified=0, rain=None, river="live", month=None, now_idx=None, cur_month=None):
    """Return per-block risk at 'now' plus a 24h forecast series. rain=None -> live weather."""
    live_ok = wx is not None and rain is None
    n = len(blocks)
    if rain is not None:
        arr = np.array([_bell(rain)] * n); now = 24
    elif wx is not None:
        arr = np.array(wx["rain"]); now = now_idx if now_idx is not None else 24
    else:
        arr = np.zeros((n, 72)); now = 24
    river_idx = RIVER_LEVELS.get(river, river_live if river_live is not None else 0.4) if river != "live" else (river_live if river_live is not None else 0.4)
    month = month or cur_month or time.localtime().tm_mon
    s, frac = model.susceptibility(month)
    series, per_t = [], []
    for t in range(24):
        i = now + t
        r24 = arr[:, max(0, i - 23):i + 1].sum(axis=1); r3 = arr[:, max(0, i - 2):i + 1].sum(axis=1)
        risks = [risk_from(s[k], r24[k], r3[k], river_idx) for k in range(n)]
        series.append({"h": t, "rain": round(float(arr[:, i].mean()), 2), "peak": round(max(risks), 3)})
        per_t.append((risks, r24, r3))
    # live -> map shows "now"; scenario -> map shows the worst hour of the next 24h
    t0 = max(range(24), key=lambda t: series[t]["peak"]) if rain is not None else 0
    risk0, r24_0, r3_0 = per_t[t0]
    out = []
    for k, b in enumerate(blocks):
        r = risk0[k]
        out.append({"id": b["block_id"], "name": b["name"], "name_hi": b["name_hi"], "verified": b["verified"],
                    "lat": b["lat"], "lon": b["lon"], "pop": b["pop"], "exposure": b["exposure"],
                    "risk": round(r, 3), "vuln": round(r * b["exposure"], 3), "rain24": round(float(r24_0[k]), 1),
                    "rain3": round(float(r3_0[k]), 1), "susceptibility": round(float(s[k]), 3),
                    "exp_inundation_pct": round(float(frac[k]) * 100, 1),
                    "level": "red" if r > 0.75 else "amber" if r > 0.45 else "green"})
    peak = max(x["risk"] for x in out)
    return {"blocks": out, "series": series, "peak": peak, "level": alert_level(peak, reports_verified),
            "river_index": round(river_idx, 2), "live_weather": live_ok, "month": month,
            "river_live": river_live is not None, "map_hour": t0}


def hotspots(top=5):
    """Data-driven hotspots: blocks most often >=10% inundated in the real satellite records."""
    import pandas as pd
    d = pd.read_csv(DATA / "inundation_bhagalpur.csv"); d["id"] = d.object_id.str[-5:]
    g = d.assign(f=d.inundation_pct >= 0.10).groupby("id").agg(freq=("f", "mean"), peak=("inundation_pct", "max"))
    blocks, _ = load_blocks(); nm = {b["block_id"]: b["name"] for b in blocks}
    g = g.sort_values("freq", ascending=False).head(top)
    return [{"id": i, "name": nm.get(i, i), "flood_freq": round(float(r.freq), 2), "peak_inundation_pct": round(float(r.peak) * 100, 1)} for i, r in g.iterrows()]


# ---------------- routing ----------------
def load_shelters(): return [dict(r, lat=float(r["lat"]), lon=float(r["lon"]), capacity=int(r["capacity"])) for r in _rows(DATA / "shelters.csv")]
def load_units(): return [dict(r, lat=float(r["lat"]), lon=float(r["lon"]), speed_kmh=float(r["speed_kmh"])) for r in _rows(DATA / "units.csv")]


def _road_graph():
    p = DATA / "roads.pkl"
    if p.exists():
        with open(p, "rb") as f: return pickle.load(f), True
    return None, False


def route(blocks_state, lat, lon, shelters):
    """Flood-aware least-cost route from (lat,lon) to the best shelter. Uses OSM roads if data/roads.pkl exists, else a block-to-block graph."""
    B = blocks_state
    bxy = np.array([[b["lat"], b["lon"]] for b in B]); brisk = np.array([b["risk"] for b in B])
    def near_block(a, o): return int(np.argmin((bxy[:, 0] - a) ** 2 + (bxy[:, 1] - o) ** 2))
    G, osm = _road_graph()
    if osm:
        ids = list(G.nodes); xy = np.array([[G.nodes[i]["y"], G.nodes[i]["x"]] for i in ids])
        nearest = lambda a, o: ids[int(np.argmin((xy[:, 0] - a) ** 2 + (xy[:, 1] - o) ** 2))]
        pos = lambda nid: (G.nodes[nid]["y"], G.nodes[nid]["x"])
        start = nearest(lat, lon); tmp = []
        targets = {s["id"]: nearest(s["lat"], s["lon"]) for s in shelters}
    else:
        G = nx.Graph()
        pts = {("b", b["id"]): (b["lat"], b["lon"]) for b in B}
        pts.update({("s", s["id"]): (s["lat"], s["lon"]) for s in shelters}); pts[("start", 0)] = (lat, lon)
        for k, (a, o) in pts.items(): G.add_node(k, y=a, x=o)
        keys = list(pts)
        for k in keys:
            ds = sorted((haversine(*pts[k], *pts[j]), j) for j in keys if j != k)[:4]
            for d, j in ds: G.add_edge(k, j, length=d * 1000 * 1.35, tunnel=False)
        pos = lambda nid: (G.nodes[nid]["y"], G.nodes[nid]["x"])
        start, targets = ("start", 0), {s["id"]: ("s", s["id"]) for s in shelters}
    nr = {}
    def node_risk(u):
        if u not in nr: nr[u] = float(brisk[near_block(*pos(u))])
        return nr[u]
    def cost(u, v, d):
        r = (node_risk(u) + node_risk(v)) / 2
        c = d["length"] * (1 + 8 * r * r)
        if max(node_risk(u), node_risk(v)) > 0.85: c += 50000
        if d.get("tunnel") and r > 0.3: c += 1500
        return c
    dist, paths = nx.single_source_dijkstra(G, start, weight=cost)
    best = None
    for s in shelters:
        t = targets[s["id"]]
        if t in dist and (best is None or dist[t] < best[0]): best = (dist[t], s, t)
    if not best: return None
    _, sh, t = best; path = paths[t]
    length = sum(G[a][b]["length"] for a, b in zip(path, path[1:]))
    rmax = max(node_risk(u) for u in path)
    return {"shelter": sh, "distance_km": round(length / 1000, 1), "walk_min": int(length / 1000 / 4.5 * 60),
            "max_risk": round(rmax, 2), "underpasses": sum(1 for a, b in zip(path, path[1:]) if G[a][b].get("tunnel")),
            "path": [list(pos(u)) for u in path], "engine": "OpenStreetMap roads" if osm else "block-to-block graph (fallback)",
            "blocked": rmax > 0.85}


# ---------------- dispatch (Hungarian) ----------------
SUIT = {"boat": {"flood"}, "pump": {"flood"}, "ambulance": {"medical", "incident"}, "ndrf": {"flood", "incident", "medical"}}


def dispatch(blocks_state, incidents, units):
    targets = [{"kind": "flood", "name": b["name"], "lat": b["lat"], "lon": b["lon"], "w": b["vuln"] + b["risk"]}
               for b in sorted(blocks_state, key=lambda x: -x["vuln"]) if b["risk"] >= 0.45][:6]
    for i in incidents[:4]:
        kind = "medical" if "medical" in i.get("needs", []) else "incident"
        targets.append({"kind": kind, "name": f"Report: {i['place']}", "lat": i["lat"], "lon": i["lon"], "w": 1.5 + i["score"]})
    if not targets or not units: return []
    C = np.zeros((len(units), len(targets)))
    for a, u in enumerate(units):
        for b, t in enumerate(targets):
            eta = haversine(u["lat"], u["lon"], t["lat"], t["lon"]) * 1.35 / u["speed_kmh"] * 60
            C[a, b] = eta / (0.5 + t["w"]) + (0 if t["kind"] in SUIT.get(u["type"], set()) else 1e4)
    r, c = linear_sum_assignment(C)
    res = []
    for a, b in zip(r, c):
        if C[a, b] >= 1e4: continue
        u, t = units[a], targets[b]
        res.append({"unit": u["name"], "type": u["type"], "target": t["name"], "kind": t["kind"],
                    "eta_min": int(haversine(u["lat"], u["lon"], t["lat"], t["lon"]) * 1.35 / u["speed_kmh"] * 60) + 1})
    return sorted(res, key=lambda x: x["eta_min"])


# ---------------- text: briefing + alert messages (templates fed only by live numbers) ----------------
def briefing(state, verified, lang="en"):
    top = sorted(state["blocks"], key=lambda b: -b["vuln"])[:3]
    peak_h = max(state["series"], key=lambda s: s["peak"])
    if lang == "hi":
        names = ", ".join(f"{b['name_hi']} ({int(b['risk']*100)}%)" for b in top)
        return (f"भागलपुर अलर्ट स्तर: {state['level']}. सबसे अधिक जोखिम×आबादी वाले ब्लॉक: {names}. "
                f"अगले 24 घंटों में अधिकतम जोखिम {int(peak_h['peak']*100)}% (लगभग {peak_h['h']} घंटे बाद). सत्यापित रिपोर्टें: {verified}.")
    names = ", ".join(f"{b['name']} ({int(b['risk']*100)}%)" for b in top)
    return (f"Bhagalpur alert level: {state['level']}. Highest risk x population: {names}. "
            f"Peak district risk in the next 24h is {int(peak_h['peak']*100)}% (in about {peak_h['h']}h). Verified reports (30 min): {verified}.")


def alert_text(level, state, lang="en"):
    top = [b for b in sorted(state["blocks"], key=lambda b: -b["risk"]) if b["risk"] > 0.45][:4] or sorted(state["blocks"], key=lambda b: -b["risk"])[:2]
    if lang == "hi":
        return (f"JalRakshak बाढ़ अलर्ट: {level}\nजोखिम वाले ब्लॉक: " + ", ".join(b["name_hi"] for b in top) +
                "\nनिचले इलाकों से दूर रहें, बिजली के उपकरण बंद करें, ऊँची जगह/राहत शिविर जाएँ। आपात: 112।")
    return (f"JalRakshak flood alert: {level}\nBlocks at risk: " + ", ".join(b["name"] for b in top) +
            "\nAvoid low-lying areas, switch off electrical mains, move to higher ground or a relief camp. Emergency: 112.")
