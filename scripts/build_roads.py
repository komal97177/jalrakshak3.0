"""OPTIONAL one-off (needs internet): download OpenStreetMap main roads for Bhagalpur and save data/roads.pkl.
The router uses it automatically when present; otherwise it falls back to a block-to-block graph.
Usage: pip install osmnx && python scripts/build_roads.py"""
import pickle
from pathlib import Path
import networkx as nx, osmnx as ox
N, S, E, W = 25.55, 25.05, 87.55, 86.65
G = ox.graph_from_bbox(bbox=(W, S, E, N), custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary"]', simplify=True)
H = nx.Graph()
for n, d in G.nodes(data=True): H.add_node(n, x=d["x"], y=d["y"])
for u, v, d in G.edges(data=True):
    tun = bool(d.get("tunnel")) or "underpass" in str(d.get("name", "")).lower() or "subway" in str(d.get("name", "")).lower()
    if H.has_edge(u, v) and H[u][v]["length"] <= d["length"]: continue
    H.add_edge(u, v, length=float(d["length"]), tunnel=tun)
H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
with open(Path(__file__).resolve().parent.parent / "data" / "roads.pkl", "wb") as f: pickle.dump(H, f)
print("roads saved:", H.number_of_nodes(), "nodes", H.number_of_edges(), "edges")
