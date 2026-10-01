"""Turn the official Census 2011 CD-block PCA file for Bhagalpur into data/census_blocks.csv and verify block names.
Download: censusindia.gov.in -> Primary Census Abstract -> CD Block wise -> Bihar -> Bhagalpur (file PCA_CDB_1022_F_Census.xls).
Usage: python scripts/prep_census.py PCA_CDB_1022_F_Census.xls   (.xls needs `pip install xlrd`; or convert to .xlsx in Excel)
Options: --out DIR (default data/) for testing; --district CODE (default 224)"""
import argparse, csv, difflib, sys
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
ap.add_argument("--district", type=int, default=224); a = ap.parse_args()
OFFSET = 1012   # inundation block id = census CD Block_Code + 1012 (checked: Bihpur 01330, Bhagwanpur/Kaimur 01455)
ALIAS = {"colgong": "Kahalgaon", "sonhaula": "Sanhaula", "naugachia": "Naugachhia", "rangrachowk": "Rangra Chowk", "pirpainty": "Pirpainti"}
x = pd.read_excel(a.file)
t = x[(x.Level == "CD BLOCK") & (x["Total/Rural/Urban"] == "Total") & (x.District_Code == a.district)].copy()
if t.empty: sys.exit(f"No CD-block rows for district code {a.district} in this file. Is it the Bhagalpur file?")
ag = [c for c in x.columns if "Agricultural Labourers" in c and "Person" in c]
pop = t["Total Population Person"].astype(float)
t["vuln"] = (t["Population in the age group 0-6 Person"] / pop + t["Illiterate Persons"] / pop + (t[ag[0]] / pop if ag else 0)) / (3 if ag else 2)
root = Path(a.out); blocks = list(csv.DictReader(open(root / "blocks.csv", encoding="utf-8"))); names = {b["name"]: b for b in blocks}
rows = []
for _, r in t.iterrows():
    bid = f"{int(r['CD Block_Code']) + OFFSET:05d}"; nm = str(r["Name"]).strip()
    match = ALIAS.get(nm.lower().replace(" ", "")) or (difflib.get_close_matches(nm, list(names), 1, 0.8) or [None])[0]
    rows.append({"block_id": bid, "name": match or nm, "households": int(r["No of Households"]), "population": int(r["Total Population Person"]),
                 "pop_0_6": int(r["Population in the age group 0-6 Person"]), "illiterate": int(r["Illiterate Persons"]),
                 "ag_labourers": int(r[ag[0]]) if ag else 0, "vuln_share": round(float(r["vuln"]), 4)})
with open(root / "census_blocks.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# authoritative id -> name mapping (keeps our coordinates, which are keyed by name)
geo = {b["name"]: b for b in blocks}; new, miss = [], []
for r in rows:
    if r["name"] in geo: g = geo[r["name"]]; new.append({**g, "block_id": r["block_id"], "verified": 1})
    else: miss.append(r["name"])
if new and not miss:
    with open(root / "blocks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new[0])); w.writeheader(); w.writerows(sorted(new, key=lambda b: b["block_id"]))
    print(f"OK: {len(rows)} blocks written to census_blocks.csv; block names verified from census. Now run: python scripts/train.py")
else:
    print("census_blocks.csv written, but blocks.csv NOT changed. Unmatched names:", miss)
