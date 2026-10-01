import json, logging, os, re, subprocess, sys, threading, datetime as dt
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import core, auth, mailer, db
from .nlp import NLP, find_place, lang_of

ROOT = core.ROOT
app = FastAPI(title="JalRakshak - Bhagalpur flood early warning", version="1.0")
BLOCKS, CENSUS_SRC = core.load_blocks()
BYNAME = {b["name"]: b for b in BLOCKS}
SHELTERS, UNITS = core.load_shelters(), core.load_units()
MODEL, NL = None, None
SAMPLES = ["Naugachhia me ghar me paani ghus gaya hai, naav chahiye", "Bihpur में बाँध टूट गया, गाँव में बाढ़, लोग फंसे हैं, बचाओ",
           "Road water logging near Kahalgaon, vehicles stuck", "Sabour mein baadh ka paani gali mein aa gaya, ek marij hai ambulance bhejo",
           "lol my phone is flooded with notifications", "Pirpainti me kamar tak paani, pump chahiye", "Sultanganj ghat pe Ganga ka paani tezi se badh raha hai",
           "Cricket match in Nathnagar today"]
_sample_i = [0]


@app.on_event("startup")
def startup():
    global MODEL, NL
    db.init()
    for script, f in (("train.py", "flood_rf.joblib"), ("train_nlp.py", "nlp.joblib")):
        if not (core.MODELS / f).exists(): subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=False)
    MODEL, NL = core.FloodModel(BLOCKS), NLP()


def session():
    s = db.Session()
    try: yield s
    finally: s.close()


def audit(s, actor, action):
    s.add(db.Audit(actor=(actor or "anon")[:12], action=action)); s.commit()


def ip(req): return (req.headers.get("x-forwarded-for") or req.client.host or "?").split(",")[0].strip()


def cur_user(req: Request, s=Depends(session)):
    d = auth.read_token(req.cookies.get("jr_session", ""))
    if not d: return None
    u = s.query(db.User).filter_by(email_hash=d["u"]).first()
    if u: u.is_admin = bool(d.get("a"))
    return u


def need_admin(u=Depends(cur_user)):
    if not u or not getattr(u, "is_admin", False): raise HTTPException(403, "Admin sign-in required")
    return u


# ---------------- state ----------------
def ist_idx(wx):
    if not wx: return None
    ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%dT%H:00")
    return wx["times"].index(ist) if ist in wx["times"] else 24


def verified_live(s):
    cut = db.now() - dt.timedelta(minutes=30)
    return s.query(db.Report).filter(db.Report.status == "Verified", db.Report.scenario == False, db.Report.created_at >= cut).count()  # noqa: E712


def build_state(s, rain=None, river="live", scen=False):
    wx = core.fetch_weather(BLOCKS); rv = core.fetch_river()
    m = dt.datetime.now().month
    month = (m if 6 <= m <= 10 else 8) if scen else m
    st = core.compute_state(BLOCKS, MODEL, wx, rv, verified_live(s), rain=rain, river=river, month=month, now_idx=ist_idx(wx))
    st["mode"] = "scenario" if rain is not None or river != "live" else "live"
    return st


class Scn(BaseModel):
    rain: float | None = None
    river: str = "live"
    scen: bool = False


@app.get("/api/state")
def state(rain: float | None = None, river: str = "live", scen: bool = False, s=Depends(session)):
    st = build_state(s, rain, river, scen or rain is not None)
    if st["mode"] == "live":
        last = s.query(db.Snapshot).order_by(db.Snapshot.id.desc()).first()
        if not last or (db.now() - last.ts).seconds > 1800:
            r = [b["rain24"] for b in st["blocks"]]
            s.add(db.Snapshot(rain24_mean=sum(r) / len(r), rain24_max=max(r), river_index=st["river_index"], peak_risk=st["peak"], level=st["level"])); s.commit()
    st["verified_reports"] = verified_live(s) if st["mode"] == "live" else 0
    st["census_source"] = CENSUS_SRC; st["hotspots"] = core.hotspots(); st["shelters"] = SHELTERS
    st["units"] = UNITS; st["district"] = core.DISTRICT
    return st


@app.get("/api/briefing")
def briefing(lang: str = "en", rain: float | None = None, river: str = "live", s=Depends(session)):
    st = build_state(s, rain, river, rain is not None)
    return {"text": core.briefing(st, verified_live(s), lang)}


# ---------------- reports ----------------
class ReportIn(BaseModel):
    text: str
    lat: float | None = None
    lon: float | None = None
    scenario: bool = False
    rain: float | None = None
    river: str = "live"


def submit_report(s, r: ReportIn):
    text = r.text.strip()[:500]
    if len(text) < 4: raise HTTPException(400, "Report too short")
    rd = NL.read_report(text)
    blk = BYNAME.get(rd["place"]) if rd["place"] else None
    if r.lat is not None and r.lon is not None:
        near = min(BLOCKS, key=lambda b: core.haversine(r.lat, r.lon, b["lat"], b["lon"])); lat, lon = r.lat, r.lon
        blk = blk or near
    elif blk: lat, lon = blk["lat"], blk["lon"]
    else: lat, lon = core.CENTER; blk = BLOCKS[0]; rd["place"] = None
    st = build_state(s, r.rain if r.scenario else None, r.river if r.scenario else "live", r.scenario)
    bs = next(b for b in st["blocks"] if b["id"] == blk["block_id"])
    sevw = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(rd["severity"], 0)
    raining = 1.0 if bs["rain24"] >= 10 or bs["rain3"] >= 1 else 0.0
    score = 0.4 * rd["flood_prob"] + 0.35 * bs["risk"] + 0.15 * raining + 0.10 * sevw
    cut = db.now() - dt.timedelta(minutes=15)
    for old in s.query(db.Report).filter(db.Report.created_at >= cut, db.Report.status != "Rejected", db.Report.scenario == r.scenario):
        if core.haversine(lat, lon, old.lat, old.lon) <= 0.2 and rd["flood_prob"] >= 0.5:
            old.dups += 1; old.score = min(1.0, old.score + 0.08); old.status = status_of(old.score); s.commit()
            return report_dict(old, merged=True, risk=bs["risk"])
    status = "Rejected" if rd["flood_prob"] < 0.5 else status_of(score)
    row = db.Report(text=text, place=rd["place"] or blk["name"], lat=lat, lon=lon, flood_prob=rd["flood_prob"], severity=rd["severity"],
                    needs=",".join(rd["needs"]), lang=rd["lang"], score=round(score, 3), status=status, scenario=r.scenario)
    s.add(row); s.commit()
    return report_dict(row, risk=bs["risk"])


def status_of(sc): return "Verified" if sc >= 0.75 else "Probable" if sc >= 0.5 else "Unverified" if sc >= 0.3 else "Rejected"


def report_dict(r, merged=False, risk=None):
    return {"id": r.id, "text": r.text, "place": r.place, "lat": r.lat, "lon": r.lon, "flood_prob": r.flood_prob, "severity": r.severity,
            "needs": [n for n in (r.needs or "").split(",") if n], "lang": r.lang, "score": r.score, "status": r.status, "dups": r.dups,
            "scenario": r.scenario, "merged": merged, "model_risk": risk, "at": r.created_at.isoformat() + "Z"}


@app.post("/api/reports")
def post_report(r: ReportIn, req: Request, s=Depends(session)):
    if auth.limited("rep" + ip(req), 15, 60): raise HTTPException(429, "Too many reports, slow down")
    return submit_report(s, r)


@app.post("/api/reports/sample")
def sample(scn: Scn, s=Depends(session)):
    t = SAMPLES[_sample_i[0] % len(SAMPLES)]; _sample_i[0] += 1
    return submit_report(s, ReportIn(text=t, scenario=scn.scen, rain=scn.rain, river=scn.river))


@app.get("/api/reports")
def list_reports(scenario: bool = False, s=Depends(session)):
    q = s.query(db.Report).filter(db.Report.scenario == scenario).order_by(db.Report.id.desc()).limit(40)
    return [report_dict(r) for r in q]


@app.delete("/api/reports")
def reset_reports(u=Depends(need_admin), s=Depends(session)):
    s.query(db.Report).delete(); s.commit(); audit(s, u.email_hash, "reset_reports"); return {"ok": True}


# ---------------- route / dispatch ----------------
@app.get("/api/route")
def get_route(lat: float, lon: float, rain: float | None = None, river: str = "live", s=Depends(session)):
    if not (24.6 <= lat <= 25.8 and 86.3 <= lon <= 87.8): raise HTTPException(400, "Start point must be inside Bhagalpur district area")
    st = build_state(s, rain, river, rain is not None)
    r = core.route(st["blocks"], lat, lon, SHELTERS)
    if not r: raise HTTPException(404, "No route found")
    return r


@app.post("/api/dispatch")
def do_dispatch(scn: Scn, u=Depends(need_admin), s=Depends(session)):
    st = build_state(s, scn.rain, scn.river, scn.scen or scn.rain is not None)
    rows = s.query(db.Report).filter(db.Report.status.in_(["Verified", "Probable"]), db.Report.scenario == (st["mode"] == "scenario")).order_by(db.Report.id.desc()).limit(6)
    inc = [{"place": r.place, "lat": r.lat, "lon": r.lon, "score": r.score, "needs": [n for n in r.needs.split(",") if n]} for r in rows]
    audit(s, u.email_hash, "dispatch"); return core.dispatch(st["blocks"], inc, UNITS)


# ---------------- alerts (email) ----------------
def run_alert(s, force=False, st=None):
    st = st or build_state(s)
    level = st["level"]
    if level == "GREEN" and not force: return {"sent": 0, "level": level, "note": "GREEN - nothing to send"}
    users = s.query(db.User).filter_by(subscribed=True).all(); msgs, tgt = [], []
    bm = {b["id"]: b for b in st["blocks"]}
    for u in users:
        if not force and u.last_alert_at and (db.now() - u.last_alert_at).total_seconds() < 6 * 3600: continue
        if level == "AMBER" and not force and u.block_id in bm and bm[u.block_id]["risk"] <= 0.45: continue
        msgs.append((auth.dec(u.email_enc), f"JalRakshak {level} flood alert - Bhagalpur", core.alert_text(level, st, u.lang or "en"))); tgt.append(u)
    sent = mailer.send_many(msgs) if msgs else 0
    for u in tgt: u.last_alert_at = db.now()
    s.add(db.Alert(level=level, message=core.alert_text(level, st, "en"), recipients=len(msgs), mode="email" if mailer.configured() else "dry-run")); s.commit()
    return {"sent": sent, "recipients": len(msgs), "level": level, "mode": "email" if mailer.configured() else "dry-run (SMTP not configured)"}


@app.post("/api/alerts/send")
def send_alert(u=Depends(need_admin), s=Depends(session)):
    audit(s, u.email_hash, "send_alert"); return run_alert(s, force=True)


@app.get("/api/alerts")
def alerts(s=Depends(session)):
    return [{"level": a.level, "message": a.message, "recipients": a.recipients, "mode": a.mode, "at": a.created_at.isoformat() + "Z"}
            for a in s.query(db.Alert).order_by(db.Alert.id.desc()).limit(15)]


@app.get("/api/cron/tick")
def cron(key: str, s=Depends(session)):
    if not os.getenv("CRON_KEY") or key != os.getenv("CRON_KEY"): raise HTTPException(403, "bad key")
    st = build_state(s); last = s.query(db.Alert).order_by(db.Alert.id.desc()).first()
    rank = {"GREEN": 0, "AMBER": 1, "RED": 2}
    due = st["level"] != "GREEN" and (not last or rank[st["level"]] > rank[last.level] or (db.now() - last.created_at).total_seconds() > 6 * 3600)
    return run_alert(s, st=st) if due else {"sent": 0, "level": st["level"], "note": "no new alert due"}


# ---------------- chat (own intent model, live facts only) ----------------
class ChatIn(BaseModel):
    message: str
    lat: float | None = None
    lon: float | None = None
    rain: float | None = None
    river: str = "live"


@app.post("/api/chat")
def chat(c: ChatIn, req: Request, s=Depends(session)):
    if auth.limited("chat" + ip(req), 20, 60): raise HTTPException(429, "Max 20 messages per minute")
    lang = lang_of(c.message); hi = lang == "hi"
    intent, conf = NL.intent(c.message)
    st = build_state(s, c.rain, c.river, c.rain is not None)
    place = find_place(c.message); B = {b["name"]: b for b in st["blocks"]}
    if not place and c.lat is not None:        # "my area": nearest block to the chosen start point / GPS
        place = min(st["blocks"], key=lambda b: core.haversine(c.lat, c.lon, b["lat"], b["lon"]))["name"]
    if place in B and (conf < 0.6 or intent in ("other", "alert", "forecast")) and re.search(r"safe|danger|risk|khatar|khatra|खतर|सुरक्षित|baadh|बाढ़|flood", c.message, re.I):
        intent, conf = "risk", max(conf, 0.9)    # a named place + danger words = a risk question
    help_ = "आपात स्थिति में 112 या जिला नियंत्रण कक्ष 1077 पर कॉल करें।" if hi else "In an emergency call 112 or the district control room 1077."
    if conf < 0.45 or intent == "other":
        txt = ("मेरे पास इसकी जानकारी नहीं है। " if hi else "I don't have facts on that. ") + help_
    elif intent == "alert":
        txt = f"भागलपुर का अलर्ट स्तर अभी {st['level']} है (अधिकतम जोखिम {int(st['peak']*100)}%)।" if hi else f"Bhagalpur alert level is {st['level']} (peak risk {int(st['peak']*100)}%)."
    elif intent == "risk":
        if place in B:
            b = B[place]; pc = int(b["risk"] * 100)
            VERDICT = {"green": ("🟢 सुरक्षित (अभी)", "🟢 SAFE for now", "अभी कोई तत्काल खतरा नहीं, पर अलर्ट पर नज़र रखें।", "No immediate danger, but keep watching alerts."),
                       "amber": ("🟠 सावधान रहें", "🟠 CAUTION - flooding possible", "तैयार रहें: ज़रूरी सामान, दवाइयाँ और दस्तावेज़ ऊँची जगह रखें।", "Be ready: keep documents, medicines and essentials on high ground."),
                       "red": ("🔴 खतरनाक", "🔴 DANGEROUS - flooding likely", "निचले इलाके से तुरंत निकलें और राहत शिविर जाएँ।", "Leave low-lying areas now and move to a relief camp.")}[b["level"]]
            extra = ""
            if b["level"] != "green":
                r = core.route(st["blocks"], b["lat"], b["lon"], SHELTERS)
                if r: extra = (f" नजदीकी शिविर: {r['shelter']['name']} ({r['distance_km']} किमी)।" if hi else f" Nearest shelter: {r['shelter']['name']} ({r['distance_km']} km).")
            if hi: txt = f"{b['name_hi']}: {VERDICT[0]} — बाढ़ जोखिम {pc}%. {VERDICT[2]}{extra} (24 घंटे वर्षा {b['rain24']} मिमी, गंगा स्तर सूचकांक {st['river_index']})"
            else: txt = f"{place}: {VERDICT[1]} - flood risk {pc}%. {VERDICT[3]}{extra} (24h rain {b['rain24']} mm, Ganga level index {st['river_index']})"
        else:
            top = sorted(st["blocks"], key=lambda b: -b["risk"])[:3]; nm = ", ".join((b["name_hi"] if hi else b["name"]) + f" {int(b['risk']*100)}%" for b in top)
            txt = ("किसी ब्लॉक का नाम लिखें (जैसे 'क्या सबौर सुरक्षित है?')। अभी सबसे अधिक जोखिम: " if hi else "Name a block (e.g. 'is Sabour safe?'). Highest risk right now: ") + nm
    elif intent == "shelter":
        lat, lon = (c.lat, c.lon) if c.lat is not None else ((B[place]["lat"], B[place]["lon"]) if place in B else core.CENTER)
        r = core.route(st["blocks"], lat, lon, SHELTERS)
        txt = (f"सबसे नजदीकी राहत शिविर: {r['shelter']['name']}, {r['distance_km']} किमी, पैदल ~{r['walk_min']} मिनट (अधिकतम जोखिम {int(r['max_risk']*100)}%)। (डेटा प्लेसहोल्डर है)" if hi else
               f"Nearest shelter: {r['shelter']['name']}, {r['distance_km']} km, ~{r['walk_min']} min walk (max route risk {int(r['max_risk']*100)}%). (Shelter list is placeholder data)") if r else help_
    elif intent == "forecast":
        pk = max(st["series"], key=lambda x: x["rain"]); tot = round(sum(x["rain"] for x in st["series"]), 1)
        txt = f"अगले 24 घंटे में औसतन {tot} मिमी बारिश; सबसे तेज़ लगभग {pk['h']} घंटे बाद।" if hi else f"About {tot} mm of rain expected in the next 24 h, heaviest in ~{pk['h']} h."
    elif intent == "tips":
        txt = ("ऊँची जगह जाएँ, बिजली बंद करें, बाढ़ के पानी में न चलें, दस्तावेज़ और दवाइयाँ साथ रखें, पीने का पानी उबालें। " if hi else
               "Move to higher ground, switch off mains, never walk or drive through flood water, carry documents and medicines, boil drinking water. ") + help_
    else: txt = help_ + (" NDMA: 1078." if not hi else " NDMA: 1078।")
    return {"reply": txt, "intent": intent, "confidence": round(conf, 2), "lang": lang}


# ---------------- auth ----------------
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


class ReqIn(BaseModel):
    email: str
    mode: str = "signin"
    name: str | None = None
    consent: bool = False


@app.post("/api/auth/request")
def auth_request(r: ReqIn, req: Request, s=Depends(session)):
    e = r.email.lower().strip()
    if not EMAIL_RE.match(e): raise HTTPException(400, "Invalid email")
    if auth.limited("otp" + auth.h(e), 5, 3600) or auth.limited("otpip" + ip(req), 20, 3600): raise HTTPException(429, "Too many codes requested. Try later.")
    eh = auth.h(e); exists = s.query(db.User).filter_by(email_hash=eh).first() is not None
    if r.mode == "signup" and not exists:
        if not (r.name and r.name.strip()) or not r.consent: raise HTTPException(400, "Name and consent are required to sign up")
    resp = {"ok": True, "message": "If the address is valid, a 6-digit code was emailed. It expires in 10 minutes."}
    if not exists and r.mode != "signup": return resp            # no account enumeration
    code = auth.new_code(); s.query(db.Otp).filter_by(email_hash=eh).delete()
    s.add(db.Otp(email_hash=eh, code_hash=auth.code_hash(e, code), expires=db.now() + dt.timedelta(minutes=10), pending_name=(r.name or "")[:80] or None)); s.commit()
    body = f"Your JalRakshak sign-in code is {code}\nIt expires in 10 minutes. If you did not request it, ignore this email. We never ask for a password."
    if mailer.configured(): mailer.send_many([(e, "Your JalRakshak sign-in code", body)])
    elif os.getenv("DEV_SHOW_OTP") == "1": resp["dev_code"] = code          # demo only: no SMTP configured
    else: logging.getLogger("jalrakshak").warning("SMTP not configured; OTP not delivered")
    return resp


class VerIn(BaseModel):
    email: str
    code: str


@app.post("/api/auth/verify")
def auth_verify(v: VerIn, response: Response, s=Depends(session)):
    e = v.email.lower().strip(); eh = auth.h(e); o = s.query(db.Otp).filter_by(email_hash=eh).first()
    if not o or o.expires < db.now() or o.attempts >= 5: raise HTTPException(400, "Code expired or too many attempts. Request a new one.")
    o.attempts += 1; s.commit()
    if not auth.hmac.compare_digest(o.code_hash, auth.code_hash(e, v.code.strip())): raise HTTPException(400, "Wrong code")
    u = s.query(db.User).filter_by(email_hash=eh).first()
    if not u:
        u = db.User(email_hash=eh, email_enc=auth.enc(e), name=o.pending_name or e.split("@")[0]); s.add(u)
    s.delete(o); s.commit(); audit(s, eh, "login")
    admin = e in auth.ADMINS
    response.set_cookie("jr_session", auth.make_token(eh, admin), max_age=auth.SESSION_SECS, httponly=True, samesite="lax",
                        secure=os.getenv("COOKIE_SECURE", "1") == "1")
    return profile(u, admin)


def profile(u, admin=False):
    return {"name": u.name, "block_id": u.block_id, "lang": u.lang, "subscribed": u.subscribed, "admin": admin,
            "created_at": u.created_at.isoformat() + "Z", "email_masked": auth.dec(u.email_enc)[:2] + "***"}


@app.get("/api/me")
def me(u=Depends(cur_user)):
    if not u: raise HTTPException(401, "Not signed in")
    return profile(u, getattr(u, "is_admin", False))


class MeIn(BaseModel):
    name: str | None = None
    block_id: str | None = None
    lang: str | None = None
    subscribed: bool | None = None


@app.put("/api/me")
def update_me(m: MeIn, u=Depends(cur_user), s=Depends(session)):
    if not u: raise HTTPException(401, "Not signed in")
    u = s.merge(u)
    if m.name is not None: u.name = m.name.strip()[:80]
    if m.block_id is not None and any(b["block_id"] == m.block_id for b in BLOCKS): u.block_id = m.block_id
    if m.lang in ("en", "hi"): u.lang = m.lang
    if m.subscribed is not None: u.subscribed = m.subscribed
    s.commit(); return profile(u, getattr(u, "is_admin", False))


@app.get("/api/me/export")
def export_me(u=Depends(cur_user)):
    if not u: raise HTTPException(401, "Not signed in")
    return {"stored_about_you": {"name": u.name, "email": auth.dec(u.email_enc), "block_id": u.block_id, "lang": u.lang, "subscribed": u.subscribed,
                                 "consent_at": u.consent_at.isoformat(), "created_at": u.created_at.isoformat()}}


@app.delete("/api/me")
def delete_me(response: Response, u=Depends(cur_user), s=Depends(session)):
    if not u: raise HTTPException(401, "Not signed in")
    s.query(db.User).filter_by(id=u.id).delete(); s.commit(); audit(s, "deleted", "account_erased"); response.delete_cookie("jr_session"); return {"ok": True}


@app.post("/api/auth/logout")
def logout(response: Response): response.delete_cookie("jr_session"); return {"ok": True}


# ---------------- meta ----------------
@app.get("/api/model-card")
def model_card():
    f = MODEL.bundle["metrics"] if MODEL.ready else {}
    n = NL.b["metrics"] if NL.ready else {}
    return {"flood_model": f, "nlp_models": n, "census": CENSUS_SRC,
            "data_sources": {"flood_labels": "Satellite-derived block-level flood inundation records, Bhagalpur (16 blocks), 22 months in 2021-2025 (uploaded dataset)",
                             "population": CENSUS_SRC, "rain": "Open-Meteo forecast (16 block points)", "river": "Open-Meteo GloFAS flood API (Ganga discharge)"},
            "placeholders": ["shelters.csv", "units.csv", "block centroid coordinates (approximate)", "block-ID to name mapping for 13 of 16 blocks until census PCA is loaded"],
            "limitations": ["Flood records have no rainfall history, so the rain/river adjustment is hand-set, not learned",
                            "Report and chat NLP trained on templated text; real-world accuracy will be lower",
                            "Decision support only - does not replace IMD / BSDMA / DDMA warnings"]}


@app.get("/api/health")
def health(s=Depends(session)):
    return {"ok": True, "flood_model": MODEL.ready, "nlp": NL.ready, "email": "smtp" if mailer.configured() else "dry-run",
            "weather": core.fetch_weather(BLOCKS) is not None, "river": core.fetch_river() is not None,
            "roads": (core.DATA / "roads.pkl").exists(), "users": s.query(db.User).count(), "reports": s.query(db.Report).count()}


app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/")
def index(): return FileResponse(ROOT / "static" / "index.html")
