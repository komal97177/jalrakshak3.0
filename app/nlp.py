"""Own NLP models (no external AI API): flood-report classifier, severity, needs, and chat intents.
Trained on template-generated English / Hindi / Hinglish text (see make_*); real reports collected in the DB can be used to retrain."""
import random, re, difflib
from pathlib import Path
import joblib
from .core import MODELS

PLACES = ["Bihpur", "Naugachhia", "Kahalgaon", "Sabour", "Sultanganj", "Pirpainti", "Nathnagar", "Jagdishpur", "Kharik", "Gopalpur",
          "Narayanpur", "Goradih", "Shahkund", "Sanhaula", "Ismailpur", "Rangra Chowk", "Bhagalpur"]
HI = {"बिहपुर": "Bihpur", "नौगछिया": "Naugachhia", "कहलगांव": "Kahalgaon", "सबौर": "Sabour", "सुल्तानगंज": "Sultanganj", "पीरपैंती": "Pirpainti",
      "नाथनगर": "Nathnagar", "जगदीशपुर": "Jagdishpur", "खरीक": "Kharik", "गोपालपुर": "Gopalpur", "नारायणपुर": "Narayanpur", "गोराडीह": "Goradih",
      "शाहकुंड": "Shahkund", "सन्हौला": "Sanhaula", "इस्माइलपुर": "Ismailpur", "रंगरा चौक": "Rangra Chowk", "भागलपुर": "Bhagalpur"}
ALIAS = {"naugachia": "Naugachhia", "colgong": "Kahalgaon", "sonhaula": "Sanhaula", "pirpaiti": "Pirpainti", "sultangunj": "Sultanganj"}
ALL_NAMES = {p.lower(): p for p in PLACES} | {k.lower(): v for k, v in HI.items()} | ALIAS
ROMAN_HI = {"hai", "hain", "mein", "me", "paani", "pani", "baadh", "badh", "bhar", "gaya", "gayi", "nahi", "kya", "kahan", "kaha", "madad", "chahiye", "bhejo", "log", "ghar", "aaj", "barish", "khatra", "kare", "karen"}


def find_place(text):
    t = text.lower()
    for k in sorted(ALL_NAMES, key=len, reverse=True):
        if k in t: return ALL_NAMES[k]
    for w in re.findall(r"[a-z]{5,}", t):
        m = difflib.get_close_matches(w, list(ALL_NAMES), n=1, cutoff=0.82)
        if m: return ALL_NAMES[m[0]]
    return None


def lang_of(text):
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    return "hi" if len(set(re.findall(r"[a-z]+", text.lower())) & ROMAN_HI) >= 2 else "en"


def norm(text):
    t = text.lower()
    for k in sorted(ALL_NAMES, key=len, reverse=True): t = t.replace(k, " PLACE ")
    return re.sub(r"\s+", " ", t)


# ---------- training data ----------
FLOOD = {  # severity -> phrases
    "low": ["Water logging on the road in {p}", "{p} mein sadak par paani jama hai", "{p} में सड़क पर पानी जमा है", "Drains overflowing in {p}, ankle deep water",
            "{p} me nala overflow ho gaya hai", "Heavy rain, road under water at {p}, bikes cannot pass", "{p} me barish se sadak pe paani bhar gaya"],
    "medium": ["Water has entered houses in {p}", "{p} mein ghar me paani ghus gaya", "{p} में घरों में पानी घुस गया है", "Knee deep flood water in {p} market",
               "{p} ka bazaar doob gaya, kamar tak paani", "Flood water in the lane, cows and belongings are being swept away at {p}", "{p} ki gali me baadh ka paani aa gaya"],
    "high": ["{p} is completely flooded, water above roof level", "{p} puri tarah doob gaya hai, chhat tak paani", "{p} पूरी तरह डूब गया है, छत तक पानी है",
             "Embankment breach near {p}, village under water", "{p} ke paas bandh toot gaya, gaon mein baadh", "Ganga water entering the village of {p}, everyone is moving out", "{p} me gaon ke gaon pani me dub gaye, chhat par baithe hain"]}
NEEDS = {"rescue": ["People are trapped, please send rescue", "log fase hue hain, bachao", "लोग फंसे हुए हैं, बचाओ"],
         "medical": ["Injured and sick people need an ambulance", "marij ko doctor chahiye, ambulance bhejo", "मरीज को डॉक्टर चाहिए, एम्बुलेंस भेजो"],
         "boat": ["We need a boat urgently", "naav chahiye jaldi", "नाव चाहिए जल्दी"],
         "pump": ["Need a water pump to drain water", "paani nikalne ka pump chahiye", "पानी निकालने के लिए पंप चाहिए"]}
NOT_FLOOD = ["Traffic jam near {p} today", "Power cut in {p} since morning", "Flood of congratulation messages, my result is out! {p}", "lol my phone got flooded with notifications",
             "Cricket match at {p} ground today", "{p} me shaadi ka function hai", "{p} में बिजली नहीं है सुबह से", "mela in {p} is very crowded", "Flooded my inbox with emails lol",
             "The Ganga aarti at {p} was beautiful", "{p} ka mausam bahut suhana hai", "गंगा आरती {p} में बहुत सुंदर थी", "Good morning friends from {p}", "Vegetable prices high in {p}", "Water bottle price increased in {p}", "Bihar election rally at {p} today", "{p} me school ka result aaj aayega", "Train late at {p} station by two hours"]
INTENTS = {
    "risk": ["Is {p} safe?", "flood risk in {p}", "will {p} flood", "kya {p} mein baadh ka khatra hai", "{p} safe hai kya", "{p} में बाढ़ का खतरा है क्या", "मेरे इलाके में खतरा कितना है", "is it dangerous in {p}", "{p} risk level", "danger level at {p}", "is my area safe", "is it safe here", "is this area dangerous", "{p} khatarnak hai kya", "mera area safe hai kya", "yahan khatra hai kya", "{p} में खतरनाक है क्या", "क्या यह इलाका सुरक्षित है", "क्या यहां खतरा है"],
    "shelter": ["where is the nearest shelter", "nearest relief camp", "I need a safe place to go", "shelter kahan hai", "najdeeki rahat shivir kahan hai", "नजदीकी राहत शिविर कहां है", "सुरक्षित जगह कहां जाएं"],
    "alert": ["what is the alert level", "is there any flood warning", "alert kya hai abhi", "abhi koi warning hai kya", "अभी अलर्ट क्या है", "कोई चेतावनी है क्या"],
    "forecast": ["will it rain today", "rain forecast for next hours", "aaj barish hogi kya", "kitni barish hogi", "आज बारिश होगी क्या", "कितनी बारिश होगी"],
    "tips": ["what should I do in a flood", "flood safety tips", "baadh me kya kare", "baadh se bachne ke upay", "बाढ़ में क्या करें", "बाढ़ से बचाव के उपाय"],
    "help": ["emergency number", "helpline number please", "I need help call someone", "madad ka number kya hai", "helpline number batao", "हेल्पलाइन नंबर क्या है", "मदद चाहिए नंबर बताओ"],
    "other": ["tell me a joke", "who won the match", "what is the capital of india", "gaana sunao", "khana kya banau", "आज खाने में क्या बनाएं", "hello how are you"]}


def _fill(t, rnd): return t.format(p=rnd.choice(PLACES)) if "{p}" in t else t


def make_report_data(seed=0):
    rnd = random.Random(seed); rows = []  # (text, is_flood, severity, needs, template_id)
    tid = 0
    for sev, tpls in FLOOD.items():
        for t in tpls:
            tid += 1
            for _ in range(14):
                ns = [n for n in NEEDS if rnd.random() < 0.22]
                txt = _fill(t, rnd) + "".join(". " + rnd.choice(NEEDS[n]) for n in ns)
                rows.append((txt, 1, sev, ns, tid))
    for t in NOT_FLOOD:
        tid += 1
        for _ in range(14): rows.append((_fill(t, rnd), 0, "none", [], tid))
    return rows


def make_intent_data(seed=1):
    rnd = random.Random(seed); rows, tid = [], 0
    for k, tpls in INTENTS.items():
        for t in tpls:
            tid += 1
            for _ in range(10): rows.append((_fill(t, rnd), k, tid))
    return rows


def train_all():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score, accuracy_score
    vec = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    rep, itn = make_report_data(), make_intent_data()
    T = [norm(r[0]) for r in rep]; g = [r[4] for r in rep]
    mlb = MultiLabelBinarizer(classes=list(NEEDS)); Y = mlb.fit_transform([r[3] for r in rep])
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=0).split(T, groups=g))
    v = vec().fit([T[i] for i in tr])
    Xtr, Xte = v.transform([T[i] for i in tr]), v.transform([T[i] for i in te])
    m = {}
    m["flood_acc_heldout_templates"] = round(accuracy_score([rep[i][1] for i in te], LogisticRegression(max_iter=2000, C=5).fit(Xtr, [rep[i][1] for i in tr]).predict(Xte)), 3)
    m["severity_acc_heldout_templates"] = round(accuracy_score([rep[i][2] for i in te], LogisticRegression(max_iter=2000, C=5).fit(Xtr, [rep[i][2] for i in tr]).predict(Xte)), 3)
    m["needs_f1_heldout_templates"] = round(f1_score(Y[te], OneVsRestClassifier(LogisticRegression(max_iter=2000, C=5)).fit(Xtr, Y[tr]).predict(Xte), average="micro", zero_division=0), 3)
    IT = [norm(r[0]) for r in itn]; ig = [r[2] for r in itn]
    a, b = next(GroupShuffleSplit(test_size=0.25, random_state=0).split(IT, groups=ig))
    iv = vec().fit([IT[i] for i in a])
    m["intent_acc_heldout_templates"] = round(accuracy_score([itn[i][1] for i in b], LogisticRegression(max_iter=2000, C=5).fit(iv.transform([IT[i] for i in a]), [itn[i][1] for i in a]).predict(iv.transform([IT[i] for i in b]))), 3)
    v = vec().fit(T); X = v.transform(T); iv = vec().fit(IT)
    bundle = {"vec": v, "flood": LogisticRegression(max_iter=2000, C=5).fit(X, [r[1] for r in rep]),
              "sev": LogisticRegression(max_iter=2000, C=5).fit(X, [r[2] for r in rep]),
              "needs": OneVsRestClassifier(LogisticRegression(max_iter=2000, C=5)).fit(X, Y), "mlb": mlb,
              "ivec": iv, "intent": LogisticRegression(max_iter=2000, C=5).fit(iv.transform(IT), [r[1] for r in itn]),
              "metrics": m | {"note": "Trained on synthetic templated text; held-out = unseen templates. Expect lower accuracy on real messy reports."}}
    MODELS.mkdir(exist_ok=True); joblib.dump(bundle, MODELS / "nlp.joblib"); return bundle["metrics"]


STRONG = re.compile(r"(paani|pani|पानी|water).{0,40}(ghus|bhar|jama|entered|logging|overflow|doob|dub|डूब|घुस|भर|जमा|submerg|under)|(doob|dub|डूब|baadh|बाढ़|flood(ed)? (water|village|area|road))|bandh toot|बांध टूट|breach", re.I)
KW_NEEDS = {"rescue": r"rescue|trapped|fase|fans|bachao|बचाओ|फंस", "medical": r"injur|ambulance|doctor|marij|मरीज|बीमार|sick|एम्बुलेंस",
            "boat": r"boat|naav|nav |नाव", "pump": r"pump|पंप"}
JOKE = re.compile(r"\blol\b|inbox|notification|result|congrat|emails?\b", re.I)


class NLP:
    def __init__(self):
        p = MODELS / "nlp.joblib"; self.b = joblib.load(p) if p.exists() else None

    @property
    def ready(self): return self.b is not None

    def read_report(self, text):
        b, t = self.b, norm(text); X = b["vec"].transform([t])
        pc = float(b["flood"].predict_proba(X)[0][1])
        lex = 0.0 if JOKE.search(text) else (1.0 if STRONG.search(text) else 0.0)
        pf = 0.5 * pc + 0.5 * lex if lex or pc < 0.5 else pc * 0.7   # classifier + keyword evidence
        sev = str(b["sev"].predict(X)[0])
        if re.search(r"chhat|roof|छत|bandh toot|बांध टूट|breach|puri tarah|पूरी तरह", text, re.I): sev = "high"
        np_ = b["needs"].predict_proba(X)[0]
        needs = sorted(set(n for n, p in zip(b["mlb"].classes_, np_) if p >= 0.5) | {n for n, rx in KW_NEEDS.items() if re.search(rx, text, re.I)})
        return {"flood_prob": round(pf, 3), "severity": sev if pf >= 0.5 else "none",
                "needs": needs if pf >= 0.5 else [],
                "place": find_place(text), "lang": lang_of(text)}

    def intent(self, text):
        b = self.b; p = b["intent"].predict_proba(b["ivec"].transform([norm(text)]))[0]
        i = int(p.argmax()); return b["intent"].classes_[i], float(p[i])
