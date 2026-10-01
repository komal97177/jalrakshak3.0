"""Passwordless auth: email one-time code -> signed session cookie. No passwords are ever created, asked for or stored."""
import base64, hashlib, hmac, json, os, random, time, datetime as dt
from collections import defaultdict, deque
from cryptography.fernet import Fernet

SECRET = os.getenv("APP_SECRET", "dev-only-change-me").encode()
FERNET = Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"enc" + SECRET).digest()))
ADMINS = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
SESSION_SECS = 7 * 24 * 3600


def h(value: str) -> str: return hmac.new(SECRET, value.lower().strip().encode(), hashlib.sha256).hexdigest()
def enc(email: str) -> str: return FERNET.encrypt(email.lower().strip().encode()).decode()
def dec(tok: str) -> str: return FERNET.decrypt(tok.encode()).decode()
def new_code() -> str: return f"{random.SystemRandom().randrange(10**6):06d}"
def code_hash(email, code) -> str: return h(f"otp:{email}:{code}")


def make_token(email_hash: str, admin: bool) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"u": email_hash, "a": admin, "e": int(time.time()) + SESSION_SECS}).encode()).decode()
    return body + "." + hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()


def read_token(tok):
    try:
        body, sig = tok.rsplit(".", 1)
        if not hmac.compare_digest(sig, hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()): return None
        d = json.loads(base64.urlsafe_b64decode(body))
        return d if d["e"] > time.time() else None
    except Exception:
        return None


_hits = defaultdict(deque)
def limited(key: str, n: int, per_sec: int) -> bool:
    q, t = _hits[key], time.time()
    while q and q[0] < t - per_sec: q.popleft()
    if len(q) >= n: return True
    q.append(t); return False
