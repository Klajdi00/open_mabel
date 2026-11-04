"""
Operator session auth for the Trainer Studio.

The studio is browsable read-only, but every *mutating* training action — launch,
stop/kill, pause, resume, retrain, deploy, eval, community approve/reject — is
gated by a styled username+passkey popup (same look as the Real-Robot Control
Center) backed by a signed session cookie, so a job cannot be started or killed
without the operator credentials. Server-enforced; the client popup is only UX.

Security model mirrors control_center/server/auth.py: constant-time credential
check, HMAC-SHA256 signed non-extendable session token, HttpOnly cookie, per-IP
throttle. Set MABEL_TRAIN_AUTH_SECRET to persist sessions across restarts.
"""
import base64
import hmac
import hashlib
import os
import threading
import time

from . import config as CFG

COOKIE = "mabel_train_session"
HEADER = "x-train-auth"
TTL_SECONDS = 12 * 3600
_B64 = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


_SECRET = (os.environ.get("MABEL_TRAIN_AUTH_SECRET") or "").encode() or os.urandom(32)

_ATTEMPTS = {}
_ATT_LOCK = threading.Lock()
_MAX_FAILS = 8
_WINDOW = 300.0


def throttled(ip, now):
    with _ATT_LOCK:
        hits = [t for t in _ATTEMPTS.get(ip, []) if now - t < _WINDOW]
        _ATTEMPTS[ip] = hits
        return len(hits) >= _MAX_FAILS


def note_fail(ip, now):
    with _ATT_LOCK:
        _ATTEMPTS.setdefault(ip, []).append(now)


def note_ok(ip):
    with _ATT_LOCK:
        _ATTEMPTS.pop(ip, None)


def check_credentials(username, passkey):
    if not CFG.TRAIN_USER or not CFG.TRAIN_KEY:  # unconfigured -> fail closed
        return False
    u = hmac.compare_digest((username or "").strip(), CFG.TRAIN_USER)
    p = hmac.compare_digest((passkey or "").strip(), CFG.TRAIN_KEY)
    return u and p


def _sign(payload):
    return hmac.new(_SECRET, payload, hashlib.sha256).digest()


def make_session(username, now=None):
    now = int(now if now is not None else time.time())
    payload = f"{username}|{now}|{now + TTL_SECONDS}".encode()
    return f"{_B64(payload)}.{_B64(_sign(payload))}"


def verify_session(token, now=None):
    if not token or "." not in token:
        return None
    now = int(now if now is not None else time.time())
    try:
        body, sig = token.split(".", 1)
        payload = _unb64(body)
        if not hmac.compare_digest(_sign(payload), _unb64(sig)):
            return None
        user, issued, exp = payload.decode().split("|")
        if now > int(exp):
            return None
        return user
    except Exception:
        return None


def request_session(request):
    """Session token from cookie OR X-Train-Auth header (SPA fetch convenience)."""
    return (request.cookies.get(COOKIE)
            or request.headers.get(HEADER)
            or "")
