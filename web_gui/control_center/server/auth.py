"""
App-level session auth for the Real-Robot Control Center.

Replaces the browser-native basic-auth prompt (ugly, and browsers refuse to
attach `Authorization` to WebSocket handshakes → 401 reprompt loop) with a
studio-styled in-app login and a signed session **cookie**. Browsers DO send
same-origin cookies on WS handshakes, so one cookie authenticates every plane
(HTTP + /ws/*) cleanly.

Security model (convenience-grade over TLS — the bridge access-code still gates
actual robot motion):
  • username+passkey compared with hmac.compare_digest (constant-time);
  • session token = b64url(user|issued|exp) + "." + HMAC-SHA256(secret, payload);
    unforgeable / non-extendable without the server secret;
  • cookie is HttpOnly + SameSite=Lax (+ Secure when served over TLS);
  • per-IP throttle blunts online guessing.
Set MABEL_CC_AUTH_SECRET in the service env to keep sessions valid across
restarts; otherwise a fresh random secret is minted per process (users re-login).
"""
import base64
import hmac
import hashlib
import os
import threading
import time

from . import config as CFG

COOKIE = "mabel_cc_session"
TTL_SECONDS = 12 * 3600
_B64 = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


_SECRET = (os.environ.get("MABEL_CC_AUTH_SECRET") or "").encode() or os.urandom(32)

# ── per-IP login throttle ────────────────────────────────────────────────────
_ATTEMPTS = {}                                   # ip -> [(ts), ...] recent fails
_ATT_LOCK = threading.Lock()
_MAX_FAILS = 8
_WINDOW = 300.0                                  # 5 min sliding window


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


# ── credentials ──────────────────────────────────────────────────────────────
def check_credentials(username, passkey):
    if not CFG.CC_USER or not CFG.CC_PASSKEY:    # unconfigured -> fail closed
        return False
    u = hmac.compare_digest((username or "").strip(), CFG.CC_USER)
    p = hmac.compare_digest((passkey or "").strip(), CFG.CC_PASSKEY)
    return u and p                               # both compared even if u fails


# ── session tokens ───────────────────────────────────────────────────────────
def _sign(payload):
    return hmac.new(_SECRET, payload, hashlib.sha256).digest()


def make_session(username, now=None):
    now = int(now if now is not None else time.time())
    payload = f"{username}|{now}|{now + TTL_SECONDS}".encode()
    return f"{_B64(payload)}.{_B64(_sign(payload))}"


def verify_session(token, now=None):
    """Return username for a valid, unexpired token, else None."""
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
