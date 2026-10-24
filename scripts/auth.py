"""One-time access codes: 6 digits, valid 3 minutes, single use. The code goes to the OWNER (e-mail or console);
the owner tells it to the visitor."""
import hashlib
import hmac
import os
import secrets
import smtplib
import threading
import time
from email.message import EmailMessage

TTL = 180               # seconds a code is valid
MAX_WRONG = 5           # wrong tries before the code is cancelled
MIN_GAP = 60            # seconds between two code requests
MAX_PER_HOUR = 6
SESSION_MINUTES = 60    # how long one accepted code lets the visitor stay

_lock = threading.Lock()
_state = {"hash": None, "expires": 0.0, "wrong": 0, "requests": []}


def _hash(code):
    return hashlib.sha256(code.encode()).hexdigest()


def request_code(who=""):
    now = time.time()
    with _lock:
        _state["requests"] = [t for t in _state["requests"] if now - t < 3600]
        if _state["requests"] and now - _state["requests"][-1] < MIN_GAP:
            return False, f"Yeni kod üçün {int(MIN_GAP - (now - _state['requests'][-1]))} saniyə gözləyin."
        if len(_state["requests"]) >= MAX_PER_HOUR:
            return False, "Saatlıq kod limiti dolub. Sonra yenidən cəhd edin."
        code = f"{secrets.randbelow(10 ** 6):06d}"
        _state.update(hash=_hash(code), expires=now + TTL, wrong=0)
        _state["requests"].append(now)
    _deliver(code, who)
    return True, "Kod tətbiqin sahibinə göndərildi. Ondan 6 rəqəmli kodu alıb aşağıya yazın (3 dəqiqə etibarlıdır)."


def _deliver(code, who):
    user, pw, to = os.getenv("SUSEN_SMTP_USER"), os.getenv("SUSEN_SMTP_PASS"), os.getenv("SUSEN_ADMIN_EMAIL")
    if user and pw and to:
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = f"SUSEN giriş kodu: {code}", user, to
            msg.set_content(f"SUSEN AI giriş kodu: {code}\nEtibarlıdır: 3 dəqiqə, bir dəfəlik.\nSessiya: {who or 'naməlum'}")
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
                s.login(user, pw)
                s.send_message(msg)
            print("[auth] code e-mailed to the owner")
            return
        except Exception as e:
            print(f"[auth] e-mail failed ({type(e).__name__}); showing the code here instead")
    print(f"[auth] ONE-TIME CODE: {code}   (valid 3 minutes)")


def verify(code):
    now = time.time()
    with _lock:
        if not _state["hash"] or now > _state["expires"]:
            return False
        if hmac.compare_digest(_hash(code.strip()), _state["hash"]):
            _state["hash"] = None                      # single use
            return True
        _state["wrong"] += 1
        if _state["wrong"] >= MAX_WRONG:
            _state["hash"] = None
        return False