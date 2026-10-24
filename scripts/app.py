"""SUSEN AI chat window. Run: python -m streamlit run scripts\\app.py
Access needs a one-time 6-digit code (auth.py). The ADMIN view is the page opened directly on this computer."""
import copy
import json
import os
import secrets
import time
import traceback
from datetime import datetime

import streamlit as st

import auth
from config import ADMIN, OUT_DIR, PDF_DIR

MAX_QUESTIONS = 60
MAX_IMAGE_BYTES = 8_000_000
AUTH_ON = os.getenv("SUSEN_AUTH", "on").lower() != "off"

st.set_page_config(page_title="SUSEN AI",page_icon="C:/susen_ai/imgs/a_clean_vector_style_logo_illustration_on_a_transp.png")
st.markdown("""
<style>
div.stButton > button {
    font-size: 24px;
    min-width: 45px;
    min-height: 40px;
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)


def is_admin_view():
    """Admin = opened directly on this computer. Visitors come through the tunnel, which adds proxy headers."""
    try:
        headers = {k.lower() for k in st.context.headers.keys()}
    except Exception:
        return ADMIN
    return ADMIN or not (headers & {"cf-connecting-ip", "cf-ray", "x-forwarded-for"})


def show_error(exc, code=500):
    incident = secrets.token_hex(2).upper()
    detail = "".join(traceback.format_exception(exc))
    with open(OUT_DIR / "errors.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  incident {incident}  code {code}\n{detail}\n")
    if is_admin_view():
        st.error(f"Xəta {code} · İD {incident}")
        st.code(detail)                                   # file, line and the failing code
    else:
        st.error(f"Xəta {code} · İD {incident}. Zəhmət olmasa bir az sonra yenidən cəhd edin.")


@st.cache_resource(show_spinner="Məlumat bazası yüklənir…")
def get_bot():
    from chatbot import Chatbot                           # imported here so a code error is caught and covered
    return Chatbot()


@st.cache_data(show_spinner=False)
def render_page(file, page):
    import pymupdf
    if not file.endswith(".pdf") or "/" in file or "\\" in file or ".." in file:
        raise ValueError("bad file")
    with pymupdf.open(PDF_DIR / file) as doc:
        return doc.load_page(page - 1).get_pixmap(dpi=110).tobytes("png")


def save_feedback(msg, value):
    with open(OUT_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "question": msg.get("question"),
                            "status": msg.get("status"), "reply": msg["content"], "value": value}, ensure_ascii=False) + "\n")
    st.toast("Təşəkkür edirik!")


def gate():
    if not AUTH_ON or time.time() < st.session_state.get("granted_until", 0):
        return True
    st.title("💊 SUSEN AI")
    st.info("Giriş üçün birdəfəlik kod lazımdır. Kodu tətbiqin sahibindən alın.")
    if st.button("Kod istə"):
        ok, msg = auth.request_code(who="sessiya " + st.session_state.setdefault("sid", secrets.token_hex(2)))
        (st.success if ok else st.warning)(msg)
    code = st.text_input("6 rəqəmli kod", max_chars=6)
    if st.button("Daxil ol"):
        if code.isdigit() and len(code) == 6 and auth.verify(code):
            st.session_state.granted_until = time.time() + auth.SESSION_MINUTES * 60
            st.rerun()
        st.error("Kod yanlışdır və ya vaxtı bitib.")
    return False


def main():
    from llm import warm_up
    admin = is_admin_view()
    st.title("💊 SUSEN AI")
    st.caption("Dərman təlimat vərəqələri üzrə məlumat köməkçisi. Mən həkim və ya əczaçı deyiləm: məlumatı rəsmi "
               "təlimat vərəqələrindən və etibarlı mənbələrdən götürürəm.")
    if "bot" not in st.session_state:                     # each visitor has an own conversation; the data is shared
        st.session_state.bot = copy.copy(get_bot())
        st.session_state.bot.reset()
    bot = st.session_state.bot
    bot.admin = admin
    st.session_state.setdefault("messages", [])

    with st.sidebar:
        if admin:
            st.header("Admin")
            bot.model = st.text_input("Model", bot.model)
        if st.button("Yeni söhbət"):
            st.session_state.messages = []
            bot.reset()
            st.rerun()
        st.divider()
        st.subheader("📷 Dərmanın şəkli")
        files = st.file_uploader("Qutunun və ya təlimatın şəkli", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if files and st.button("Şəkli oxu"):
            if any(f.size > MAX_IMAGE_BYTES for f in files):
                st.error("Xəta 400 · Şəkil çox böyükdür (maksimum 8 MB).")
            else:
                with st.spinner("Şəkil oxunur…"):
                    try:
                        info = bot.identify_photo([f.getvalue() for f in files])
                        st.session_state.photo_text = info["text"]
                        st.session_state.messages.append({"role": "assistant", "content": info["message"], "sources": [],
                                                          "status": info["status"], "question": "[şəkil]",
                                                          "candidates": info["candidates"]})
                    except ValueError:
                        st.error("Xəta 400 · Şəkil oxunmadı (format və ya ölçü).")
                    except Exception as e:
                        show_error(e, 500)
        if st.session_state.get("photo_text"):
            with st.expander("Oxunan mətn (səhvləri yoxlayın)"):
                st.text(st.session_state.photo_text)
        if bot.photo and st.button("Şəkil rejimini söndür"):
            bot.photo = None
            st.rerun()

    if "warm" not in st.session_state:
        with st.spinner("Model yüklənir (ilk dəfə bir az çəkə bilər)…"):
            try:
                warm_up(bot.model)
                st.session_state.warm = True
            except Exception as e:
                show_error(e, 503)

    if not st.session_state.messages:
        st.info("Salam! Dərmanın adını və sualınızı yazın, ya da qutunun şəklini yükləyin. Cavabın hansı vərəqədən və hansı səhifədən götürüldüyünü göstərirəm.")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("excerpt"):
                    with st.expander("Vərəqədən orijinal mətn"):
                        st.write(m["excerpt"]["text"])
                        st.caption(m["excerpt"]["label"])
                for j, s in enumerate(m.get("sources", [])[:1]):
                    if s["file"].endswith(".pdf") and st.toggle("Səhifəni göstər", key=f"pg{i}_{j}"):
                        try:
                            st.image(render_page(s["file"], s["page"]), caption=s["label"])
                        except Exception:
                            st.warning("Səhifə göstərilmədi.")
                if m.get("question") != "[şəkil]":
                    c1, c2, _ = st.columns([1, 1, 8])
                    if c1.button("🖒", key=f"up{i}", help="Helpful answer"):
                        save_feedback(m, 1)
                    if c2.button("🖓", key=f"dn{i}", help="Unhelpful answer"):
                        save_feedback(m, 0)

    last = st.session_state.messages[-1] if st.session_state.messages else None
    if last and last["role"] == "assistant" and last.get("status") == "ambiguous":
        for k, c in enumerate(last.get("candidates", [])):
            if st.button(f"{c['name']} — {c['form']}, {c['dose'] or 'doza göstərilməyib'}", key=f"cand{k}"):
                st.session_state.queued = f"{c['name']} {c['dose']}"

    prompt = st.chat_input("Dərman haqqında sualınızı yazın…") or st.session_state.pop("queued", None)
    if prompt:
        if sum(m["role"] == "user" for m in st.session_state.messages) >= MAX_QUESTIONS:
            st.warning("Bu sessiyada sual limitinə çatdınız. Səhifəni yeniləyin.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Cavab hazırlanır…"):
            r = bot.ask(prompt)
        st.session_state.messages.append({"role": "assistant", "content": r["reply"], "sources": r.get("sources", []),
                                          "status": r["status"], "question": prompt, "candidates": r.get("candidates", []),
                                          "excerpt": r.get("excerpt")})
        st.rerun()


if gate():
    try:
        main()
    except Exception as e:
        show_error(e)