"""SUSEN AI chat window. Run: python -m streamlit run scripts\\app.py   (admin view: $env:SUSEN_ADMIN="1")"""
import json
from datetime import datetime

import pymupdf
import streamlit as st

from chatbot import Chatbot
from config import ADMIN, OUT_DIR, PDF_DIR
from llm import warm_up

MAX_QUESTIONS = 60            # per browser session
MAX_IMAGE_BYTES = 8_000_000

st.set_page_config(page_title="SUSEN AI", page_icon="💊")
st.title("💊 SUSEN AI")
st.caption("Dərman təlimat vərəqələri üzrə məlumat köməkçisi. Mən həkim və ya əczaçı deyiləm: məlumatı rəsmi "
           "təlimat vərəqələrindən və etibarlı mənbələrdən götürürəm.")


@st.cache_resource(show_spinner="Məlumat bazası yüklənir…")
def get_bot():
    return Chatbot()


@st.cache_data(show_spinner=False)
def render_page(file, page):
    if not file.endswith(".pdf") or "/" in file or "\\" in file or ".." in file:
        raise ValueError("bad file")
    with pymupdf.open(PDF_DIR / file) as doc:
        return doc.load_page(page - 1).get_pixmap(dpi=110).tobytes("png")


def save_feedback(msg, value):
    with open(OUT_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "question": msg.get("question"),
                            "status": msg.get("status"), "reply": msg["content"], "value": value}, ensure_ascii=False) + "\n")
    st.toast("Təşəkkür edirik!")


bot = get_bot()
st.session_state.setdefault("messages", [])

with st.sidebar:
    if ADMIN:
        st.header("Admin")
        bot.model = st.text_input("Model", bot.model)
    if st.button("Yeni söhbət"):
        st.session_state.messages = []
        bot.reset()
        st.rerun()
    st.divider()
    st.subheader("📷 Təlimat şəkli")
    files = st.file_uploader("Aldığınız dərmanın təlimatının şəkli", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files and st.button("Şəkli oxu"):
        if any(f.size > MAX_IMAGE_BYTES for f in files):
            st.error("Şəkil çox böyükdür (maksimum 8 MB).")
        else:
            with st.spinner("Şəkil oxunur…"):
                try:
                    from ocr import ocr_bytes
                    texts = [ocr_bytes(f.getvalue()) for f in files]
                    st.session_state.photo_info = bot.load_photo(texts)
                    st.session_state.photo_text = "\n\n".join(texts)
                except Exception as e:
                    st.error("Şəkli oxumaq alınmadı." + (f" {e}" if ADMIN else " Daha aydın şəkil yükləməyi yoxlayın."))
    if bot.photo:
        info = st.session_state.get("photo_info", {})
        st.success(f"Şəkil rejimi aktivdir. Tanınan dərman: {info.get('drug') or 'tanınmadı'}")
        with st.expander("Oxunan mətn (səhvləri yoxlayın)"):
            st.text(st.session_state.get("photo_text", ""))
        if st.button("Şəkli sil"):
            bot.photo = None
            st.rerun()

if "warm" not in st.session_state:
    with st.spinner("Model yüklənir (ilk dəfə bir az çəkə bilər)…"):
        try:
            warm_up(bot.model)
            st.session_state.warm = True
        except Exception as e:
            st.error("Model hazırda əlçatan deyil." + (f" {e}" if ADMIN else ""))

if not st.session_state.messages:
    st.info("Salam! Dərmanın adını və sualınızı yazın, məsələn: «Abizol 10 mq yan təsirləri nədir?» "
            "Hər cavabda hansı təlimat vərəqəsindən və hansı səhifədən götürüldüyünü göstərirəm.")

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            for j, s in enumerate(m.get("sources", [])):
                st.caption("📄 " + s["label"])
                if s["file"].endswith(".pdf") and st.toggle("Səhifəni göstər", key=f"pg{i}_{j}"):
                    try:
                        st.image(render_page(s["file"], s["page"]), caption=s["label"])
                    except Exception:
                        st.warning("Səhifə göstərilmədi.")
            c1, c2, _ = st.columns([1, 1, 8])
            if c1.button("👍", key=f"up{i}"):
                save_feedback(m, 1)
            if c2.button("👎", key=f"dn{i}"):
                save_feedback(m, 0)

last = st.session_state.messages[-1] if st.session_state.messages else None
if last and last["role"] == "assistant" and last.get("status") == "ambiguous":
    st.write("Seçin:")
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
                                      "status": r["status"], "question": prompt, "candidates": r.get("candidates", [])})
    st.rerun()