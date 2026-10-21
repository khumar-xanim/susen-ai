"""SUSEN AI chat window. Run: streamlit run scripts\\app.py"""
import json
from datetime import datetime

import pymupdf
import streamlit as st

from chatbot import Chatbot
from config import OUT_DIR, PDF_DIR
from llm import warm_up

st.set_page_config(page_title="SUSEN AI", page_icon="💊")
st.title("💊 SUSEN AI")
st.caption("Dərman təlimat vərəqələri üzrə məlumat köməkçisi. Mən həkim və ya əczaçı deyiləm: məlumatı rəsmi "
           "təlimat vərəqələrindən və kataloqdan götürürəm.")


@st.cache_resource(show_spinner="Məlumat bazası yüklənir…")
def get_bot():
    return Chatbot()


@st.cache_data(show_spinner=False)
def render_page(file, page):
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
    st.header("Ayarlar")
    model = st.selectbox("Model", ["qwen3.5:4b", "gemma3:4b", "qwen3.5:9b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    if st.button("Yeni söhbət"):
        st.session_state.messages = []
        bot.reset()
        st.rerun()
    st.divider()
    st.subheader("📷 Təlimat şəkli")
    files = st.file_uploader("Aldığınız dərmanın təlimatının şəkli", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files and st.button("Şəkli oxu"):
        with st.spinner("Şəkil oxunur…"):
            try:
                from ocr import ocr_bytes
                texts = [ocr_bytes(f.getvalue()) for f in files]
                st.session_state.photo_info = bot.load_photo(texts)
                st.session_state.photo_text = "\n\n".join(texts)
            except Exception as e:
                st.error(f"Şəkli oxumaq alınmadı: {e}")
    if bot.photo:
        info = st.session_state.get("photo_info", {})
        st.success(f"Şəkil rejimi aktivdir. Tanınan dərman: {info.get('drug') or 'tanınmadı'}")
        with st.expander("Oxunan mətn (səhvləri yoxlayın)"):
            st.text(st.session_state.get("photo_text", ""))
        if st.button("Şəkli sil"):
            bot.photo = None
            st.rerun()

if st.session_state.get("warm") != model:
    with st.spinner(f"{model} yüklənir (ilk dəfə bir az çəkə bilər)…"):
        try:
            warm_up(model)
            st.session_state.warm = model
        except Exception as e:
            st.error(f"Ollama ilə əlaqə yoxdur: {e}")

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
                        st.image(render_page(s["file"], s["page"]), caption=f"{s['file']}, səh. {s['page']}")
                    except Exception as e:
                        st.warning(f"Səhifə göstərilmədi: {e}")
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
    bot.model, bot.options = model, {"temperature": temperature}
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Cavab hazırlanır…"):
        r = bot.ask(prompt)
    st.session_state.messages.append({"role": "assistant", "content": r["reply"], "sources": r.get("sources", []),
                                      "status": r["status"], "question": prompt, "candidates": r.get("candidates", [])})
    st.rerun()