import requests
import streamlit as st

API = "http://localhost:8000"
st.set_page_config(page_title="NeuroVoice")
st.title("NeuroVoice")
st.warning("Screening-support research prototype. Not a diagnosis. Small cohorts, "
           "three languages, no clinical validation. Talk to a doctor with any concerns.")
st.write("In a quiet room, read a sentence or two aloud at your normal pace for 5 to 20 seconds.")
audio = st.audio_input("Record") or st.file_uploader("...or upload", type=["wav", "mp3", "m4a", "webm"])
if audio is not None:
    r = requests.post(f"{API}/predict", files={"file": (audio.name or "rec.wav", audio.getvalue())})
    if r.ok:
        res = r.json()
        words = {"lower": "lower than the model's uncertainty band",
                 "uncertain": "not enough confidence to say",
                 "higher": "higher than the model's uncertainty band"}[res["band"]]
        st.metric("Calibrated probability", f"{res['probability']:.2f}")
        st.write(f"Result: **{words}**.")
        st.caption(res["notice"])
        st.caption("Your audio is deleted right after the prediction.")
    else:
        st.error(r.json().get("detail", "Something went wrong."))
with st.expander("About the model"):
    try:
        st.json(requests.get(f"{API}/model-info", timeout=5).json())
    except Exception:
        st.write("Model info unavailable.")
