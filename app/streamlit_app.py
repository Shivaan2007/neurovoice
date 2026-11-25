import requests
import streamlit as st

API = "http://localhost:8000"
TIMEOUT_S = 60  # feature extraction (pyin) can take a while on a slow host


def error_detail(r: requests.Response) -> str:
    try:
        return r.json().get("detail", "Something went wrong.")
    except ValueError:  # non-JSON error, e.g. a proxy error page
        return f"Something went wrong (HTTP {r.status_code})."


def model_meta() -> dict:
    try:
        return requests.get(f"{API}/model-info", timeout=5).json().get("meta", {})
    except Exception:
        return {}


def higher_reliability() -> str:
    """Honest, cross-fitted reliability of the "higher" band, read from the model's metadata."""
    bp = model_meta().get("band_precision", {})
    est, folds = bp.get("higher_crossfit"), [f for f in bp.get("crossfit_higher_per_fold", []) if f]
    if est is None:
        return "How often this level is right on new people has not been measured."
    spread = f" It ranged from {min(folds):.0%} to {max(folds):.0%} across test groups." if folds else ""
    return (f"On speakers the model had never heard, this level was right about {est:.0%} of "
            f"the time, tested on a small group of speakers.{spread} Some people without "
            "Parkinson's would also land here.")


st.set_page_config(page_title="NeuroVoice")
st.title("NeuroVoice")
st.warning("Screening-support research prototype. Not a diagnosis. Trained on English "
           "speech from 37 people in one dataset; not tested on other languages, microphones "
           "or rooms; no clinical validation. Talk to a doctor with any concerns.")
st.write("In a quiet room, read a sentence or two aloud at your normal pace for 5 to 20 seconds. "
         "Only speech works: the model has never heard other sounds, so clips that don't "
         "sound like someone talking (music, tones, noise, silence) are turned away.")
audio = st.audio_input("Record") or st.file_uploader("...or upload", type=["wav", "mp3", "m4a", "webm"])
if audio is not None:
    try:
        r = requests.post(f"{API}/predict",
                          files={"file": (audio.name or "rec.wav", audio.getvalue())},
                          timeout=TIMEOUT_S)
    except requests.RequestException:
        st.error("Could not reach the prediction service. Please try again.")
        r = None
    if r is not None and r.ok:
        res = r.json()
        words = {"lower": "lower than the model's uncertainty band",
                 "uncertain": "not enough confidence to say",
                 "higher": "higher than the model's uncertainty band"}[res["band"]]
        st.metric("Calibrated probability", f"{res['probability']:.2f}")
        st.write(f"Result: **{words}**.")
        if res["band"] == "higher":
            st.info("This is a weak signal, not a finding about you. " + higher_reliability())
        elif res["band"] == "uncertain":
            st.info("The model is not confident either way. That is the most common outcome.")
        st.caption(res["notice"])
        st.caption("Your audio is deleted right after the prediction.")
    elif r is not None:
        st.error(error_detail(r))
with st.expander("About the model"):
    meta = model_meta()
    if not meta:
        st.write("Model info unavailable.")
    else:
        cv = meta.get("cv", {})
        bp = meta.get("band_precision", {})
        st.write(f"**Trained on:** {meta.get('trained_on', 'unknown')} "
                 f"({meta.get('n_speakers', '?')} speakers).")
        if cv:
            st.write(f"**On speakers it had never heard** (not a random split): "
                     f"AUROC {cv.get('auroc_window')} per 5 s window "
                     f"(95% CI {cv.get('window_ci95')}), {cv.get('auroc_speaker')} per speaker. "
                     "0.5 is chance. The interval is wide because the group is small.")
        if bp.get("higher_crossfit") is not None:
            st.write(f"**\"Higher\" band:** right about {bp['higher_crossfit']:.0%} of the time "
                     "on unseen speakers, and much less reliable in some test groups.")
        st.write("**Not yet tested:** other datasets, languages, microphones or rooms. "
                 "Expect lower performance there.")
        with st.expander("Raw model details"):
            st.json(meta)
