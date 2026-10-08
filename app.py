"""Cadenza: a local studio for the trained piano LSTM."""
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from musicgen.generate import Generator, export
from musicgen.io import read_json
from musicgen.midi import Note
from musicgen.paths import EXAMPLES, GENERATED, MODEL_DIR, REPORTS, ROOT
from musicgen.visualize import piano_roll

st.set_page_config(page_title="Cadenza | AI Music Studio", page_icon="🎹", layout="wide")
st.markdown("""<style>
 .block-container {max-width:1250px; padding-top:2.5rem; padding-bottom:2rem;}
 [data-testid="stHeader"] {background:transparent;}
 h1 {font-size:3.6rem !important; font-weight:600 !important; letter-spacing:-.065em;}
 h2 {letter-spacing:-.04em;}
 h3 {font-weight:500 !important; letter-spacing:-.025em;}
 .studio-label {font-size:11px;letter-spacing:.18em;color:#93a58f;margin-bottom:10px;}
 .brand {font-size:17px;letter-spacing:-.03em;font-weight:600;color:#eaf4e3;}
 .ready {color:#b7f397;font-size:11px;letter-spacing:.1em;text-align:right;padding-top:8px;}
 .hero-copy {font-size:17px;color:#9aaa94;max-width:600px;margin-bottom:26px;line-height:1.6;}
 [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {border-color:#303a2d !important; border-radius:16px !important;}
 [data-testid="stMetricValue"] {font-size:1.5rem !important;}
 [data-testid="stMetricLabel"] {color:#93a58f;}
 [data-testid="stTabs"] {margin-top:12px;}
 button[kind="primary"], button[kind="primaryFormSubmit"] {font-weight:600;min-height:48px;color:#162012;}
 button[kind="primary"] p, button[kind="primaryFormSubmit"] p {color:#162012;}
 .footer {color:#72816d;font-size:12px;margin-top:35px;border-top:1px solid #2b3528;padding-top:18px;}
 @media (max-width:700px) {h1 {font-size:2.8rem !important;} .block-container {padding-top:1.5rem;} .ready {text-align:left;}}
 </style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: int, generation_modified: int):
    return Generator(Path(path))


def stored_result(directory: Path, name: str):
    metadata = read_json(directory / f"{name}.json")
    return {"midi": directory / f"{name}.mid", "wav": directory / f"{name}.wav",
            "metadata": metadata, "name": name}


def display_composition(result, title: str):
    notes = [Note(**item) for item in result["metadata"]["notes"]]
    metadata = result["metadata"]
    st.markdown('<div class="studio-label">YOUR LISTENING ROOM</div>', unsafe_allow_html=True)
    st.subheader(title)
    columns = st.columns(3)
    columns[0].metric("Duration", f"{metadata['duration_seconds']:.1f}s")
    columns[1].metric("Notes", len(notes))
    columns[2].metric("Tempo", f"{metadata['bpm']} BPM")
    st.audio(result["wav"].read_bytes(), format="audio/wav")
    fig = piano_roll(notes, metadata["bpm"])
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Piano roll · horizontal position is time; vertical position is pitch.")
    midi, wav = st.columns(2)
    midi.download_button("Download MIDI", result["midi"].read_bytes(),
                         file_name=f"{result['name']}.mid", mime="audio/midi", width="stretch")
    wav.download_button("Download WAV", result["wav"].read_bytes(),
                        file_name=f"{result['name']}.wav", mime="audio/wav", width="stretch")
    settings = metadata["settings"]
    st.caption(f"Seed {settings['seed']} · creativity {settings['temperature']:.2f} · "
               "piano-like synthesized preview")


brand, badge = st.columns([3, 2])
brand.markdown('<div class="brand">♬ &nbsp; CADENZA</div>', unsafe_allow_html=True)
checkpoint = MODEL_DIR / "piano_lstm.pt"
if not checkpoint.exists():
    st.title("A little inspiration.\nA new composition.")
    st.info("Train your piano model to begin composing.")
    st.code("python -m musicgen.dataset\npython -m musicgen.train", language="bash")
    st.stop()
try:
    generator = load_model(str(checkpoint), checkpoint.stat().st_mtime_ns,
                           (ROOT / "musicgen" / "generate.py").stat().st_mtime_ns)
except (ValueError, OSError, RuntimeError, KeyError) as error:
    st.error(f"The trained model could not be loaded: {error}")
    st.info("Restore models/piano_lstm.pt or retrain with python -m musicgen.train.")
    st.stop()
badge.markdown('<div class="ready">● &nbsp; TRAINED MODEL · READY</div>', unsafe_allow_html=True)
st.markdown('<div class="studio-label" style="margin-top:35px">GENERATIVE PIANO STUDIO</div>', unsafe_allow_html=True)
st.title("Make room for a new melody.")
st.markdown('<div class="hero-copy">Turn a spark of curiosity into a piano composition. '
            'Shape the mood, find your rhythm, and hear what comes next.</div>', unsafe_allow_html=True)

compose, library, model_tab = st.tabs(["Compose", "Example library", "About the model"])
with compose:
    controls, listening = st.columns([1, 1.9], gap="large")
    with controls:
        with st.form("composition_controls"):
            st.markdown('<div class="studio-label">01 / SHAPE YOUR SOUND</div>', unsafe_allow_html=True)
            st.subheader("The starting point")
            length_label = st.selectbox("Composition length", ["Sketch · 64 notes", "Standard · 128 notes", "Extended · 256 notes", "Long · 512 notes"], index=1)
            count = {"Sketch · 64 notes": 64, "Standard · 128 notes": 128,
                     "Extended · 256 notes": 256, "Long · 512 notes": 512}[length_label]
            bpm = st.slider("Tempo · BPM", min_value=60, max_value=180, value=100, step=5,
                            help="A higher tempo plays the composition faster.")
            temperature = st.slider("Creativity", min_value=.2, max_value=1.8, value=.9, step=.05,
                                    help="Lower values favor familiar patterns; higher values explore more varied notes.")
            seed = st.number_input("Variation seed", min_value=0, max_value=2**32 - 1, value=42, step=1,
                                   help="Use the same settings and seed to reproduce a composition.")
            with st.expander("Fine-tune the variation"):
                top_k = st.slider("Candidate notes", min_value=1, max_value=40, value=12,
                                   help="How many likely next notes the model can choose from.")
            st.caption("Each composition is capped at two minutes. Longer note counts may stop at this limit.")
            submitted = st.form_submit_button("Generate composition  →", type="primary", width="stretch")
        st.caption("Local generation · no API key needed")
    if submitted:
        try:
            with st.spinner("Composing your next melody…"):
                settings = {"notes": count, "temperature": temperature, "top_k": top_k, "seed": int(seed)}
                notes = generator.generate(count, temperature, top_k, int(seed), bpm)
                name = f"cadenza_{int(seed)}_{uuid.uuid4().hex[:8]}"
                result = export(notes, GENERATED, name, bpm, settings)
                result["name"] = name
                st.session_state["composition"] = result
            st.success("Your composition is ready. Press play to listen.")
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"Could not generate the composition: {error}")
    if "composition" not in st.session_state and (EXAMPLES / "balanced.json").exists():
        st.session_state["composition"] = stored_result(EXAMPLES, "balanced")
    with listening:
        with st.container(border=True):
            if "composition" in st.session_state:
                result = st.session_state["composition"]
                title = "Featured composition" if result["name"] == "balanced" else "Your new composition"
                display_composition(result, title)
            else:
                st.subheader("Your next melody starts here.")
                st.write("Choose your settings and generate a composition to hear it and explore its piano roll.")
with library:
    st.subheader("A few places to begin")
    st.write("Three variations from the same trained model. Listen, then explore your own settings in Compose.")
    for column, (name, title) in zip(st.columns(3), [("focused", "01 / Focused"), ("balanced", "02 / Balanced"), ("exploratory", "03 / Exploratory")]):
        with column:
            with st.container(border=True):
                st.subheader(title)
                if (EXAMPLES / f"{name}.json").exists():
                    result = stored_result(EXAMPLES, name)
                    st.audio(result["wav"].read_bytes(), format="audio/wav")
                    settings = result["metadata"]["settings"]
                    st.caption(f"{result['metadata']['note_count']} notes · seed {settings['seed']} · creativity {settings['temperature']}")
                    st.download_button("Save MIDI", result["midi"].read_bytes(), file_name=f"{name}.mid",
                                       mime="audio/midi", key=f"library_{name}", width="stretch")
                else:
                    st.caption("No saved example yet. Generate a composition in the Compose tab.")
with model_tab:
    st.subheader("Learned from real piano performances")
    st.write("A two-layer LSTM learns sequences of pitch, spacing, duration, and velocity from classical piano MIDI. "
             "A training excerpt provides hidden-state context; newly sampled notes form the output.")
    counts = generator.metadata["dataset_counts"]
    a, b, c = st.columns(3)
    a.metric("Training performances", counts["train"]["pieces"])
    b.metric("Training notes", f"{counts['train']['notes']:,}")
    c.metric("Best training epoch", generator.metadata["epoch"])
    report_path = REPORTS / "evaluation.json"
    if report_path.exists():
        report = read_json(report_path)
        st.write(f"Held-out test pitch accuracy: **{report['test']['accuracy']['pitch']:.1%}**. "
                 f"Test loss: **{report['test']['loss']:.3f}**, compared with "
                 f"**{report['unigram_baseline_test']['loss']:.3f}** for a frequency-only baseline.")
        if (REPORTS / "training_curves.png").exists():
            st.image(str(REPORTS / "training_curves.png"), caption="Recorded training and validation results.")
    st.caption("This compact model produces short musical sketches. It can repeat patterns and does not guarantee "
               "a complete musical structure. Audio uses a portable piano-like synthesizer; MIDI can be played "
               "with your preferred instrument in a music application. Sustain pedal is not modeled.")
    st.markdown("Dataset: [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro), "
                "provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).")
st.markdown('<div class="footer">CADENZA &nbsp; / &nbsp; Music Generation with AI &nbsp; · &nbsp; CodeAlpha Task 3</div>', unsafe_allow_html=True)
