import streamlit as st
import numpy as np
import pandas as pd
import joblib
import keras
import altair as alt
import random
import time

# ----------------------------------------------------------------------
# Config / encodings
# ----------------------------------------------------------------------

# Binary / ordinal encodings assumed during training
ENCODINGS = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},           # frequent high-caloric food
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},            # monitors calories
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},   # eating between meals
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},   # alcohol consumption
}

MTRANS_OPTIONS = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
# "Automobile" was dropped as the one-hot baseline (all MTRANS_* columns = 0)


def bmi_to_label(bmi: float) -> str:
    if bmi < 18.5: return "Insufficient_Weight"
    elif bmi < 25: return "Normal_Weight"
    elif bmi < 27.5: return "Overweight_Level_I"
    elif bmi < 30: return "Overweight_Level_II"
    elif bmi < 35: return "Obesity_Type_I"
    elif bmi < 40: return "Obesity_Type_II"
    else: return "Obesity_Type_III"


FEATURE_ORDER = [
    "Gender", "Age", "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC",
    "MTRANS_Bike", "MTRANS_Motorbike", "MTRANS_Public_Transportation", "MTRANS_Walking",
]


LABEL_MAP = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

# Palette kept intuitive (blue -> green -> amber -> red) but tuned to sit
# nicely against the new light theme.
LABEL_COLORS = {
    "Insufficient_Weight": "#7FB8F0",
    "Normal_Weight": "#4FAE93",
    "Overweight_Level_I": "#E3B94D",
    "Overweight_Level_II": "#F0A868",
    "Obesity_Type_I": "#FF8F6B",
    "Obesity_Type_II": "#E86A6A",
    "Obesity_Type_III": "#C1447A",
}

MODEL_COLORS = {
    "Random Forest": "#4FAE93",   # lake mint
    "SVM": "#FF9B72",             # eating-card coral
    "KNN": "#E3B94D",             # hero gold
    "ANN": "#6152C4",             # sidebar violet
}


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(c))) for c in rgb))


def _shade(hex_color: str, factor: float) -> str:
    """factor > 0 lightens toward white, factor < 0 darkens toward black."""
    r, g, b = _hex_to_rgb(hex_color)
    if factor >= 0:
        r, g, b = (c + (255 - c) * factor for c in (r, g, b))
    else:
        r, g, b = (c * (1 + factor) for c in (r, g, b))
    return _rgb_to_hex((r, g, b))


def metallic_gradient(hex_color: str) -> dict:
    """A Vega-Lite linear-gradient spec that gives a flat bar a brushed-metal,
    glossy-pill look: a bright highlight streak near the top fading into the
    base color, then a deeper shade toward the bottom — matching the
    gel-button/card styling used throughout the rest of the UI."""
    return {
        "gradient": "linear",
        "x1": 0, "y1": 0, "x2": 0, "y2": 1,
        "stops": [
            {"offset": 0.0, "color": _shade(hex_color, 0.6)},
            {"offset": 0.16, "color": _shade(hex_color, 0.18)},
            {"offset": 0.45, "color": hex_color},
            {"offset": 0.78, "color": _shade(hex_color, -0.18)},
            {"offset": 1.0, "color": _shade(hex_color, -0.38)},
        ],
    }

# ----------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------

@st.cache_resource
def load_models():
    rf = joblib.load("obesity_rf_model.joblib")
    knn = joblib.load("obesity_knn_model.joblib")
    svm = joblib.load("obesity_svm_model.joblib")
    ann = keras.models.load_model("obesity_ann_model.keras")
    return rf, knn, svm, ann


@st.cache_data
def load_comparison():
    return pd.read_csv("model_comparison.csv")


@st.cache_resource
def load_scaler():
    """The real MinMaxScaler fit on the training set during cleaning
    (see: scaler.fit_transform(X_train[num_cols]) in the cleaning
    notebook). Loading the actual fitted object instead of guessing
    min/max bounds means this app can never drift out of sync with
    what the models were actually trained on.

    Bundled into a single joblib file: {"scaler": ..., "columns": ...}
    so the fitted scaler and its expected column order always travel
    together and can't drift out of sync with each other.
    """
    bundle = joblib.load("obesity_scaler_bundle.joblib")
    return bundle["scaler"], bundle["columns"]


def scale_numeric_inputs(inputs: dict, scaler, scaler_cols) -> dict:
    """Scales the numeric fields using the real fitted scaler, in the
    exact column order it was fit with, instead of a per-field guessed
    (value - lo) / (hi - lo) formula."""
    raw_row = pd.DataFrame([[inputs[c] for c in scaler_cols]], columns=scaler_cols)
    scaled_row = scaler.transform(raw_row)[0]
    return dict(zip(scaler_cols, scaled_row))


@st.cache_resource
def load_svm_scaler():
    """The real StandardScaler the SVM notebook itself fit and saved
    (svm_scaler.pkl) — no need to guess or reverse-engineer this one,
    the training notebook exports it directly."""
    return joblib.load("svm_scaler.pkl")


@st.cache_resource
def load_ann_scaler():
    """The real StandardScaler the ANN notebook fit on the full 17-column
    feature vector and saved (ann_scaler.pkl). Like the SVM scaler, this
    is applied on top of the already-encoded/MinMax-scaled feature row,
    right before the ANN's own forward pass."""
    return joblib.load("ann_scaler.pkl")


def build_feature_vector(inputs: dict, scaler, scaler_cols) -> pd.DataFrame:
    scaled_numeric = scale_numeric_inputs(inputs, scaler, scaler_cols)
    row = {
        "Gender": ENCODINGS["Gender"][inputs["Gender"]],
        "Age": scaled_numeric["Age"],
        "family_history_with_overweight": ENCODINGS["family_history_with_overweight"][inputs["family_history"]],
        "FAVC": ENCODINGS["FAVC"][inputs["FAVC"]],
        "FCVC": scaled_numeric["FCVC"],
        "NCP": scaled_numeric["NCP"],
        "CAEC": ENCODINGS["CAEC"][inputs["CAEC"]],
        "SMOKE": ENCODINGS["SMOKE"][inputs["SMOKE"]],
        "CH2O": scaled_numeric["CH2O"],
        "SCC": ENCODINGS["SCC"][inputs["SCC"]],
        "FAF": scaled_numeric["FAF"],
        "TUE": scaled_numeric["TUE"],
        "CALC": ENCODINGS["CALC"][inputs["CALC"]],
        "MTRANS_Bike": inputs["MTRANS"] == "Bike",
        "MTRANS_Motorbike": inputs["MTRANS"] == "Motorbike",
        "MTRANS_Public_Transportation": inputs["MTRANS"] == "Public_Transportation",
        "MTRANS_Walking": inputs["MTRANS"] == "Walking",
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------

st.set_page_config(page_title="Obesity Level Predictor", page_icon="⚖️", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Playfair+Display:wght@600;700;800;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, span, label, div {
    font-family: 'Nunito', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
h1, h2, h3, h4, h5, h6, .hero-title {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
}

/* ================= Sky/lake backdrop ================= */
/* Back to the mountain/lake photo (Gary Ellis, Unsplash — free to use, no
   attribution required: https://unsplash.com/photos/rr7BbHdxvZQ), this time
   with a much heavier white wash on top so text stays readable over the
   busy mountain/water detail without giving up the photo entirely. */
.stApp {
    background-image:
        linear-gradient(180deg, rgba(255,255,255,0.5) 0%, rgba(255,255,255,0.34) 45%, rgba(255,255,255,0.46) 100%),
        url('https://images.unsplash.com/photo-1607602274042-161d6cba839a?auto=format&fit=crop&w=1920&q=80');
    background-size: cover;
    background-position: 50% 30%;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* ================= Hero: cream "memory popup" card ================= */
.hero-banner {
    position: relative;
    background: linear-gradient(180deg, #FBEFC9 0%, #F3DFA0 55%, #E9CE85 100%);
    border: 3px solid #C9A855;
    padding: 1.7rem 2.2rem;
    border-radius: 22px;
    margin-bottom: 1.8rem;
    box-shadow:
        inset 0 2px 0 rgba(255,255,255,0.7),
        0 10px 24px -10px rgba(60,40,10,0.45);
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 45%;
    background: linear-gradient(180deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0) 100%);
    pointer-events: none;
}
.hero-title {
    position: relative;
    color: #5B4522;
    font-size: 2.15rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
    text-shadow: 0 1px 0 rgba(255,255,255,0.5);
}
.hero-caption {
    position: relative;
    color: #6B5A34;
    font-size: 1rem;
    max-width: 760px;
    line-height: 1.55;
}
.page-pill {
    position: relative;
    display: inline-block;
    background: radial-gradient(circle at 35% 30%, #7FC8FF 0%, #2E8FE0 55%, #1665B8 100%);
    color: #FFFFFF;
    padding: 0.22rem 0.9rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.8),
        0 2px 4px rgba(0,0,0,0.35);
    border: 2px solid #124E8E;
}

/* ================= Sidebar: violet glass panel ================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #7A6FCE 0%, #5E4FB8 55%, #4A3B9E 100%);
    border-right: 3px solid #3A2E82;
}
section[data-testid="stSidebar"] * {
    color: #F3F0FF !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.35);
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: #D3CBFA !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2);
}

/* Radio nav items styled like the game's glossy purple pill buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: linear-gradient(180deg, #A79BEF 0%, #8171DE 45%, #6152C4 100%);
    border: 2px solid #4A3B9E;
    border-radius: 999px;
    padding: 0.55rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow:
        inset 0 2px 2px rgba(255,255,255,0.55),
        inset 0 -6px 10px rgba(40,25,110,0.25),
        0 2px 5px rgba(0,0,0,0.25);
    transition: all 0.12s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: linear-gradient(180deg, #BCB2FA 0%, #9686EA 45%, #7264D6 100%);
    border-color: #2F2470;
}

/* ================= Glossy purple pill buttons (the pie-menu look) ================= */
.stButton > button {
    background: linear-gradient(180deg, #ADA1F5 0%, #8676E8 42%, #6152C4 55%, #7768D9 100%);
    color: #FFFFFF;
    border: 2px solid #4A3B9E;
    border-radius: 999px;
    font-weight: 800;
    padding: 0.65rem 1.6rem;
    box-shadow:
        inset 0 2px 2px rgba(255,255,255,0.6),
        inset 0 -8px 12px rgba(40,25,110,0.3),
        0 5px 12px -3px rgba(40,25,110,0.55);
    text-shadow: 0 1px 2px rgba(30,20,70,0.4);
    transition: all 0.12s ease;
}
.stButton > button:hover {
    background: linear-gradient(180deg, #C0B6FA 0%, #9A8BEF 42%, #7264D6 55%, #8878E5 100%);
    color: #FFFFFF;
    box-shadow:
        inset 0 2px 2px rgba(255,255,255,0.7),
        inset 0 -8px 12px rgba(40,25,110,0.3),
        0 7px 16px -3px rgba(40,25,110,0.65);
}
.stButton > button:active {
    filter: brightness(0.92);
}

/* ================= Containers / cards: soft lavender glass panels ================= */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    border: 2px solid rgba(74, 59, 158, 0.35) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(235,232,255,0.92) 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 6px 16px -6px rgba(40,25,110,0.25);
    color: #3A2E70;
}
div[data-testid="stExpander"] {
    border-radius: 16px;
    border: 2px solid rgba(74, 59, 158, 0.3);
    background: linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(232,228,255,0.9) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 3px 8px rgba(40,25,110,0.15);
}
div[data-testid="stExpander"] summary {
    font-weight: 800;
    color: #4A3B9E;
}

/* ---- Metric ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EAE6FF 100%);
    border: 2px solid rgba(74, 59, 158, 0.3);
    border-radius: 18px;
    padding: 0.8rem 1rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 3px 8px rgba(40,25,110,0.15);
}

/* ---- Tabs: violet selected pill ---- */
button[data-baseweb="tab"] {
    font-weight: 700;
    color: #4A3B9E;
    opacity: 0.65;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4A3B9E;
    opacity: 1;
}
div[data-baseweb="tab-highlight"] {
    background-color: #6152C4 !important;
}

/* ---- Dataframe / table ---- */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid rgba(74, 59, 158, 0.3);
    box-shadow: 0 3px 8px rgba(40,25,110,0.15);
}

/* ---- Selects / inputs: rounded lavender pill fields ----
   Fully clip to the pill radius on the OUTER control too, otherwise the
   outer box stays square and its background peeks out past the rounded
   corners of the inner element. */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div,
div[data-testid="stNumberInput"] > div,
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="input"] {
    border-radius: 999px !important;
    overflow: hidden !important;
    border-color: rgba(74, 59, 158, 0.4) !important;
    box-shadow: inset 0 1px 3px rgba(74,59,158,0.15) !important;
    background: #FFFFFF !important;
}
div[data-baseweb="select"] > div > div,
input, textarea {
    border-radius: 999px !important;
    box-shadow: none !important;
    background: transparent !important;
}
/* number input's +/- stepper buttons: round the whole cluster, not the box */
div[data-testid="stNumberInput"] button {
    background: transparent !important;
}

/* ---- Slider handle: glossy violet orb ----
   (Reverted to styling just the thumb via the reliable role="slider"
   selector — the track/value-label part-name selectors weren't reliably
   matching Streamlit's internal markup across versions, so rather than
   risk breaking things further we're keeping this to the one selector
   that's guaranteed correct.) */
div[data-testid="stSlider"] div[role="slider"] {
    background: radial-gradient(circle at 35% 30%, #C0B6FA 0%, #6152C4 70%, #4A3B9E 100%) !important;
    border: 2px solid #3A2E82 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 2px 5px rgba(40,25,110,0.4) !important;
}

hr {
    border-color: rgba(74, 59, 158, 0.25);
}

/* ================================================================
   Darker body text + glossy light-blue "chip" backgrounds for captions
   and secondary text that sit directly on the photo backdrop (outside
   the colored cards), so they're never just grey text floating on a
   busy background — mirrors the rounded gradient chip / badge look
   from the reference UI kit.
   ================================================================ */

/* body copy: darker than the previous muted grey, for contrast against
   the photo backdrop */
p, .stMarkdown p, li, .stMarkdown li {
    color: #1F2430;
}

/* st.caption() text: single flat solid color, square corners — a gradient
   made from sky-toned blues just blended into the cloud photo behind it,
   so this needs to be an unmistakably solid, more saturated block instead */
[data-testid="stCaptionContainer"] {
    display: block;
    width: 100%;
    background: linear-gradient(180deg, #FFFFFF 0%, #EAF4FF 45%, #BFE0FF 100%) !important;
    opacity: 1 !important;
    background-blend-mode: normal !important;
    backdrop-filter: none !important;
    border: 1px solid #9EC4EA;
    border-radius: 0;
    padding: 0.75rem 1.1rem;
    box-shadow: 0 3px 8px -3px rgba(20,50,90,0.2);
}
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] * {
    color: #16324F !important;
    font-weight: 700;
}

/* ...except inside the sidebar, where it clashed badly with the violet
   panel — restore plain, unboxed text there */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    display: block;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
    width: auto;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #D3CBFA !important;
    font-weight: 500;
}


   ================================================================ */

@keyframes skyDrift {
    0%   { background-position: 50% 30%, 50% 30%; }
    50%  { background-position: 50% 30%, 52% 36%; }
    100% { background-position: 50% 30%, 50% 30%; }
}
@keyframes blobFloat {
    0%   { transform: translate(0px, 0px) scale(1); }
    33%  { transform: translate(18px, -14px) scale(1.05); }
    66%  { transform: translate(-14px, 10px) scale(0.97); }
    100% { transform: translate(0px, 0px) scale(1); }
}
@keyframes heroGlow {
    0%, 100% { box-shadow: inset 0 2px 0 rgba(255,255,255,0.7), 0 10px 24px -10px rgba(60,40,10,0.45), 0 0 0px rgba(255,214,120,0); }
    50%      { box-shadow: inset 0 2px 0 rgba(255,255,255,0.8), 0 10px 30px -8px rgba(60,40,10,0.5), 0 0 26px rgba(255,214,120,0.55); }
}
@keyframes badgePulse {
    0%, 100% { box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 2px 4px rgba(0,0,0,0.35), 0 0 0px rgba(46,143,224,0); }
    50%      { box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 2px 8px rgba(0,0,0,0.4), 0 0 16px rgba(46,143,224,0.75); }
}
@keyframes logoBounce {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    25%      { transform: translateY(-4px) rotate(-8deg); }
    50%      { transform: translateY(0) rotate(0deg); }
    75%      { transform: translateY(-2px) rotate(8deg); }
}
@keyframes cardPopIn {
    0%   { opacity: 0; transform: translateY(14px) scale(0.98); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes sparkleTwinkle {
    0%, 100% { opacity: 0.25; transform: scale(0.85); }
    50%      { opacity: 1; transform: scale(1.15); }
}

/* animated sky/lake backdrop: gentle photo pan instead of static */
.stApp {
    animation: skyDrift 30s ease-in-out infinite;
}

/* soft drifting light blobs across the whole app, purely decorative */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 12% 20%, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0) 22%),
        radial-gradient(circle at 85% 15%, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0) 20%),
        radial-gradient(circle at 75% 80%, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 25%),
        radial-gradient(circle at 20% 85%, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 22%);
    animation: blobFloat 14s ease-in-out infinite;
}

/* hero card: soft golden glow pulse, like a memory-popup shimmer */
.hero-banner {
    animation: heroGlow 5s ease-in-out infinite;
}

/* the little "PREDICT"-style page badge gets a matching heartbeat glow */
.page-pill {
    animation: badgePulse 2.6s ease-in-out infinite;
}

/* sidebar logo icon: a small playful wobble */
.sidebar-logo-icon {
    display: inline-block;
    animation: logoBounce 3.2s ease-in-out infinite;
}

/* buttons: lift + brighten on hover instead of just a color swap */
.stButton > button {
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.015);
}
.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

/* sidebar nav pills: lift + glow on hover, gentle press on click */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    transform: translateX(3px) scale(1.02);
}

/* cards fade/pop into place instead of just appearing */
div[data-testid="stVerticalBlockBorderWrapper"] {
    animation: cardPopIn 0.45s ease both;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 12px 24px -8px rgba(40,25,110,0.35);
}

/* ---- Individual scenery per input card (Personal / Eating / Lifestyle) ----
   Each st.container(border=True, key=...) call gets a stable `st-key-...`
   class straight from Streamlit, so we target that directly instead of a
   fragile sibling-selector guess at the internal DOM structure. The class
   lands on a wrapper that CONTAINS stVerticalBlockBorderWrapper, so we
   style both that wrapper itself and the inner one, and force an opaque
   fill so the page backdrop can't show through underneath. */

.st-key-personal_card,
.st-key-personal_card > div[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    overflow: hidden;
    border-radius: 20px !important;
    border: 2px solid rgba(94, 110, 210, 0.4) !important;
    background:
        radial-gradient(circle at 88% 8%, rgba(120,178,255,0.55) 0%, rgba(120,178,255,0) 45%),
        radial-gradient(circle at 6% 92%, rgba(167,155,239,0.45) 0%, rgba(167,155,239,0) 42%),
        linear-gradient(160deg, #EEF3FF 0%, #E4E8FF 55%, #DCE2FB 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 6px 16px -6px rgba(40,25,110,0.25) !important;
}
.st-key-personal_card::after {
    content: "🧍";
    position: absolute;
    right: -18px;
    bottom: -28px;
    font-size: 8rem;
    opacity: 0.12;
    transform: rotate(-8deg);
    pointer-events: none;
}

.st-key-eating_card,
.st-key-eating_card > div[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    overflow: hidden;
    border-radius: 20px !important;
    border: 2px solid rgba(224, 130, 70, 0.4) !important;
    background:
        radial-gradient(circle at 90% 10%, rgba(255,178,120,0.5) 0%, rgba(255,178,120,0) 45%),
        radial-gradient(circle at 8% 90%, rgba(255,140,140,0.4) 0%, rgba(255,140,140,0) 42%),
        linear-gradient(160deg, #FFF4E8 0%, #FFEBDD 55%, #FFE0D6 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 6px 16px -6px rgba(40,25,110,0.25) !important;
}
.st-key-eating_card::after {
    content: "🍽️";
    position: absolute;
    right: -14px;
    bottom: -28px;
    font-size: 8rem;
    opacity: 0.12;
    transform: rotate(8deg);
    pointer-events: none;
}

.st-key-lifestyle_card,
.st-key-lifestyle_card > div[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    overflow: hidden;
    border-radius: 20px !important;
    border: 2px solid rgba(60, 165, 120, 0.4) !important;
    background:
        radial-gradient(circle at 90% 10%, rgba(120,222,170,0.5) 0%, rgba(120,222,170,0) 45%),
        radial-gradient(circle at 8% 90%, rgba(120,200,230,0.4) 0%, rgba(120,200,230,0) 42%),
        linear-gradient(160deg, #EAFBF1 0%, #E1F6EA 55%, #D9F1E6 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 6px 16px -6px rgba(40,25,110,0.25) !important;
}
.st-key-lifestyle_card::after {
    content: "🏃";
    position: absolute;
    right: -14px;
    bottom: -30px;
    font-size: 8rem;
    opacity: 0.12;
    transform: rotate(-6deg);
    pointer-events: none;
}

/* keep card headings/text above the decorative watermark */
.st-key-personal_card > div > div,
.st-key-eating_card > div > div,
.st-key-lifestyle_card > div > div {
    position: relative;
    z-index: 1;
}

/* a few extra twinkles scattered near the hero, matching the little
   floating fruit/veg accents in the reference screenshot */
.sparkle-layer {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 2;
}
.sparkle-layer span {
    position: absolute;
    font-size: 1.4rem;
    animation: sparkleTwinkle 2.4s ease-in-out infinite;
    filter: drop-shadow(0 2px 3px rgba(0,0,0,0.15));
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Falling-food background animation (burger / veg / water, snow-effect style)
# ----------------------------------------------------------------------

FOOD_EMOJIS = ["🍔", "🍔", "🥦", "🥕", "🍅", "💧", "💧"]


def render_food_rain(n_items: int = 26, seed: int = 7):
    """Continuous background animation: food emoji drift down the whole
    app the same way Streamlit's built-in st.snow() does — fixed-position
    layer, non-interactive (pointer-events: none), looping keyframe fall
    with per-item randomized size/duration/delay/horizontal drift/rotation.

    A fixed seed keeps the layout stable across Streamlit reruns (every
    widget interaction reruns the whole script) instead of the items
    jumping to new random spots on every click.
    """
    rng = random.Random(seed)
    items_html = []
    for i in range(n_items):
        emoji = rng.choice(FOOD_EMOJIS)
        left = rng.uniform(0, 100)                # vw position
        size = rng.uniform(1.4, 2.8)               # rem
        duration = rng.uniform(9, 18)               # seconds to fall
        delay = rng.uniform(-18, 0)                 # negative = already mid-fall on load
        drift = rng.uniform(-60, 60)                # px horizontal sway
        spin = rng.choice([1, -1]) * rng.uniform(180, 540)  # deg rotation over the fall
        items_html.append(
            f'<div class="food-item" style="'
            f'left:{left:.2f}vw; '
            f'font-size:{size:.2f}rem; '
            f'animation-duration:{duration:.2f}s; '
            f'animation-delay:{delay:.2f}s; '
            f'--drift:{drift:.1f}px; '
            f'--spin:{spin:.0f}deg;">{emoji}</div>'
        )

    st.markdown(
        f"""
        <style>
        @keyframes food-fall {{
            0%   {{ transform: translateY(-10vh) translateX(0) rotate(0deg); opacity: 0; }}
            8%   {{ opacity: 0.9; }}
            50%  {{ transform: translateY(50vh) translateX(var(--drift)) rotate(calc(var(--spin) * 0.5)); }}
            92%  {{ opacity: 0.9; }}
            100% {{ transform: translateY(110vh) translateX(0) rotate(var(--spin)); opacity: 0; }}
        }}
        .food-rain-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            pointer-events: none;
            z-index: 3;
        }}
        .food-item {{
            position: absolute;
            top: 0;
            will-change: transform, opacity;
            animation-name: food-fall;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
            filter: drop-shadow(0 2px 3px rgba(0,0,0,0.15));
        }}
        @media (prefers-reduced-motion: reduce) {{
            .food-rain-container {{ display: none; }}
        }}
        </style>
        <div class="food-rain-container">
            {''.join(items_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


render_food_rain()


def render_sparkles():
    """A handful of static, gently twinkling sparkle/star accents pinned to
    fixed spots on screen — cheap 'this app is alive' polish that doesn't
    compete with the falling-food layer or block any clicks underneath."""
    spots = [
        ("✨", 6, 18), ("✨", 92, 12), ("⭐", 88, 55),
        ("✨", 4, 62), ("⭐", 50, 6), ("✨", 96, 80),
    ]
    spans = "".join(
        f'<span style="left:{left}vw; top:{top}vh; animation-delay:{i * 0.4:.1f}s;">{icon}</span>'
        for i, (icon, left, top) in enumerate(spots)
    )
    st.markdown(f'<div class="sparkle-layer">{spans}</div>', unsafe_allow_html=True)


render_sparkles()


def hero(pill: str, title: str, caption: str):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="page-pill">{pill}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Bar-growth animation helper
# ----------------------------------------------------------------------

def _ease_out_back(t: float, punch: float = 1.4) -> float:
    """Ease-out-back: overshoots past the target then settles — this is
    what actually reads as 'shoot up' rather than a gentle slide-in.
    Returns exactly 1.0 at t=1.0 (no permanent offset), peaks a bit above
    1.0 partway through. Larger `punch` = more overshoot/bounce."""
    c3 = punch + 1
    t -= 1
    return 1 + c3 * (t ** 3) + punch * (t ** 2)


def animate_bar_growth(df: pd.DataFrame, value_col: str, chart_builder, key: str,
                        frames: int = 20, duration: float = 0.7,
                        start_value=0.0, punch: float = 1.4):
    """Fakes a 'bars shoot up' animation for an Altair bar chart.

    Altair/Vega-Lite (as rendered by st.altair_chart) has no built-in
    transition API for animating a mark's height, so this does it the
    way Streamlit apps normally fake it: redraw the chart into a single
    placeholder several times, moving `value_col` from `start_value` up
    to its real value on each frame, with an ease-out-BACK curve (a
    small overshoot past the final height before settling) so the
    motion is actually visible instead of a subtle fade.

    `start_value` can be 0 (grow from nothing — good for a 0-1 probability
    axis) or a baseline like the chart's y-domain minimum (good for a
    zoomed-in, non-zero axis, so bars still visibly shoot up from the
    bottom of the *visible* chart instead of from an invisible zero far
    below it). It can be a scalar (same baseline for every bar) or an
    array-like matching df's row order (a different baseline per bar).

    `chart_builder(frame_df) -> alt.Chart` must build its bar height
    encoding from `value_col`, and its y-scale domain must be wide enough
    to show the overshoot (chart_builder should compute that domain from
    the ORIGINAL final values, not from frame_df, so the axis doesn't
    itself jump around frame to frame). It should pull on-bar text labels
    and tooltips from `f"{value_col}_final"` instead, so the printed
    numbers stay put and only the bar height animates.
    """
    placeholder = st.empty()
    final_values = df[value_col].astype(float).copy()
    start_values = pd.Series(start_value, index=final_values.index, dtype=float)
    frame_df = df.copy()
    frame_df[f"{value_col}_final"] = final_values

    for i in range(1, frames + 1):
        t = i / frames
        eased = 1.0 if i == frames else _ease_out_back(t, punch=punch)
        frame_df[value_col] = start_values + (final_values - start_values) * eased
        placeholder.altair_chart(chart_builder(frame_df), use_container_width=True, key=key)
        if i < frames:
            time.sleep(duration / frames)


# ----------------------------------------------------------------------
# Model comparison charts (accuracy, precision, recall, F1, ROC AUC)
# ----------------------------------------------------------------------

def render_comparison_charts(df: pd.DataFrame):
    """Bar charts comparing all models across every metric in model_comparison.csv.

    Uses Altair (bundled with Streamlit, no extra dependency) instead of
    st.bar_chart so each model's bar can have its own distinct, consistent
    color — st.bar_chart only colors by column/series, not by category,
    so it can't do per-model colors on a single-metric chart.

    Expects df with a 'Model' column plus one or more numeric metric
    columns (e.g. Accuracy, Precision_Macro, Recall_Macro, F1_Macro,
    ROC_AUC_Macro), values already in percent (0-100).
    """
    metric_cols = [c for c in df.columns if c != "Model"]
    models_present = df["Model"].tolist()
    flat_color_scale = alt.Scale(
        domain=models_present,
        range=[MODEL_COLORS.get(m, "#94A3B8") for m in models_present],
    )

    def bar_chart_for(metric: str):
        y_scale = alt.Scale(zero=False)
        x_scale = alt.Scale(domain=models_present)

        # Vega-Lite only allows a gradient as a literal per-mark fill value,
        # not inside a scale's `range` array — so instead of one bar mark
        # colored by a category scale, we layer one bar mark PER model, each
        # with its own flat metallic-gradient fill applied directly.
        bar_layers = [
            alt.Chart(df[df["Model"] == m]).mark_bar(
                clip=False, cornerRadiusTopLeft=8, cornerRadiusTopRight=8,
                fill=metallic_gradient(MODEL_COLORS.get(m, "#94A3B8")),
                stroke="#FFFFFF", strokeWidth=1, strokeOpacity=0.35,
            ).encode(
                x=alt.X("Model:N", sort=models_present, scale=x_scale, title=None),
                y=alt.Y(f"{metric}:Q", title="%", scale=y_scale),
                tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".2f")],
            )
            for m in models_present
        ]

        labels = alt.Chart(df).mark_text(
            clip=False,
            align="center",
            baseline="bottom",
            dy=-8,
            fontSize=14,
            fontWeight="bold",
            color="#16324F",
        ).encode(
            x=alt.X("Model:N", sort=models_present, scale=x_scale),
            y=alt.Y(f"{metric}:Q", scale=y_scale),
            text=alt.Text(f"{metric}:Q", format=".2f"),
        )

        return alt.layer(*bar_layers, labels).properties(
            height=320,
            padding={"top": 25, "left": 5, "right": 5, "bottom": 5},
            background="transparent",
        ).configure_axis(
            labelColor="#16324F", titleColor="#16324F", labelFontWeight="bold",
            gridColor="rgba(22,50,79,0.15)", domainColor="rgba(22,50,79,0.3)",
        )

    metric_labels = [m.replace("_", " ") for m in metric_cols]
    selected_label = st.radio(
        "Metric", metric_labels, horizontal=True, key="cmp_metric_view", label_visibility="collapsed",
    )
    metric = metric_cols[metric_labels.index(selected_label)]

    st.caption(
        f"Range shown: {df[metric].min():.2f}% – {df[metric].max():.2f}% "
        "(zoomed in — not 0-100 — so small differences between models are visible)"
    )
    st.altair_chart(bar_chart_for(metric), use_container_width=True, key=f"cmp_chart_{metric}")

    with st.expander("Show all metrics side-by-side (grouped)"):
        melted = df.melt(id_vars="Model", value_vars=metric_cols,
                          var_name="Metric", value_name="Value")
        base = alt.Chart(melted).encode(
            x=alt.X("Model:N", sort=None, title=None),
            y=alt.Y("Value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
        )
        bars = base.mark_bar().encode(
            color=alt.Color("Model:N", scale=flat_color_scale, legend=alt.Legend(title="Model")),
            tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2f")],
        )
        labels = base.mark_text(
            align="center", baseline="bottom", dy=-2, fontSize=9,
        ).encode(
            text=alt.Text("Value:Q", format=".1f"),
        )
        layered = alt.layer(bars, labels).properties(height=280, width=120)
        grouped = layered.facet(column=alt.Column("Metric:N", title=None)).properties(
            background="transparent"
        ).configure_axis(
            labelColor="#16324F", titleColor="#16324F", labelFontWeight="bold",
            gridColor="rgba(22,50,79,0.15)", domainColor="rgba(22,50,79,0.3)",
        )
        st.altair_chart(grouped, use_container_width=False, key="cmp_chart_grouped")


# ----------------------------------------------------------------------
# Page: Exploratory Data Analysis (only built when this page is selected)
# ----------------------------------------------------------------------

def render_eda_page():
    hero(
        "Explore",
        "🔍 Exploratory Data Analysis",
        "Dataset: 2,111 survey responses (2,087 after removing 24 duplicates) on eating "
        "habits and physical condition, labeled with one of 7 obesity levels (NObeyesdad). "
        "Height, Weight, and BMI are shown here for context only — they're excluded from "
        "the model features to avoid leaking the answer into the prediction.",
    )

    with st.expander("📊 Target class distribution", expanded=True):
        st.markdown(
            "Classes are fairly balanced, ranging from **12.8%** "
            "(Insufficient Weight) to **16.8%** (Obesity Type I) of the "
            "dataset — so no class-imbalance correction (e.g. SMOTE) was "
            "needed before training."
        )
        st.image("eda_assets/fig1_target_distribution.png", use_container_width=True)

    with st.expander("📦 Boxplot: numeric feature spread & outliers"):
        st.markdown(
            "Shows each numeric feature's raw spread before cleaning. "
            "**Age** and **NCP** were deliberately left *out* of IQR capping — "
            "Age's higher values are real adult ages, not errors, and NCP is "
            "a near-discrete \"meals per day\" value where IQR flags its "
            "natural clustering as false outliers. The other four "
            "(FCVC, CH2O, FAF, TUE) were IQR-capped."
        )
        st.image("eda_assets/fig2_boxplot.png", use_container_width=True)

    with st.expander("📈 Distribution shape & skewness"):
        st.markdown(
            "Histogram + skewness score for each numeric feature. **Weight** "
            "and **BMI** are right-skewed (a longer tail toward higher "
            "values), which lines up with the dataset having more severe "
            "obesity classes than a general population sample would."
        )
        st.image("eda_assets/fig5_histograms.png", use_container_width=True)

    with st.expander("🧩 Pairwise relationships between key features"):
        st.markdown(
            "Age, BMI, FCVC (vegetable intake), FAF (physical activity), and "
            "CH2O (water intake), colored by obesity level. **BMI vs. Age** "
            "shows the clearest separation — higher obesity classes cluster "
            "at higher BMI across most ages, as expected."
        )
        st.image("eda_assets/fig3_pairplot.png", use_container_width=True)

    with st.expander("🔗 Correlation between numeric features"):
        st.markdown(
            "As expected, **Weight and BMI are strongly correlated** "
            "(BMI is derived from Weight and Height), which is exactly why "
            "both are excluded from the model's input features. Lifestyle "
            "features (FCVC, FAF, CH2O, TUE) show only weak correlation with "
            "each other, meaning they contribute fairly independent signal."
        )
        st.image("eda_assets/fig4_correlation.png", use_container_width=True)

    with st.expander("🧬 All attributes vs. obesity level"):
        st.markdown(
            "Categorical features (Gender, family history, MTRANS, etc.) as "
            "grouped counts; FCVC, NCP, CH2O, FAF, and TUE as density curves "
            "instead of counts, since those five are continuous "
            "(FCVC alone has 810 distinct values across 2,111 rows) rather "
            "than the small set of discrete answers they might look like. "
            "**Family history of overweight** shows the starkest split: it's "
            "heavily skewed toward \"yes\" in every obesity class above "
            "Normal Weight."
        )
        st.image("eda_assets/fig6_attributes_by_obesity.png", use_container_width=True)

    with st.expander("📋 Categorical features: plain frequency counts"):
        st.markdown(
            "Same 8 categorical features as above, but without splitting by "
            "obesity level — just how common each answer is overall. Most "
            "are heavily one-sided: **SMOKE** and **SCC** (calorie "
            "monitoring) are both over 95% \"no\", and **MTRANS** is "
            "dominated by Public Transportation, which is worth keeping in "
            "mind since these low-frequency categories give the model very "
            "few examples to learn from."
        )
        st.image("eda_assets/fig8_categorical_univariate.png", use_container_width=True)

    with st.expander("🌡️ Average feature values by obesity level"):
        st.markdown(
            "Mean value of each numeric feature per obesity class, ordered "
            "by severity. **Weight and BMI climb steadily** from Insufficient "
            "Weight through Obesity Type III, while lifestyle features like "
            "FAF (physical activity) trend gently downward — consistent with "
            "the behavioral story the model is trying to learn."
        )
        st.image("eda_assets/fig7_mean_heatmap.png", use_container_width=True)


# ----------------------------------------------------------------------
# Page: Model comparison (only built when this page is selected)
# ----------------------------------------------------------------------

def render_comparison_page(comparison_df: pd.DataFrame):
    hero(
        "Benchmark",
        "📊 Model Comparison",
        "All metrics computed once on the held-out test set (not from any single "
        "prediction). This is the basis for choosing which model performs best overall.",
    )
    render_comparison_charts(comparison_df)


# ----------------------------------------------------------------------
# Page: Predict
# ----------------------------------------------------------------------

def render_predict_page(model_choice, rf_model, knn_model, svm_model, ann_model,
                         real_scaler, ann_scaler, minmax_scaler, minmax_cols):
    hero(
        "Predict",
        "⚖️ Obesity Level Predictor",
        "Predicts obesity category from eating habits and physical condition, using 4 "
        "trained models (Random Forest, KNN (baseline), SVM, ANN).",
    )

    with st.expander("ℹ️ About this app's inputs"):
        st.markdown(
            "Numeric inputs (Age, FCVC, NCP, CH2O, FAF, TUE) are scaled using the "
            "**actual fitted MinMaxScaler** from training (`obesity_scaler.joblib`), "
            "not guessed ranges. Categorical/ordinal encodings in the `CONFIG` "
            "block match the mappings used in the cleaning notebook directly. "
            "The SVM and ANN models each additionally apply their own fitted "
            "`StandardScaler` (`svm_scaler.pkl` / `ann_scaler.pkl`) on top of "
            "the full feature vector, exactly as done during training."
        )

    st.markdown("### Enter your information")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True, key="personal_card"):
            st.markdown("#### 🧍 Personal")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 14, 61, 25)
            family_history = st.selectbox("Family history of overweight?", ["no", "yes"])
            smoke = st.selectbox("Do you smoke?", ["no", "yes"])
            st.caption(
                "Height & weight (optional) — used only to show your BMI for "
                "reference. They are **not** sent to the prediction model."
            )
            height_m = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)

    with col2:
        with st.container(border=True, key="eating_card"):
            st.markdown("#### 🍽️ Eating habits")
            favc = st.selectbox("Frequent consumption of high-caloric food?", ["yes", "no"])
            fcvc = st.slider("Vegetable consumption frequency (1=never, 3=always)", 1.0, 3.0, 2.0, 0.1)
            ncp = st.slider("Number of main meals per day", 1.0, 4.0, 3.0, 0.5)
            caec = st.selectbox("Eating food between meals", ["no", "Sometimes", "Frequently", "Always"])
            calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
            scc = st.selectbox("Do you monitor calorie intake?", ["no", "yes"])

    with col3:
        with st.container(border=True, key="lifestyle_card"):
            st.markdown("#### 🏃 Lifestyle")
            ch2o = st.slider("Daily water intake (liters)", 1.0, 3.0, 2.0, 0.1)
            faf = st.slider("Physical activity frequency (days/week, 0-3)", 0.0, 3.0, 1.0, 0.5)
            tue = st.slider("Time using technology devices (0=low, 2=high)", 0.0, 2.0, 1.0, 0.1)
            mtrans = st.selectbox("Main mode of transportation", MTRANS_OPTIONS)

    inputs = {
        "Gender": gender,
        "Age": age,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
    }

    # Signature of everything the results section depends on (model inputs +
    # height/weight for the BMI card), so we can tell "user just changed
    # something" apart from "user just clicked Predict" or "switched which
    # model's tab they're viewing".
    current_signature = {**inputs, "height_m": height_m, "weight_kg": weight_kg}

    if (st.session_state.get("show_prediction")
            and st.session_state.get("last_predicted_signature") != current_signature):
        # An input changed since the last prediction — hide the stale result
        # immediately rather than leaving it on screen (or silently updating
        # it) until the user clicks Predict again.
        st.session_state["show_prediction"] = False

    st.divider()

    if st.button("🔍 Predict obesity level", type="primary", use_container_width=True):
        st.session_state["show_prediction"] = True
        st.session_state["last_predicted_signature"] = current_signature

    if st.session_state.get("show_prediction"):
        # --- BMI reference (independent of the model, calculated directly) ---
        bmi = weight_kg / (height_m ** 2)
        bmi_label = bmi_to_label(bmi)
        bmi_color = LABEL_COLORS.get(bmi_label, "#666")

        st.subheader("📏 Your BMI (direct calculation)")
        bcol1, bcol2 = st.columns([1, 2])
        with bcol1:
            st.metric("BMI", f"{bmi:.1f}")
        with bcol2:
            st.markdown(
                f"Category: <span style='color:{bmi_color}; font-weight:600'>"
                f"{bmi_label.replace('_', ' ')}</span>",
                unsafe_allow_html=True,
            )
        st.caption(
            "This is a direct BMI formula result, not a model prediction. "
            "Compare it to the estimate below — they answer different "
            "questions: this is 'what BMI category are you in right now', "
            "the model below is 'what obesity category does your current "
            "lifestyle pattern-match to' (both describe your PRESENT state, "
            "not a future forecast)."
        )
        st.divider()

        X = build_feature_vector(inputs, minmax_scaler, minmax_cols)
        X_values = X.values.astype(float)

        def predict_proba(model_name):
            if model_name == "Random Forest":
                return rf_model.predict_proba(X_values)[0]

            if model_name == "KNN (Baseline)":
                return knn_model.predict_proba(X_values)[0]

            if model_name == "SVM":
                X_std = real_scaler.transform(X_values)
                return svm_model.predict_proba(X_std)[0]

            if model_name == "ANN":
                X_std = ann_scaler.transform(X_values)
                return ann_model.predict(X_std, verbose=0)[0]
            raise ValueError(model_name)

        def show_result(model_name, proba):
            pred_idx = int(np.argmax(proba))
            pred_label = LABEL_MAP[pred_idx]
            confidence = float(proba[pred_idx]) * 100
            color = LABEL_COLORS.get(pred_label, "#666")

            st.markdown(
                f"### {model_name} prediction: "
                f"<span style='color:{color}'>{pred_label.replace('_',' ')}</span> "
                f"({confidence:.1f}% confidence)",
                unsafe_allow_html=True,
            )

            # 1. Create the DataFrame in the specific order of LABEL_MAP
            categories = [LABEL_MAP[i].replace("_", " ") for i in range(len(proba))]
            category_colors = [LABEL_COLORS[LABEL_MAP[i]] for i in range(len(proba))]

            proba_df = pd.DataFrame({
                "Category": categories,
                "Probability": proba,
            })

            x_scale = alt.Scale(domain=categories)

            # 2. Use Altair for the chart to force the categorical order and use custom colors.
            #    Bar height comes from "Probability" (which animate_bar_growth scales
            #    from 0 up to the real value frame by frame); text/tooltip read from
            #    "Probability_final" so the printed numbers don't flicker mid-animation.
            #    Vega-Lite only allows a gradient as a literal per-mark fill value, not
            #    inside a scale's `range` array, so each category gets its own layer
            #    with a flat metallic-gradient fill applied directly instead of one bar
            #    mark colored through a shared category scale.
            def build_proba_chart(frame_df):
                y_scale = alt.Scale(domain=[0, 1.08])

                bar_layers = [
                    alt.Chart(frame_df[frame_df["Category"] == cat]).mark_bar(
                        clip=False, cornerRadiusTopLeft=8, cornerRadiusTopRight=8,
                        fill=metallic_gradient(hex_color),
                        stroke="#FFFFFF", strokeWidth=1, strokeOpacity=0.35,
                    ).encode(
                        x=alt.X("Category:N", sort=categories, scale=x_scale, title="Obesity Level"),
                        y=alt.Y("Probability:Q", title="Probability", scale=y_scale),
                        tooltip=["Category", alt.Tooltip("Probability_final:Q", format=".2%")],
                    )
                    for cat, hex_color in zip(categories, category_colors)
                ]

                labels = alt.Chart(frame_df).mark_text(
                    clip=False,
                    align="center",
                    baseline="bottom",
                    dy=-8,
                    fontSize=13,
                    fontWeight="bold",
                    color="#16324F",
                ).encode(
                    x=alt.X("Category:N", sort=categories, scale=x_scale),
                    y=alt.Y("Probability:Q", scale=y_scale),
                    text=alt.Text("Probability_final:Q", format=".1%"),
                )

                return alt.layer(*bar_layers, labels).properties(
                    height=350,
                    padding={"top": 20, "left": 5, "right": 5, "bottom": 5},
                    background="transparent",
                ).configure_axis(
                    labelColor="#16324F", titleColor="#16324F", labelFontWeight="bold",
                    gridColor="rgba(22,50,79,0.15)", domainColor="rgba(22,50,79,0.3)",
                )

            animate_bar_growth(proba_df, "Probability", build_proba_chart, key=f"proba_chart_{model_name}")

        st.subheader("🧠 Predicted current obesity category (from your habits)")
        st.caption(
            "This is the model's estimate of which obesity category best "
            "matches your CURRENT lifestyle habits — it is not a future "
            "forecast or a risk score. It can differ from your BMI category "
            "above because it's inferred indirectly from behaviour patterns, "
            "not calculated from your actual height/weight."
        )

        if model_choice == "Compare all 4":
            compare_names = ["Random Forest", "SVM", "KNN (Baseline)", "ANN"]
            selected_compare_model = st.radio(
                "View prediction for:", compare_names, horizontal=True, key="compare_model_view",
            )
            show_result(selected_compare_model, predict_proba(selected_compare_model))
        else:
            show_result(model_choice, predict_proba(model_choice))

        with st.expander("See the encoded feature vector sent to the model"):
            X_display = X.copy()
            X_display.insert(
                X_display.columns.get_loc("MTRANS_Bike"),
                "MTRANS_Automobile",
                inputs["MTRANS"] == "Automobile",
            )
            st.dataframe(X_display, use_container_width=True)


# ----------------------------------------------------------------------
# Sidebar navigation — pages only render their content when selected
# ----------------------------------------------------------------------

st.sidebar.markdown(
    '## <span class="sidebar-logo-icon">⚖️</span> Obesity Predictor',
    unsafe_allow_html=True,
)
st.sidebar.caption("Navigate between sections")

page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict", "📊 Model Comparison", "🔍 Exploratory Data Analysis"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.divider()

comparison_df = load_comparison()

if page == "🔮 Predict":
    st.sidebar.header("Model selection")
    model_choice = st.sidebar.radio(
        "Which model should make the prediction?",
        ["Random Forest", "SVM", "KNN (Baseline)", "ANN", "Compare all 4"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.subheader("Model performance (test set)")
    st.sidebar.dataframe(comparison_df.set_index("Model"), use_container_width=True)

    rf_model, knn_model, svm_model, ann_model = load_models()
    real_scaler = load_svm_scaler()               # SVM's own fitted StandardScaler (SVM input only)
    ann_scaler = load_ann_scaler()                 # ANN's own fitted StandardScaler (ANN input only)
    minmax_scaler, minmax_cols = load_scaler()     # actual training-time MinMaxScaler (numeric inputs)

    render_predict_page(
        model_choice, rf_model, knn_model, svm_model, ann_model,
        real_scaler, ann_scaler, minmax_scaler, minmax_cols,
    )

elif page == "📊 Model Comparison":
    render_comparison_page(comparison_df)

else:
    render_eda_page()
