# ═══════════════════════════════════════════════════════════════════════════════
#  MMM PRO — Marketing Mix Modeling  |  Budget Optimizer + What-If Simulator
#  Pharma Commercial Analytics  |  Power & Hill Response Curves
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Optional, Tuple, List
import re as _re, time , hashlib as _hashlib

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.optimize import curve_fit
    SCIPY = True
except Exception:
    SCIPY = False

try:
    from gekko import GEKKO
    GEKKO_OK = True
except Exception:
    GEKKO_OK = False


try:
    import google.generativeai as _genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False


# ─────────────────────────────────────────────────────────────
# AI CALL PROXY (safe forward reference)
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
        "You are a senior Marketing Mix Modeling (MMM) analyst specialising in pharma commercial analytics. "
        "Be concise, analytical, and structured. Use bullet points. "
        "Always reference specific numbers from the data provided. "
        "Avoid generic statements — every observation must be tied to a specific metric."
    )

_PREFERRED_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]

def call_ai(prompt: str) -> str:
    """
    Proxy for the actual AI engine defined later in the file.
    Allows AI calls from earlier tabs (e.g., Response Curves).
    """
    if "_call_ai_internal" not in globals():
        return (
            "⚠️ AI engine is not initialized yet.\n\n"
            "Please visit the AI Interpretation tab once and try again."
        )
    return _call_ai_internal(prompt)

def _extract_retry_seconds(err_text, default=30):
    try:
        m = _re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", err_text)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return default


def _prompt_hash(model_name, prompt):
    return _hashlib.sha256((model_name + "||" + prompt).encode()).hexdigest()


def _pick_gemini_model(key, preferred):
    if not _GENAI_OK:
        return preferred[0], "Install: pip install google-generativeai"
    try:
        _genai.configure(api_key=key)
        models = list(_genai.list_models())
        available = {
            m.name
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        }
        for name in preferred:
            if name in available or ("models/" + name) in available:
                return name, None
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name.replace("models/", ""), None
    except Exception as e:
        return preferred[0], str(e)

    return preferred[0], None


def _call_gemini(full_prompt, key, preferred, max_retries=1):
    if not _GENAI_OK:
        return None, "Install google-generativeai: pip install google-generativeai"

    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}

    model_name, warn = _pick_gemini_model(key, preferred)
    cache_key = _prompt_hash(model_name, full_prompt)

    if cache_key in st.session_state["ai_cache"]:
        return st.session_state["ai_cache"][cache_key], None

    try:
        _genai.configure(api_key=key)
        gm = _genai.GenerativeModel(model_name)
    except Exception as e:
        return None, f"Gemini init error: {e}"

    attempt = 0
    while attempt <= max_retries:
        try:
            resp = gm.generate_content(full_prompt)
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                st.session_state["ai_cache"][cache_key] = text
                return text, None
            return None, "Empty response from Gemini"
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries:
                wait = _extract_retry_seconds(err)
                time.sleep(wait)
                attempt += 1
            else:
                return None, f"Gemini error: {err}"

    return None, "Max retries exceeded"


def _call_groq(prompt, key, model):
    import requests
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1200,
                "temperature": 0.3,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return None, f"Groq error: {detail or str(e)}"



def _call_ai_internal(prompt):
        key      = (st.session_state["ai_key"] or "").strip()
        provider = st.session_state.get("ai_provider", "")
        full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
        if not key:
            return ("Enter your API key in the sidebar.\n\n"
                    "- Google Gemini (free): https://aistudio.google.com/app/apikey\n"
                    "- Groq (free): https://console.groq.com")
        if provider == "Google Gemini (free)":
            text, err = _call_gemini(full_prompt, key, _PREFERRED_MODELS)
            return text if text else f"AI error: {err}"
        else:
            # Retrieve model selection from session state (set in AI tab sidebar)
            groq_model = st.session_state.get("ai_model_groq", "llama-3.3-70b-versatile")
            text, err = _call_groq(prompt, key, groq_model)
            return text if text else f"AI error: {err}"

# ─── SHARED AI BUTTON HELPER ──────────────────────────────────────────────────

def render_ai_button(prompt: str, btn_key: str, btn_label: str = "🤖 Interpret using AI"):
    """
    Renders an AI interpret button + persistent output + follow-up Q&A chat.

    Fix: result display and chat are driven by session_state cache, NOT by
    the button click state. This prevents the reset-on-rerun bug where
    typing a follow-up question caused clicked=False and wiped the output.
    """
    cache_k    = f"ai_btn_cache_{btn_key}"
    chat_hist_k = f"ai_chat_{btn_key}"

    # ── Button row: Interpret button + Clear button side by side ─────────────
    st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)
    has_result = cache_k in st.session_state

    if has_result:
        # Show Interpret + Clear buttons side by side once result exists
        btn_l, btn_c, btn_r = st.columns([0.35, 0.30, 0.35])
        with btn_l:
            clicked = st.button(btn_label, key=btn_key, type="primary", width="stretch")
        with btn_c:
            if st.button("💬 Ask AI a Question", key=f"scroll_chat_{btn_key}",
                         width="stretch"):
                # No-op — just scrolls user attention; chat is always shown below
                pass
        with btn_r:
            if st.button("🗑 Clear", key=f"clear_{btn_key}", width="stretch"):
                del st.session_state[cache_k]
                if chat_hist_k in st.session_state:
                    del st.session_state[chat_hist_k]
                st.rerun()
    else:
        btn_l, _, btn_r = st.columns([0.30, 0.40, 0.30])
        with btn_l:
            clicked = st.button(btn_label, key=btn_key, type="primary", width="stretch")

    # ── On click: generate and cache the AI response ─────────────────────────
    if clicked:
        if cache_k not in st.session_state:
            with st.spinner("Generating AI interpretation…"):
                st.session_state[cache_k] = call_ai(prompt)
        st.rerun()   # rerun so the result block below always renders cleanly

    # ── Always render result if cached (NOT gated on clicked) ────────────────
    if cache_k in st.session_state:
        result = st.session_state[cache_k]
        st.markdown("---")
        st.markdown("#### 🤖 AI Interpretation")
        st.markdown(
            f"<div style='padding:1rem 1.25rem;border-radius:10px;"
            f"background:#FAFAFA;border:1px solid #E5E7EB;margin-top:.5rem'>"
            f"{result}</div>",
            unsafe_allow_html=True
        )

        # ── Persistent Q&A follow-up chat ─────────────────────────────────────
        if chat_hist_k not in st.session_state:
            st.session_state[chat_hist_k] = []

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.markdown("##### 💬 Ask a follow-up question")
        st.caption("e.g. *What happens if I cut 15% from HCP PLD?* · *Which channel has best ROI headroom?*")

        # Render existing chat history
        for turn in st.session_state[chat_hist_k]:
            with st.chat_message("user"):      st.write(turn["q"])
            with st.chat_message("assistant"): st.markdown(turn["a"])

        # Chat input — always rendered while result is cached so it never disappears
        user_q = st.chat_input("Your question…", key=f"chat_input_{btn_key}")
        if user_q:
            qa_prompt = (
                f"ANALYSIS CONTEXT:\n{prompt}\n\n"
                f"INTERPRETATION GIVEN:\n{result}\n\n"
                f"USER QUESTION: {user_q}\n\n"
                f"Answer in 150-250 words. Use specific numbers from the context. "
                f"For hypotheticals (e.g. cut X%), estimate the impact. "
                f"End with one clear recommendation."
            )
            with st.chat_message("user"):      st.write(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    ans = call_ai(qa_prompt)
                st.markdown(ans)
            st.session_state[chat_hist_k].append({"q": user_q, "a": ans})
            st.rerun()   # rerun to append answer to history cleanly


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MMM Pro · Budget Optimizer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DESIGN SYSTEM — Clean Professional Light Theme ───────────────────────────
PALETTE = {
    "bg":        "#F7F8FA",
    "surface":   "#FFFFFF",
    "surface2":  "#F0F2F5",
    "border":    "#E2E5EA",
    "border2":   "#CBD0D8",
    "accent":    "#1A56DB",   # deep blue — primary action
    "accent_lt": "#EBF0FD",   # light blue tint
    "accent2":   "#057A55",   # green — positive
    "accent2_lt":"#ECFDF3",
    "accent3":   "#C81E1E",   # red — negative / warning
    "accent3_lt":"#FEF2F2",
    "accent4":   "#6C2BD9",   # purple — Hill / secondary
    "accent4_lt":"#F5F0FF",
    "text":      "#111928",
    "text2":     "#374151",
    "muted":     "#6B7280",
    "muted2":    "#9CA3AF",
    "gold":      "#B45309",
    "gold_lt":   "#FFFBEB",
    "sidebar_bg":"#1E2433",   # dark navy sidebar — contrast anchor
    "sidebar_tx":"#E5E9F2",
    "sidebar_mu":"#8B95A8",
    "sidebar_br":"#2D3548",
}

CHANNEL_COLORS = [
    "#1A56DB","#057A55","#9C27B0","#E65100",
    "#0277BD","#2E7D32","#AD1457","#00838F",
    "#6A1B9A","#558B2F","#D84315","#00695C",
]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*='css'] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background: {PALETTE['bg']} !important;
    color: {PALETTE['text']} !important;
}}
.stApp {{ background: {PALETTE['bg']} !important; }}
/* Ensure content clears the fixed top toolbar */
div[data-testid='stDecoration'] {{ display: none !important; }}
header {{ background: rgba(247,248,250,0.95) !important; backdrop-filter: blur(8px); border-bottom: 1px solid {PALETTE['border']} !important; }}
.block-container {{ padding: 2.5rem 1.75rem 3rem !important; max-width: 1560px; }}

/* ── Sidebar: dark navy ── */
[data-testid="stSidebar"] {{
    background: {PALETTE['sidebar_bg']} !important;
    border-right: 1px solid {PALETTE['sidebar_br']};
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{ color: {PALETTE['sidebar_tx']} !important; }}
[data-testid="stSidebar"] .stMarkdown {{ color: {PALETTE['sidebar_mu']} !important; }}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {{
    background: {PALETTE['sidebar_br']} !important;
    border: 1.5px dashed #3D4C68 !important;
    border-radius: 10px !important;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {PALETTE['surface']} !important;
    border: 1.5px solid {PALETTE['border']} !important;
    border-radius: 14px !important;
    padding: .9rem 1.1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.05) !important;
    transition: box-shadow .18s, border-color .18s !important;
}}
[data-testid="stMetric"]:hover {{
    border-color: {PALETTE['accent']} !important;
    box-shadow: 0 3px 12px rgba(26,86,219,.12) !important;
}}
[data-testid="stMetricLabel"] {{
    color: {PALETTE['muted']} !important;
    font-size: .7rem !important;
    font-weight: 600 !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] {{
    color: {PALETTE['text']} !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -.02em !important;
}}
[data-testid="stMetricDelta"] {{ font-size: .75rem !important; font-weight: 500 !important; }}
[data-testid="stMetricDelta"] svg {{ display: none !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {PALETTE['surface']} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1.5px solid {PALETTE['border']} !important;
    gap: 2px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px !important;
    color: {PALETTE['muted']} !important;
    font-weight: 500 !important;
    font-size: .84rem !important;
    padding: .38rem 1.05rem !important;
    letter-spacing: .01em !important;
}}
.stTabs [aria-selected="true"] {{
    background: {PALETTE['accent']} !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(26,86,219,.3) !important;
}}

/* ── Primary buttons — exclude file uploader browse button ── */
.stButton > button {{
    background: {PALETTE['accent']} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: .86rem !important;
    padding: .48rem 1.2rem !important;
    letter-spacing: .01em !important;
    box-shadow: 0 2px 6px rgba(26,86,219,.22) !important;
    transition: all .18s !important;
}}
.stButton > button:hover {{
    background: #1648C0 !important;
    box-shadow: 0 4px 14px rgba(26,86,219,.35) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button[kind="secondary"] {{
    background: {PALETTE['surface']} !important;
    border: 1.5px solid {PALETTE['border']} !important;
    color: {PALETTE['text2']} !important;
    box-shadow: none !important;
}}
/* Browse files button — orange, overrides global blue button rule */
[data-testid="stFileUploadDropzone"] button {{
    background-color: #CC5500 !important;
    background: #CC5500 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transform: none !important;
}}
[data-testid="stFileUploadDropzone"] button:hover {{
    background-color: #A84300 !important;
    background: #A84300 !important;
    color: #ffffff !important;
    box-shadow: none !important;
    transform: none !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1.5px solid {PALETTE['border']} !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.05) !important;
}}

/* ── Select / dropdowns ── */
[data-baseweb="select"] > div {{
    background: {PALETTE['surface']} !important;
    border-color: {PALETTE['border']} !important;
    border-radius: 9px !important;
    color: {PALETTE['text']} !important;
    font-size: .85rem !important;
}}
[data-baseweb="popover"] {{
    background: {PALETTE['surface']} !important;
    border: 1.5px solid {PALETTE['border']} !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.1) !important;
}}

/* ── Sidebar select boxes override ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background: #2D3548 !important;
    border-color: #3D4C68 !important;
    color: {PALETTE['sidebar_tx']} !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: {PALETTE['surface']} !important;
    border: 1.5px solid {PALETTE['border']} !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
}}
[data-testid="stExpander"] summary {{
    font-size: .83rem !important;
    font-weight: 500 !important;
    color: {PALETTE['text2']} !important;
}}

/* ── Sidebar expander ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: #2D3548 !important;
    border-color: #3D4C68 !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    color: {PALETTE['sidebar_tx']} !important;
}}

/* ── Sliders ── */
[data-testid="stSlider"] {{ padding: .15rem 0; }}
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stThumbValue"] {{
    color: {PALETTE['sidebar_tx']} !important;
}}

/* ── Number inputs ── */
[data-baseweb="input"] {{
    background: {PALETTE['surface']} !important;
    border-color: {PALETTE['border']} !important;
    border-radius: 9px !important;
    font-size: .85rem !important;
}}
[data-testid="stSidebar"] [data-baseweb="input"] {{
    background: #2D3548 !important;
    border-color: #3D4C68 !important;
    color: {PALETTE['sidebar_tx']} !important;
}}

/* ── Radio ── */
[data-baseweb="radio"] span {{ color: {PALETTE['text2']} !important; font-size: .85rem !important; }}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {{
    background: {PALETTE['surface']} !important;
    border: 1.5px solid {PALETTE['border']} !important;
    color: {PALETTE['text2']} !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: .82rem !important;
}}

/* ── Checkbox ── */
[data-testid="stCheckbox"] span {{ color: {PALETTE['sidebar_tx']} !important; font-size: .83rem !important; }}

/* ── Form labels ── */
div[data-testid="stSelectbox"] > label,
div[data-testid="stMultiselect"] > label,
div[data-testid="stSlider"] > label,
div[data-testid="stNumberInput"] > label {{
    color: {PALETTE['sidebar_mu']} !important;
    font-size: .68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
}}

/* ── Headings ── */
h1 {{ font-weight: 800 !important; letter-spacing: -.03em !important; color: {PALETTE['text']} !important; }}
h2 {{ font-weight: 700 !important; color: {PALETTE['text']} !important; }}
hr {{ border-color: {PALETTE['border']} !important; margin: 1rem 0 !important; }}

/* ── Info/alert boxes ── */
[data-testid="stInfo"]  {{ border-radius: 10px !important; }}
[data-testid="stSuccess"] {{ border-radius: 10px !important; }}
[data-testid="stWarning"] {{ border-radius: 10px !important; }}
[data-testid="stError"]   {{ border-radius: 10px !important; }}

/* ── Custom component classes ── */
.section-header {{
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: {PALETTE['muted']};
    padding: .4rem 0 .65rem;
    border-bottom: 1.5px solid {PALETTE['border']};
    margin-bottom: .9rem;
}}

.chip {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: .69rem;
    font-weight: 600;
    border: 1px solid;
    line-height: 1.6;
}}
.chip-blue   {{ background: {PALETTE['accent_lt']};  color: {PALETTE['accent']};  border-color: #BFCFFA; }}
.chip-green  {{ background: {PALETTE['accent2_lt']}; color: {PALETTE['accent2']}; border-color: #A7F3D0; }}
.chip-red    {{ background: {PALETTE['accent3_lt']}; color: {PALETTE['accent3']}; border-color: #FCA5A5; }}
.chip-gold   {{ background: {PALETTE['gold_lt']};    color: {PALETTE['gold']};    border-color: #FCD34D; }}
.chip-purple {{ background: {PALETTE['accent4_lt']}; color: {PALETTE['accent4']}; border-color: #C4B5FD; }}

.alert-box {{
    border-radius: 10px;
    padding: .65rem .9rem;
    margin: .45rem 0;
    font-size: .83rem;
    font-weight: 500;
    border-left: 3px solid;
}}
.alert-info    {{ background: {PALETTE['accent_lt']};  border-color: {PALETTE['accent']};  color: #1E40AF; }}
.alert-success {{ background: {PALETTE['accent2_lt']}; border-color: {PALETTE['accent2']}; color: #065F46; }}
.alert-warn    {{ background: {PALETTE['gold_lt']};    border-color: {PALETTE['gold']};    color: {PALETTE['gold']}; }}

/* Channel card in RC tab */
.ch-card {{
    background: {PALETTE['surface']};
    border: 1.5px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}}
.ch-card-label {{ color: {PALETTE['muted']}; font-size: .65rem; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; }}
.ch-card-val   {{ color: {PALETTE['text']};  font-size: 1.05rem; font-weight: 700; margin-top: .1rem; }}

/* KPI strip */
.kpi-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px,1fr));
    gap: .6rem;
    margin: .5rem 0 1rem;
}}
.kpi-item {{
    background: {PALETTE['surface']};
    border: 1.5px solid {PALETTE['border']};
    border-radius: 12px;
    padding: .75rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
    transition: border-color .15s;
}}
.kpi-item:hover {{ border-color: {PALETTE['accent']}; }}
.kpi-lbl {{ color: {PALETTE['muted']}; font-size: .63rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }}
.kpi-val {{ color: {PALETTE['text']};  font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; margin-top: .15rem; }}
.kpi-delta {{ font-size: .72rem; font-weight: 500; margin-top: .15rem; }}
.kpi-delta-pos {{ color: {PALETTE['accent2']}; }}
.kpi-delta-neg {{ color: {PALETTE['accent3']}; }}
.kpi-delta-neu {{ color: {PALETTE['muted2']}; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & REQUIRED SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED = [
    'channel', 'total_activity', 'total_spend', 'total_sales', 'coefficient',
    'type_transformation', 'alpha', 'lock_spend', 'lower_bound_pct',
    'upper_bound_pct', 'total_segments', 'net_per_unit'
]

EPS = 1e-9

def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Convert #RRGGBB hex to rgba() string — Plotly does not accept 8-digit hex."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE CURVE ENGINE (Power / Log / Hill)
# ═══════════════════════════════════════════════════════════════════════════════

def power_impact(aps: float, coef: float, alpha: float, segments: float, npu: float) -> float:
    return float(coef) * (max(float(aps), 0.0) ** float(alpha)) * float(segments) * float(npu)

def log_impact(aps: float, coef: float, alpha: float, segments: float, npu: float) -> float:
    return float(coef) * np.log1p(max(float(aps), 0.0) ** float(alpha)) * float(segments) * float(npu)

def hill_impact(aps: float, ec50: float, slope: float, max_resp: float) -> float:
    aps = max(float(aps), 0.0)
    ec50 = max(float(ec50), EPS)
    slope = max(float(slope), EPS)
    return float(max_resp) * (aps ** slope) / (ec50 ** slope + aps ** slope)

def unadj_impact(aps, coef, alpha, segments, npu, tform, hill_params=None):
    t = str(tform).strip().lower()
    if t == 'hill' and hill_params is not None:
        return hill_impact(aps, *hill_params)
    elif t == 'log':
        return log_impact(aps, coef, alpha, segments, npu)
    else:
        return power_impact(aps, coef, alpha, segments, npu)

def revenue_from_spend(spend: float, row: pd.Series,
                        use_hill: bool = False, use_log: bool = False) -> float:
    """
    Revenue calculation supporting Power / Log / Hill curves.
    Priority: Hill (if use_hill + params) > Log (if use_log) > type_transformation > Power
    """
    spend = max(float(spend), 0.0)
    cpm  = float(row.get('cost_per_mention', 0))
    segs = float(row['total_segments'])
    if cpm <= 0 or segs <= 0:
        return 0.0
    aps   = spend / (cpm * segs)
    tform = row.get('type_transformation', 'power')
    hill_p = None
    if use_hill and pd.notnull(row.get('hill_ec50', np.nan)):
        hill_p = (float(row['hill_ec50']), float(row['hill_slope']), float(row['hill_max_response']))
        tform  = 'hill'
    elif use_log:
        tform = 'log'
    unadj = unadj_impact(aps, float(row['coefficient']), float(row['alpha']),
                          segs, float(row['net_per_unit']), tform, hill_p)
    adj = float(row.get('Adj_Factor', 1.0))
    return adj * unadj

def mroi(spend: float, row: pd.Series, use_hill: bool = False,
           use_log: bool = False, h_rel: float = 1e-4) -> float:
    """Instantaneous marginal ROI (derivative-based). Matches Excel mROI closely."""
    h = max(abs(spend) * h_rel, 1.0)
    return (revenue_from_spend(spend + h, row, use_hill, use_log) -
            revenue_from_spend(spend - h, row, use_hill, use_log)) / (2 * h)

def mroi_excel_style(spend: float, row: pd.Series, use_hill: bool = False) -> float:
    """Excel-style mROI: secant slope between (spend - delta_sp) and spend,
    where delta_sp corresponds to Δaps = 0.2 (matches Excel's =POWER(aps-0.2,...) formula).
    Use this for display comparison with Excel solver output."""
    cpm  = float(row.get('cost_per_mention', 0))
    segs = float(row['total_segments'])
    if cpm <= 0 or segs <= 0:
        return 0.0
    # Δaps = 0.2 → Δactivity = 0.2 * segs → Δspend = 0.2 * segs / cpm
    delta_sp = max(0.2 * segs / cpm if cpm > 0 else spend * 0.01, 1.0)
    sp_lo = max(spend - delta_sp, 0.01)
    rev_hi = revenue_from_spend(spend,   row, use_hill)
    rev_lo = revenue_from_spend(sp_lo,   row, use_hill)
    d_sp   = spend - sp_lo
    return (rev_hi - rev_lo) / d_sp if d_sp > 0 else 0.0

@st.cache_data(show_spinner=False)
def simulate_power_xy(aps: float, alpha: float, coef: float, npu: float, segs: float,
                       steps: int = 500, mult: float = 2.0):
    x = np.linspace(0.01, max(aps, 0.01) * mult, steps)
    y = coef * (x ** alpha) * npu * segs
    return x, y

@st.cache_data(show_spinner=False)
def fit_hill(x_arr, y_arr):
    if not SCIPY:
        return (np.nan, np.nan, np.nan)
    def hill_fn(x, ec50, slope, max_r):
        ec50, slope = max(ec50, EPS), max(slope, EPS)
        return max_r * (x ** slope) / (ec50 ** slope + x ** slope)
    try:
        p0 = [float(np.mean(x_arr)), 1.0, float(np.max(y_arr))]
        bounds = ([EPS, 1e-6, EPS], [np.inf, 10.0, np.inf])
        params, _ = curve_fit(hill_fn, x_arr, y_arr, p0=p0, bounds=bounds, maxfev=30000)
        return tuple(float(p) for p in params)
    except Exception:
        return (np.nan, np.nan, np.nan)

@st.cache_data(show_spinner=False)
def compute_all_hill_fits(df_hash: str, channel_col, aps_col, alpha_col, coef_col, npu_col, segs_col):
    """Cache key uses hash; actual data passed as columns"""
    return None  # placeholder, actual logic in main

def _compute_hill_for_df(df: pd.DataFrame):
    results = []
    for _, r in df.iterrows():
        aps = float(r['activity_per_segment'])
        x, y = simulate_power_xy(aps, float(r['alpha']), float(r['coefficient']),
                                  float(r['net_per_unit']), float(r['total_segments']))
        ec50, slope, max_r = fit_hill(tuple(x), tuple(y))
        results.append({'channel': r['channel'],
                        'hill_ec50': ec50, 'hill_slope': slope, 'hill_max_response': max_r})
    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_bounds_spend(row: pd.Series):
    """Spend bounds for a channel — lock means fixed at baseline."""
    base = float(row['total_spend'])
    if int(row.get('lock_spend', 0)) == 1:
        return (base, base)
    lo = float(row.get('lower_bound_pct', 0.5)) * base
    hi = float(row.get('upper_bound_pct', 1.5)) * base
    if lo > hi: lo, hi = hi, lo
    return (max(lo, 0.0), hi)

def get_bounds_aps(row: pd.Series):
    """
    APS bounds — exact Excel model:
      lo_aps = aps0 * lower_bound_pct  (e.g. 0.5 → 50% of baseline aps)
      hi_aps = aps0 * upper_bound_pct  (e.g. 1.5 → 150% of baseline aps)
    Locked channels are fixed at baseline APS.
    """
    aps0 = float(row['activity_per_segment'])
    if int(row.get('lock_spend', 0)) == 1:
        return (aps0, aps0)
    lo = float(row.get('lower_bound_pct', 0.5)) * aps0
    hi = float(row.get('upper_bound_pct', 1.5)) * aps0
    if lo > hi: lo, hi = hi, lo
    return (max(lo, 0.0), hi)

# Keep spend-based helpers for legacy charts/WhatIf
def get_bounds(row: pd.Series):
    return get_bounds_spend(row)

def total_profit(spends: np.ndarray, df: pd.DataFrame, use_hill: bool = False) -> float:
    return sum(
        revenue_from_spend(spends[i], df.iloc[i], use_hill) - spends[i]
        for i in range(len(spends))
    )

def total_revenue(spends: np.ndarray, df: pd.DataFrame, use_hill: bool = False) -> float:
    return sum(revenue_from_spend(spends[i], df.iloc[i], use_hill) for i in range(len(spends)))

def channel_roas_weighted(spends: np.ndarray, df: pd.DataFrame, use_hill: bool = False) -> float:
    total = 0.0
    for i in range(len(spends)):
        s = max(float(spends[i]), EPS)
        r = revenue_from_spend(s, df.iloc[i], use_hill)
        total += r / s
    return total


def optimize_budget_gekko(df: pd.DataFrame,
                          budget_constraint: float,
                          objective: str = "profit",
                          use_hill: bool = False) -> dict:
    """
    GEKKO/APOPT optimizer — HARD CONSTRAINT VERSION

    FIXES:
    ✅ Never auto-expands channel bounds
    ✅ If budget outside feasible range -> returns warning
    ✅ Locked Face-to-Face + huge budget handled correctly
    ✅ Detailed results will now respect channel constraints
    """

    if not GEKKO_OK:
        return {
            "spends": None,
            "success": False,
            "message": "GEKKO not installed. Run: pip install gekko"
        }

    n = len(df)
    rows = [df.iloc[i] for i in range(n)]

    aps0_arr = np.array([
        float(r['activity_per_segment']) for r in rows
    ])

    coef_arr = np.array([
        float(r['coefficient']) for r in rows
    ])

    alpha_arr = np.array([
        float(r['alpha']) for r in rows
    ])

    segs_arr = np.array([
        float(r['total_segments']) for r in rows
    ])

    npu_arr = np.array([
        float(r['net_per_unit']) for r in rows
    ])

    adjf_arr = np.array([
        float(r.get('Adj_Factor', 1.0)) for r in rows
    ])

    spend0_arr = np.array([
        float(r['total_spend']) for r in rows
    ])

    rev0_arr = (
        adjf_arr
        * coef_arr
        * (aps0_arr ** alpha_arr)
        * segs_arr
        * npu_arr
    )

    # ----------------------------------------------------------
    # TRUE HARD BOUNDS
    # ----------------------------------------------------------
    locked_arr = np.array([
        1 if int(r.get('lock_spend', 0)) == 1 else 0
        for r in rows
    ])

    lo_arr = np.array([
        get_bounds_aps(r)[0] / max(aps0_arr[i], EPS)
        for i, r in enumerate(rows)
    ])

    hi_arr = np.array([
        get_bounds_aps(r)[1] / max(aps0_arr[i], EPS)
        for i, r in enumerate(rows)
    ])

    # ----------------------------------------------------------
    # FEASIBILITY CHECK
    # ----------------------------------------------------------
    min_total = float((spend0_arr * lo_arr).sum())
    max_total = float((spend0_arr * hi_arr).sum())

    if budget_constraint < min_total:
        return {
            "spends": spend0_arr,
            "success": False,
            "message":
                f"Budget {budget_constraint:,.0f} is BELOW minimum feasible "
                f"channel spend {min_total:,.0f}.",
            "fun": None,
            "constraint_violation": True
        }

    if budget_constraint > max_total:
        return {
            "spends": spend0_arr,
            "success": False,
            "message":
                f"Budget {budget_constraint:,.0f} EXCEEDS maximum feasible "
                f"channel spend {max_total:,.0f}.",
            "fun": None,
            "constraint_violation": True
        }

    # ----------------------------------------------------------
    # Warm start
    # ----------------------------------------------------------
    base_total = float(spend0_arr.sum())
    ratio = budget_constraint / base_total if base_total > 0 else 1.0

    seed_arr = np.clip(
        np.full(n, ratio),
        lo_arr,
        hi_arr
    )

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------
    m = GEKKO(remote=False)

    m.options.SOLVER = 1
    m.options.MAX_ITER = 2000
    m.options.IMODE = 3
    m.options.RTOL = 1e-9
    m.options.OTOL = 1e-9

    x_v = [
        m.Var(
            value=float(seed_arr[i]),
            lb=float(lo_arr[i]),
            ub=float(hi_arr[i])
        )
        for i in range(n)
    ]

    # HARD exact budget
    m.Equation(
        m.sum([
            x_v[i] * float(spend0_arr[i])
            for i in range(n)
        ]) == budget_constraint
    )

    # ----------------------------------------------------------
    # Objective functions
    # ----------------------------------------------------------
    def revenue_term(i):
        return float(rev0_arr[i]) * (
            x_v[i] ** float(alpha_arr[i])
        )

    def profit_term(i):
        rev_i = revenue_term(i)
        spd_i = float(spend0_arr[i]) * x_v[i]
        return rev_i - spd_i

    def roas_term(i):
        rev_i = revenue_term(i)
        spd_i = float(spend0_arr[i]) * x_v[i] + EPS
        return rev_i / spd_i

    if objective == "profit":
        m.Maximize(
            m.sum([profit_term(i) for i in range(n)])
        )

    elif objective == "revenue":
        m.Maximize(
            m.sum([revenue_term(i) for i in range(n)])
        )

    else:
        m.Maximize(
            m.sum([roas_term(i) for i in range(n)])
        )

    # ----------------------------------------------------------
    # Solve
    # ----------------------------------------------------------
    try:
        m.solve(disp=False)

        x_opt = np.array([
            float(x_v[i].value[0])
            for i in range(n)
        ])

        opt_spend = spend0_arr * x_opt

        return {
            "spends": opt_spend,
            "success": True,
            "message": "GEKKO solved with hard constraints.",
            "fun": None,
            "constraint_violation": False
        }

    except Exception as e:

        return {
            "spends": spend0_arr,
            "success": False,
            "message": f"Optimization failed: {e}",
            "fun": None,
            "constraint_violation": True
        }


def optimize_budget_slsqp(df: pd.DataFrame, budget_constraint: float,
                           objective: str = "profit", use_hill: bool = False,
                           method: str = "SLSQP") -> dict:
    """Fallback SLSQP / Differential Evolution optimizer (spend-space)."""
    bounds_base = [get_bounds_spend(df.iloc[i]) for i in range(len(df))]

    # Auto-scale bounds to make new budget feasible (same logic as GEKKO)
    locked_arr  = np.array([1 if int(df.iloc[i].get('lock_spend',0))==1 else 0 for i in range(len(df))])
    spend0_arr  = np.array([float(df.iloc[i]['total_spend']) for i in range(len(df))])
    flex_mask   = locked_arr == 0
    locked_sp   = float((spend0_arr * locked_arr).sum())
    flex_budget = budget_constraint - locked_sp
    if flex_budget <= 0:
        return {"spends": spend0_arr, "success": False,
                "message": (
                    f"Budget ${budget_constraint:,.0f} is at or below locked channel spend "
                    f"(${locked_sp:,.0f}). Increase budget above ${locked_sp:,.0f}."
                ), "fun": None}
    lo_v = np.array([b[0] for b in bounds_base])
    hi_v = np.array([b[1] for b in bounds_base])
    if flex_mask.any():
        flex_min = lo_v[flex_mask].sum()
        flex_max = hi_v[flex_mask].sum()
        if flex_budget > flex_max and flex_max > 0:
            scale = flex_budget / flex_max
            hi_v  = np.where(flex_mask, hi_v * scale, hi_v)
        elif flex_budget < flex_min and flex_min > 0:
            scale = flex_budget / flex_min
            lo_v  = np.where(flex_mask, lo_v * scale, lo_v)
    bounds = list(zip(lo_v, hi_v))
    flex_base = float(spend0_arr[flex_mask].sum())
    seed_ratio = flex_budget / flex_base if flex_base > 0 else 1.0
    seed = np.where(locked_arr == 1, spend0_arr,
                    np.clip(spend0_arr * seed_ratio, lo_v, hi_v))

    if not SCIPY:
        return {"spends": seed, "success": False, "message": "SciPy not available"}

    if objective == "profit":
        neg_obj = lambda x: -total_profit(x, df, use_hill)
    elif objective == "revenue":
        neg_obj = lambda x: -total_revenue(x, df, use_hill)
    else:
        neg_obj = lambda x: -channel_roas_weighted(x, df, use_hill)

    budget_tol = budget_constraint * 0.0001
    cons = [
        {"type": "ineq", "fun": lambda x:  np.sum(x) - budget_constraint + budget_tol},
        {"type": "ineq", "fun": lambda x: -np.sum(x) + budget_constraint + budget_tol},
    ]

    if method == "differential_evolution":
        def penalized(x):
            overage  = max(0, np.sum(x) - budget_constraint - budget_tol)
            underage = max(0, budget_constraint - budget_tol - np.sum(x))
            return neg_obj(x) + 1e6 * (overage + underage)
        rng = np.random.default_rng(42)
        n_m = max(15, len(seed) * 5)
        lo_v = np.array([b[0] for b in bounds])
        hi_v = np.array([b[1] for b in bounds])
        init_pop = np.clip(seed * (1.0 + rng.uniform(-0.08, 0.08, (n_m, len(seed)))), lo_v, hi_v)
        init_pop[0] = np.clip(seed, lo_v, hi_v)
        res = differential_evolution(penalized, bounds, init=init_pop, seed=42,
                                     maxiter=500, tol=1e-6, workers=1,
                                     mutation=(0.5, 1.5), recombination=0.7)
        return {"spends": res.x, "success": res.success, "message": res.message}

    best = None
    rng2 = np.random.default_rng(99)
    lo_v = np.array([b[0] for b in bounds])
    hi_v = np.array([b[1] for b in bounds])
    for trial in range(8):
        x0 = seed.copy() if trial == 0 else np.clip(
            rng2.uniform(lo_v, hi_v) / rng2.uniform(lo_v, hi_v).sum() * budget_constraint,
            lo_v, hi_v)
        r = minimize(neg_obj, x0=x0, bounds=bounds, constraints=cons,
                     method='SLSQP', options={'maxiter': 1500, 'ftol': 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    return {"spends": best.x, "success": best.success, "message": best.message}


def optimize_budget(df: pd.DataFrame, budget_constraint: float,
                    objective: str = "profit", use_hill: bool = False,
                    use_log: bool = False, method: str = "GEKKO") -> dict:
    """
    Main optimizer dispatcher.
    method: 'GEKKO' (default, recommended) | 'SLSQP' | 'Differential Evolution'
    """
    if method == "GEKKO":
        return optimize_budget_gekko(df, budget_constraint, objective, use_hill)
    else:
        return optimize_budget_slsqp(df, budget_constraint, objective, use_hill, method)


def find_optimal_budget(df: pd.DataFrame,
                        use_hill: bool = False,
                        use_log: bool = False,
                        lo_pct: float = 0.7,
                        hi_pct: float = 1.5,
                        increment: float = 1_000_000.0,
                        respect_bounds: bool = False,
                        strict_feasible: bool = False,
                        objective: str = "profit") -> dict:
    """
    Scan total budgets from lo_pct to hi_pct of baseline in fixed $ increments
    to find the budget that maximises the selected objective (profit or revenue).

    NEW ROBUST FIX:
    - respect_bounds=True      -> validates channel min/max bounds
    - strict_feasible=True    -> excludes infeasible budgets from scan
    - returns infeasible_points count
    - returns constraint_warning flag
    - objective: "profit" (default) or "revenue" — controls which metric is maximised
    """

    locked_arr = df['lock_spend'].astype(int).values
    spend0_arr = df['total_spend'].values

    locked_spend = float((spend0_arr * locked_arr).sum())
    baseline_budget = float(spend0_arr.sum())

    # Clean rounded scan range
    lo_raw = max(baseline_budget * lo_pct, locked_spend * 1.01)
    hi_raw = baseline_budget * hi_pct

    lo_budget = max(round(lo_raw / increment) * increment, increment)
    hi_budget = round(hi_raw / increment) * increment

    if hi_budget <= lo_budget:
        hi_budget = lo_budget + increment

    budgets = np.arange(
        lo_budget,
        hi_budget + increment * 0.5,
        increment
    )

    results = []
    infeasible_points = 0
    any_constraint_warning = False

    # ──────────────────────────────────────────────────────────────
    # Scan all candidate budgets
    # ──────────────────────────────────────────────────────────────
    for bgt in budgets:

        res = optimize_budget_gekko(
            df,
            float(bgt),
            objective=objective,
            use_hill=use_hill
        )

        if res["success"] and res["spends"] is not None:

            spends = np.array(res["spends"]).astype(float)

            # ------------------------------------------------------
            # Check channel bounds
            # ------------------------------------------------------
            violations = []

            if respect_bounds:

                for i in range(len(df)):

                    row = df.iloc[i]

                    base_spend = float(row["total_spend"])
                    opt_spend = float(spends[i])

                    lb = base_spend * float(row["lower_bound_pct"])
                    ub = base_spend * float(row["upper_bound_pct"])

                    if opt_spend < lb:
                        violations.append({
                            "channel": row["channel"],
                            "type": "below",
                            "actual": opt_spend,
                            "min": lb
                        })

                    if opt_spend > ub:
                        violations.append({
                            "channel": row["channel"],
                            "type": "above",
                            "actual": opt_spend,
                            "max": ub
                        })

            # If strict mode, skip infeasible budgets
            if strict_feasible and len(violations) > 0:
                infeasible_points += 1
                continue

            if len(violations) > 0:
                any_constraint_warning = True

            # ------------------------------------------------------
            # Compute revenue / profit
            # ------------------------------------------------------
            opt_sp = float(spends.sum())

            opt_rev = sum(
                revenue_from_spend(
                    spends[i],
                    df.iloc[i],
                    use_hill,
                    use_log
                )
                for i in range(len(df))
            )

            opt_pf = opt_rev - opt_sp

            results.append({
                "budget": bgt,
                "opt_spend": opt_sp,
                "opt_revenue": opt_rev,
                "opt_profit": opt_pf
            })

        else:
            results.append({
                "budget": bgt,
                "opt_spend": bgt,
                "opt_revenue": np.nan,
                "opt_profit": np.nan
            })

    # ──────────────────────────────────────────────────────────────
    # Convert results
    # ──────────────────────────────────────────────────────────────
    scan_df = pd.DataFrame(results).dropna()

    if scan_df.empty:
        return {
            "optimal_budget": baseline_budget,
            "optimal_profit": np.nan,
            "scan_df": pd.DataFrame(),
            "success": False,
            "infeasible_points": infeasible_points,
            "constraint_warning": True
        }

    # Select best budget based on the correct objective metric
    _scan_metric = "opt_revenue" if objective == "revenue" else "opt_profit"
    best_idx = scan_df[_scan_metric].idxmax()

    # ──────────────────────────────────────────────────────────────
    # Refine using quadratic interpolation
    # ──────────────────────────────────────────────────────────────
    try:
        n = len(scan_df)

        lo_i = max(0, best_idx - 1)
        hi_i = min(n - 1, best_idx + 1)

        _fit_x = scan_df["budget"].iloc[lo_i:hi_i + 1].values
        _fit_y = scan_df[_scan_metric].iloc[lo_i:hi_i + 1].values

        if len(_fit_x) >= 3:

            _coeffs = np.polyfit(_fit_x, _fit_y, 2)

            _a = _coeffs[0]
            _b = _coeffs[1]

            if _a < 0:

                _interp_budget = -_b / (2 * _a)

                _interp_budget = float(
                    np.clip(
                        _interp_budget,
                        scan_df["budget"].min(),
                        scan_df["budget"].max()
                    )
                )

                _interp_budget = round(
                    _interp_budget / 500_000
                ) * 500_000

                _ref_res = optimize_budget_gekko(
                    df,
                    _interp_budget,
                    objective=objective,
                    use_hill=use_hill
                )

                if (
                    _ref_res.get("success")
                    and _ref_res.get("spends") is not None
                ):

                    _ref_spends = np.array(_ref_res["spends"]).astype(float)

                    _ref_rev = sum(
                        revenue_from_spend(
                            _ref_spends[i],
                            df.iloc[i],
                            use_hill,
                            use_log
                        )
                        for i in range(len(df))
                    )

                    _ref_pf = _ref_rev - _interp_budget
                    _ref_metric = _ref_rev if objective == "revenue" else _ref_pf

                    opt_budget = _interp_budget
                    opt_profit = _ref_metric

                else:
                    opt_budget = float(scan_df.loc[best_idx, "budget"])
                    opt_profit = float(scan_df.loc[best_idx, _scan_metric])

            else:
                opt_budget = float(scan_df.loc[best_idx, "budget"])
                opt_profit = float(scan_df.loc[best_idx, _scan_metric])

        else:
            opt_budget = float(scan_df.loc[best_idx, "budget"])
            opt_profit = float(scan_df.loc[best_idx, _scan_metric])

    except Exception:

        opt_budget = float(scan_df.loc[best_idx, "budget"])
        opt_profit = float(scan_df.loc[best_idx, _scan_metric])

    # ──────────────────────────────────────────────────────────────
    # Final return
    # ──────────────────────────────────────────────────────────────
    return {
        "optimal_budget": opt_budget,
        "optimal_profit": opt_profit,
        "scan_df": scan_df,
        "success": True,
        "infeasible_points": infeasible_points,
        "constraint_warning": any_constraint_warning
    }
# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def read_file(data_bytes: bytes, name: str):
    try:
        if name.lower().endswith('.csv'):
            return pd.read_csv(io.BytesIO(data_bytes))
        else:
            return pd.read_excel(io.BytesIO(data_bytes), engine='openpyxl')
    except Exception as e:
        return None

def validate(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return f"Missing columns: {', '.join(missing)}"
    return None

def fmt(v, prefix="", suffix="", decimals=0):
    if abs(v) >= 1e6:
        return f"{prefix}{v/1e6:.1f}M{suffix}"
    elif abs(v) >= 1e3:
        return f"{prefix}{v/1e3:.0f}K{suffix}"
    return f"{prefix}{v:,.{decimals}f}{suffix}"

def plotly_dark_layout(fig, height=400, **kwargs):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(size=11, color=PALETTE["text2"])
        ),
        xaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                   tickfont=dict(size=10, color=PALETTE["muted"]), zeroline=False),
        yaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                   tickfont=dict(size=10, color=PALETTE["muted"]), zeroline=False),
        **kwargs
    )
    return fig

def channel_color(i: int) -> str:
    return CHANNEL_COLORS[i % len(CHANNEL_COLORS)]

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

_hdr_mu   = PALETTE["muted"];   _hdr_surf2 = PALETTE["surface2"]
_hdr_bdr  = PALETTE["border"]; _hdr_mu2   = PALETTE["muted2"]
st.markdown(
    f"<div style='display:flex;align-items:center;gap:1rem;padding:0 0 .75rem'>"
    f"<div style='width:40px;height:40px;border-radius:10px;background:#065AC7D2;"
    f"display:flex;align-items:center;justify-content:center;"
    f"font-size:1.3rem;box-shadow:0 4px 12px rgba(26,86,219,.3);flex-shrink:0'>&#x2B21;</div>"
    f"<div>"
    f"<div style='font-size:1.45rem;font-weight:800;letter-spacing:-.03em;"
    f"color:#CC5500;line-height:1.1'>"
    f"MMM <span style='color:#002F6C'>Scenario Recommendation &amp; Optimization Engine</span>"
    f"<span style='font-size:.75rem;font-weight:500;color:{_hdr_mu};"
    f"background:{_hdr_surf2};border:1.5px solid {_hdr_bdr};"
    f"border-radius:6px;padding:1px 8px;margin-left:.5rem;"
    f"vertical-align:middle;letter-spacing:.04em'>PRO</span>"
    f"</div>"
    f"<div style='font-size:.68rem;color:{_hdr_mu2};margin-top:.2rem;"
    f"letter-spacing:.08em;text-transform:uppercase'>"
    f"Marketing Mix Modeling &nbsp;&#183;&nbsp; Power &amp; Hill Response Curves "
    f"&nbsp;&#183;&nbsp; Multi-Objective Optimization"
    f"</div></div></div>",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════
# AI CONFIGURATION (INITIALIZED EARLY – SAFE / NO LOGIC CHANGE)
# ═══════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='margin-top:.75rem;padding:0.75rem 1rem;"
    "border:1px solid #E5E7EB;border-radius:10px;background:#FAFAFA'>"
    "<div style='font-size:.75rem;font-weight:700;letter-spacing:.08em;"
    "text-transform:uppercase;color:#6B7280;margin-bottom:.4rem'>"
    "AI Configuration</div>"
    "</div>",
    unsafe_allow_html=True
)

# Initialize AI session state safely (does not affect existing AI tab)
if "ai_provider" not in st.session_state:
    st.session_state["ai_provider"] = "Google Gemini (free)"

if "ai_key" not in st.session_state:
    st.session_state["ai_key"] = ""

ai_c1, ai_c2, ai_c3 = st.columns([0.28, 0.42, 0.30])

with ai_c1:
    st.selectbox(
        "AI Provider",
        ["Google Gemini (free)", "Groq (free)"],
        key="ai_provider",
        help="LLM provider used for AI interpretation"
    )

with ai_c2:
    st.text_input(
        "AI API Key",
        type="password",
        key="ai_key",
        placeholder="Paste your API key here",
        help="Stored only in Streamlit session memory"
    )

with ai_c3:
    if st.session_state["ai_key"].strip():
        st.success("✅ AI key read successfully")
    else:
        st.info("Enter your AI key to enable AI interpretation")

st.markdown("<div style='margin-bottom:.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════




# st.markdown(
#     """
#     <style>
#     /* White background ONLY for the logo image */
#     section[data-testid="stSidebar"] img {
#         background-color: #ffffff;
#         padding: 0.5rem 0.75rem;
#         border-radius: 8px;
#         display: block;
#         margin: 0.75rem auto 0.5rem auto;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.sidebar.image("eversana_logo.png", width=200)

st.markdown(
    """
    <style>
    /* Logo image container — white pill on dark sidebar */
    section[data-testid="stSidebar"] img[data-testid="stImage"] {
        background-color: #ffffff;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        display: block;
        margin: 0;
    }
    /* Fallback for older Streamlit versions that don't add data-testid to img */
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        background-color: #ffffff;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
    }
    /* Remove extra top padding from sidebar */
    section[data-testid="stSidebar"] {
        padding-top: 0;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load logo safely — falls back gracefully if file not found
import os as _os
_logo_path = "eversana_logo.png"
if _os.path.exists(_logo_path):
    st.sidebar.image(_logo_path, width=200)
else:
    # Fallback: text logo if image file is missing
    st.sidebar.markdown(
        "<div style='background:#fff;border-radius:8px;padding:.5rem .75rem;"
        "font-weight:800;font-size:1rem;color:#CC5500;letter-spacing:.05em;"
        "margin-bottom:.25rem'>⬡ EVERSANA</div>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("<hr style='border:1px solid #F27C38;'>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    /* ── File uploader dropzone — dark navy background ── */
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploadDropzone"] > div,
    section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
        background: #1A2540 !important;
        background-image: none !important;
        border-radius: 12px !important;
        border: 1.5px dashed #3D5080 !important;
        padding: 1rem !important;
    }

    /* All text in dropzone — white */
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] label {
        color: #C8D0E0 !important;
    }

    /* Browse files button — orange background, white text */
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploadDropzone"] button span {
        background-color: #CC5500 !important;
        background: #CC5500 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploadDropzone"] button:hover,
    [data-testid="stFileUploadDropzone"] button:hover span {
        background-color: #A84300 !important;
        background: #A84300 !important;
        color: #ffffff !important;
    }

    /* Uploaded file name row */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] p {
        color: #E5E9F2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown(f"<div style='font-size:.65rem;font-weight:700;color:{PALETTE['sidebar_mu']};text-transform:uppercase;letter-spacing:.1em;padding:.5rem 0'>DATA INPUT</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"],
                                  help="Required columns: channel, total_activity, total_spend, total_sales, coefficient, type_transformation, alpha, lock_spend, lower_bound_pct, upper_bound_pct, total_segments, net_per_unit")

    st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.65rem;font-weight:700;color:{PALETTE['sidebar_mu']};text-transform:uppercase;letter-spacing:.1em;padding:.5rem 0'>CURVE MODEL</div>", unsafe_allow_html=True)
    curve_model_choice = st.radio(
        "Curve model",
        ["Power (default)"],
        help=(
            "Power: revenue = AdjF × coef × aps^alpha × segs × npu  (Excel-equivalent, recommended)\n"
            "Log: revenue = AdjF × coef × ln(1 + aps^alpha) × segs × npu  (softer saturation)\n"
            "Hill: auto-fits EC50/slope from Power simulation  (S-curve, R-equivalent)"
        )
    )
    use_hill_global   = (curve_model_choice == "Hill (auto-fitted)")
    use_log_global    = (curve_model_choice == "Log")
    #show_hill_params  = st.checkbox("Show Hill fit parameters", value=False,
    #                                 disabled=not use_hill_global)

    st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.65rem;font-weight:700;color:{PALETTE['sidebar_mu']};text-transform:uppercase;letter-spacing:.1em;padding:.5rem 0'>OPTIMIZATION</div>", unsafe_allow_html=True)
    opt_objective = st.selectbox("Objective", ["Maximize Profit", "Maximize Revenue"],
                                  help="What to optimize for")
    obj_map = {"Maximize Profit": "profit", "Maximize Revenue": "revenue", "Maximize ROI": "roi"}

    opt_method = st.selectbox("Solver", [
                                "GEKKO / IPOPT (recommended)",
                                "SLSQP (fast)",
                                "Differential Evolution (robust)"
                               ],
                               help="GEKKO/IPOPT matches Excel Solver exactly. SLSQP is faster but may miss global optimum. DE is slowest but most robust.")
    method_map = {
        "GEKKO / IPOPT (recommended)": "GEKKO",
        "SLSQP (fast)":                "SLSQP",
        "Differential Evolution (robust)": "differential_evolution"
    }

    st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)
    with st.expander("📋 Schema reference", expanded=False):
        _schema_muted = PALETTE["muted"]
        st.markdown(
            f"<div style='font-size:.75rem;color:{_schema_muted}'>"
            f"<b>Required columns:</b><br>"
            f"• channel, total_activity, total_spend<br>"
            f"• total_sales, coefficient<br>"
            f"• type_transformation (power/log/hill)<br>"
            f"• alpha, lock_spend (0/1)<br>"
            f"• lower_bound_pct, upper_bound_pct<br>"
            f"• total_segments, net_per_unit<br><br>"
            f"<b>Optional:</b> hill_ec50, hill_slope, hill_max_response"
            f"</div>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════════════════════

if uploaded is None:
    _up_surf = PALETTE["surface"]; _up_bdr = PALETTE["border"]
    _up_txt  = PALETTE["text"];    _up_mu  = PALETTE["muted"]
    st.markdown(
        f"<div style='margin-top:3rem;text-align:center;padding:3rem;"
        f"background:{_up_surf};border-radius:16px;border:2px dashed {_up_bdr}'>"
        f"<div style='font-size:2.5rem;margin-bottom:1rem'>⬡</div>"
        f"<div style='font-size:1.1rem;font-weight:600;color:{_up_txt};margin-bottom:.5rem'>"
        f"Upload your MMM input file to begin</div>"
        f"<div style='font-size:.82rem;color:{_up_mu}'>"
        f"Accepts CSV or Excel • Power, Log & Hill response curves • Multi-channel optimization</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.stop()

raw_bytes = uploaded.read()
df_raw = read_file(raw_bytes, uploaded.name)
# Reset Hill fits whenever file changes (new file hash may differ)
_file_hash = hash(raw_bytes)
if st.session_state.get('_file_hash') != _file_hash:
    st.session_state['_file_hash'] = _file_hash
    st.session_state['hill_df'] = None
    # Clear stale opt results from old file
    for _k in list(st.session_state.keys()):
        if _k.startswith(('profit|','revenue|','roas|','opt_result','opt_success','opt_message','opt_objective','opt_method','frontier')):
            del st.session_state[_k]

if df_raw is None:
    st.error("Could not read file. Please check the format.")
    st.stop()

err = validate(df_raw)
if err:
    st.error(f"Schema error: {err}")
    st.stop()

df = df_raw.copy()

# ─── DERIVED COLUMNS ──────────────────────────────────────────────────────────
df['activity_per_segment'] = df['total_activity'] / df['total_segments'].replace(0, np.nan)
# cost_per_mention = dollars per activity ($/activity) — matches Excel model exactly
# Excel formula: cost_per_mention = total_spend / total_activity
# Used in revenue calc as: spend = aps * cost_per_mention * total_segments
df['cost_per_mention'] = df['total_spend'] / df['total_activity'].replace(0, np.nan)

# Unadjusted impact + adjustment factor
ui_vals = []
for _, r in df.iterrows():
    ui_vals.append(unadj_impact(
        r['activity_per_segment'], r['coefficient'], r['alpha'],
        r['total_segments'], r['net_per_unit'], r['type_transformation']
    ))
df['Unadjusted_impact'] = ui_vals
df['Adj_Factor'] = np.where(df['Unadjusted_impact'] > 0,
                             df['total_sales'] / df['Unadjusted_impact'], 1.0)

# Hill fits (auto-compute if checked)
if 'hill_df' not in st.session_state:
    st.session_state['hill_df'] = None

if use_hill_global and st.session_state['hill_df'] is None:
    with st.spinner("Computing Hill fits from Power curves..."):
        st.session_state['hill_df'] = _compute_hill_for_df(df)

if st.session_state['hill_df'] is not None:
    df = df.drop(columns=[c for c in ['hill_ec50','hill_slope','hill_max_response'] if c in df.columns], errors='ignore')
    df = df.merge(st.session_state['hill_df'], on='channel', how='left')
else:
    for c in ['hill_ec50', 'hill_slope', 'hill_max_response']:
        if c not in df.columns:
            df[c] = np.nan

# Baseline revenue & profit
baseline_rev = [revenue_from_spend(float(r['total_spend']), r, use_hill_global, use_log_global) for _, r in df.iterrows()]
df['baseline_revenue'] = baseline_rev
df['baseline_profit'] = df['baseline_revenue'] - df['total_spend']
df['baseline_roi'] = df['baseline_revenue'] / df['total_spend'].replace(0, np.nan)
df['baseline_mroi'] = [mroi(float(df.iloc[i]['total_spend']), df.iloc[i], use_hill_global, use_log_global) for i in range(len(df))]
df['channel_color'] = [channel_color(i) for i in range(len(df))]

# channel_desc — human-readable channel description for AI prompts
if 'channel_desc' in df.columns:
    df['channel_desc'] = df['channel_desc'].fillna(df['channel'])
else:
    df['channel_desc'] = df['channel']   # fallback: use channel name
ch_desc_map = dict(zip(df['channel'], df['channel_desc']))


# Assign channel index for colors
ch_color_map = {r['channel']: r['channel_color'] for _, r in df.iterrows()}

# ─── TOP KPIs ─────────────────────────────────────────────────────────────────
tot_spend   = df['total_spend'].sum()
tot_rev     = df['baseline_revenue'].sum()
tot_profit  = df['baseline_profit'].sum()
avg_roas    = tot_rev / max(tot_spend, EPS)
tot_sales   = df['total_sales'].sum()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Spend", fmt(tot_spend, "$"), help="Sum of channel spend (baseline)")
with k2:
    st.metric("Calibrated Revenue", fmt(tot_rev, "$"),
              delta=f"{fmt(tot_rev - tot_sales, '$')} vs actual sales")
with k3:
    st.metric("Baseline Profit", fmt(tot_profit, "$"))
with k4:
    st.metric("Avg ROI", f"{avg_roas:.2f}×")
with k5:
    locked = int(df['lock_spend'].sum())
    st.metric("Channels", f"{len(df)} total", delta=f"{locked} locked")

st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_rc, tab_opt, tab_wi, tab_data, tab_ai = st.tabs([
    "📈  Response Curves",
    "⚡  Budget Optimization",
   # "🚀  Launch Brand Optimization",
    "🎛  What-If Scenarios",
    "📋  Data & Diagnostics",
    "🧠  AI Interpretation"
])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — RESPONSE CURVES                                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

with tab_rc:

    # ── Controls row ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([.28, .18, .18, .18, .18])
    with ctrl1:
        selected_ch = st.selectbox("Channel", df['channel'].tolist(),
                                    key="rc_channel", label_visibility="collapsed")
    with ctrl2:
        show_all = st.checkbox("Overlay all channels", value=False, key="rc_show_all")
    with ctrl3:
        show_all_curves = st.checkbox(
            "Show all response curves together",
            value=False,
            key="rc_show_all_curves"
        )
    with ctrl4:
        show_table = st.checkbox("Show response table", value=True, key="rc_show_tbl")
    with ctrl5:
        curve_type = st.radio(
            "Curve Type",
            options=["Profit Curve", "Revenue Curve"],
            horizontal=True,
            key="rc_curve_type"
        )

    rrow      = df[df['channel'] == selected_ch].iloc[0]
    base_sp   = float(rrow['total_spend'])
    base_act  = float(rrow['total_activity'])
    base_aps  = float(rrow['activity_per_segment'])
    segs      = float(rrow['total_segments'])
    cpm       = float(rrow.get('cost_per_mention', 0))
    mroi_val = float(rrow['baseline_mroi'])
    roas_val  = float(rrow['baseline_roi'])
    lo_pct    = float(rrow.get('lower_bound_pct', 0.5))
    hi_pct    = float(rrow.get('upper_bound_pct', 1.5))
    lock_val  = int(rrow.get('lock_spend', 0))

    # ── Curve grid: X = % of baseline investment, 0 → max(150%, hi_pct) ─────
    n_pts    = 500
    disp_hi  = max(hi_pct * 100, 300)
    pct_grid = np.linspace(0.0, disp_hi, n_pts)
    spd_grid = base_sp * pct_grid / 100
    act_grid = (spd_grid / cpm) if cpm > 0 else spd_grid
    rev_grid = np.array([revenue_from_spend(s, rrow, use_hill_global, use_log_global) for s in spd_grid])
    prof_grid = rev_grid - spd_grid
    mroi_grid = np.array([mroi(s, rrow, use_hill_global, use_log_global) for s in spd_grid])

    _rev_range   = float(rev_grid.max() - rev_grid.min())
    _curve_flat  = (_rev_range < 1.0)

    # ── Annotation points ────────────────────────────────────────────────────
    # Current (always at 100%)
    curr_pct   = 100.0
    curr_sp    = base_sp
    curr_act   = base_act
    curr_avg_a = base_aps
    curr_rev   = revenue_from_spend(curr_sp, rrow, use_hill_global, use_log_global)
    curr_prof  = curr_rev - curr_sp
    curr_mroi = mroi_val

    # Optimal from optimizer (if available)
    opt_df_stored = st.session_state.get('opt_result_df')
    has_opt = opt_df_stored is not None and selected_ch in opt_df_stored['channel'].values
    if has_opt:
        opt_row   = opt_df_stored[opt_df_stored['channel'] == selected_ch].iloc[0]
        opt_sp    = float(opt_row['opt_spend'])
        opt_pct   = opt_sp / base_sp * 100 if base_sp > 0 else 100.0
        opt_act   = opt_sp / cpm if cpm > 0 else opt_sp
        opt_avg_a = opt_act / segs if segs > 0 else np.nan
        opt_rev   = float(opt_row['opt_revenue'])
        opt_prof  = float(opt_row['opt_profit'])
        opt_mroi = mroi(opt_sp, rrow, use_hill_global, use_log_global)
    else:
        opt_sp = opt_pct = opt_act = opt_avg_a = opt_rev = opt_prof = opt_mroi = None

    # Peak profit point
    peak_idx  = int(np.argmax(prof_grid))
    peak_pct  = float(pct_grid[peak_idx])
    peak_sp   = float(spd_grid[peak_idx])
    peak_prof = float(prof_grid[peak_idx])

    # ── Saturation alert ──────────────────────────────────────────────────────
    if mroi_val > 1.2:
        sat_msg, sat_cls = f"📈 Underinvested — mROI {mroi_val:.2f}×. Consider increasing spend.", "alert-success"
    elif mroi_val < 0.8:
        sat_msg, sat_cls = f"📉 Saturated — mROI {mroi_val:.2f}×. Diminishing returns.", "alert-warn"
    else:
        sat_msg, sat_cls = f"✅ Near-optimal — mROI {mroi_val:.2f}×.", "alert-info"

    if _curve_flat:
        st.warning(f"⚠️ Flat curve for {selected_ch} — check total_activity and total_spend in your input.")

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN LAYOUT: left = channel card, right = chart + table
    # ═════════════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([.25, .75])

    with left_col:
        col_hex  = ch_color_map.get(selected_ch, PALETTE["accent"])
        lock_str = "Opt. locked" if lock_val else "Opt. flexible"
        lo_act_disp = (base_sp * lo_pct / cpm) if cpm > 0 else base_sp * lo_pct
        hi_act_disp = (base_sp * hi_pct / cpm) if cpm > 0 else base_sp * hi_pct

        _c_surf  = PALETTE["surface"];  _c_bdr   = PALETTE["border"]
        _c_mu    = PALETTE["muted"];    _c_mu2   = PALETTE["muted2"]
        _c_acc4  = PALETTE["accent4"];  _c_gold  = PALETTE["gold"]
        _c_acc2  = PALETTE["accent2"]
        st.markdown(
            f"<div style='background:{_c_surf};border:1.5px solid {_c_bdr};"
            f"border-radius:14px;padding:.9rem 1rem;box-shadow:0 1px 4px rgba(0,0,0,.05);"
            f"border-top:3px solid {col_hex}'>"
            f"<div style='font-size:.62rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.09em;color:{_c_mu};margin-bottom:.6rem'>"
            f"{selected_ch}"
            f"&nbsp;<span style='color:{_c_acc4}'>{rrow['type_transformation'].upper()}</span>"
            f"&nbsp;<span style='color:{_c_gold}'>{lock_str}</span>"
            f"</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:.45rem'>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>Spend</div>"
            f"<div style='font-size:.92rem;font-weight:700'>{fmt(curr_sp,'$')}</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>Activity</div>"
            f"<div style='font-size:.92rem;font-weight:700'>{fmt(curr_act)}</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>Revenue</div>"
            f"<div style='font-size:.92rem;font-weight:700'>{fmt(curr_rev,'$')}</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>Profit</div>"
            f"<div style='font-size:.92rem;font-weight:700;color:{_c_acc2}'>{fmt(curr_prof,'$')}</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>ROI</div>"
            f"<div style='font-size:.92rem;font-weight:700'>{roas_val:.2f}×</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>mROI</div>"
            f"<div style='font-size:.92rem;font-weight:700'>{mroi_val:.2f}×</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>Alpha</div>"
            f"<div style='font-size:.92rem;font-weight:700;color:{_c_acc4}'>{float(rrow['alpha']):.3f}</div></div>"
            f"<div><div style='font-size:.58rem;color:{_c_mu2}'>Adj Factor</div>"
            f"<div style='font-size:.92rem;font-weight:700'>{float(rrow['Adj_Factor']):.3f}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

        st.markdown(f"<div class='alert-box {sat_cls}' style='margin-top:.6rem;font-size:.77rem'>{sat_msg}</div>",
                    unsafe_allow_html=True)

        # Optimizer bounds
        bounds_txt = (f"Fixed at {fmt(base_sp,'$')}" if lock_val
                      else f"{fmt(base_sp*lo_pct,'$')} – {fmt(base_sp*hi_pct,'$')}")
        _ob_mu   = PALETTE["muted"]; _ob_surf2 = PALETTE["surface2"]; _ob_bdr = PALETTE["border"]
        st.markdown(
            f"<div style='font-size:.68rem;color:{_ob_mu};margin-top:.5rem;"
            f"background:{_ob_surf2};border-radius:8px;padding:.45rem .65rem;"
            f"border:1px solid {_ob_bdr}'>"
            f"<b>Optimizer bounds:</b> {bounds_txt}"
            f"</div>",
            unsafe_allow_html=True
        )

        if has_opt:
            delta_p = opt_prof - curr_prof
            _oa2    = PALETTE["accent2"]; _oa2lt = PALETTE["accent2_lt"]
            _dp_col = "#057A55" if delta_p >= 0 else "#C81E1E"
            _dp_val = fmt(delta_p, "$", "+") if delta_p >= 0 else fmt(delta_p, "$")
            st.markdown(
                f"<div style='background:{_oa2lt};border:1.5px solid #A7F3D0;"
                f"border-radius:10px;padding:.65rem .85rem;margin-top:.55rem'>"
                f"<div style='font-size:.6rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:.07em;color:{_oa2};margin-bottom:.35rem'>Optimized</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:.35rem;font-size:.82rem'>"
                f"<div><div style='font-size:.55rem;color:{_oa2}'>Spend</div>"
                f"<b>{fmt(opt_sp,'$')}</b></div>"
                f"<div><div style='font-size:.55rem;color:{_oa2}'>Activity</div>"
                f"<b>{fmt(opt_act)}</b></div>"
                f"<div><div style='font-size:.55rem;color:{_oa2}'>Profit</div>"
                f"<b style='color:{_oa2}'>{fmt(opt_prof,'$')}</b></div>"
                f"<div><div style='font-size:.55rem;color:{_oa2}'>Delta</div>"
                f"<b style='color:{_dp_col}'>{_dp_val}</b></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<div class='alert-box alert-info' style='margin-top:.55rem;font-size:.72rem'>Run Budget Optimization to annotate the optimal point.</div>",
                        unsafe_allow_html=True)

    # ─── RIGHT COLUMN: chart title + chart + table ────────────────────────────
    with right_col:
        lock_note = "  ·  🔒 Spend locked for optimizer" if lock_val else ""
        curve_title = "Profit Response Curve" if curve_type == "Profit Curve" else "Revenue Response Curve"
        st.markdown(f"<div class='section-header'>{selected_ch} – {curve_title}{lock_note}</div>",
                    unsafe_allow_html=True)

        if PLOTLY and not show_all:
            # ── Single-channel: clean Profit or Revenue curve, X = % investment ──────────
            fig_rc = go.Figure()

            # Determine which curve to plot
            if curve_type == "Profit Curve":
                curve_data = prof_grid
                curve_label = "Profit"
                curve_color = PALETTE["accent"]
                y_title = "Profit ($)"
                hover_format = "Investment: %{x:.0f}%<br>Profit: $%{y:,.0f}<extra></extra>"
            else:
                curve_data = rev_grid
                curve_label = "Revenue"
                curve_color = PALETTE["accent4"]
                y_title = "Revenue ($)"
                hover_format = "Investment: %{x:.0f}%<br>Revenue: $%{y:,.0f}<extra></extra>"

            # Main curve
            fig_rc.add_trace(go.Scatter(
                x=pct_grid, y=curve_data,
                mode='lines', name=selected_ch,
                line=dict(color=curve_color, width=3),
                hovertemplate=hover_format
            ))

            # Zero line
            fig_rc.add_hline(y=0, line_color="#CBD0D8", line_width=1.2, line_dash="solid")

            # ── Current investment annotation ─────────────────────────────────
            curr_curve_val = curr_prof if curve_type == "Profit Curve" else curr_rev
            fig_rc.add_trace(go.Scatter(
                x=[curr_pct], y=[curr_curve_val],
                mode='markers',
                marker=dict(size=14, color=PALETTE["accent3"],
                            symbol='circle', line=dict(color='white', width=2)),
                name='Current Investment',
                hovertemplate=(f"<b>Current</b><br>"
                               f"Investment: {curr_pct:.0f}%<br>"
                               f"Spend: ${curr_sp:,.0f}<br>"
                               f"Activity: {curr_act:,.0f}<br>"
                               f"Avg/Seg: {curr_avg_a:.2f}<br>"
                               f"Revenue: ${curr_rev:,.0f}<br>"
                               f"Profit: ${curr_prof:,.0f}<br>"
                               f"mROI: {curr_mroi:.2f}×<extra></extra>")
            ))
            
            # Peak line and annotation (for profit curve only)
            if curve_type == "Profit Curve":
                fig_rc.add_vline(
                    x=peak_pct, line=dict(color="#7B0361", width=2, dash="dash"),
                    annotation_text=f"Max Profit:<br>{fmt(peak_prof,'$')}", 
                    annotation_position="top left", 
                    annotation_font=dict(size=9, color="#790258"), 
                    annotation_bgcolor="rgba(255,255,255,0.85)", 
                    annotation_borderwidth=1, 
                    annotation_bordercolor="#7B0361", 
                    annotation_xshift=-15
                )

            # Callout label
            callout_text = f"<b>Current</b><br>{fmt(curr_curve_val,'$')}"
            fig_rc.add_annotation(
                x=curr_pct, y=curr_curve_val,
                text=callout_text,
                showarrow=True, arrowhead=2, arrowcolor=PALETTE["accent3"],
                ax=50, ay=-45, bgcolor="white",
                bordercolor=PALETTE["accent3"], borderwidth=1.5,
                font=dict(size=9, color=PALETTE["text2"])
            )
            fig_rc.add_vline(x=curr_pct, line_dash="dash",
                              line_color=PALETTE["accent3"], line_width=1.2,
                              opacity=0.6)

            # ── Optimal investment annotation ─────────────────────────────────
            if has_opt:
                opt_curve_val = opt_prof if curve_type == "Profit Curve" else opt_rev
                fig_rc.add_trace(go.Scatter(
                    x=[opt_pct], y=[opt_curve_val],
                    mode='markers',
                    marker=dict(size=14, color=PALETTE["accent2"],
                                symbol='circle', line=dict(color='white', width=2)),
                    name='Optimal Investment',
                    hovertemplate=(f"<b>Optimal</b><br>"
                                   f"Investment: {opt_pct:.1f}%<br>"
                                   f"Spend: ${opt_sp:,.0f}<br>"
                                   f"Activity: {opt_act:,.0f}<br>"
                                   f"Avg/Seg: {opt_avg_a:.2f}<br>"
                                   f"Revenue: ${opt_rev:,.0f}<br>"
                                   f"Profit: ${opt_prof:,.0f}<br>"
                                   f"mROI: {opt_mroi:.2f}×<extra></extra>")
                ))
                ax_off = 55 if opt_pct > curr_pct else -55
                fig_rc.add_annotation(
                    x=opt_pct, y=opt_curve_val,
                    text=f"<b>Suggested Optimal</b><br>{fmt(opt_curve_val,'$')}",
                    showarrow=True, arrowhead=2, arrowcolor=PALETTE["accent2"],
                    ax=ax_off, ay=-45, bgcolor="white",
                    bordercolor=PALETTE["accent2"], borderwidth=1.5,
                    font=dict(size=9, color=PALETTE["text2"])
                )
                fig_rc.add_vline(x=opt_pct, line_dash="dash",
                                  line_color=PALETTE["accent2"], line_width=1.2,
                                  opacity=0.6)

            # ── Bound lines (skip for locked channels) ────────────────────────
            if not lock_val:
                lo_pct_line = lo_pct * 100   # lower_bound_pct → %
                hi_pct_line = hi_pct * 100   # upper_bound_pct → %
                lo_sp_line  = base_sp * lo_pct
                hi_sp_line  = base_sp * hi_pct
                # Lower bound
                fig_rc.add_vline(
                    x=lo_pct_line,
                    line=dict(color="#F59E0B", width=1.5, dash="dot"),
                    opacity=0.75,
                    annotation_text=f"Min: {fmt(lo_sp_line,'$')}",
                    annotation_position="top right",
                    annotation_font=dict(size=8, color="#B45309"),
                    annotation_bgcolor="rgba(255,251,235,0.9)",
                    annotation_bordercolor="#F59E0B",
                    annotation_borderwidth=1,
                )
                # Upper bound
                fig_rc.add_vline(
                    x=hi_pct_line,
                    line=dict(color="#F59E0B", width=1.5, dash="dot"),
                    opacity=0.75,
                    annotation_text=f"Max: {fmt(hi_sp_line,'$')}",
                    annotation_position="top left",
                    annotation_font=dict(size=8, color="#B45309"),
                    annotation_bgcolor="rgba(255,251,235,0.9)",
                    annotation_bordercolor="#F59E0B",
                    annotation_borderwidth=1,
                )
                # Shaded feasible region between bounds
                lo_y = float(min(curve_data))
                hi_y = float(max(curve_data)) * 1.05
                fig_rc.add_vrect(
                    x0=lo_pct_line, x1=hi_pct_line,
                    fillcolor="rgba(245,158,11,0.06)",
                    line_width=0,
                    layer="below"
                )

            # Add legend entries for bound lines if not locked
            if not lock_val:
                fig_rc.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color="#F59E0B", width=1.5, dash="dot"),
                    name="Optimizer Bounds (Min/Max)"
                ))

            fig_rc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                height=420,
                margin=dict(l=10, r=20, t=20, b=50),
                legend=dict(
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=PALETTE["border"], borderwidth=1,
                    orientation="h", y=-0.13, font=dict(size=11)
                ),
                xaxis=dict(
                    title=f"% of Baseline Investment  [{curve_model_choice}]",
                    ticksuffix="%",
                    gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                    tickfont=dict(size=10, color=PALETTE["muted"]),
                    range=[0, disp_hi], zeroline=False
                ),
                yaxis=dict(
                    title=y_title,
                    gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                    tickfont=dict(size=10, color=PALETTE["muted"]),
                    tickformat="$.3s",
                    zeroline=True, zerolinecolor="#CBD0D8", zerolinewidth=1.5
                )
            )
            st.plotly_chart(fig_rc, width='stretch')

        elif PLOTLY and show_all:
            # ── All-channels overlay ──────────────────────────────────────────
            fig_all = go.Figure()
            for ri, (_, row) in enumerate(df.iterrows()):
                b_sp   = float(row['total_spend'])
                b_cpm  = float(row.get('cost_per_mention', 0))
                _hp    = max(float(row.get('upper_bound_pct', 1.5)) * 100, 150.0)
                p_g    = np.linspace(0, _hp, 300)
                s_g    = b_sp * p_g / 100
                r_g    = np.array([revenue_from_spend(s, row, use_hill_global, use_log_global) for s in s_g])
                
                if curve_type == "Profit Curve":
                    plot_data = r_g - s_g
                    y_label = "Profit ($)"
                else:
                    plot_data = r_g
                    y_label = "Revenue ($)"
                
                col_c  = channel_color(ri)
                b_rev  = revenue_from_spend(b_sp, row, use_hill_global, use_log_global)
                b_prf  = b_rev - b_sp

                fig_all.add_trace(go.Scatter(
                    x=p_g, y=plot_data, mode='lines', name=row['channel'],
                    line=dict(color=col_c, width=2),
                    hovertemplate=(f"<b>{row['channel']}</b><br>"
                                   f"Investment: %{{x:.0f}}%<br>"
                                   f"{'Profit' if curve_type == 'Profit Curve' else 'Revenue'}: $%{{y:,.0f}}<extra></extra>")
                ))
                # Current dot at 100%
                curr_val = b_prf if curve_type == "Profit Curve" else b_rev
                fig_all.add_trace(go.Scatter(
                    x=[100.0], y=[curr_val], mode='markers',
                    marker=dict(size=9, color=col_c, symbol='circle',
                                line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertemplate=(f"<b>{row['channel']} — Current</b><br>"
                                   f"Spend: ${b_sp:,.0f}<br>"
                                   f"{'Profit' if curve_type == 'Profit Curve' else 'Revenue'}: ${curr_val:,.0f}<extra></extra>")
                ))
                # Optimal dot if available
                if opt_df_stored is not None and row['channel'] in opt_df_stored['channel'].values:
                    o_r   = opt_df_stored[opt_df_stored['channel'] == row['channel']].iloc[0]
                    o_sp  = float(o_r['opt_spend'])
                    o_pct = o_sp / b_sp * 100 if b_sp > 0 else 100
                    o_rev = revenue_from_spend(o_sp, row, use_hill_global, use_log_global)
                    o_pf  = o_rev - o_sp
                    opt_val = o_pf if curve_type == "Profit Curve" else o_rev
                    fig_all.add_trace(go.Scatter(
                        x=[o_pct], y=[opt_val], mode='markers',
                        marker=dict(size=9, color=col_c, symbol='diamond',
                                    line=dict(color='white', width=2)),
                        showlegend=False,
                        hovertemplate=(f"<b>{row['channel']} — Optimal</b><br>"
                                       f"Spend: ${o_sp:,.0f}<br>"
                                       f"{'Profit' if curve_type == 'Profit Curve' else 'Revenue'}: ${opt_val:,.0f}<extra></extra>")
                    ))

                # ── Bound lines for this channel (colour-matched) ─────────────
                if not int(row.get('lock_spend', 0)) == 1:
                    _lo_pct_ov = float(row.get('lower_bound_pct', 0.5)) * 100
                    _hi_pct_ov = float(row.get('upper_bound_pct', 1.5)) * 100
                    _lo_sp_ov  = b_sp * float(row.get('lower_bound_pct', 0.5))
                    _hi_sp_ov  = b_sp * float(row.get('upper_bound_pct', 1.5))
                    for _bpct, _bsp, _blabel in [
                        (_lo_pct_ov, _lo_sp_ov, "Min Bound"),
                        (_hi_pct_ov, _hi_sp_ov, "Max Bound"),
                    ]:
                        fig_all.add_trace(go.Scatter(
                            x=[_bpct, _bpct], y=[0, 1],
                            mode='lines',
                            line=dict(color=col_c, width=1, dash="dot"),
                            opacity=0.45,
                            showlegend=False,
                            hovertemplate=(f"<b>{row['channel']} — {_blabel}</b><br>"
                                           f"Spend: ${_bsp:,.0f} ({_bpct:.0f}%)<extra></extra>"),
                        ))

            fig_all.add_hline(y=0, line_color="#CBD0D8", line_width=1)
            fig_all.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                height=440, margin=dict(l=10, r=10, t=20, b=50),
                legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=PALETTE["border"],
                            borderwidth=1, orientation="h", y=-0.13, font=dict(size=10)),
                xaxis=dict(title="% of Baseline Investment", ticksuffix="%",
                           gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                           tickfont=dict(size=10, color=PALETTE["muted"]),
                           range=[0, None], zeroline=False),
                yaxis=dict(title=y_label, gridcolor="#EAECF0",
                           linecolor=PALETTE["border2"],
                           tickfont=dict(size=10, color=PALETTE["muted"]),
                           tickformat="$.3s", zeroline=True,
                           zerolinecolor="#CBD0D8", zerolinewidth=1.5)
            )
            st.plotly_chart(fig_all, width='stretch')

        # ═══════════════════════════════════════════════════════════════════
        # INVESTMENT RESPONSE TABLE — matches the reference snapshot
        # ═══════════════════════════════════════════════════════════════════
        if show_table and not show_all:
            st.markdown("<div class='section-header' style='margin-top:.1rem'>Investment Response Table</div>",
                        unsafe_allow_html=True)

            # ── Build table rows at 10% steps 0→disp_hi, always include current & opt ──
            step_pcts = list(range(0, int(disp_hi) + 1, 10))
            key_pcts  = {round(curr_pct)}
            if has_opt:
                key_pcts.add(round(opt_pct))
            all_pcts = sorted(set(step_pcts) | key_pcts)

            tbl_rows = []
            prev_rv  = None   # for mROI_Sales calculation
            prev_sp  = None

            # Pre-compute bound spend values for this channel
            _tbl_lo_sp = base_sp * lo_pct   # lower_bound_pct × baseline_spend
            _tbl_hi_sp = base_sp * hi_pct   # upper_bound_pct × baseline_spend
            _tbl_lo_pct = lo_pct * 100
            _tbl_hi_pct = hi_pct * 100

            for pct in all_pcts:
                sp   = base_sp * pct / 100
                act  = sp / cpm if cpm > 0 else sp   # total_activity at this spend
                rv   = revenue_from_spend(sp, rrow, use_hill_global, use_log_global)
                pf   = rv - sp

                # ROI_Profit  = profit / spend
                roi_pf = pf / sp if sp > 0 else np.nan
                # ROI_Sales   = revenue / spend
                roi_sl = rv / sp if sp > 0 else np.nan

                # mROI (marginal) — central difference on revenue
                h_     = max(sp * 1e-4, 1.0)
                if sp > 0:
                    rv_up  = revenue_from_spend(sp + h_, rrow, use_hill_global, use_log_global)
                    rv_dn  = revenue_from_spend(sp - h_, rrow, use_hill_global, use_log_global)
                    mroi_sl = (rv_up - rv_dn) / (2 * h_)           # mROI_Sales
                    mroi_pf = mroi_sl - 1.0                         # mROI_Profit = d(rev)/d(spend) - 1
                else:
                    mroi_sl = mroi_pf = np.nan

                is_curr = (round(pct) == round(curr_pct))
                is_opt  = has_opt and (round(pct) == round(opt_pct))

                # Bound status
                if lock_val:
                    in_bounds = True   # locked channel — always "at bound"
                    bound_label = "🔒 Locked"
                elif sp < _tbl_lo_sp - 1:
                    in_bounds = False
                    bound_label = "⬇ Below Min"
                elif sp > _tbl_hi_sp + 1:
                    in_bounds = False
                    bound_label = "⬆ Above Max"
                else:
                    in_bounds = True
                    bound_label = "✓"

                def _fmt(v, decimals=2):
                    if pd.isna(v): return "—"
                    return f"{v:,.{decimals}f}"

                tbl_rows.append({
                    '_pct':    pct,
                    '_is_curr': is_curr,
                    '_is_opt':  is_opt,
                    '_in_bounds': in_bounds,
                    '_bound_label': bound_label,
                    '_sp': sp, '_rv': rv, '_pf': pf,
                    'Investment_%':  str(round(pct)) + "%",
                    'Total_Activity':_fmt(act, 0),
                    'Spend':         fmt(sp,"$"),
                    'Min Bound':     fmt(_tbl_lo_sp,"$"),
                    'Max Bound':     fmt(_tbl_hi_sp,"$"),
                    'Bounds':        bound_label,
                    'Impact':        fmt(rv,"$"),
                    'Profit':        fmt(pf, "$"),
                    'ROI_Profit':    _fmt(roi_pf,  2),
                    'mROI_Profit':   _fmt(mroi_pf, 2),
                    'ROI_Sales':     _fmt(roi_sl,  2),
                    'mROI_Sales':    _fmt(mroi_sl, 2),
                })

            # ── Colour scheme ──────────────────────────────────────────────────
            curr_color  = PALETTE["accent_lt"]
            opt_color   = PALETTE["accent2_lt"]
            both_color  = "#FFF7ED"
            muted_bg    = PALETTE["surface2"]
            border      = PALETTE["border"]
            muted_txt   = PALETTE["muted"]
            text_col    = PALETTE["text2"]
            green       = PALETTE["accent2"]
            red         = PALETTE["accent3"]
            blue        = PALETTE["accent"]
            orange      = "#E66800"
            out_of_bounds_bg = "#FEF3C7"   # amber tint for rows outside bounds

            # ── Legend ─────────────────────────────────────────────────────────
            lc = (f"<span style='display:inline-block;background:{curr_color};"
                  f"border:1.5px solid {blue};border-radius:5px;"
                  f"padding:2px 10px;font-size:.72rem;font-weight:600;color:{blue}'>&#11044; Current Investment</span>")
            lo = (f"<span style='display:inline-block;background:{opt_color};"
                  f"border:1.5px solid {green};border-radius:5px;"
                  f"padding:2px 10px;font-size:.72rem;font-weight:600;color:{green};margin-left:6px'>&#11044; Optimal Investment</span>") if has_opt else ""
            lb_note = "" if lock_val else (
                f"<span style='display:inline-block;background:{out_of_bounds_bg};"
                f"border:1.5px solid #F59E0B;border-radius:5px;"
                f"padding:2px 10px;font-size:.72rem;font-weight:600;color:#B45309;margin-left:6px'>"
                f"&#9888; Outside optimizer bounds ({fmt(_tbl_lo_sp,'$')} – {fmt(_tbl_hi_sp,'$')})</span>"
            )

            # ── Column headers ─────────────────────────────────────────────────
            th_style  = f"padding:.5rem .7rem;font-weight:700;font-size:.67rem;letter-spacing:.05em;text-transform:uppercase;color:#cc5500;white-space:nowrap;text-align:center"
            th_bound  = f"padding:.5rem .7rem;font-weight:700;font-size:.67rem;letter-spacing:.05em;text-transform:uppercase;color:#B45309;white-space:nowrap;text-align:center;background:#FFFBEB"
            tr_hdr    = (f"<tr style='background:{muted_bg};border-bottom:2px solid {border}'>"
                         f"<th style='{th_style}'>Investment %</th>"
                         f"<th style='{th_style}'>Total Activity</th>"
                         f"<th style='{th_style}'>Spend</th>"
                         f"<th style='{th_bound}'>Min Bound</th>"
                         f"<th style='{th_bound}'>Max Bound</th>"
                         f"<th style='{th_bound}'>In Bounds?</th>"
                         f"<th style='{th_style}'>Impact (Sales)</th>"
                         f"<th style='{th_style}'>Profit</th>"
                         f"<th style='{th_style}'>ROI Profit<br>(Profit / Spend)</th>"
                         f"<th style='{th_style}'>mROI Profit<br>Δ(Profit)/Δ(Spend)</th>"
                         f"<th style='{th_style}'>ROI Sales<br>(Sales / Spend)</th>"
                         f"<th style='{th_style}'>mROI Sales<br>Δ(Sales)/Δ(Spend)</th>"
                         f"</tr>")

            # ── Row HTML ───────────────────────────────────────────────────────
            td = "padding:.4rem .7rem;white-space:nowrap"
            rows_html = ""
            for r in tbl_rows:
                if r['_is_curr'] and r['_is_opt']:
                    bg, fw, marker = both_color, "700", " &#11044;&#11044;"
                elif r['_is_curr']:
                    bg, fw, marker = curr_color, "700", " &#11044;"
                elif r['_is_opt']:
                    bg, fw, marker = opt_color, "700", " &#11044;"
                else:
                    bg, fw, marker = "transparent", "400", ""

                # Rows outside the optimizer bounds get an amber tint (only if no other highlight)
                if not r['_in_bounds'] and bg == "transparent":
                    bg = out_of_bounds_bg

                # profit colour
                try:
                    pf_num = float(r['_pf'])
                    pf_col = green if pf_num >= 0 else red
                except Exception:
                    pf_col = text_col

                # bounds cell colour
                if r['_bound_label'] == "✓":
                    bnd_col = green; bnd_bg = "#ECFDF3"
                elif r['_bound_label'] == "🔒 Locked":
                    bnd_col = "#B45309"; bnd_bg = "#FFFBEB"
                else:
                    bnd_col = red; bnd_bg = "#FEF2F2"

                _inv_col = blue if bg not in ("transparent", out_of_bounds_bg) else text_col
                rows_html += (
                    f"<tr style='background:{bg};border-bottom:1px solid {border}'>"
                    f"<td style='text-align:center;{td};font-weight:{fw};color:{_inv_col}'>"
                    f"{r['Investment_%']}{marker}</td>"
                    f"<td style='text-align:center;{td}'>{r['Total_Activity']}</td>"
                    f"<td style='text-align:center;{td}'>{r['Spend']}</td>"
                    f"<td style='text-align:center;{td};background:#FFFBEB;color:#B45309;font-weight:600'>{r['Min Bound']}</td>"
                    f"<td style='text-align:center;{td};background:#FFFBEB;color:#B45309;font-weight:600'>{r['Max Bound']}</td>"
                    f"<td style='text-align:center;{td};background:{bnd_bg};color:{bnd_col};font-weight:600'>{r['Bounds']}</td>"
                    f"<td style='text-align:center;{td}'>{r['Impact']}</td>"
                    f"<td style='text-align:center;{td};font-weight:{fw};color:{pf_col}'>{r['Profit']}</td>"
                    f"<td style='text-align:center;{td}'>{r['ROI_Profit']}</td>"
                    f"<td style='text-align:center;{td}'>{r['mROI_Profit']}</td>"
                    f"<td style='text-align:center;{td}'>{r['ROI_Sales']}</td>"
                    f"<td style='text-align:center;{td}'>{r['mROI_Sales']}</td>"
                    f"</tr>"
                )

            # ── Assemble & render ──────────────────────────────────────────────
            html = (
                    f"<div style='margin-bottom:.5rem;text-align:left'>{lc}{lo}{lb_note}</div>"
                    f"<div style='width:100%;overflow-x:auto;"
                    f"border:1.5px solid {border};"
                    f"border-radius:12px;"
                    f"overflow:hidden;"
                    f"margin-left:0;"
                    f"margin-right:auto;"
                    f"text-align:left;'>"
                    f"<table style='width:100%;"
                    f"border-collapse:collapse;"
                    f"font-size:.82rem;"
                    f"font-family:Inter,sans-serif;"
                    f"margin-left:0;"
                    f"margin-right:auto;"
                    f"text-align:left;'>"
                    f"<thead>{tr_hdr}</thead>"
                    f"<tbody>{rows_html}</tbody>"
                    f"</table>"
                    f"</div>"


            )
            st.markdown(html, unsafe_allow_html=True)

            # CSV download
            import io as _io
            _csv_buf = _io.StringIO()
            _csv_df  = pd.DataFrame([{
                'Investment_%':   r['Investment_%'],
                'Total_Activity': r['Total_Activity'],
                'Spend':          r['Spend'],
                'Min_Bound':      r['Min Bound'],
                'Max_Bound':      r['Max Bound'],
                'In_Bounds':      r['Bounds'],
                'Impact':         r['Impact'],
                'Profit':         r['Profit'],
                'ROI_Profit':     r['ROI_Profit'],
                'mROI_Profit':    r['mROI_Profit'],
                'ROI_Sales':      r['ROI_Sales'],
                'mROI_Sales':     r['mROI_Sales'],
            } for r in tbl_rows])
            _csv_df.to_csv(_csv_buf, index=False)

            # ── Download button — inside show_table block so _csv_buf always exists ──
            st.markdown(f"<div style='height:.3rem'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download table (CSV)",
                data=_csv_buf.getvalue(),
                file_name=f"response_table_{selected_ch.replace(' ','_')}.csv",
                mime="text/csv",
                key="rc_tbl_download"
            )

    # ─────────────────────────────────────────────────────────────
    # AI Interpret button — always visible regardless of show_table
    # ─────────────────────────────────────────────────────────────
    st.markdown(f"<div style='height:.4rem'></div>", unsafe_allow_html=True)
    _, btn_ai_col, _ = st.columns([0.25, 0.50, 0.25])
    with btn_ai_col:
        run_ai_curve = st.button(
            "🤖 Interpret using AI",
            key=f"ai_curve_interp_{selected_ch}",
            type="primary",
            width="stretch",
            help="AI explanation of the response curve and next steps"
        )


    # ─────────────────────────────────────────────────────────────
    # AI Interpretation Logic
    # ─────────────────────────────────────────────────────────────

    if run_ai_curve:

        _ch_desc_rc  = ch_desc_map.get(selected_ch, selected_ch)
        _all_ch_ctx  = "\n".join([
            f"  {r['channel']} ({ch_desc_map.get(r['channel'], r['channel'])}): "
            f"spend=${float(r['total_spend']):,.0f}, ROI={float(r['baseline_roi']):.2f}x, "
            f"mROI={float(r['baseline_mroi']):.2f}x, alpha={float(r['alpha']):.3f}"
            for _, r in df.iterrows()
        ])
        _opt_note = (
            f"Optimizer recommended: ${opt_sp:,.0f} spend (Δ{opt_pct-100:+.1f}%), "
            f"profit ${opt_prof:,.0f}" if has_opt else "Optimizer not yet run."
        )
        ai_prompt = (
            f"RESPONSE CURVE ANALYSIS — {selected_ch} ({_ch_desc_rc})\n\n"
            f"CHANNEL:\n"
            f"  Type: {_ch_desc_rc}\n"
            f"  Transformation: {rrow['type_transformation']} | "
            f"Alpha: {float(rrow['alpha']):.3f} | "
            f"Coefficient: {float(rrow['coefficient']):.5f} | "
            f"Adj. Factor: {float(rrow['Adj_Factor']):.3f}\n"
            f"  Segments (DMAs): {int(rrow['total_segments'])} | "
            f"Activity: {float(rrow['total_activity']):,.0f}\n\n"
            f"PERFORMANCE:\n"
            f"  Spend: ${curr_sp:,.0f} | Revenue: ${curr_rev:,.0f} | "
            f"Profit: ${curr_prof:,.0f}\n"
            f"  ROI: {roas_val:.2f}x | mROI: {mroi_val:.2f}x\n"
            f"  {_opt_note}\n\n"
            f"PORTFOLIO (all channels):\n{_all_ch_ctx}\n\n"
            f"Analyse using the four-section structure. "
            f"Focus on: (1) what alpha= {float(rrow['alpha']):.3f} means for this specific "
            f"channel type ({_ch_desc_rc}) — saturation and diminishing returns; "
            f"(2) whether mROI = {mroi_val:.2f}x indicates under/over-investment; "
            f"(3) specific spend recommendation with estimated impact; "
            f"(4) how this channel compares to others in the pharma mix."
        )

        with st.spinner("Generating AI interpretation…"):
            ai_output = call_ai(_SYSTEM_PROMPT + "\n\n" + ai_prompt)

        st.markdown("---")
        st.markdown("### 🤖 AI Interpretation")
        st.markdown(ai_output)


# ─────────────────────────────────────────────────────────────
# Mini per-channel response curves (checkbox controlled)
# Updated: respects Profit/Revenue selection
# ─────────────────────────────────────────────────────────────

if show_all_curves and PLOTLY:

    curve_header = (
        "All channels · Profit response (Activity domain)"
        if curve_type == "Profit Curve"
        else "All channels · Revenue response (Activity domain)"
    )

    st.markdown(
        f"<div class='section-header' style='margin-top:.75rem'>{curve_header}</div>",
        unsafe_allow_html=True
    )

    n_cols_mini = min(len(df), 4)
    mini_cols = st.columns(n_cols_mini)

    for i, (_, row) in enumerate(df.iterrows()):
        with mini_cols[i % n_cols_mini]:

            sp_r   = float(row['total_spend'])
            col_c  = channel_color(i)
            _hp2   = max(float(row.get('upper_bound_pct', 1.5)) * 100, 150.0)

            pct_m = np.linspace(0, _hp2, 300)
            s_m   = sp_r * pct_m / 100

            # Revenue curve
            rev_m = np.array([
                revenue_from_spend(s, row, use_hill_global, use_log_global)
                for s in s_m
            ])

            # Dynamic curve selection
            if curve_type == "Profit Curve":
                curve_m = rev_m - s_m
                y_title = "Profit ($)"
            else:
                curve_m = rev_m
                y_title = "Revenue ($)"

            _curve_min = float(curve_m.min())
            _curve_max = float(curve_m.max())

            _y_lo = min(_curve_min * 0.88 if _curve_min < 0 else 0, 0)
            _y_hi = _curve_max * 1.12

            f_m = go.Figure()

            # Main curve
            f_m.add_trace(go.Scatter(
                x=pct_m,
                y=curve_m,
                mode='lines',
                line=dict(color=col_c, width=2),
                hovertemplate=(
                    f"<b>{row['channel']}</b><br>"
                    f"Investment: %{{x:.0f}}%<br>"
                    f"{'Profit' if curve_type == 'Profit Curve' else 'Revenue'}: "
                    f"$%{{y:,.0f}}<extra></extra>"
                )
            ))

            # Current baseline marker (100%)
            f_m.add_vline(
                x=100.0,
                line_dash='dot',
                line_color=PALETTE["accent3"],
                line_width=1
            )

            # Bound lines (amber dotted) — skip for locked channels
            if not int(row.get('lock_spend', 0)) == 1:
                _lo_m = float(row.get('lower_bound_pct', 0.5)) * 100
                _hi_m = float(row.get('upper_bound_pct', 1.5)) * 100
                _lo_sp_m = sp_r * float(row.get('lower_bound_pct', 0.5))
                _hi_sp_m = sp_r * float(row.get('upper_bound_pct', 1.5))
                f_m.add_vline(
                    x=_lo_m,
                    line=dict(color="#F59E0B", width=1.2, dash="dot"),
                    opacity=0.8,
                    annotation_text=f"Min {fmt(_lo_sp_m,'$')}",
                    annotation_position="top right",
                    annotation_font=dict(size=7, color="#B45309"),
                )
                f_m.add_vline(
                    x=_hi_m,
                    line=dict(color="#F59E0B", width=1.2, dash="dot"),
                    opacity=0.8,
                    annotation_text=f"Max {fmt(_hi_sp_m,'$')}",
                    annotation_position="top left",
                    annotation_font=dict(size=7, color="#B45309"),
                )
                f_m.add_vrect(
                    x0=_lo_m, x1=_hi_m,
                    fillcolor="rgba(245,158,11,0.06)",
                    line_width=0,
                    layer="below"
                )

            f_m.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#FAFBFC',
                height=250,
                margin=dict(l=5, r=5, t=36, b=40),
                showlegend=False,
                title=dict(
                    text=row['channel'],
                    font=dict(size=10)
                ),
                xaxis=dict(
                    title='% of Baseline',
                    gridcolor='#EAECF0',
                    linecolor=PALETTE['border2'],
                    ticksuffix='%',
                    tickfont=dict(
                        size=8,
                        color=PALETTE['muted']
                    ),
                    range=[0, _hp2]
                ),
                yaxis=dict(
                    title=y_title,
                    gridcolor='#EAECF0',
                    linecolor=PALETTE['border2'],
                    tickfont=dict(
                        size=8,
                        color=PALETTE['muted']
                    ),
                    tickformat='$.3s',
                    range=[_y_lo, _y_hi]
                )
            )

            st.plotly_chart(f_m, width='stretch')


with tab_opt:
    opt_ctrl, opt_results = st.columns([.28, .72])

    with opt_ctrl:
        st.markdown(f"<div class='section-header'>Optimization Settings</div>", unsafe_allow_html=True)

        budget_mode = st.radio(
            "Budget mode",
            ["Keep total budget", "Model recommends budget", "Set new budget"],
            help=(
                "Keep total budget: redistribute the same total spend optimally.\n"
                "Model recommends budget: scans 50%–300% of baseline to find the "
                "budget that maximises profit (takes ~30s).\n"
                "Set new budget: manually specify a total budget."
            )
        )

        _locked_spend_min = float((df['total_spend'] * df['lock_spend'].astype(int)).sum())
        _baseline_total   = float(df['total_spend'].sum())
        
        # Check for face-to-face channel (common pharma channel name)
        _f2f_channels = [ch for ch in df['channel'].tolist() if 'face' in ch.lower() or 'f2f' in ch.lower() or 'f2f' in ch.lower().replace(' ','')]
        _has_f2f = len(_f2f_channels) > 0
        _f2f_channel = _f2f_channels[0] if _has_f2f else None

        # Default scan df — overridden in Model recommends budget mode if unlock is checked
        df_scan = df.copy()

        # Read accept flag directly from session_state so it's always current
        # regardless of whether the checkbox has rendered yet this pass.
        _accept_budget_warning = bool(st.session_state.get("snb_accept_warning", False))
        _budget_warning = False   # set True inside Set new budget block when infeasible

        # ── Pre-populate override_bounds from session_state ───────────────────
        # The Per-channel bounds expander (rendered BELOW the feasibility check)
        # stores widget values under keys "lo_{channel}" / "hi_{channel}".
        # By reading session_state here we ensure the feasibility check always
        # uses the user's current bound values, not the empty-dict default.
        override_bounds = {}
        for _, _row in df.iterrows():
            if int(_row.get('lock_spend', 0)) == 1:
                continue
            _ch = _row['channel']
            _lo_key = f"lo_{_ch}"
            _hi_key = f"hi_{_ch}"
            _lo_default = float(_row.get('lower_bound_pct', 0.5))
            _hi_default = float(_row.get('upper_bound_pct', 1.5))
            # Use session_state value if widget has been rendered before,
            # otherwise fall back to the value from the input data.
            _lo_val = st.session_state.get(_lo_key, _lo_default * 100) / 100.0
            _hi_val = st.session_state.get(_hi_key, _hi_default * 100) / 100.0
            override_bounds[_ch] = (_lo_val, _hi_val)

        if budget_mode == "Keep total budget":
            opt_budget = _baseline_total
            st.markdown(f"<div class='alert-box alert-info'>Total budget: "
                        f"<b>{fmt(opt_budget, '$')}</b></div>", unsafe_allow_html=True)
            
            # ── Face-to-Face unlock option for Keep total budget ─────────────
            if _has_f2f:
                _f2f_spend = float(df[df['channel'] == _f2f_channel]['total_spend'].iloc[0])
                st.markdown(
                    f"<div style='font-size:.7rem;margin-top:.5rem;color:{PALETTE['muted']}'>"
                    f"Channel Unlock Options</div>",
                    unsafe_allow_html=True
                )
                _unlock_f2f_ktb = st.checkbox(
                    f"🔓 Unlock {_f2f_channel} for reallocation",
                    value=False,
                    key="ktb_unlock_f2f",
                    help=(
                        f"Unlock {_f2f_channel} channel to allow the optimizer to "
                        f"redistribute its budget of {fmt(_f2f_spend,'$')} across other channels."
                    )
                )
                if _unlock_f2f_ktb:
                    # Apply unlock to df_scan
                    mask = df_scan['channel'] == _f2f_channel
                    df_scan.loc[mask, 'lock_spend'] = 0
                    df_scan.loc[mask, 'lower_bound_pct'] = 0.5
                    df_scan.loc[mask, 'upper_bound_pct'] = 3.0
                    st.markdown(
                        f"<div class='alert-box alert-info' style='font-size:.75rem'>"
                        f"✅ {_f2f_channel} unlocked · will be included in optimization</div>",
                        unsafe_allow_html=True
                    )
            else:
                _unlock_f2f_ktb = False

        elif budget_mode == "Model recommends budget":

            # ── Scan range controls ──────────────────────────────────────────
            st.markdown(
                f"<div style='font-size:.7rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:.06em;color:{PALETTE['muted']};padding:.3rem 0 .1rem'>"
                f"Scan Range</div>",
                unsafe_allow_html=True
            )

            # Show which objective the scan will use
            _mrb_obj_preview = obj_map.get(opt_objective, "profit")
            _mrb_obj_color = PALETTE["accent2"] if _mrb_obj_preview == "profit" else PALETTE["accent"]
            _mrb_obj_icon  = "💰" if _mrb_obj_preview == "profit" else "📈"
            st.markdown(
                f"<div class='alert-box alert-info' style='font-size:.75rem;margin-bottom:.4rem'>"
                f"{_mrb_obj_icon} Scan will <b>maximise {_mrb_obj_preview.upper()}</b> "
                f"at each budget level (matches your Objective setting above). "
                f"Per-channel bounds are strictly enforced during the scan.</div>",
                unsafe_allow_html=True
            )
            _sr_col1, _sr_col2 = st.columns(2)

            with _sr_col1:
                _scan_lo_pct = st.number_input(
                    "Min budget (% of baseline)",
                    value=70,
                    min_value=10,
                    max_value=200,
                    step=5,
                    key="mrb_lo_pct",
                    help=f"Scan starts at this % of baseline {fmt(_baseline_total,'$')}. "
                         f"Default 70% = {fmt(_baseline_total*0.7,'$')}"
                ) / 100.0

            with _sr_col2:
                _scan_hi_pct = st.number_input(
                    "Max budget (% of baseline)",
                    value=150,
                    min_value=50,
                    max_value=500,
                    step=10,
                    key="mrb_hi_pct",
                    help=f"Scan ends at this % of baseline {fmt(_baseline_total,'$')}. "
                         f"Default 150% = {fmt(_baseline_total*1.5,'$')}"
                ) / 100.0

            # ── Budget increment control ──────────────────────────────────────
            _incr_options = {
                "$500K steps":   500_000,
                "$1M steps":   1_000_000,
                "$2M steps":   2_000_000,
                "$5M steps":   5_000_000,
                "$10M steps": 10_000_000,
            }

            _scan_incr_label = st.selectbox(
                "Budget increment",
                list(_incr_options.keys()),
                index=3,
                key="mrb_increment",
                help="Each scan point is this many dollars apart. "
                     "Smaller = more scan points = slower but more precise."
            )

            _scan_increment = _incr_options[_scan_incr_label]

            # ── Preview scan points ──────────────────────────────────────────
            _scan_lo_abs = max(
                round(_baseline_total * _scan_lo_pct / _scan_increment) * _scan_increment,
                _scan_increment
            )

            _scan_hi_abs = round(
                _baseline_total * _scan_hi_pct / _scan_increment
            ) * _scan_increment

            _scan_pts_arr = np.arange(
                _scan_lo_abs,
                _scan_hi_abs + _scan_increment * 0.5,
                _scan_increment
            )

            _n_pts = len(_scan_pts_arr)
            _est_secs = _n_pts * 2

            st.markdown(
                f"<div class='alert-box alert-info' style='font-size:.76rem'>"
                f"📊 <b>{_n_pts} scan points</b>: "
                f"{fmt(_scan_lo_abs,'$')} → {fmt(_scan_hi_abs,'$')} "
                f"every {fmt(_scan_increment,'$')}. "
                f"Est. time: ~{_est_secs}s.</div>",
                unsafe_allow_html=True
            )

            # ── Lock override toggle ─────────────────────────────────────────
            _locked_channels = df[
                df['lock_spend'].astype(int) == 1
            ]['channel'].tolist()

            _has_locks = len(_locked_channels) > 0

            if _has_locks:
                st.markdown(
                    f"<div class='alert-box alert-warn' style='font-size:.76rem;margin-top:.4rem'>"
                    f"⚠️ <b>{', '.join(_locked_channels)}</b> "
                    f"{'is' if len(_locked_channels)==1 else 'are'} locked "
                    f"({fmt(_locked_spend_min,'$')}). "
                    f"Unlock to allow full reallocation.</div>",
                    unsafe_allow_html=True
                )

                _unlock_for_scan = st.checkbox(
                    f"🔓 Unlock {', '.join(_locked_channels)} for this scan",
                    value=True,
                    key="mrb_unlock_locked",
                    help="Freed for this mode only — lock is NOT permanently changed."
                )
            else:
                _unlock_for_scan = False

            # ── Build scan dataframe ─────────────────────────────────────────
            df_scan = df.copy()

            if _has_locks and _unlock_for_scan:
                df_scan['lock_spend'] = 0

                for ch in _locked_channels:
                    mask = df_scan['channel'] == ch
                    df_scan.loc[mask, 'lower_bound_pct'] = 0.5
                    df_scan.loc[mask, 'upper_bound_pct'] = 3.0

            # ── Face-to-Face unlock option ──────────────────────────────────
            if _has_f2f and _f2f_channel not in _locked_channels:

                _f2f_spend = float(
                    df[df['channel'] == _f2f_channel]['total_spend'].iloc[0]
                )

                _f2f_locked = int(
                    df[df['channel'] == _f2f_channel]['lock_spend'].iloc[0]
                ) == 1

                st.markdown(
                    f"<div style='font-size:.7rem;margin-top:.5rem;"
                    f"color:{PALETTE['muted']}'>"
                    f"Additional Channel Options</div>",
                    unsafe_allow_html=True
                )

                _unlock_f2f_mrb = st.checkbox(
                    f"🔓 Unlock {_f2f_channel} for this scan",
                    value=not _f2f_locked,
                    key="mrb_unlock_f2f",
                    help=(
                        f"Unlock {_f2f_channel} "
                        f"(current spend: {fmt(_f2f_spend,'$')}) "
                        f"to allow reallocation during scan."
                    )
                )

                if _unlock_f2f_mrb:
                    mask = df_scan['channel'] == _f2f_channel
                    df_scan.loc[mask, 'lock_spend'] = 0
                    df_scan.loc[mask, 'lower_bound_pct'] = 0.5
                    df_scan.loc[mask, 'upper_bound_pct'] = 3.0

                    st.markdown(
                        f"<div class='alert-box alert-info' style='font-size:.75rem'>"
                        f"✅ {_f2f_channel} unlocked for scan</div>",
                        unsafe_allow_html=True
                    )
            else:
                _unlock_f2f_mrb = False

            # ── Scan key ─────────────────────────────────────────────────────
            _mrb_obj_label = obj_map[opt_objective]  # "profit" or "revenue"
            scan_key = (
                f"optimal_budget_scan|{use_hill_global}|{use_log_global}|"
                f"{_unlock_for_scan}|{_scan_lo_pct:.2f}|"
                f"{_scan_hi_pct:.2f}|{_scan_increment:.0f}|{_mrb_obj_label}"
            )

            st.session_state["_mrb_scan_key"] = scan_key

            # ── Run frontier scan ────────────────────────────────────────────
            if st.button(
                "🔍 Find Optimal Budget",
                key="find_opt_bgt_btn",
                width="stretch"
            ):

                _spinner_msg = (
                    f"Scanning {_n_pts} budget levels "
                    f"({fmt(_scan_lo_abs,'$')} → {fmt(_scan_hi_abs,'$')} "
                    f"every {fmt(_scan_increment,'$')})…"
                )

                with st.spinner(_spinner_msg):

                    scan_result = find_optimal_budget(
                        df_scan,
                        use_hill=use_hill_global,
                        use_log=use_log_global,
                        lo_pct=_scan_lo_pct,
                        hi_pct=_scan_hi_pct,
                        increment=_scan_increment,
                        respect_bounds=True,
                        strict_feasible=True,
                        objective=_mrb_obj_label
                    )

                st.session_state[scan_key] = scan_result

            # ── Read scan result ─────────────────────────────────────────────
            scan_result = st.session_state.get(scan_key)

            if scan_result and scan_result.get("success"):

                # Infeasible budgets excluded
                if scan_result.get("infeasible_points", 0) > 0:

                    st.markdown(
                        f"<div class='alert-box alert-warn' "
                        f"style='font-size:.76rem'>"
                        f"⚠️ <b>{scan_result['infeasible_points']} "
                        f"budget level(s)</b> were skipped because channel "
                        f"constraints could not be satisfied."
                        f"<br>This often happens when Face-to-Face is locked "
                        f"or some channels hit max bounds."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                opt_budget = scan_result["optimal_budget"]

                # ── Detect scan ceiling hit (revenue-max always hits top of range) ──
                _scan_hi_abs_val = _baseline_total * _scan_hi_pct
                _at_ceiling = abs(opt_budget - _scan_hi_abs_val) < _scan_increment * 0.6
                if _at_ceiling and _mrb_obj_label == "revenue":
                    st.markdown(
                        f"<div class='alert-box alert-warn' style='font-size:.76rem'>"
                        f"📈 <b>Revenue-max hit the scan ceiling ({fmt(opt_budget,'$')}).</b>"
                        f"<br>For a power/log response curve, revenue is always increasing, "
                        f"so the revenue-optimal budget is the highest budget scanned. "
                        f"Consider increasing the <b>Max budget %</b> input above to explore "
                        f"larger budgets, or accept the current ceiling as a practical cap."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                # Detect if recommended budget still needs warning
                _constraint_warning = scan_result.get(
                    "constraint_warning",
                    False
                )

                if _constraint_warning:

                    st.markdown(
                        f"<div class='alert-box alert-warn' "
                        f"style='font-size:.78rem'>"
                        f"⚠️ <b>Recommended budget "
                        f"{fmt(opt_budget,'$')}</b> may exceed one or more "
                        f"channel bounds due to locked channels."
                        f"<br>Please confirm before viewing optimized results."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    _accept_warning = st.checkbox(
                        "✅ I understand — show optimized table anyway",
                        key="mrb_accept_warning"
                    )

                    if _accept_warning:

                        st.markdown(
                            f"<div class='alert-box alert-success' "
                            f"style='font-size:.78rem'>"
                            f"✅ <b>{fmt(opt_budget,'$')}</b> recommended "
                            f"({(opt_budget/_baseline_total-1)*100:+.1f}% "
                            f"vs baseline)</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        opt_budget = _baseline_total

                else:

                    _lock_note = (
                        " · all channels unlocked"
                        if (_has_locks and _unlock_for_scan)
                        else ""
                    )

                    st.markdown(
                        f"<div class='alert-box alert-success' "
                        f"style='font-size:.78rem'>"
                        f"✅ <b>{fmt(opt_budget,'$')}</b> recommended "
                        f"({(opt_budget/_baseline_total-1)*100:+.1f}% "
                        f"vs baseline){_lock_note}</div>",
                        unsafe_allow_html=True
                    )

            else:
                opt_budget = _baseline_total

        else:  # Set new budget
            df_scan = df.copy()

            opt_budget = st.number_input(
                "Total budget ($)",
                value=_baseline_total,
                min_value=_locked_spend_min if _locked_spend_min > 0 else 1.0,
                step=50000.0,
                format="%.0f"
            )

            delta_b = opt_budget - _baseline_total
            _budget_warning = False

            # ── Face-to-Face unlock option for Set new budget ────────────────
            if _has_f2f:
                _f2f_spend = float(
                    df[df['channel'] == _f2f_channel]['total_spend'].iloc[0]
                )

                _f2f_locked = int(
                    df[df['channel'] == _f2f_channel]['lock_spend'].iloc[0]
                ) == 1

                st.markdown(
                    f"<div style='font-size:.7rem;margin-top:.5rem;"
                    f"color:{PALETTE['muted']}'>"
                    f"Channel Unlock Options</div>",
                    unsafe_allow_html=True
                )

                _unlock_f2f_snb = st.checkbox(
                    f"🔓 Unlock {_f2f_channel} for reallocation",
                    value=not _f2f_locked,
                    key="snb_unlock_f2f",
                    help=(
                        f"Unlock {_f2f_channel} channel "
                        f"(current spend: {fmt(_f2f_spend,'$')}) "
                        f"to allow optimizer redistribution."
                    )
                )

                if _unlock_f2f_snb:
                    mask = df_scan['channel'] == _f2f_channel
                    df_scan.loc[mask, 'lock_spend'] = 0
                    df_scan.loc[mask, 'lower_bound_pct'] = 0.5
                    df_scan.loc[mask, 'upper_bound_pct'] = 3.0

                    st.markdown(
                        f"<div class='alert-box alert-info' "
                        f"style='font-size:.75rem'>"
                        f"✅ {_f2f_channel} unlocked · included in optimization"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                _unlock_f2f_snb = False

            # ── Standard budget message ──────────────────────────────────────
            if _locked_spend_min > 0 and opt_budget <= _locked_spend_min:
                st.markdown(
                    f"<div class='alert-box alert-warn'>"
                    f"⚠️ Budget {fmt(opt_budget,'$')} is at or below "
                    f"locked spend ({fmt(_locked_spend_min,'$')}). "
                    f"No budget remains for flexible channels."
                    f"</div>",
                    unsafe_allow_html=True
                )
                _budget_warning = True

            else:
                _flex_remain = opt_budget - _locked_spend_min
                cls_ = "alert-success" if delta_b >= 0 else "alert-warn"

                st.markdown(
                    f"<div class='alert-box {cls_}'>"
                    f"Delta vs baseline: "
                    f"<b>{fmt(delta_b,'$','+' if delta_b>=0 else '')}</b>"
                    + (
                        f" &nbsp;·&nbsp; Flexible budget: "
                        f"<b>{fmt(_flex_remain,'$')}</b>"
                        if _locked_spend_min > 0 else ""
                    )
                    + "</div>",
                    unsafe_allow_html=True
                )

            # ── HIGH BUDGET IMPRACTICALITY WARNING ───────────────────────────
            _budget_ratio = opt_budget / _baseline_total if _baseline_total > 0 else 1.0
            if _budget_ratio >= 2.5:
                st.markdown(
                    f"<div class='alert-box alert-warn' style='font-size:.78rem'>"
                    f"🚨 <b>Very high budget: {(_budget_ratio-1)*100:.0f}% above baseline.</b>"
                    f"<br>Budgets above 250% of baseline are typically impractical in pharma "
                    f"commercial planning. Channels will be deeply in diminishing-returns territory "
                    f"and mROI will be very low. Consider whether this scenario is actionable."
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif _budget_ratio >= 1.75:
                st.markdown(
                    f"<div class='alert-box alert-warn' style='font-size:.78rem'>"
                    f"⚠️ <b>High budget: {(_budget_ratio-1)*100:.0f}% above baseline.</b>"
                    f"<br>This is a stretch scenario. At this level, most channels will show "
                    f"significant diminishing returns (mROI well below 1). "
                    f"Ensure this budget level is operationally feasible."
                    f"</div>",
                    unsafe_allow_html=True
                )

            # ── HARD FEASIBILITY CHECK (RESPECT LOCKS + OVERRIDE BOUNDS) ─────
            # Build effective bounds: start from df_scan (which has unlock applied),
            # then layer on any per-channel overrides the user set in the expander.
            _mins = []
            _maxs = []

            for _, r in df_scan.iterrows():
                b      = float(r["total_spend"])
                locked = int(r.get("lock_spend", 0)) == 1
                ch     = r["channel"]

                if locked:
                    _mins.append(b)
                    _maxs.append(b)
                else:
                    # Use override if user set it, otherwise use df_scan bound
                    if ch in override_bounds:
                        lo_eff, hi_eff = override_bounds[ch]
                    else:
                        lo_eff = float(r["lower_bound_pct"])
                        hi_eff = float(r["upper_bound_pct"])
                    _mins.append(b * lo_eff)
                    _maxs.append(b * hi_eff)

            _min_total = float(sum(_mins))
            _max_total = float(sum(_maxs))

            if opt_budget < _min_total:
                _budget_warning = True

                st.markdown(
                    f"<div class='alert-box alert-warn' "
                    f"style='font-size:.78rem'>"
                    f"⚠️ <b>Budget too LOW for current channel constraints.</b>"
                    f"<br>Entered: <b>{fmt(opt_budget,'$')}</b>"
                    f"<br>Minimum feasible: <b>{fmt(_min_total,'$')}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            elif opt_budget > _max_total:
                _budget_warning = True
                _needed_ratio_ctrl = (opt_budget - float(
                    (df_scan[df_scan['lock_spend'].astype(int)==1]['total_spend']).sum()
                )) / max(float(
                    (df_scan[df_scan['lock_spend'].astype(int)==0]['total_spend']).sum()
                ), 1.0)

                if _has_f2f and not _unlock_f2f_snb:
                    _warn_msg = (
                        f"⚠️ <b>Budget exceeds feasible maximum.</b>"
                        f"<br>Entered: <b>{fmt(opt_budget,'$')}</b>"
                        f"<br>Maximum feasible with locked {_f2f_channel}: "
                        f"<b>{fmt(_max_total,'$')}</b>"
                        f"<br><br>Unlock {_f2f_channel} or lower budget."
                    )
                else:
                    _warn_msg = (
                        f"⚠️ <b>Budget exceeds channel max bounds.</b>"
                        f"<br>Entered: <b>{fmt(opt_budget,'$')}</b>"
                        f"<br>Maximum feasible: <b>{fmt(_max_total,'$')}</b>"
                        f"<br>Tip: increase channel Max% bounds below."
                    )

                st.markdown(
                    f"<div class='alert-box alert-warn' "
                    f"style='font-size:.78rem'>{_warn_msg}</div>",
                    unsafe_allow_html=True
                )

                # Show per-channel guidance on what Max% to set
                _ctrl_flex = df_scan[df_scan['lock_spend'].astype(int) == 0]
                _ctrl_guide = []
                for _, _cgr in _ctrl_flex.iterrows():
                    _cg_base   = float(_cgr['total_spend'])
                    _cg_cur_hi = override_bounds.get(_cgr['channel'], (None, float(_cgr.get('upper_bound_pct',1.5))))[1]
                    _cg_cur_max = _cg_base * _cg_cur_hi
                    _cg_needed  = _needed_ratio_ctrl * 1.05
                    if _cg_needed > _cg_cur_hi:
                        _ctrl_guide.append({
                            "Channel":      _cgr['channel'],
                            "Current Max%": f"{_cg_cur_hi*100:.0f}%",
                            "Current Max $": fmt(_cg_cur_max, "$"),
                            "Suggested Max%": f"{_cg_needed*100:.0f}%",
                            "Suggested Max $": fmt(_cg_base * _cg_needed, "$"),
                        })
                if _ctrl_guide:
                    st.markdown(
                        f"<div style='font-size:.72rem;font-weight:600;"
                        f"margin:.4rem 0 .2rem'>💡 Suggested Max% increases:</div>",
                        unsafe_allow_html=True
                    )
                    st.dataframe(pd.DataFrame(_ctrl_guide), hide_index=True,
                                 use_container_width=True)

            elif not _budget_warning:
                st.markdown(
                    f"<div class='alert-box alert-success' "
                    f"style='font-size:.76rem'>"
                    f"✅ Budget is feasible within all channel constraints."
                    f"</div>",
                    unsafe_allow_html=True
                )

            # ── CONSTRAINT SUMMARY for Set new budget ────────────────────────
            with st.expander("📐 Channel constraint summary for this budget", expanded=False):
                _snb_rows = []
                for _, _r in df_scan.iterrows():
                    _b      = float(_r["total_spend"])
                    _locked = int(_r.get("lock_spend", 0)) == 1
                    _ch     = _r["channel"]
                    if _locked:
                        _snb_rows.append({
                            "Channel": _ch,
                            "Status": "🔒 Locked",
                            "Min ($)": fmt(_b, "$"),
                            "Max ($)": fmt(_b, "$"),
                            "Feasible?": "🔒",
                            "Share of Baseline": f"{_b/_baseline_total*100:.1f}%",
                        })
                    else:
                        # Use effective bound (override takes precedence)
                        if _ch in override_bounds:
                            _lo_pct_eff, _hi_pct_eff = override_bounds[_ch]
                        else:
                            _lo_pct_eff = float(_r["lower_bound_pct"])
                            _hi_pct_eff = float(_r["upper_bound_pct"])
                        _lo_amt = _b * _lo_pct_eff
                        _hi_amt = _b * _hi_pct_eff
                        _ch_feasible = (_lo_amt <= opt_budget) and (opt_budget <= _max_total)
                        _snb_rows.append({
                            "Channel": _ch,
                            "Status": "🔓 Flexible",
                            "Min ($)": fmt(_lo_amt, "$"),
                            "Max ($)": fmt(_hi_amt, "$"),
                            "Feasible?": "✓" if _ch_feasible else "⚠️",
                            "Share of Baseline": f"{_b/_baseline_total*100:.1f}%",
                        })
                st.dataframe(pd.DataFrame(_snb_rows), hide_index=True, width="stretch")
                _muted_color = PALETTE["muted"]
                st.markdown(
                    f"<div style='font-size:.72rem;color:{_muted_color};margin-top:.3rem'>"
                    f"Total feasible range: <b>{fmt(_min_total,'$')}</b> – <b>{fmt(_max_total,'$')}</b> "
                    f"&nbsp;·&nbsp; Your budget: <b>{fmt(opt_budget,'$')}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # ── Confirmation required if warning exists ──────────────────────
            if _budget_warning:
                _accept_budget_warning = st.checkbox(
                    "✅ I understand — run optimization anyway",
                    key="snb_accept_warning"
                )

        st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)
        _muted = PALETTE["muted"]

        # ── How allocation works explanation ─────────────────────────────────
        _alloc_bg    = PALETTE["surface2"]
        _alloc_bdr   = PALETTE["border"]
        _alloc_txt   = PALETTE["text2"]
        _alloc_green = PALETTE["accent2"]
        st.markdown(
            f"<div style='background:{_alloc_bg};border:1px solid {_alloc_bdr};"
            f"border-radius:10px;padding:.65rem .85rem;font-size:.76rem;"
            f"color:{_alloc_txt};margin-bottom:.5rem'>"
            f"<b>How budget allocation works:</b><br>"
            f"The optimizer automatically distributes the total budget across channels "
            f"to maximise your objective — no manual allocation needed.<br><br>"
            f"<b>Bounds (optional guard-rails):</b> Min% and Max% limit how far "
            f"each channel's spend can move from its baseline.<br><br>"
            f"<span style='color:{_alloc_green}'>✅ Constraints are now strictly enforced.</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='font-size:.72rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.06em;"
            f"color:{_muted};padding:.4rem 0'>"
            f"Channel Bounds (optional guard-rails)</div>",
            unsafe_allow_html=True
        )

        with st.expander("Per-channel bounds", expanded=False):

            for _, row in df.iterrows():

                if int(row.get('lock_spend', 0)) == 1:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;"
                        f"gap:.5rem;padding:.3rem 0;font-size:.8rem'>"
                        f"<span class='chip chip-gold'>"
                        f"🔒 {row['channel']} — locked at "
                        f"{fmt(float(row['total_spend']),'$')}</span></div>",
                        unsafe_allow_html=True
                    )
                    continue

                base = float(row['total_spend'])
                lo_pct = float(row.get('lower_bound_pct', 0.5)) * 100
                hi_pct = float(row.get('upper_bound_pct', 1.5)) * 100

                c0, c1, c2, c3 = st.columns([.16, .28, .28, .16])

                c0.markdown(
                f"<div style='font-size:.85rem;font-weight:400;"
                f"padding-top:.35rem'>{row['channel']}</div>",
                unsafe_allow_html=True
                           )


                lo_ov = c1.number_input(
                    "Min%",
                    value=lo_pct,
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key=f"lo_{row['channel']}"
                )

                hi_ov = c2.number_input(
                    "Max%",
                    value=hi_pct,
                    min_value=0.0,
                    max_value=500.0,
                    step=5.0,
                    key=f"hi_{row['channel']}"
                )

                c3.markdown(
                    f"<div style='font-size:.7rem;"
                    f"color:{PALETTE['muted']};padding-top:.45rem'>"
                    f"Base: {fmt(base*lo_ov/100,'$')} – "
                    f"{fmt(base*hi_ov/100,'$')}</div>",
                    unsafe_allow_html=True
                )

                override_bounds[row['channel']] = (
                    lo_ov / 100.0,
                    hi_ov / 100.0
                )

        st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)

        run_opt = st.button(
            "🚀 Run Optimization",
            type="primary",
            width='stretch'
        )

        if _budget_warning and not _accept_budget_warning:
            run_opt = False

        _bounds_sig = "|".join(
                    f"{ch}:{lo:.2f}:{hi:.2f}"
                    for ch, (lo, hi) in sorted(override_bounds.items())
                            )    

    with opt_results:
        # ── Frontier chart — shown at the top of results when model recommends budget ──
        if budget_mode == "Model recommends budget":
            _mrb_key  = st.session_state.get("_mrb_scan_key",
                           f"optimal_budget_scan|{use_hill_global}|{use_log_global}|False")
            _scan_res = st.session_state.get(_mrb_key)
            if _scan_res and _scan_res["success"]:
                _scan_df    = _scan_res["scan_df"]
                _rec_budget = _scan_res["optimal_budget"]
                _rec_profit = _scan_res["optimal_profit"]
                _base_pf    = float(df['baseline_profit'].sum())
                _base_rev   = float(df['baseline_revenue'].sum())
                # Determine which scan metric was used based on current objective
                _mrb_is_revenue = (obj_map[opt_objective] == "revenue")
                _mrb_scan_metric = "opt_revenue" if _mrb_is_revenue else "opt_profit"
                _mrb_frontier_label = "Revenue ($)" if _mrb_is_revenue else "Profit ($)"

                # ── Detect stale scan: objective changed since last scan ───────────
                # The scan key encodes the objective at the END ("|profit" or "|revenue")
                _stored_scan_key = st.session_state.get("_mrb_scan_key", "")
                _scan_obj_in_key = _stored_scan_key.split("|")[-1] if _stored_scan_key else ""
                _current_obj     = obj_map[opt_objective]
                if _scan_obj_in_key and _scan_obj_in_key != _current_obj:
                    st.warning(
                        f"⚠️ The frontier was scanned with **{_scan_obj_in_key.upper()}** maximization "
                        f"but you have now selected **{_current_obj.upper()}**. "
                        f"Click **Find Optimal Budget** again to re-scan with the current objective.",
                        icon="⚠️"
                    )

                # ── Build tiered budget options from the scan data ────────────────
                # Tiers: Conservative (75% optimal), Moderate (90%), Optimal (100%),
                #        Accelerated (115%), Aggressive (130%)
                _tier_defs = [
                    ("🔵 Conservative",  0.75, "chip-blue",   PALETTE["accent"]),
                    ("🟢 Moderate",      0.90, "chip-green",  PALETTE["accent2"]),
                    ("⭐ Optimal",        1.00, "chip-gold",   PALETTE["gold"]),
                    ("🟣 Accelerated",   1.15, "chip-purple", PALETTE["accent4"]),
                    ("🔴 Aggressive",    1.30, "chip-red",    PALETTE["accent3"]),
                ]

                # For each tier find the closest scan point
                def _closest_scan_row(target_budget):
                    idx = (_scan_df["budget"] - target_budget).abs().idxmin()
                    return _scan_df.loc[idx]

                _tier_rows = []
                for _tlabel, _tfrac, _tchip, _tcolor in _tier_defs:
                    _tbgt  = _rec_budget * _tfrac
                    _tbgt  = max(_tbgt, _baseline_total * 0.5)   # floor at 50% baseline
                    _trow  = _closest_scan_row(_tbgt)
                    # Use the correct revenue/profit columns from scan
                    _tpf   = float(_trow["opt_profit"])
                    _trev  = float(_trow["opt_revenue"]) if "opt_revenue" in _trow else _tpf + float(_trow.get("budget", _tbgt))
                    _troi  = _trev / max(float(_trow.get("budget", _tbgt)), 1)
                    _incr_profit = _tpf - _base_pf
                    _incr_rev    = _trev - _base_rev
                    _incr_primary = _incr_rev if _mrb_is_revenue else _incr_profit
                    _incr_spend  = float(_trow.get("budget", _tbgt)) - _baseline_total
                    _tier_rows.append({
                        "label":        _tlabel,
                        "chip":         _tchip,
                        "color":        _tcolor,
                        "budget":       float(_trow.get("budget", _tbgt)),
                        "revenue":      _trev,
                        "profit":       _tpf,
                        "roi":          _troi,
                        "incr_profit":  _incr_profit,
                        "incr_revenue": _incr_rev,
                        "incr_primary": _incr_primary,
                        "incr_spend":   _incr_spend,
                        "is_optimal":   _tfrac == 1.00,
                    })

                # ── Tiered budget cards ───────────────────────────────────────────
                st.markdown(
                    f"<div class='section-header'>Budget Scenario Options — pick your investment level</div>",
                    unsafe_allow_html=True
                )
                _tcols = st.columns(len(_tier_rows))
                for _ti, (_tc, _tr) in enumerate(zip(_tcols, _tier_rows)):
                    _border_style = (f"border:2px solid {_tr['color']}"
                                     if _tr["is_optimal"]
                                     else f"border:1.5px solid {PALETTE['border']}")
                    _bg       = PALETTE["surface"]
                    _mu       = PALETTE["muted"]
                    _mu2      = PALETTE["muted2"]
                    _bdr      = PALETTE["border"]
                    _acc2     = PALETTE["accent2"]
                    _acc3     = PALETTE["accent3"]
                    _tr_chip  = _tr["chip"]
                    _tr_label = _tr["label"]
                    _tr_color = _tr["color"]
                    _tr_bgt   = _tr["budget"]
                    _tr_rev   = _tr["revenue"]
                    _tr_pf    = _tr["profit"]
                    _tr_roi   = _tr["roi"]
                    _tr_incr  = _tr["incr_primary"]
                    _tr_opt   = _tr["is_optimal"]
                    _rec_note = "&nbsp;⭐ Recommended" if _tr_opt else ""
                    _pf_col   = _acc2 if _tr_pf >= _base_pf else _acc3
                    _ip_col   = _acc2 if _tr_incr >= 0 else _acc3
                    _ip_sign  = "+" if _tr_incr >= 0 else ""
                    _bgt_vs   = (_tr_bgt / _baseline_total - 1) * 100
                    _delta_label = "Δ Revenue" if _mrb_is_revenue else "Δ Profit"
                    _primary_val_label = "Revenue" if _mrb_is_revenue else "Profit"
                    _primary_val = _tr_rev if _mrb_is_revenue else _tr_pf
                    _primary_col = _acc2 if _primary_val >= (_base_rev if _mrb_is_revenue else _base_pf) else _acc3
                    with _tc:
                        st.markdown(
                            f"<div style='{_border_style};border-radius:12px;padding:.75rem .8rem;"
                            f"background:{_bg};box-shadow:0 1px 4px rgba(0,0,0,.05)'>"
                            f"<div style='font-size:.68rem;font-weight:700;margin-bottom:.35rem'>"
                            f"<span class='{_tr_chip}'>{_tr_label}</span>{_rec_note}</div>"
                            f"<div style='font-size:.65rem;color:{_mu};margin-bottom:.1rem'>Budget</div>"
                            f"<div style='font-size:1rem;font-weight:800;color:{_tr_color}'>{fmt(_tr_bgt,'$')}</div>"
                            f"<div style='font-size:.62rem;color:{_mu2}'>{_bgt_vs:+.1f}% vs baseline</div>"
                            f"<hr style='margin:.4rem 0;border-color:{_bdr}'>"
                            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:.2rem;font-size:.68rem'>"
                            f"<div><div style='color:{_mu2}'>{_primary_val_label}</div>"
                            f"<div style='font-weight:600;color:{_primary_col}'>{fmt(_primary_val,'$')}</div></div>"
                            f"<div><div style='color:{_mu2}'>{'Profit' if _mrb_is_revenue else 'Revenue'}</div>"
                            f"<div style='font-weight:600'>{fmt(_tr_pf if _mrb_is_revenue else _tr_rev,'$')}</div></div>"
                            f"<div><div style='color:{_mu2}'>ROI</div>"
                            f"<div style='font-weight:600'>{_tr_roi:.2f}×</div></div>"
                            f"<div><div style='color:{_mu2}'>{_delta_label}</div>"
                            f"<div style='font-weight:600;color:{_ip_col}'>{fmt(_tr_incr,'$',_ip_sign)}</div></div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                st.markdown(f"<div style='height:.75rem'></div>", unsafe_allow_html=True)

                # ── Tiered comparison table ───────────────────────────────────────
                _delta_col_name = "Δ Revenue" if _mrb_is_revenue else "Δ Profit"
                _tier_table_rows = []
                for _tr in _tier_rows:
                    _tier_table_rows.append({
                        "Scenario":      _tr["label"],
                        "Budget":        fmt(_tr["budget"], "$"),
                        "vs Baseline":   f"{(_tr['budget']/_baseline_total-1)*100:+.1f}%",
                        "Revenue":       fmt(_tr["revenue"], "$"),
                        "Profit":        fmt(_tr["profit"], "$"),
                        _delta_col_name: fmt(_tr["incr_primary"], "$", "+" if _tr["incr_primary"] >= 0 else ""),
                        "ROI":           f"{_tr['roi']:.2f}×",
                        "Extra Spend":   fmt(_tr["incr_spend"], "$", "+" if _tr["incr_spend"] >= 0 else ""),
                    })
                st.dataframe(pd.DataFrame(_tier_table_rows), width='stretch', hide_index=True)

                # ── Select budget tier to use for optimization ────────────────────
                st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)
                _tier_labels = [r["label"] for r in _tier_rows]
                _selected_tier_label = st.selectbox(
                    "Select budget tier to use for optimization",
                    _tier_labels,
                    index=2,   # default = Optimal
                    key="frontier_tier_sel",
                    help="This sets the budget for Run Optimization below"
                )
                _selected_tier = next(r for r in _tier_rows if r["label"] == _selected_tier_label)
                opt_budget = _selected_tier["budget"]
                st.markdown(
                    f"<div class='alert-box alert-success' style='font-size:.8rem'>"
                    f"✅ Selected: <b>{_selected_tier_label}</b> — Budget set to "
                    f"<b>{fmt(opt_budget,'$')}</b> for Run Optimization.</div>",
                    unsafe_allow_html=True
                )

                # ── Frontier chart with all tiers marked ─────────────────────────
                if PLOTLY and not _scan_df.empty:
                    # Use the correct Y column depending on objective
                    _frontier_y_col  = _mrb_scan_metric  # "opt_revenue" or "opt_profit"
                    _frontier_y_vals = _scan_df[_frontier_y_col] if _frontier_y_col in _scan_df.columns else _scan_df["opt_profit"]
                    _frontier_y_label = "Optimized Revenue ($)" if _mrb_is_revenue else "Optimized Profit ($)"
                    _frontier_hover_label = "Revenue" if _mrb_is_revenue else "Profit"
                    _tier_y_key = "revenue" if _mrb_is_revenue else "profit"

                    fig_scan = go.Figure()
                    fig_scan.add_trace(go.Scatter(
                        x=_scan_df["budget"], y=_frontier_y_vals,
                        fill='tozeroy',
                        fillcolor=f"rgba(5,122,85,0.07)",
                        line=dict(color=PALETTE["accent2"], width=0),
                        showlegend=False, hoverinfo='skip'
                    ))
                    fig_scan.add_trace(go.Scatter(
                        x=_scan_df["budget"], y=_frontier_y_vals,
                        mode='lines',
                        line=dict(color=PALETTE["accent2"], width=3, dash='dashdot'),
                        hovertemplate=f"Budget: $%{{x:,.0f}}<br>{_frontier_hover_label}: $%{{y:,.0f}}<extra></extra>",
                        name="Efficient frontier"
                    ))
                    # Baseline
                    fig_scan.add_vline(
                        x=_baseline_total, line_dash="dot",
                        line_color=PALETTE["muted"], line_width=1.5,
                        annotation_text=f"Baseline: {fmt(_baseline_total,'$')}",
                        annotation_font=dict(size=9, color=PALETTE["muted"]),
                        annotation_position="bottom left"
                    )
                    # Plot each tier as a marker
                    for _tr in _tier_rows:
                        _star = "star" if _tr["is_optimal"] else "circle"
                        _sz   = 16 if _tr["is_optimal"] else 11
                        _tier_y_val = _tr[_tier_y_key]
                        fig_scan.add_trace(go.Scatter(
                            x=[_tr["budget"]], y=[_tier_y_val],
                            mode='markers+text',
                            marker=dict(size=_sz, color=_tr["color"],
                                        symbol=_star, line=dict(color='white', width=2)),
                            text=[_tr["label"].split(" ", 1)[1]],   # strip emoji
                            textposition="top center",
                            textfont=dict(size=9, color=_tr["color"]),
                            name=_tr["label"],
                            hovertemplate=(
                                f"<b>{_tr['label']}</b><br>"
                                f"Budget: ${_tr['budget']:,.0f}<br>"
                                f"{_frontier_hover_label}: ${_tier_y_val:,.0f}<br>"
                                f"ROI: {_tr['roi']:.2f}×<extra></extra>"
                            )
                        ))
                    fig_scan.update_layout(
                        title=dict(
                            text=f"Efficient Frontier — 5 Budget Scenarios ({_frontier_hover_label} Maximization)",
                            font=dict(size=12, color=PALETTE["text2"])
                        ),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                        height=400, margin=dict(l=10, r=20, t=45, b=60),
                        legend=dict(bgcolor="rgba(255,255,255,0.92)",
                                    bordercolor=PALETTE["border"], borderwidth=1,
                                    orientation="h", y=-0.18, font=dict(size=10)),
                        xaxis=dict(title="Total Budget ($)", tickformat="$,.0f",
                                   gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                   tickfont=dict(size=10, color=PALETTE["muted"])),
                        yaxis=dict(title=_frontier_y_label, tickformat="$,.0f",
                                   gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                   tickfont=dict(size=10, color=PALETTE["muted"]))
                    )
                    st.plotly_chart(fig_scan, width='stretch')
                    _frontier_note = (
                        "The ⭐ Optimal point is the budget level where additional spend generates maximum revenue."
                        if _mrb_is_revenue
                        else "The ⭐ Optimal point is where mROI → 1 across all channels."
                    )
                    st.markdown(
                        f"<div class='alert-box alert-info' style='font-size:.8rem'>"
                        f"💡 <b>How to read this:</b> Each marker is a selectable budget scenario. "
                        f"The curve rises steeply when channels are underinvested (high mROI) and "
                        f"flattens as channels saturate. {_frontier_note} "
                        f"Select a tier above and click Run Optimization.</div>",
                        unsafe_allow_html=True
                    )
            else:
                # No scan run yet — show placeholder
                _ph_surf = PALETTE["surface"]; _ph_bdr = PALETTE["border"]
                _ph_txt  = PALETTE["text"];    _ph_mu  = PALETTE["muted"]
                st.markdown(
                    f"<div style='margin-top:2rem;text-align:center;padding:3rem;"
                    f"background:{_ph_surf};border-radius:16px;border:2px dashed {_ph_bdr}'>"
                    f"<div style='font-size:2rem;margin-bottom:1rem'>🔍</div>"
                    f"<div style='font-size:1rem;font-weight:600;color:{_ph_txt};margin-bottom:.5rem'>"
                    f"Click \"Find Optimal Budget\" to scan the efficient frontier</div>"
                    f"<div style='font-size:.82rem;color:{_ph_mu}'>"
                    f"Set your scan range (Min/Max % of baseline) in the left panel, "
                    f"then click Find Optimal Budget. Default: 70%–150% of baseline "
                    f"for realistic planning scenarios.</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Cache key encodes ALL settings — any change produces a new key ─────
        _opt_key = (f"{obj_map[opt_objective]}|{method_map[opt_method]}|"
                    f"{opt_budget:.2f}|{budget_mode}|"
                    f"{use_hill_global}|{use_log_global}|{_bounds_sig}")

        # Only show results if Run was just clicked, OR we have a result for
        # exactly these settings. Never show stale results from different settings.
        _has_cached = isinstance(st.session_state.get(_opt_key), pd.DataFrame)

        # In "Model recommends budget" mode, warn if scan hasn't run yet
        _mrb_key_check = st.session_state.get("_mrb_scan_key",
                            f"optimal_budget_scan|{use_hill_global}|{use_log_global}|False")
        _scan_done = (st.session_state.get(_mrb_key_check) or {}).get("success", False)
        _needs_scan = (budget_mode == "Model recommends budget" and not _scan_done)

        if run_opt and _needs_scan:
            st.warning(
                "⚠️ Please click **Find Optimal Budget** first to determine the recommended "
                "budget level, then click Run Optimization."
            )

        if (run_opt and not _needs_scan) or _has_cached:
            if run_opt:
                # Start from df_scan so lock overrides from any mode are respected.
                df_opt_in = df_scan.copy()

                # ── Step 1: Apply user per-channel bound overrides ──────────────
                for ch, (lo_ov, hi_ov) in override_bounds.items():
                    mask = df_opt_in['channel'] == ch
                    is_locked = int(df_opt_in.loc[mask, 'lock_spend'].iloc[0]) == 1
                    if is_locked:
                        continue
                    df_opt_in.loc[mask, 'lower_bound_pct'] = lo_ov
                    df_opt_in.loc[mask, 'upper_bound_pct'] = hi_ov

                # ── Step 2: Compute strict feasibility range ────────────────────
                _locked_sp_total = float(
                    (df_opt_in['total_spend'] * df_opt_in['lock_spend'].astype(int)).sum()
                )
                _flex_rows = df_opt_in[df_opt_in['lock_spend'].astype(int) == 0]
                _flex_min  = float((_flex_rows['total_spend'] * _flex_rows['lower_bound_pct']).sum())
                _flex_max  = float((_flex_rows['total_spend'] * _flex_rows['upper_bound_pct']).sum())
                _strict_min = _locked_sp_total + _flex_min
                _strict_max = _locked_sp_total + _flex_max
                _flex_budget = opt_budget - _locked_sp_total
                _flex_sp_total = float(_flex_rows['total_spend'].sum())

                # ── Step 3: Feasibility gate (only for Set new budget) ──────────
                _run_blocked = False

                if budget_mode == "Set new budget" and opt_budget < _strict_min:
                    st.error(
                        f"⛔ Budget {fmt(opt_budget,'$')} is below the minimum required "
                        f"by channel bounds ({fmt(_strict_min,'$')}). "
                        f"Please raise the budget or lower channel Min% bounds.",
                        icon="⛔"
                    )
                    _run_blocked = True

                elif budget_mode == "Set new budget" and opt_budget > _strict_max:
                    # Calculate how much extra budget needs to be absorbed
                    _excess         = opt_budget - _strict_max
                    _needed_ratio   = (_flex_budget / _flex_sp_total) if _flex_sp_total > 0 else 1.0

                    # Show per-channel guidance: what Max% each channel would need
                    _guidance_rows = []
                    for _, _gr in _flex_rows.iterrows():
                        _g_base    = float(_gr['total_spend'])
                        _g_cur_hi  = float(_gr.get('upper_bound_pct', 1.5))
                        _g_cur_max = _g_base * _g_cur_hi
                        _g_needed_hi = _needed_ratio * 1.05
                        if _g_needed_hi > _g_cur_hi:
                            _guidance_rows.append({
                                "Channel":       _gr['channel'],
                                "Current Max%":  f"{_g_cur_hi*100:.0f}%  (${_g_cur_max:,.0f})",
                                "Needed Max%":   f"{_g_needed_hi*100:.0f}%  (${_g_base*_g_needed_hi:,.0f})",
                            })

                    if _guidance_rows:
                        _guide_df = pd.DataFrame(_guidance_rows)
                        st.warning(
                            f"⚠️ Budget {fmt(opt_budget,'$')} needs more room in channel Max% bounds. "
                            f"Either increase Max% bounds below, or check "
                            f"'I understand' to run anyway (optimizer will concentrate spend "
                            f"in highest-ROI channels up to their current bounds).",
                            icon="⚠️"
                        )
                        st.dataframe(_guide_df, hide_index=True, use_container_width=True)

                    if _accept_budget_warning:
                        # User accepted — run WITHOUT modifying bounds.
                        # The optimizer will spend as much as bounds allow;
                        # any unallocatable excess stays unspent (shown in gap warning).
                        st.info(
                            f"ℹ️ Running with current bounds. The optimizer will allocate "
                            f"up to {fmt(_strict_max,'$')} (the channel bound maximum). "
                            f"To spend the full {fmt(opt_budget,'$')}, increase Max% bounds "
                            f"in the Per-channel bounds expander.",
                            icon="ℹ️"
                        )
                        # Use _strict_max as effective budget so optimizer isn't overconstrained
                        opt_budget = _strict_max
                    else:
                        _run_blocked = True

                if not _run_blocked:
                    _obj_label      = obj_map[opt_objective]
                    _method_label   = method_map[opt_method]
                    _display_budget = opt_budget   # may have been capped to _strict_max above
                    with st.spinner(
                        f"Optimizing · {opt_objective} · {opt_method} · "
                        f"Budget: {fmt(opt_budget, '$')}..."
                    ):
                        result = optimize_budget(
                            df_opt_in, opt_budget,
                            objective=_obj_label,
                            use_hill=use_hill_global,
                            use_log=use_log_global,
                            method=_method_label
                        )

                    spends_opt = result['spends']
                    df_opt = df_opt_in.copy()
                    df_opt['opt_spend']        = spends_opt
                    df_opt['opt_revenue']      = [
                        revenue_from_spend(spends_opt[i], df_opt.iloc[i],
                                           use_hill_global, use_log_global)
                        for i in range(len(df_opt))
                    ]
                    df_opt['opt_profit']       = df_opt['opt_revenue'] - df_opt['opt_spend']
                    df_opt['opt_roi']          = df_opt['opt_revenue'] / df_opt['opt_spend'].replace(0, np.nan)
                    df_opt['delta_spend']      = df_opt['opt_spend']   - df_opt['total_spend']
                    df_opt['delta_profit']     = df_opt['opt_profit']  - df_opt['baseline_profit']
                    df_opt['delta_revenue']    = df_opt['opt_revenue'] - df_opt['baseline_revenue']
                    df_opt['delta_pct_spend']  = (df_opt['delta_spend']
                                                  / df_opt['total_spend'].replace(0, np.nan) * 100)
                    df_opt['opt_budget_used']  = opt_budget

                    # ── Validate: check actual spends sum to budget ────────────
                    _actual_total = float(np.sum(spends_opt))
                    _budget_gap   = abs(_actual_total - opt_budget)
                    if _budget_gap > opt_budget * 0.02:   # >2% gap = solver issue
                        st.warning(
                            f"⚠️ Optimizer allocated {fmt(_actual_total,'$')} "
                            f"but budget was {fmt(opt_budget,'$')} "
                            f"(gap: {fmt(_budget_gap,'$')}). "
                            f"Try increasing Max% bounds or switching solver method.",
                            icon="⚠️"
                        )

                    st.session_state[_opt_key]             = df_opt
                    st.session_state['opt_result_df']      = df_opt
                    st.session_state['opt_success']        = result['success']
                    st.session_state['opt_message']        = result.get('message', '')
                    st.session_state['opt_objective_used'] = opt_objective
                    st.session_state['opt_method_used']    = opt_method
                    st.session_state['opt_budget_used']    = opt_budget
                    st.session_state['opt_budget_mode']    = budget_mode
                
                # ── Check if any channel exceeds its bounds ───────────────────
                if not _run_blocked and 'df_opt' in dir():
                    _bound_violations = []
                    for idx, row in df_opt.iterrows():
                        ch         = row['channel']
                        base_spend = float(row['total_spend'])
                        opt_spend  = float(row['opt_spend'])
                        lo_pct     = float(row.get('lower_bound_pct', 0.5))
                        hi_pct     = float(row.get('upper_bound_pct', 1.5))
                        lo_bound   = base_spend * lo_pct
                        hi_bound   = base_spend * hi_pct
                        if opt_spend < lo_bound * 0.99:
                            _bound_violations.append({'channel': ch, 'type': 'below',
                                                       'actual': opt_spend, 'min': lo_bound, 'max': hi_bound})
                        elif opt_spend > hi_bound * 1.01:
                            _bound_violations.append({'channel': ch, 'type': 'above',
                                                       'actual': opt_spend, 'min': lo_bound, 'max': hi_bound})
                    st.session_state['opt_bound_violations'] = _bound_violations
                elif _run_blocked:
                    # Don't show stale results from a previous run
                    st.session_state.pop(_opt_key, None)

            if not isinstance(st.session_state.get(_opt_key), pd.DataFrame):
                # No result yet — show placeholder and wait for Run click
                _ph_surf = PALETTE["surface"]; _ph_bdr = PALETTE["border"]
                _ph_txt  = PALETTE["text"];    _ph_mu  = PALETTE["muted"]
                st.markdown(
                    f"<div style='margin-top:2rem;text-align:center;padding:3rem;"
                    f"background:{_ph_surf};border-radius:16px;border:2px dashed {_ph_bdr}'>"
                    f"<div style='font-size:2rem;margin-bottom:1rem'>⚡</div>"
                    f"<div style='font-size:1rem;font-weight:600;color:{_ph_txt};"
                    f"margin-bottom:.5rem'>Configure settings and run optimization</div>"
                    f"<div style='font-size:.82rem;color:{_ph_mu}'>"
                    f"Supports Maximize Profit · Revenue · ROI with SLSQP &amp; "
                    f"Differential Evolution</div></div>",
                    unsafe_allow_html=True
                )

            if isinstance(st.session_state.get(_opt_key), pd.DataFrame):
                df_opt            = st.session_state[_opt_key]
                success_          = st.session_state.get('opt_success', True)
                _bound_violations = st.session_state.get('opt_bound_violations', [])

            # Show which budget was actually used in this result
            _budget_used = float(df_opt.get('opt_budget_used', pd.Series([opt_budget])).iloc[0]) \
                           if 'opt_budget_used' in df_opt.columns else opt_budget
            _bmode_used  = st.session_state.get('opt_budget_mode', budget_mode)

            # Show which objective/method produced this result
            _obj_used    = st.session_state.get('opt_objective_used', opt_objective)
            _method_used = st.session_state.get('opt_method_used', opt_method)
            obj_chip_map = {
                "Maximize Profit":  ("chip-green",  "Profit"),
                "Maximize Revenue": ("chip-blue",   "Revenue"),
                "Maximize ROI":    ("chip-purple",  "ROI"),
            }
            _chip_cls, _chip_lbl = obj_chip_map.get(_obj_used, ("chip-blue", _obj_used))

            # Objective explanation banner
            _obj_desc = {
                "Maximize Profit":  ("🟢", "Maximize Profit (Excel-equivalent)",
                    "Optimal when mROI = 1 per channel. "
                    "Balances revenue vs spend — pulls budget away from over-invested channels."),
                "Maximize Revenue": ("🔵", "Maximize Revenue",
                    "Pushes budget toward channels with highest marginal revenue regardless of cost. "
                    "More aggressive concentration than profit; ignores diminishing cost efficiency."),
                "Maximize ROI":    ("🟣", "Maximize ROI",
                    "Concentrates spend on highest revenue-per-dollar channels. "
                    "Cuts low-ROI channels to their minimum bound — most aggressive reallocation."),
            }
            _icon, _title, _desc = _obj_desc.get(_obj_used, ("⚡","",""))
            _budget_chip = "chip-gold" if _bmode_used == "Set new budget" else "chip-blue"
            _budget_label = f"New budget: {fmt(_budget_used, '$')}" if _bmode_used == "Set new budget" else f"Budget kept: {fmt(_budget_used, '$')}"
            _banner_surf = PALETTE["surface"]; _banner_bdr = PALETTE["border"]
            _banner_txt  = PALETTE["text"];    _banner_mu  = PALETTE["muted"]
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:.75rem;"
                f"background:{_banner_surf};border:1.5px solid {_banner_bdr};"
                f"border-radius:12px;padding:.8rem 1rem;margin-bottom:.75rem;"
                f"box-shadow:0 1px 4px rgba(0,0,0,.04)'>"
                f"<div style='font-size:1.4rem;line-height:1'>{_icon}</div>"
                f"<div style='flex:1'>"
                f"<div style='font-size:.78rem;font-weight:700;color:{_banner_txt};margin-bottom:.2rem'>"
                f"{_title} &nbsp;<span class='{_chip_cls} chip'>{_method_used}</span>"
                f"&nbsp;<span class='{_budget_chip} chip'>{_budget_label}</span>"
                f"</div>"
                f"<div style='font-size:.76rem;color:{_banner_mu}'>{_desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

            if not success_:
                st.markdown(f"<div class='alert-box alert-warn'>⚠️ {st.session_state.get('opt_message', 'Solver did not fully converge. Showing best solution found.')}</div>",
                            unsafe_allow_html=True)
            else:
                _gekko_note = "  · GEKKO/IPOPT" if "GEKKO" in _method_used else ""
                st.markdown(f"<div class='alert-box alert-success'>✅ Converged · {_obj_used} · {_method_used}{_gekko_note} · Budget: {fmt(_budget_used, '$')}</div>",
                            unsafe_allow_html=True)
            
            # ── Boundary Exceed Warning ───────────────────────────────────────
            _show_results = True
            if _bound_violations and (budget_mode == "Model recommends budget" or budget_mode == "Set new budget"):
                # Build warning message with violation details
                _violation_details = []
                for v in _bound_violations:
                    if v['type'] == 'below':
                        _violation_details.append(
                            f"• {v['channel']}: ${v['actual']:,.0f} is below minimum ${v['min']:,.0f}"
                        )
                    else:
                        _violation_details.append(
                            f"• {v['channel']}: ${v['actual']:,.0f} exceeds maximum ${v['max']:,.0f}"
                        )
                
                st.markdown(
                    f"<div class='alert-box alert-warn' style='font-size:.78rem;margin-top:.5rem'>"
                    f"⚠️ <b>Channel Boundary Warning:</b> The optimization results exceed "
                    f"the per-channel bounds for <b>{len(_bound_violations)} channel(s)</b>. "
                    f"This may indicate that the selected budget cannot be fully optimized "
                    f"within the specified channel constraints.<br><br>"
                    f"<b>Violations:</b><br>" + "<br>".join(_violation_details) +
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Ask user if they want to see the results
                _proceed_key = f"proceed_with_violations_{_opt_key}"
                st.markdown(
                    f"<div style='font-size:.76rem;margin-top:.5rem;color:{PALETTE['text2']}'>"
                    f"Do you still want to view the optimization results?</div>",
                    unsafe_allow_html=True
                )
                _col_proceed_yes, _col_proceed_no = st.columns(2)
                with _col_proceed_yes:
                    _show_results = st.button("✅ Yes, show results", key=f"yes_{_opt_key}")
                with _col_proceed_no:
                    st.button("❌ No, hide results", key=f"no_{_opt_key}")
                
                # If user clicked No, hide results
                if st.session_state.get(f"no_{_opt_key}", False):
                    _show_results = False
                    st.markdown(
                        f"<div class='alert-box alert-info' style='font-size:.78rem;margin-top:.5rem'>"
                        f"ℹ️ Results hidden due to boundary violations. "
                        f"Try adjusting the budget or channel bounds to satisfy constraints.</div>",
                        unsafe_allow_html=True
                    )
            
            # Only show results if user confirmed or no violations
            if _show_results:
                # Result KPIs
                new_rev    = df_opt['opt_revenue'].sum()
                new_profit = df_opt['opt_profit'].sum()
                new_spend  = df_opt['opt_spend'].sum()
                new_roas   = new_rev / max(new_spend, EPS)
                uplift_p   = new_profit - df_opt['baseline_profit'].sum()
                uplift_r   = new_rev - df_opt['baseline_revenue'].sum()

                rk1, rk2, rk3, rk4 = st.columns(4)
                rk1.metric("Optimized Revenue", fmt(new_rev, "$"),
                           delta=f"{fmt(uplift_r, '$', '+' if uplift_r >= 0 else '')}")
            rk2.metric("Optimized Profit", fmt(new_profit, "$"),
                       delta=f"{fmt(uplift_p, '$', '+' if uplift_p >= 0 else '')}")
            rk3.metric("Optimized ROI", f"{new_roas:.2f}×",
                       delta=f"{new_roas - avg_roas:+.2f}× vs baseline")
            rk4.metric("Total Spend", fmt(new_spend, "$"),
                       delta=f"{fmt(new_spend - tot_spend, '$', '+' if new_spend >= tot_spend else '')}")

            st.markdown(f"<div style='height:.5rem'></div>", unsafe_allow_html=True)

            if PLOTLY:
                _ch_names = df_opt['channel'].tolist()
                _n        = len(_ch_names)

                def _opt_layout(title_txt, height=300, b=65):
                    return dict(
                        title=dict(text=title_txt, font=dict(size=11, color=PALETTE["text2"])),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                        height=height, margin=dict(l=5, r=10, t=38, b=b),
                        showlegend=False,
                        xaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                   tickangle=-30, tickfont=dict(size=9, color=PALETTE["muted"])),
                    )

                # ── ROW 1: Spend horizontal bar  |  Δ Spend % ────────────────
                r1a, r1b = st.columns([.6, .4])

                with r1a:
                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Bar(
                        name='Baseline', y=_ch_names, x=df_opt['total_spend'],
                        orientation='h',
                        marker_color=hex_to_rgba(PALETTE["accent"], 0.45),
                        marker_line_color=PALETTE["accent"], marker_line_width=1.5,
                        text=[fmt(v, "$") for v in df_opt['total_spend']],
                        textposition='inside', textfont=dict(size=8, color=PALETTE["accent"]),
                        hovertemplate="<b>%{y}</b><br>Baseline: $%{x:,.0f}<extra></extra>"
                    ))
                    fig_sp.add_trace(go.Bar(
                        name='Optimized', y=_ch_names, x=df_opt['opt_spend'],
                        orientation='h',
                        marker_color=hex_to_rgba(PALETTE["accent2"], 0.45),
                        marker_line_color=PALETTE["accent2"], marker_line_width=1.5,
                        text=[fmt(v, "$") for v in df_opt['opt_spend']],
                        textposition='inside', textfont=dict(size=8, color=PALETTE["accent2"]),
                        hovertemplate="<b>%{y}</b><br>Optimized: $%{x:,.0f}<extra></extra>"
                    ))
                    fig_sp.update_layout(
                        barmode='group',
                        title=dict(text="Spend  —  Baseline vs Optimized",
                                   font=dict(size=11, color=PALETTE["text2"])),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                        height=max(260, _n * 42),
                        margin=dict(l=5, r=10, t=38, b=10),
                        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=PALETTE["border"],
                                    borderwidth=1, orientation="h", y=-0.12, font=dict(size=10)),
                        xaxis=dict(title="Spend ($)", gridcolor="#EAECF0", tickformat="$,.0f",
                                   tickfont=dict(size=9, color=PALETTE["muted"])),
                        yaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                   tickfont=dict(size=9, color=PALETTE["muted"]), autorange="reversed")
                    )
                    st.plotly_chart(fig_sp, width='stretch')

                with r1b:
                    # If any channel shift exceeds 200%, % is misleading (tiny baseline).
                    # Switch to absolute Δ Spend ($) in that case and show a note.
                    _max_abs_pct   = df_opt['delta_pct_spend'].abs().max()
                    _use_abs_delta = _max_abs_pct > 200

                    dp_colors = [PALETTE["accent2"] if v >= 0 else PALETTE["accent3"]
                                 for v in df_opt['delta_pct_spend']]
                    fig_dpct = go.Figure()

                    if _use_abs_delta:
                        # Show absolute dollar change instead
                        fig_dpct.add_trace(go.Bar(
                            y=_ch_names, x=df_opt['delta_spend'], orientation='h',
                            marker_color=[hex_to_rgba(c, 0.55) for c in dp_colors],
                            marker_line_color=dp_colors, marker_line_width=1.5,
                            text=[fmt(v, "$", "+" if v >= 0 else "") for v in df_opt['delta_spend']],
                            textposition='outside', textfont=dict(size=8),
                            customdata=df_opt['delta_pct_spend'],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Δ Spend: $%{x:,.0f}<br>"
                                "Δ Spend %: %{customdata:+.1f}%<extra></extra>"
                            )
                        ))
                        fig_dpct.add_vline(x=0, line_color="#CBD0D8", line_width=1.5)
                        fig_dpct.update_layout(
                            title=dict(
                                text="Spend Change ($) vs Baseline",
                                font=dict(size=11, color=PALETTE["text2"])
                            ),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                            font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                            height=max(260, _n * 42),
                            margin=dict(l=5, r=60, t=38, b=10),
                            showlegend=False,
                            xaxis=dict(title="Δ Spend ($)", gridcolor="#EAECF0",
                                       tickformat="$,.0f", zeroline=False,
                                       tickfont=dict(size=9, color=PALETTE["muted"])),
                            yaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                       tickfont=dict(size=9, color=PALETTE["muted"]),
                                       autorange="reversed")
                        )
                        st.plotly_chart(fig_dpct, width='stretch')
                        st.markdown(
                            f"<div class='alert-box alert-info' style='font-size:.75rem'>"
                            f"ℹ️ Showing <b>absolute Δ Spend ($)</b> instead of % because some channels "
                            f"have very small baselines (e.g. {fmt(df_opt['total_spend'].min(),'$')} minimum), "
                            f"making % shifts misleadingly large (+{_max_abs_pct:,.0f}%). "
                            f"Hover each bar to see both $ and % change. "
                            f"This is expected when the selected budget is significantly larger than baseline.</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        fig_dpct.add_trace(go.Bar(
                            y=_ch_names, x=df_opt['delta_pct_spend'], orientation='h',
                            marker_color=[hex_to_rgba(c, 0.55) for c in dp_colors],
                            marker_line_color=dp_colors, marker_line_width=1.5,
                            text=[f"{v:+.1f}%" for v in df_opt['delta_pct_spend']],
                            textposition='outside', textfont=dict(size=8),
                            customdata=df_opt['delta_spend'],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Δ Spend %: %{x:+.1f}%<br>"
                                "Δ Spend $: $%{customdata:,.0f}<extra></extra>"
                            )
                        ))
                        fig_dpct.add_vline(x=0, line_color="#CBD0D8", line_width=1.5)
                        fig_dpct.update_layout(
                            title=dict(text="Spend Shift %  (vs Baseline)",
                                       font=dict(size=11, color=PALETTE["text2"])),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                            font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                            height=max(260, _n * 42),
                            margin=dict(l=5, r=50, t=38, b=10),
                            showlegend=False,
                            xaxis=dict(title="Δ Spend %", gridcolor="#EAECF0", ticksuffix="%",
                                       zeroline=False, tickfont=dict(size=9, color=PALETTE["muted"])),
                            yaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                       tickfont=dict(size=9, color=PALETTE["muted"]),
                                       autorange="reversed")
                        )
                        st.plotly_chart(fig_dpct, width='stretch')

                # ── ROW 2: Δ Profit  |  Δ Revenue ────────────────────────────
                r2a, r2b = st.columns(2)

                with r2a:
                    pf_colors = [PALETTE["accent2"] if v >= 0 else PALETTE["accent3"]
                                 for v in df_opt['delta_profit']]
                    fig_dpf = go.Figure()
                    fig_dpf.add_trace(go.Bar(
                        x=_ch_names, y=df_opt['delta_profit'],
                        marker_color=[hex_to_rgba(c, 0.6) for c in pf_colors],
                        marker_line_color=pf_colors, marker_line_width=1.5,
                        text=[fmt(v, "$", "+" if v >= 0 else "") for v in df_opt['delta_profit']],
                        textposition='outside', textfont=dict(size=8),
                        hovertemplate="<b>%{x}</b><br>Δ Profit: $%{y:,.0f}<extra></extra>"
                    ))
                    fig_dpf.add_hline(y=0, line_color="#CBD0D8", line_width=1.5)
                    fig_dpf.update_layout(
                        **_opt_layout("Profit Change by Channel"),
                        yaxis=dict(title="Δ Profit ($)", tickformat="$,.0f",
                                   gridcolor="#EAECF0", zeroline=False,
                                   tickfont=dict(size=9, color=PALETTE["muted"]))
                    )
                    st.plotly_chart(fig_dpf, width='stretch')

                with r2b:
                    rv_colors = [PALETTE["accent"] if v >= 0 else PALETTE["accent3"]
                                 for v in df_opt['delta_revenue']]
                    fig_drv = go.Figure()
                    fig_drv.add_trace(go.Bar(
                        x=_ch_names, y=df_opt['delta_revenue'],
                        marker_color=[hex_to_rgba(c, 0.6) for c in rv_colors],
                        marker_line_color=rv_colors, marker_line_width=1.5,
                        text=[fmt(v, "$", "+" if v >= 0 else "") for v in df_opt['delta_revenue']],
                        textposition='outside', textfont=dict(size=8),
                        hovertemplate="<b>%{x}</b><br>Δ Revenue: $%{y:,.0f}<extra></extra>"
                    ))
                    fig_drv.add_hline(y=0, line_color="#CBD0D8", line_width=1.5)
                    fig_drv.update_layout(
                        **_opt_layout("Revenue Change by Channel"),
                        yaxis=dict(title="Δ Revenue ($)", tickformat="$,.0f",
                                   gridcolor="#EAECF0", zeroline=False,
                                   tickfont=dict(size=9, color=PALETTE["muted"]))
                    )
                    st.plotly_chart(fig_drv, width='stretch')

                # ── ROW 3: ROI grouped bars ──────────────────────────────────
                fig_roas_cmp = go.Figure()
                fig_roas_cmp.add_trace(go.Bar(
                    name='Baseline ROI', x=_ch_names, y=df_opt['baseline_roi'],
                    offsetgroup=0,
                    marker_color=hex_to_rgba(PALETTE["accent"], 0.45),
                    marker_line_color=PALETTE["accent"], marker_line_width=1.5,
                    text=[f"{v:.2f}x" for v in df_opt['baseline_roi']],
                    textposition='outside', textfont=dict(size=8, color=PALETTE["accent"]),
                    hovertemplate="<b>%{x}</b><br>Baseline ROI: %{y:.2f}x<extra></extra>"
                ))
                fig_roas_cmp.add_trace(go.Bar(
                    name='Optimized ROI', x=_ch_names, y=df_opt['opt_roi'].fillna(0),
                    offsetgroup=1,
                    marker_color=hex_to_rgba(PALETTE["accent2"], 0.45),
                    marker_line_color=PALETTE["accent2"], marker_line_width=1.5,
                    text=[f"{v:.2f}x" if pd.notnull(v) else "-" for v in df_opt['opt_roi']],
                    textposition='outside', textfont=dict(size=8, color=PALETTE["accent2"]),
                    hovertemplate="<b>%{x}</b><br>Optimized ROI: %{y:.2f}x<extra></extra>"
                ))
                fig_roas_cmp.add_hline(y=1.0, line_dash="dash",
                                        line_color=PALETTE["muted"], line_width=1,
                                        annotation_text="ROI = 1",
                                        annotation_font=dict(size=8, color=PALETTE["muted"]))
                fig_roas_cmp.update_layout(
                    barmode='group',
                    title=dict(text="ROI  —  Baseline vs Optimized",
                               font=dict(size=11, color=PALETTE["text2"])),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                    font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                    height=320, margin=dict(l=5, r=10, t=38, b=70),
                    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=PALETTE["border"],
                                borderwidth=1, orientation="h", y=-0.22, font=dict(size=10)),
                    xaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                               tickangle=-30, tickfont=dict(size=9, color=PALETTE["muted"])),
                    yaxis=dict(title="ROI", gridcolor="#EAECF0",
                               tickformat=".2f", ticksuffix="x", zeroline=False,
                               tickfont=dict(size=9, color=PALETTE["muted"]))
                )
                st.plotly_chart(fig_roas_cmp, width='stretch')

            # Results table
            st.markdown(f"<div class='section-header'>Detailed results</div>", unsafe_allow_html=True)

            # Pre-compute bound columns from the df_opt_in bounds
            df_opt['lower_bound_$'] = df_opt['total_spend'] * df_opt['lower_bound_pct'].astype(float)
            df_opt['upper_bound_$'] = df_opt['total_spend'] * df_opt['upper_bound_pct'].astype(float)
            # Locked channels: show "Locked" instead of a range
            df_opt['_is_locked'] = df_opt['lock_spend'].astype(int) == 1

            display_cols = ['channel', 'total_spend', 'lower_bound_$', 'upper_bound_$',
                            'opt_spend', 'delta_spend', 'delta_pct_spend',
                            'baseline_revenue', 'opt_revenue', 'delta_revenue',
                            'baseline_profit', 'opt_profit', 'delta_profit', 'opt_roi']
            display_df = df_opt[display_cols].copy()
            display_df.columns = ['Channel', 'Baseline Spend', 'Min Bound', 'Max Bound',
                                   'Opt Spend', 'Δ Spend', 'Δ%',
                                   'Baseline Rev', 'Opt Rev', 'Δ Rev',
                                   'Baseline Profit', 'Opt Profit', 'Δ Profit', 'ROI']

            num_cols_fmt = ['Baseline Spend', 'Opt Spend', 'Δ Spend',
                            'Baseline Rev', 'Opt Rev', 'Δ Rev',
                            'Baseline Profit', 'Opt Profit', 'Δ Profit']
            for c in num_cols_fmt:
                display_df[c] = display_df[c].apply(lambda x: f"${x:,.0f}")

            # Format bound columns — show "Locked" for locked channels
            for i, row in df_opt.iterrows():
                ch_idx = display_df.index[display_df['Channel'] == row['channel']]
                if int(row.get('lock_spend', 0)) == 1:
                    display_df.loc[ch_idx, 'Min Bound'] = '🔒 Locked'
                    display_df.loc[ch_idx, 'Max Bound'] = '🔒 Locked'
                else:
                    display_df.loc[ch_idx, 'Min Bound'] = f"${float(row['lower_bound_$']):,.0f}"
                    display_df.loc[ch_idx, 'Max Bound'] = f"${float(row['upper_bound_$']):,.0f}"

            # Cap display of Δ% — show ">1000%" for extreme cases to avoid visual noise
            def _fmt_dpct(x):
                if abs(x) > 1000:
                    return f"{'+' if x>0 else ''}>{int(abs(x)//1000)*1000}%*"
                return f"{x:+.1f}%"
            display_df['Δ%'] = display_df['Δ%'].apply(_fmt_dpct)
            display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.2f}×" if pd.notnull(x) else "—")
            st.dataframe(display_df, width='stretch', hide_index=True)

            buf = io.StringIO()
            _csv_cols = ['channel', 'total_spend', 'lower_bound_$', 'upper_bound_$',
                         'opt_spend', 'delta_spend', 'delta_pct_spend',
                         'baseline_revenue', 'opt_revenue', 'delta_revenue',
                         'baseline_profit', 'opt_profit', 'delta_profit', 'opt_roi']
            df_opt[_csv_cols].to_csv(buf, index=False)
            st.download_button("⬇️ Download optimization results (CSV)",
                                data=buf.getvalue(),
                                file_name="mmm_optimized_budget.csv",
                                mime="text/csv")

            # ── AI Interpretation — Optimization ─────────────────────────────
            _ch_ctx_opt = "\n".join([
                f"  {r['channel']} ({ch_desc_map.get(r['channel'], r['channel'])}): "
                f"baseline=${float(r['total_spend']):,.0f} → opt=${float(r['opt_spend']):,.0f} "
                f"(Δ{float(r['delta_pct_spend']):+.1f}%), "
                f"profit Δ={fmt(float(r['delta_profit']),'$','+' if r['delta_profit']>=0 else '')}, "
                f"ROI {float(r['baseline_roi']):.2f}x → {float(r['opt_roi']):.2f}x"
                for _, r in df_opt.iterrows()
            ])
            _opt_ai_prompt = (
                f"BUDGET OPTIMIZATION RESULTS\n\n"
                f"Objective: {_obj_used} | Solver: {_method_used} | Budget: {fmt(_budget_used, '$')}\n\n"
                f"CHANNEL OUTCOMES:\n{_ch_ctx_opt}\n\n"
                f"PORTFOLIO IMPACT:\n"
                f"  Baseline profit: {fmt(float(df_opt['baseline_profit'].sum()),'$')} → "
                f"Optimized: {fmt(new_profit,'$')} (Δ {fmt(uplift_p,'$','+' if uplift_p>=0 else '')})\n"
                f"  Baseline ROI: {avg_roas:.2f}x → Optimized: {new_roas:.2f}x\n\n"
                f"Channel descriptions for context:\n"
                + "\n".join([f"  {k}: {v}" for k,v in ch_desc_map.items()])
                + "\n\nAnalyse using the four-section structure. Focus on: "
                f"(1) the headline strategic reallocation — which channel types gain/lose and the economic logic; "
                f"(2) mROI convergence — are channels moving toward mROI=1; "
                f"(3) implementation risks for shifts >30%; "
                f"(4) pharma-specific channel mix benchmarks and what best-in-class looks like."
            )
            render_ai_button(_opt_ai_prompt, "ai_opt_btn", "🤖 Interpret Optimization Results")
        elif not _needs_scan:
            # Show "run optimization" prompt only when scan is done (or not needed)
            # but no optimization has been run yet for these settings
            _s2 = PALETTE["surface"]
            _b2 = PALETTE["border"]
            _t2 = PALETTE["text"]
            _m2 = PALETTE["muted"]
            st.markdown(f"""
            <div style='margin-top:2rem;text-align:center;padding:3rem;background:{_s2};
                        border-radius:16px;border:2px dashed {_b2}'>
                <div style='font-size:2rem;margin-bottom:1rem'>⚡</div>
                <div style='font-size:1rem;font-weight:600;color:{_t2};margin-bottom:.5rem'>
                    Configure settings and run optimization
                </div>
                <div style='font-size:.82rem;color:{_m2}'>
                    Supports Maximize Profit · Revenue · ROI with SLSQP & Differential Evolution
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── LAUNCH TAB SHARED HELPERS ────────────────────────────────────────────────

def _launch_empty_state(icon, title, subtitle):
    """Renders an empty-state placeholder card for launch tab modes."""
    _les_surf = PALETTE["surface"]; _les_bdr = PALETTE["border"]
    _les_txt  = PALETTE["text"];    _les_mu  = PALETTE["muted"]
    st.markdown(
        f"<div style='margin-top:2rem;text-align:center;padding:3rem;"
        f"background:{_les_surf};border-radius:16px;border:2px dashed {_les_bdr}'>"
        f"<div style='font-size:2.5rem;margin-bottom:1rem'>{icon}</div>"
        f"<div style='font-size:1rem;font-weight:600;color:{_les_txt};margin-bottom:.5rem'>{title}</div>"
        f"<div style='font-size:.82rem;color:{_les_mu}'>{subtitle}</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def _render_launch_results(df, spends, lb_total, objective, method,
                            use_hill, use_log, ch_desc_map,
                            ref_rev, ref_profit, ref_roas, cache_suffix):
    """Shared results renderer for optimizer-based launch modes."""
    df_out = df.copy()
    df_out['launch_spend']   = spends
    df_out['launch_revenue'] = [revenue_from_spend(spends[i], df_out.iloc[i], use_hill, use_log)
                                 for i in range(len(df_out))]
    df_out['launch_profit']  = df_out['launch_revenue'] - df_out['launch_spend']
    df_out['launch_roi']     = df_out['launch_revenue'] / df_out['launch_spend'].replace(0, np.nan)
    df_out['launch_pct']     = df_out['launch_spend'] / lb_total * 100

    _tot_rev  = df_out['launch_revenue'].sum()
    _tot_prof = df_out['launch_profit'].sum()
    _tot_sp   = df_out['launch_spend'].sum()
    _roas     = _tot_rev / max(_tot_sp, EPS)

    lk1, lk2, lk3, lk4 = st.columns(4)
    lk1.metric("Launch Budget", fmt(lb_total, "$"))
    lk2.metric("Projected Revenue", fmt(_tot_rev, "$"),
               delta=f"{fmt(_tot_rev - ref_rev, '$', '+' if _tot_rev >= ref_rev else '')}")
    lk3.metric("Projected Profit", fmt(_tot_prof, "$"),
               delta=f"{fmt(_tot_prof - ref_profit, '$', '+' if _tot_prof >= ref_profit else '')}")
    lk4.metric("Projected ROI", f"{_roas:.2f}×",
               delta=f"{_roas - ref_roas:+.2f}× vs baseline")

    st.markdown(
        f"<div class='alert-box alert-success' style='margin:.5rem 0;font-size:.8rem'>"
        f"✅ {objective} · {method} · Budget utilisation: {_tot_sp/lb_total*100:.1f}%</div>",
        unsafe_allow_html=True
    )

    if PLOTLY:
        _ch = df_out['channel'].tolist()
        fig_r1 = go.Figure(go.Pie(
            labels=df_out['channel'], values=df_out['launch_spend'],
            marker=dict(colors=[channel_color(i) for i in range(len(df_out))],
                        line=dict(color=PALETTE["bg"], width=2)),
            hole=0.5, textinfo='label+percent', textfont=dict(size=9)
        ))
        fig_r1.update_layout(
            title=dict(text="Launch Budget Allocation", font=dict(size=11, color=PALETTE["text2"])),
            paper_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(l=5, r=5, t=38, b=10), showlegend=False,
            font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"])
        )

        _roi_cols = [PALETTE["accent2"] if v >= 1 else PALETTE["accent3"]
                     for v in df_out['launch_roi'].fillna(0)]
        fig_r2 = go.Figure(go.Bar(
            x=_ch, y=df_out['launch_roi'].fillna(0),
            marker_color=[hex_to_rgba(c, 0.55) for c in _roi_cols],
            marker_line_color=_roi_cols, marker_line_width=1.5,
            text=[f"{v:.2f}×" if pd.notnull(v) else "—" for v in df_out['launch_roi']],
            textposition='outside', textfont=dict(size=8)
        ))
        fig_r2.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["muted"],
                          annotation_text="ROI=1", annotation_font=dict(size=9))
        fig_r2.update_layout(
            title=dict(text="Projected ROI by Channel",
                       font=dict(size=11, color=PALETTE["text2"])),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
            font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
            height=280, margin=dict(l=5, r=10, t=38, b=70), showlegend=False,
            xaxis=dict(gridcolor="#EAECF0", tickangle=-30,
                       tickfont=dict(size=9, color=PALETTE["muted"])),
            yaxis=dict(title="ROI", tickformat=".2f", ticksuffix="×",
                       gridcolor="#EAECF0", zeroline=False,
                       tickfont=dict(size=9, color=PALETTE["muted"]))
        )
        rc1, rc2 = st.columns(2)
        with rc1: st.plotly_chart(fig_r1, width='stretch')
        with rc2: st.plotly_chart(fig_r2, width='stretch')

    st.markdown(f"<div class='section-header'>Launch allocation detail</div>", unsafe_allow_html=True)
    _disp = df_out[['channel', 'launch_spend', 'launch_pct', 'launch_revenue', 'launch_profit', 'launch_roi']].copy()
    _disp.columns = ['Channel', 'Launch Spend', '% of Budget', 'Revenue', 'Profit', 'ROI']
    for c in ['Launch Spend', 'Revenue', 'Profit']:
        _disp[c] = _disp[c].apply(lambda x: f"${x:,.0f}")
    _disp['% of Budget'] = _disp['% of Budget'].apply(lambda x: f"{x:.1f}%")
    _disp['ROI'] = _disp['ROI'].apply(lambda x: f"{x:.2f}×" if pd.notnull(x) else "—")
    st.dataframe(_disp, width='stretch', hide_index=True)

    _csv = io.StringIO()
    df_out[['channel', 'launch_spend', 'launch_pct', 'launch_revenue',
            'launch_profit', 'launch_roi']].to_csv(_csv, index=False)
    st.download_button("⬇️ Download launch plan (CSV)", data=_csv.getvalue(),
                        file_name=f"mmm_launch_{cache_suffix}.csv", mime="text/csv")


# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — LAUNCH BRAND BUDGET OPTIMIZATION - Uncomment the whole tab to add launch brand - Line 3295 - 4242  
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

# with tab_launch:
#     st.markdown(
#         f"<div class='section-header'>Launch Brand Budget Optimization — Pre-launch channel planning</div>",
#         unsafe_allow_html=True
#     )

#     # ── Mode selector ────────────────────────────────────────────────────────
#     lb_mode = st.radio(
#         "Planning method",
#         [
#             "🎯  Optimizer (Response Curves)",
#             "📊  Benchmark-Driven Allocation",
#             "⚖️  Parametric Scaling",
#             "🔍  Goal-Seek (Revenue Target)",
#         ],
#         horizontal=True,
#         label_visibility="collapsed",
#         key="lb_mode_sel"
#     )

#     # ── Shared: total budget input shown at top for all modes ────────────────
#     _lb_bgt_col, _lb_info_col = st.columns([0.35, 0.65])
#     with _lb_bgt_col:
#         lb_total = st.number_input(
#             "Total launch budget ($)",
#             value=float(df['total_spend'].sum()),
#             min_value=1.0, step=10000.0, format="%.0f",
#             key="lb_total_budget",
#             help="Total available budget for the launch brand"
#         )
#     with _lb_info_col:
#         _baseline_total_lb = float(df['total_spend'].sum())
#         _delta_bgt = lb_total - _baseline_total_lb
#         _cls_bgt = "alert-success" if _delta_bgt >= 0 else "alert-warn"
#         st.markdown(
#             f"<div class='alert-box {_cls_bgt}' style='margin-top:1.4rem;font-size:.8rem'>"
#             f"Baseline portfolio: <b>{fmt(_baseline_total_lb,'$')}</b> &nbsp;·&nbsp; "
#             f"Launch budget: <b>{fmt(lb_total,'$')}</b> &nbsp;·&nbsp; "
#             f"Delta: <b>{fmt(_delta_bgt,'$','+' if _delta_bgt>=0 else '')}</b></div>",
#             unsafe_allow_html=True
#         )

#     st.markdown(f"<div style='height:.4rem'></div>", unsafe_allow_html=True)

#     # ═══════════════════════════════════════════════════════════════════════
#     # MODE 1 — OPTIMIZER (Response Curves) — original mode
#     # ═══════════════════════════════════════════════════════════════════════
#     if "Optimizer" in lb_mode:
#         st.markdown(
#             f"<div class='alert-box alert-info' style='margin-bottom:.75rem;font-size:.8rem'>"
#             f"<b>How it works:</b> Uses your uploaded model's response curves to find the "
#             f"spend allocation that maximises the selected objective. Set per-channel min/max "
#             f"bounds as % of the total launch budget.</div>",
#             unsafe_allow_html=True
#         )

#         lb_left, lb_right = st.columns([0.35, 0.65])

#         with lb_left:
#             lb_objective = st.selectbox(
#                 "Optimization objective",
#                 ["Maximize Profit", "Maximize Revenue", "Maximize ROI"],
#                 key="lb_objective"
#             )
#             lb_obj_map = {"Maximize Profit": "profit", "Maximize Revenue": "revenue", "Maximize ROI": "roi"}
#             lb_method = st.selectbox(
#                 "Solver",
#                 ["GEKKO / IPOPT (recommended)", "SLSQP (fast)", "Differential Evolution (robust)"],
#                 key="lb_solver"
#             )

#             st.markdown(
#                 f"<div style='font-size:.7rem;font-weight:700;text-transform:uppercase;"
#                 f"letter-spacing:.06em;color:{PALETTE['muted']};padding:.5rem 0 .2rem'>"
#                 f"Per-Channel Bounds (% of total budget)</div>",
#                 unsafe_allow_html=True
#             )
#             st.caption("Min/Max % of the total launch budget each channel can receive.")

#             lb_bounds = {}
#             for _, row in df.iterrows():
#                 ch       = row['channel']
#                 ch_desc  = ch_desc_map.get(ch, ch)
#                 base_pct = float(row['total_spend']) / _baseline_total_lb * 100 if _baseline_total_lb > 0 else (100.0 / len(df))
#                 lbc1, lbc2 = st.columns(2)
#                 lb_lo = lbc1.number_input(
#                     f"{ch} Min%", value=max(round(base_pct * 0.3, 1), 0.0),
#                     min_value=0.0, max_value=100.0, step=1.0,
#                     key=f"lb_lo_{ch}", help=f"Min % of budget for {ch_desc}"
#                 )
#                 lb_hi = lbc2.number_input(
#                     f"{ch} Max%", value=min(round(base_pct * 2.5, 1), 100.0),
#                     min_value=0.0, max_value=100.0, step=1.0,
#                     key=f"lb_hi_{ch}", help=f"Max % of budget for {ch_desc}"
#                 )
#                 lb_bounds[ch] = (lb_lo / 100.0, lb_hi / 100.0)

#             st.markdown(f"<div style='height:.4rem'></div>", unsafe_allow_html=True)
#             run_lb = st.button("🚀 Optimize Launch Budget", type="primary",
#                                width="stretch", key="run_launch_btn")

#         with lb_right:
#             # ── Bounds conversion with feasibility fix ──────────────────────
#             df_lb = df.copy()
#             df_lb['lock_spend'] = 0

#             abs_lo, abs_hi = {}, {}
#             for _, row in df_lb.iterrows():
#                 ch = row['channel']
#                 lo_frac, hi_frac = lb_bounds.get(ch, (0.0, 1.0))
#                 abs_lo[ch] = lo_frac * lb_total
#                 abs_hi[ch] = hi_frac * lb_total

#             sum_lo = sum(abs_lo.values())
#             sum_hi = sum(abs_hi.values())
#             if sum_lo > lb_total and sum_lo > 0:
#                 s = lb_total / sum_lo
#                 abs_lo = {ch: v * s for ch, v in abs_lo.items()}
#             if sum_hi < lb_total and sum_hi > 0:
#                 s = lb_total / sum_hi
#                 abs_hi = {ch: v * s for ch, v in abs_hi.items()}

#             for _, row in df_lb.iterrows():
#                 ch = row['channel']
#                 bsp = float(row['total_spend'])
#                 if bsp > 0:
#                     df_lb.loc[df_lb['channel'] == ch, 'lower_bound_pct'] = abs_lo[ch] / bsp
#                     df_lb.loc[df_lb['channel'] == ch, 'upper_bound_pct'] = abs_hi[ch] / bsp
#                 else:
#                     df_lb.loc[df_lb['channel'] == ch, 'lower_bound_pct'] = 0.0
#                     df_lb.loc[df_lb['channel'] == ch, 'upper_bound_pct'] = 2.0

#             _orig_sum_lo = sum(lb_bounds.get(ch, (0.0, 1.0))[0] * lb_total for ch in df_lb['channel'])
#             _orig_sum_hi = sum(lb_bounds.get(ch, (0.0, 1.0))[1] * lb_total for ch in df_lb['channel'])
#             if _orig_sum_lo > lb_total:
#                 st.warning(f"⚠️ Min bounds summed to {fmt(_orig_sum_lo,'$')} > budget. Scaled down proportionally.")
#             if _orig_sum_hi < lb_total:
#                 st.warning(f"⚠️ Max bounds summed to {fmt(_orig_sum_hi,'$')} < budget. Scaled up proportionally.")

#             _lb_cache_key = (
#                 f"launch_opt|{lb_obj_map[lb_objective]}|{method_map[lb_method]}|"
#                 f"{lb_total:.2f}|{use_hill_global}|{use_log_global}|"
#                 + "|".join(f"{ch}:{lo:.3f}:{hi:.3f}" for ch, (lo, hi) in sorted(lb_bounds.items()))
#             )

#             if run_lb:
#                 with st.spinner("Optimizing launch budget…"):
#                     _lb_res = optimize_budget(
#                         df_lb, lb_total,
#                         objective=lb_obj_map[lb_objective],
#                         use_hill=use_hill_global, use_log=use_log_global,
#                         method=method_map[lb_method]
#                     )
#                 st.session_state[_lb_cache_key] = _lb_res

#             _lb_result = st.session_state.get(_lb_cache_key)

#             if _lb_result and _lb_result.get("success") and _lb_result.get("spends") is not None:
#                 _render_launch_results(df, _lb_result["spends"], lb_total, lb_objective, lb_method,
#                                        use_hill_global, use_log_global, ch_desc_map,
#                                        tot_rev, tot_profit, avg_roas, "lb_opt")
#             elif _lb_result and not _lb_result.get("success"):
#                 st.error(f"Optimization failed: {_lb_result.get('message','Unknown error')}")
#             else:
#                 _launch_empty_state("🎯", "Configure settings and click Optimize",
#                                     "Optimizer allocates spend to maximise your objective using response curves.")

#     # ═══════════════════════════════════════════════════════════════════════
#     # MODE 2 — BENCHMARK-DRIVEN ALLOCATION
#     # ═══════════════════════════════════════════════════════════════════════
#     elif "Benchmark" in lb_mode:
#         st.markdown(
#             f"<div class='alert-box alert-info' style='margin-bottom:.75rem;font-size:.8rem'>"
#             f"<b>How it works:</b> Seed the channel mix from pharma launch benchmarks, then "
#             f"adjust sliders to your brand's situation. No optimizer — you control the allocation "
#             f"directly. Projected revenue/profit are calculated from the response curves.</div>",
#             unsafe_allow_html=True
#         )

#         bm_left, bm_right = st.columns([0.38, 0.62])

#         # ── Pharma launch benchmark presets ──────────────────────────────
#         BENCHMARKS = {
#             "Specialty Launch (typical)": {
#                 "field_force": 0.45, "hcp_digital": 0.20, "dtc": 0.15,
#                 "speaker": 0.08, "samples": 0.07, "other": 0.05
#             },
#             "Primary Care Launch": {
#                 "field_force": 0.55, "hcp_digital": 0.15, "dtc": 0.18,
#                 "speaker": 0.05, "samples": 0.05, "other": 0.02
#             },
#             "Rare Disease / Specialty Lite": {
#                 "field_force": 0.30, "hcp_digital": 0.35, "dtc": 0.05,
#                 "speaker": 0.15, "samples": 0.08, "other": 0.07
#             },
#             "Digital-First Launch": {
#                 "field_force": 0.20, "hcp_digital": 0.40, "dtc": 0.25,
#                 "speaker": 0.05, "samples": 0.05, "other": 0.05
#             },
#             "Custom (equal split)": None
#         }

#         # Channel type mapping — user assigns each channel to a benchmark bucket
#         BUCKET_LABELS = ["field_force", "hcp_digital", "dtc", "speaker", "samples", "other"]
#         BUCKET_DISPLAY = {
#             "field_force": "Field Force / MSL",
#             "hcp_digital": "HCP Digital / PLD",
#             "dtc":         "DTC / Consumer",
#             "speaker":     "Speaker Programs / SERMO",
#             "samples":     "Samples / Co-pay",
#             "other":       "Other / DSE / Social"
#         }

#         with bm_left:
#             bm_preset = st.selectbox(
#                 "Pharma launch benchmark",
#                 list(BENCHMARKS.keys()),
#                 key="bm_preset",
#                 help="Industry benchmarks for launch-year channel mix by brand type"
#             )

#             st.markdown(
#                 f"<div style='font-size:.7rem;font-weight:700;text-transform:uppercase;"
#                 f"letter-spacing:.06em;color:{PALETTE['muted']};padding:.5rem 0 .2rem'>"
#                 f"Map channels → benchmark bucket</div>",
#                 unsafe_allow_html=True
#             )
#             st.caption("Assign each model channel to its benchmark category.")

#             ch_bucket_map = {}
#             for _, row in df.iterrows():
#                 ch = row['channel']
#                 ch_bucket_map[ch] = st.selectbox(
#                     f"{ch}", BUCKET_LABELS,
#                     format_func=lambda x: BUCKET_DISPLAY[x],
#                     key=f"bm_bucket_{ch}"
#                 )

#         with bm_right:
#             # Compute benchmark-seeded allocation
#             preset_weights = BENCHMARKS[bm_preset]
#             if preset_weights is None:
#                 # Equal split
#                 preset_weights = {b: 1.0 / len(BUCKET_LABELS) for b in BUCKET_LABELS}

#             # Aggregate baseline spend per bucket for proportional intra-bucket split
#             bucket_spend = {b: 0.0 for b in BUCKET_LABELS}
#             for _, row in df.iterrows():
#                 bucket_spend[ch_bucket_map[row['channel']]] += float(row['total_spend'])

#             # Seed spend: benchmark_weight × lb_total, split within bucket by baseline share
#             bm_seed_spend = {}
#             for _, row in df.iterrows():
#                 ch     = row['channel']
#                 bucket = ch_bucket_map[ch]
#                 bkt_sp = bucket_spend[bucket]
#                 ch_share = float(row['total_spend']) / bkt_sp if bkt_sp > 0 else 1.0
#                 bm_seed_spend[ch] = preset_weights.get(bucket, 0.0) * lb_total * ch_share

#             # Show benchmark reference bar
#             if PLOTLY:
#                 bm_ref_labels = [BUCKET_DISPLAY[b] for b in BUCKET_LABELS]
#                 bm_ref_vals   = [preset_weights.get(b, 0) * 100 for b in BUCKET_LABELS]
#                 fig_bm_ref = go.Figure(go.Bar(
#                     x=bm_ref_labels, y=bm_ref_vals,
#                     marker_color=[channel_color(i) for i in range(len(BUCKET_LABELS))],
#                     text=[f"{v:.0f}%" for v in bm_ref_vals],
#                     textposition='outside', textfont=dict(size=9)
#                 ))
#                 fig_bm_ref.update_layout(
#                     title=dict(text=f"Benchmark Mix: {bm_preset}",
#                                font=dict(size=11, color=PALETTE["text2"])),
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
#                     font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
#                     height=220, margin=dict(l=5, r=10, t=38, b=80), showlegend=False,
#                     xaxis=dict(gridcolor="#EAECF0", tickangle=-20,
#                                tickfont=dict(size=9, color=PALETTE["muted"])),
#                     yaxis=dict(title="% Budget", ticksuffix="%", gridcolor="#EAECF0",
#                                zeroline=False, tickfont=dict(size=9, color=PALETTE["muted"]))
#                 )
#                 st.plotly_chart(fig_bm_ref, width='stretch')

#             st.markdown(
#                 f"<div style='font-size:.7rem;font-weight:700;text-transform:uppercase;"
#                 f"letter-spacing:.06em;color:{PALETTE['muted']};padding:.4rem 0 .2rem'>"
#                 f"Adjust allocation (% of total budget)</div>",
#                 unsafe_allow_html=True
#             )
#             st.caption("Sliders are pre-seeded from the benchmark. Adjust to fit your brand.")

#             bm_alloc_pct = {}
#             _remaining   = 100.0
#             _ch_list     = df['channel'].tolist()
#             slider_cols  = st.columns(2)
#             for i, (_, row) in enumerate(df.iterrows()):
#                 ch       = row['channel']
#                 seed_pct = bm_seed_spend[ch] / lb_total * 100 if lb_total > 0 else 100.0 / len(df)
#                 with slider_cols[i % 2]:
#                     bm_alloc_pct[ch] = st.slider(
#                         f"{ch} (%)", 0.0, 100.0, round(seed_pct, 1),
#                         step=0.5, format="%.1f%%",
#                         key=f"bm_sl_{ch}"
#                     )

#             total_alloc = sum(bm_alloc_pct.values())
#             if abs(total_alloc - 100.0) > 0.5:
#                 st.warning(
#                     f"⚠️ Allocations sum to {total_alloc:.1f}% — must equal 100%. "
#                     f"Adjust sliders or they will be normalised automatically."
#                 )

#             # Normalise to 100%
#             norm_factor = total_alloc / 100.0 if total_alloc > 0 else 1.0
#             bm_spends = {ch: (pct / 100.0 / norm_factor) * lb_total
#                          for ch, pct in bm_alloc_pct.items()}

#             # Compute revenue/profit from response curves
#             df_bm = df.copy()
#             df_bm['bm_spend']   = [bm_spends[ch] for ch in df_bm['channel']]
#             df_bm['bm_revenue'] = [revenue_from_spend(df_bm.iloc[i]['bm_spend'], df_bm.iloc[i],
#                                                        use_hill_global, use_log_global)
#                                     for i in range(len(df_bm))]
#             df_bm['bm_profit']  = df_bm['bm_revenue'] - df_bm['bm_spend']
#             df_bm['bm_roi']     = df_bm['bm_revenue'] / df_bm['bm_spend'].replace(0, np.nan)
#             df_bm['bm_pct']     = df_bm['bm_spend'] / lb_total * 100

#             _bm_rev  = df_bm['bm_revenue'].sum()
#             _bm_prof = df_bm['bm_profit'].sum()
#             _bm_sp   = df_bm['bm_spend'].sum()
#             _bm_roas = _bm_rev / max(_bm_sp, EPS)

#             # KPIs
#             bk1, bk2, bk3, bk4 = st.columns(4)
#             bk1.metric("Total Allocated", fmt(_bm_sp, "$"),
#                        delta=f"{_bm_sp/lb_total*100:.1f}% of budget")
#             bk2.metric("Projected Revenue", fmt(_bm_rev, "$"),
#                        delta=f"{fmt(_bm_rev - tot_rev, '$', '+' if _bm_rev >= tot_rev else '')}")
#             bk3.metric("Projected Profit", fmt(_bm_prof, "$"),
#                        delta=f"{fmt(_bm_prof - tot_profit, '$', '+' if _bm_prof >= tot_profit else '')}")
#             bk4.metric("Projected ROI", f"{_bm_roas:.2f}×",
#                        delta=f"{_bm_roas - avg_roas:+.2f}× vs baseline")

#             if PLOTLY:
#                 # Pie
#                 fig_bm_pie = go.Figure(go.Pie(
#                     labels=df_bm['channel'], values=df_bm['bm_spend'],
#                     marker=dict(colors=[channel_color(i) for i in range(len(df_bm))],
#                                 line=dict(color=PALETTE["bg"], width=2)),
#                     hole=0.5, textinfo='label+percent', textfont=dict(size=9)
#                 ))
#                 fig_bm_pie.update_layout(
#                     title=dict(text="Benchmark Allocation", font=dict(size=11, color=PALETTE["text2"])),
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                     font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
#                     height=280, margin=dict(l=5, r=5, t=38, b=10), showlegend=False
#                 )

#                 # ROI bars
#                 _bm_roi_colors = [PALETTE["accent2"] if v >= 1 else PALETTE["accent3"]
#                                   for v in df_bm['bm_roi'].fillna(0)]
#                 fig_bm_roi = go.Figure(go.Bar(
#                     x=df_bm['channel'], y=df_bm['bm_roi'].fillna(0),
#                     marker_color=[hex_to_rgba(c, 0.55) for c in _bm_roi_colors],
#                     marker_line_color=_bm_roi_colors, marker_line_width=1.5,
#                     text=[f"{v:.2f}×" if pd.notnull(v) else "—" for v in df_bm['bm_roi']],
#                     textposition='outside', textfont=dict(size=8)
#                 ))
#                 fig_bm_roi.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["muted"],
#                                       annotation_text="ROI=1", annotation_font=dict(size=9))
#                 fig_bm_roi.update_layout(
#                     title=dict(text="Projected ROI by Channel",
#                                font=dict(size=11, color=PALETTE["text2"])),
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
#                     font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
#                     height=260, margin=dict(l=5, r=10, t=38, b=70), showlegend=False,
#                     xaxis=dict(gridcolor="#EAECF0", tickangle=-30,
#                                tickfont=dict(size=9, color=PALETTE["muted"])),
#                     yaxis=dict(title="ROI", tickformat=".2f", ticksuffix="×",
#                                gridcolor="#EAECF0", zeroline=False,
#                                tickfont=dict(size=9, color=PALETTE["muted"]))
#                 )

#                 pc1, pc2 = st.columns(2)
#                 with pc1:
#                     st.plotly_chart(fig_bm_pie, width='stretch')
#                 with pc2:
#                     st.plotly_chart(fig_bm_roi, width='stretch')

#             # Results table
#             st.markdown(f"<div class='section-header'>Benchmark allocation detail</div>",
#                         unsafe_allow_html=True)
#             _bm_disp = df_bm[['channel', 'bm_spend', 'bm_pct', 'bm_revenue', 'bm_profit', 'bm_roi']].copy()
#             _bm_disp.columns = ['Channel', 'Launch Spend', '% of Budget', 'Revenue', 'Profit', 'ROI']
#             for c in ['Launch Spend', 'Revenue', 'Profit']:
#                 _bm_disp[c] = _bm_disp[c].apply(lambda x: f"${x:,.0f}")
#             _bm_disp['% of Budget'] = _bm_disp['% of Budget'].apply(lambda x: f"{x:.1f}%")
#             _bm_disp['ROI'] = _bm_disp['ROI'].apply(lambda x: f"{x:.2f}×" if pd.notnull(x) else "—")
#             st.dataframe(_bm_disp, width='stretch', hide_index=True)

#             _bm_csv = io.StringIO()
#             df_bm[['channel', 'bm_spend', 'bm_pct', 'bm_revenue', 'bm_profit', 'bm_roi']].to_csv(_bm_csv, index=False)
#             st.download_button("⬇️ Download benchmark plan (CSV)", data=_bm_csv.getvalue(),
#                                 file_name="mmm_launch_benchmark.csv", mime="text/csv")

#             _bm_prompt = (
#                 f"BENCHMARK-DRIVEN LAUNCH ALLOCATION\n\n"
#                 f"Benchmark: {bm_preset} | Budget: {fmt(lb_total,'$')}\n\n"
#                 f"CHANNEL ALLOCATION:\n"
#                 + "\n".join([
#                     f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
#                     f"spend={fmt(float(r['bm_spend']),'$')} ({float(r['bm_pct']):.1f}%), "
#                     f"ROI={float(r['bm_roi']):.2f}x"
#                     for _, r in df_bm.iterrows()
#                 ])
#                 + f"\n\nPROJECTED: Revenue={fmt(_bm_rev,'$')}, Profit={fmt(_bm_prof,'$')}, ROI={_bm_roas:.2f}x\n\n"
#                 f"This is a BENCHMARK-DRIVEN analysis. Analyse: "
#                 f"(1) how this mix compares to the {bm_preset} benchmark norms; "
#                 f"(2) which channels are over/under-weighted vs pharma best practice; "
#                 f"(3) projected ROI feasibility for a launch brand; "
#                 f"(4) recommended adjustments for months 1-6 vs 7-12."
#             )
#             render_ai_button(_bm_prompt, "ai_bm_btn", "🤖 Interpret Benchmark Allocation")

#     # ═══════════════════════════════════════════════════════════════════════
#     # MODE 3 — PARAMETRIC SCALING
#     # ═══════════════════════════════════════════════════════════════════════
#     elif "Parametric" in lb_mode:
#         st.markdown(
#             f"<div class='alert-box alert-info' style='margin-bottom:.75rem;font-size:.8rem'>"
#             f"<b>How it works:</b> Apply per-channel <b>effectiveness multipliers</b> to the "
#             f"response curves before optimizing. A multiplier > 1 makes a channel more "
#             f"effective (e.g. 1.3 = 30% uplift vs modelled brand — typical for new molecule "
#             f"novelty effect). A multiplier < 1 reflects lower effectiveness "
#             f"(e.g. 0.7 = limited formulary access at launch).</div>",
#             unsafe_allow_html=True
#         )

#         ps_left, ps_right = st.columns([0.35, 0.65])

#         with ps_left:
#             ps_objective = st.selectbox(
#                 "Optimization objective",
#                 ["Maximize Profit", "Maximize Revenue", "Maximize ROI"],
#                 key="ps_objective"
#             )
#             ps_obj_map = {"Maximize Profit": "profit", "Maximize Revenue": "revenue", "Maximize ROI": "roi"}
#             ps_method = st.selectbox(
#                 "Solver",
#                 ["GEKKO / IPOPT (recommended)", "SLSQP (fast)", "Differential Evolution (robust)"],
#                 key="ps_solver"
#             )

#             st.markdown(
#                 f"<div style='font-size:.7rem;font-weight:700;text-transform:uppercase;"
#                 f"letter-spacing:.06em;color:{PALETTE['muted']};padding:.5rem 0 .2rem'>"
#                 f"Channel Effectiveness Multipliers</div>",
#                 unsafe_allow_html=True
#             )
#             st.caption("Scale each channel's Adj_Factor for launch-specific dynamics.")

#             ps_multipliers = {}
#             _preset_col = st.selectbox(
#                 "Quick preset",
#                 ["Custom", "Conservative launch (−20% all)", "Strong launch (+20% all)",
#                  "Digital-heavy launch (HCP digital +30%, FF −10%)"],
#                 key="ps_quick_preset"
#             )

#             for _, row in df.iterrows():
#                 ch      = row['channel']
#                 ch_desc = ch_desc_map.get(ch, ch)
#                 # Default multiplier based on quick preset
#                 if "Conservative" in _preset_col:
#                     default_mult = 0.8
#                 elif "Strong" in _preset_col:
#                     default_mult = 1.2
#                 else:
#                     default_mult = 1.0

#                 ps_multipliers[ch] = st.slider(
#                     f"{ch} multiplier",
#                     min_value=0.1, max_value=3.0,
#                     value=default_mult, step=0.05,
#                     format="%.2f×",
#                     key=f"ps_mult_{ch}",
#                     help=(f"Scales the Adj_Factor for {ch_desc}. "
#                           f"Baseline Adj_Factor = {float(row['Adj_Factor']):.3f}")
#                 )

#             st.markdown(f"<div style='height:.4rem'></div>", unsafe_allow_html=True)
#             run_ps = st.button("🚀 Run Parametric Optimization", type="primary",
#                                width="stretch", key="run_ps_btn")

#         with ps_right:
#             # Build scaled df
#             df_ps = df.copy()
#             df_ps['lock_spend'] = 0
#             for _, row in df_ps.iterrows():
#                 ch = row['channel']
#                 orig_adj = float(row['Adj_Factor'])
#                 df_ps.loc[df_ps['channel'] == ch, 'Adj_Factor'] = orig_adj * ps_multipliers.get(ch, 1.0)

#             # Set bounds: full budget freedom (0% min, 100% max per channel)
#             for _, row in df_ps.iterrows():
#                 ch  = row['channel']
#                 bsp = float(row['total_spend'])
#                 if bsp > 0:
#                     df_ps.loc[df_ps['channel'] == ch, 'lower_bound_pct'] = 0.0
#                     df_ps.loc[df_ps['channel'] == ch, 'upper_bound_pct'] = lb_total / bsp
#                 else:
#                     df_ps.loc[df_ps['channel'] == ch, 'lower_bound_pct'] = 0.0
#                     df_ps.loc[df_ps['channel'] == ch, 'upper_bound_pct'] = 2.0

#             # Show multiplier impact preview
#             if PLOTLY:
#                 _mult_vals  = [ps_multipliers.get(ch, 1.0) for ch in df['channel']]
#                 _mult_cols  = [PALETTE["accent2"] if v > 1 else
#                                (PALETTE["accent3"] if v < 1 else PALETTE["muted"])
#                                for v in _mult_vals]
#                 fig_mult = go.Figure(go.Bar(
#                     x=df['channel'], y=_mult_vals,
#                     marker_color=[hex_to_rgba(c, 0.55) for c in _mult_cols],
#                     marker_line_color=_mult_cols, marker_line_width=1.5,
#                     text=[f"{v:.2f}×" for v in _mult_vals],
#                     textposition='outside', textfont=dict(size=9)
#                 ))
#                 fig_mult.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["muted"],
#                                     line_width=1.5, annotation_text="Baseline (1.0×)",
#                                     annotation_font=dict(size=9))
#                 fig_mult.update_layout(
#                     title=dict(text="Effectiveness Multipliers vs Baseline",
#                                font=dict(size=11, color=PALETTE["text2"])),
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
#                     font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
#                     height=240, margin=dict(l=5, r=10, t=38, b=70), showlegend=False,
#                     xaxis=dict(gridcolor="#EAECF0", tickangle=-30,
#                                tickfont=dict(size=9, color=PALETTE["muted"])),
#                     yaxis=dict(title="Multiplier", gridcolor="#EAECF0", zeroline=False,
#                                tickformat=".2f", ticksuffix="×",
#                                tickfont=dict(size=9, color=PALETTE["muted"]))
#                 )
#                 st.plotly_chart(fig_mult, width='stretch')

#             _ps_cache_key = (
#                 f"launch_ps|{ps_obj_map[ps_objective]}|{method_map[ps_method]}|"
#                 f"{lb_total:.2f}|{use_hill_global}|{use_log_global}|"
#                 + "|".join(f"{ch}:{ps_multipliers.get(ch,1.0):.3f}" for ch in sorted(df['channel']))
#             )

#             if run_ps:
#                 with st.spinner("Running parametric optimization…"):
#                     _ps_res = optimize_budget(
#                         df_ps, lb_total,
#                         objective=ps_obj_map[ps_objective],
#                         use_hill=use_hill_global, use_log=use_log_global,
#                         method=method_map[ps_method]
#                     )
#                 st.session_state[_ps_cache_key] = _ps_res

#             _ps_result = st.session_state.get(_ps_cache_key)

#             if _ps_result and _ps_result.get("success") and _ps_result.get("spends") is not None:
#                 # Use SCALED df for revenue calc so multipliers are applied
#                 _ps_spends = _ps_result["spends"]
#                 df_ps_out  = df_ps.copy()
#                 df_ps_out['ps_spend']   = _ps_spends
#                 df_ps_out['ps_revenue'] = [revenue_from_spend(_ps_spends[i], df_ps_out.iloc[i],
#                                                                use_hill_global, use_log_global)
#                                             for i in range(len(df_ps_out))]
#                 df_ps_out['ps_profit']  = df_ps_out['ps_revenue'] - df_ps_out['ps_spend']
#                 df_ps_out['ps_roi']     = df_ps_out['ps_revenue'] / df_ps_out['ps_spend'].replace(0, np.nan)
#                 df_ps_out['ps_pct']     = df_ps_out['ps_spend'] / lb_total * 100
#                 df_ps_out['mult']       = [ps_multipliers.get(ch, 1.0) for ch in df_ps_out['channel']]

#                 _ps_rev  = df_ps_out['ps_revenue'].sum()
#                 _ps_prof = df_ps_out['ps_profit'].sum()
#                 _ps_sp   = df_ps_out['ps_spend'].sum()
#                 _ps_roas = _ps_rev / max(_ps_sp, EPS)

#                 # Baseline revenue at same budget for comparison (unscaled curves)
#                 _ps_base_rev  = sum(revenue_from_spend(float(df.iloc[i]['total_spend']), df.iloc[i],
#                                                         use_hill_global, use_log_global)
#                                      for i in range(len(df)))

#                 pk1, pk2, pk3, pk4 = st.columns(4)
#                 pk1.metric("Launch Budget", fmt(lb_total, "$"))
#                 pk2.metric("Projected Revenue", fmt(_ps_rev, "$"),
#                            delta=f"{fmt(_ps_rev - tot_rev, '$', '+' if _ps_rev >= tot_rev else '')}")
#                 pk3.metric("Projected Profit", fmt(_ps_prof, "$"),
#                            delta=f"{fmt(_ps_prof - tot_profit, '$', '+' if _ps_prof >= tot_profit else '')}")
#                 pk4.metric("Projected ROI", f"{_ps_roas:.2f}×",
#                            delta=f"{_ps_roas - avg_roas:+.2f}× vs baseline")

#                 st.markdown(
#                     f"<div class='alert-box alert-success' style='margin:.5rem 0;font-size:.8rem'>"
#                     f"✅ Parametric solve complete · {ps_objective} · {ps_method}</div>",
#                     unsafe_allow_html=True
#                 )

#                 if PLOTLY:
#                     # Spend allocation + multiplier overlay
#                     fig_ps1 = go.Figure()
#                     fig_ps1.add_trace(go.Bar(
#                         name='Baseline Spend', x=df_ps_out['channel'], y=df_ps_out['total_spend'],
#                         marker_color=hex_to_rgba(PALETTE["accent"], 0.4),
#                         marker_line_color=PALETTE["accent"], marker_line_width=1.5,
#                         text=[fmt(v, "$") for v in df_ps_out['total_spend']],
#                         textposition='outside', textfont=dict(size=8, color=PALETTE["accent"])
#                     ))
#                     fig_ps1.add_trace(go.Bar(
#                         name='Parametric Launch Spend', x=df_ps_out['channel'], y=df_ps_out['ps_spend'],
#                         marker_color=hex_to_rgba(PALETTE["accent4"], 0.5),
#                         marker_line_color=PALETTE["accent4"], marker_line_width=1.5,
#                         text=[fmt(v, "$") for v in df_ps_out['ps_spend']],
#                         textposition='outside', textfont=dict(size=8, color=PALETTE["accent4"])
#                     ))
#                     fig_ps1.update_layout(
#                         barmode='group',
#                         title=dict(text="Parametric Launch vs Baseline Spend",
#                                    font=dict(size=11, color=PALETTE["text2"])),
#                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
#                         font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
#                         height=300, margin=dict(l=5, r=10, t=38, b=70),
#                         legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=PALETTE["border"],
#                                     borderwidth=1, orientation="h", y=-0.25, font=dict(size=10)),
#                         xaxis=dict(gridcolor="#EAECF0", tickangle=-30,
#                                    tickfont=dict(size=9, color=PALETTE["muted"])),
#                         yaxis=dict(title="Spend ($)", tickformat="$,.0f", gridcolor="#EAECF0",
#                                    zeroline=False, tickfont=dict(size=9, color=PALETTE["muted"]))
#                     )
#                     st.plotly_chart(fig_ps1, width='stretch')

#                 # Results table with multiplier column
#                 st.markdown(f"<div class='section-header'>Parametric results detail</div>",
#                             unsafe_allow_html=True)
#                 _ps_disp = df_ps_out[['channel', 'mult', 'ps_spend', 'ps_pct',
#                                        'ps_revenue', 'ps_profit', 'ps_roi']].copy()
#                 _ps_disp.columns = ['Channel', 'Multiplier', 'Launch Spend', '% of Budget',
#                                      'Revenue', 'Profit', 'ROI']
#                 _ps_disp['Multiplier']   = _ps_disp['Multiplier'].apply(lambda x: f"{x:.2f}×")
#                 for c in ['Launch Spend', 'Revenue', 'Profit']:
#                     _ps_disp[c] = _ps_disp[c].apply(lambda x: f"${x:,.0f}")
#                 _ps_disp['% of Budget'] = _ps_disp['% of Budget'].apply(lambda x: f"{x:.1f}%")
#                 _ps_disp['ROI'] = _ps_disp['ROI'].apply(lambda x: f"{x:.2f}×" if pd.notnull(x) else "—")
#                 st.dataframe(_ps_disp, width='stretch', hide_index=True)

#                 _ps_csv = io.StringIO()
#                 df_ps_out[['channel', 'mult', 'ps_spend', 'ps_pct',
#                             'ps_revenue', 'ps_profit', 'ps_roi']].to_csv(_ps_csv, index=False)
#                 st.download_button("⬇️ Download parametric plan (CSV)", data=_ps_csv.getvalue(),
#                                     file_name="mmm_launch_parametric.csv", mime="text/csv")

#                 _ps_ch_ctx = "\n".join([
#                     f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
#                     f"mult={float(r['mult']):.2f}×, spend={fmt(float(r['ps_spend']),'$')} "
#                     f"({float(r['ps_pct']):.1f}%), ROI={float(r['ps_roi']):.2f}x"
#                     for _, r in df_ps_out.iterrows()
#                 ])
#                 _ps_prompt = (
#                     f"PARAMETRIC SCALING LAUNCH OPTIMIZATION\n\n"
#                     f"Objective: {ps_objective} | Budget: {fmt(lb_total,'$')}\n"
#                     f"Preset: {_preset_col}\n\n"
#                     f"CHANNEL EFFECTIVENESS MULTIPLIERS & ALLOCATION:\n{_ps_ch_ctx}\n\n"
#                     f"PROJECTED: Revenue={fmt(_ps_rev,'$')}, Profit={fmt(_ps_prof,'$')}, ROI={_ps_roas:.2f}x\n\n"
#                     f"This is a PARAMETRIC SCALING analysis. Analyse: "
#                     f"(1) how the effectiveness multipliers reflect realistic launch dynamics for each channel type; "
#                     f"(2) which channels benefit most from launch novelty effect vs which are constrained; "
#                     f"(3) how the optimized allocation changes vs the unscaled model; "
#                     f"(4) sensitivity — which multiplier assumptions most affect the projected ROI?"
#                 )
#                 render_ai_button(_ps_prompt, "ai_ps_btn", "🤖 Interpret Parametric Results")

#             elif _ps_result and not _ps_result.get("success"):
#                 st.error(f"Optimization failed: {_ps_result.get('message','Unknown error')}")
#             else:
#                 _launch_empty_state("⚖️", "Set multipliers and click Run",
#                                     "Parametric scaling adjusts response curve effectiveness before optimizing.")

#     # ═══════════════════════════════════════════════════════════════════════
#     # MODE 4 — GOAL-SEEK (Revenue Target)
#     # ═══════════════════════════════════════════════════════════════════════
#     elif "Goal-Seek" in lb_mode:
#         st.markdown(
#             f"<div class='alert-box alert-info' style='margin-bottom:.75rem;font-size:.8rem'>"
#             f"<b>How it works:</b> Flip the optimization question — instead of "
#             f"<i>'given budget, maximize revenue'</i>, ask "
#             f"<i>'what is the minimum budget to hit $X revenue?'</i> "
#             f"The solver scans budget levels and finds the minimum spend required to reach "
#             f"your target, with optimal channel allocation at each level.</div>",
#             unsafe_allow_html=True
#         )

#         gs_left, gs_right = st.columns([0.35, 0.65])

#         with gs_left:
#             # Revenue target input
#             _default_target = tot_rev * 1.10
#             gs_target = st.number_input(
#                 "Revenue target ($)",
#                 value=float(round(_default_target / 1000) * 1000),
#                 min_value=1.0, step=100000.0, format="%.0f",
#                 key="gs_revenue_target",
#                 help="The revenue level the brand needs to achieve"
#             )

#             gs_method = st.selectbox(
#                 "Solver",
#                 ["SLSQP (fast)", "GEKKO / IPOPT (recommended)", "Differential Evolution (robust)"],
#                 key="gs_solver"
#             )

#             gs_n_scan = st.slider(
#                 "Scan resolution",
#                 min_value=10, max_value=60, value=25, step=5,
#                 key="gs_scan_pts",
#                 help="Number of budget levels to scan. More = smoother curve but slower."
#             )

#             st.markdown(
#                 f"<div class='alert-box alert-warn' style='margin-top:.5rem;font-size:.78rem'>"
#                 f"ℹ️ Budget scan range: {fmt(_baseline_total_lb*0.3,'$')} – "
#                 f"{fmt(_baseline_total_lb*3.0,'$')} (30%–300% of baseline portfolio)</div>",
#                 unsafe_allow_html=True
#             )

#             run_gs = st.button("🔍 Find Minimum Budget", type="primary",
#                                width="stretch", key="run_gs_btn")

#         with gs_right:
#             _gs_cache_key = (
#                 f"launch_gs|{method_map[gs_method]}|{gs_target:.2f}|"
#                 f"{gs_n_scan}|{lb_total:.2f}|{use_hill_global}|{use_log_global}"
#             )

#             if run_gs:
#                 _gs_budgets = np.linspace(
#                     _baseline_total_lb * 0.3,
#                     _baseline_total_lb * 3.0,
#                     gs_n_scan
#                 )
#                 _gs_scan_rows = []
#                 _prog = st.progress(0, text="Scanning budget levels…")

#                 for _bi, _bgt in enumerate(_gs_budgets):
#                     _prog.progress(int((_bi + 1) / len(_gs_budgets) * 100),
#                                    text=f"Scanning {fmt(_bgt,'$')}…")
#                     _res_i = optimize_budget(
#                         df.assign(lock_spend=0), _bgt,
#                         objective="revenue",
#                         use_hill=use_hill_global, use_log=use_log_global,
#                         method=method_map[gs_method]
#                     )
#                     if _res_i.get("success") and _res_i.get("spends") is not None:
#                         _rev_i = sum(revenue_from_spend(_res_i["spends"][i], df.iloc[i],
#                                                          use_hill_global, use_log_global)
#                                       for i in range(len(df)))
#                         _prf_i = _rev_i - _bgt
#                         _gs_scan_rows.append({
#                             "budget": _bgt, "opt_revenue": _rev_i,
#                             "opt_profit": _prf_i, "spends": _res_i["spends"]
#                         })

#                 _prog.empty()
#                 st.session_state[_gs_cache_key] = _gs_scan_rows

#             _gs_rows = st.session_state.get(_gs_cache_key)

#             if _gs_rows:
#                 _gs_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'spends'}
#                                         for r in _gs_rows])

#                 # Find minimum budget that meets the revenue target
#                 _meets = _gs_df[_gs_df['opt_revenue'] >= gs_target]
#                 if not _meets.empty:
#                     _min_bgt_row  = _meets.iloc[0]
#                     _min_bgt      = float(_min_bgt_row['budget'])
#                     _min_bgt_rev  = float(_min_bgt_row['opt_revenue'])
#                     _min_bgt_prof = float(_min_bgt_row['opt_profit'])
#                     _target_met   = True
#                 else:
#                     _max_rev_row  = _gs_df.loc[_gs_df['opt_revenue'].idxmax()]
#                     _min_bgt      = float(_max_rev_row['budget'])
#                     _min_bgt_rev  = float(_max_rev_row['opt_revenue'])
#                     _min_bgt_prof = float(_max_rev_row['opt_profit'])
#                     _target_met   = False

#                 # KPIs
#                 gk1, gk2, gk3, gk4 = st.columns(4)
#                 gk1.metric(
#                     "Min Budget to Hit Target" if _target_met else "Max Achievable Revenue",
#                     fmt(_min_bgt, "$"),
#                     delta=f"{fmt(_min_bgt - _baseline_total_lb, '$', '+' if _min_bgt >= _baseline_total_lb else '')} vs baseline"
#                 )
#                 gk2.metric("Revenue at That Budget", fmt(_min_bgt_rev, "$"),
#                            delta=f"{fmt(_min_bgt_rev - gs_target, '$', '+' if _min_bgt_rev >= gs_target else '')} vs target")
#                 gk3.metric("Profit at That Budget", fmt(_min_bgt_prof, "$"))
#                 gk4.metric("Budget Efficiency",
#                            f"{_min_bgt_rev/max(_min_bgt,EPS):.2f}×",
#                            delta=f"ROI at minimum budget")

#                 if not _target_met:
#                     st.error(
#                         f"⚠️ Target {fmt(gs_target,'$')} is not achievable within the scan range "
#                         f"(max revenue = {fmt(_min_bgt_rev,'$')} at {fmt(_min_bgt,'$')} budget). "
#                         f"Increase the target or check your response curve parameters."
#                     )
#                 else:
#                     st.markdown(
#                         f"<div class='alert-box alert-success' style='font-size:.8rem'>"
#                         f"✅ Target {fmt(gs_target,'$')} achieved at minimum budget "
#                         f"<b>{fmt(_min_bgt,'$')}</b> "
#                         f"({(_min_bgt/_baseline_total_lb-1)*100:+.1f}% vs baseline portfolio)</div>",
#                         unsafe_allow_html=True
#                     )

#                 if PLOTLY:
#                     fig_gs = go.Figure()
#                     # Shaded area
#                     fig_gs.add_trace(go.Scatter(
#                         x=_gs_df["budget"], y=_gs_df["opt_revenue"],
#                         fill='tozeroy', fillcolor="rgba(26,86,219,0.06)",
#                         line=dict(width=0), showlegend=False, hoverinfo='skip'
#                     ))
#                     # Revenue curve
#                     fig_gs.add_trace(go.Scatter(
#                         x=_gs_df["budget"], y=_gs_df["opt_revenue"],
#                         mode='lines+markers',
#                         line=dict(color=PALETTE["accent"], width=3),
#                         marker=dict(size=6, color=PALETTE["accent"],
#                                     line=dict(color="white", width=1.5)),
#                         name="Achievable Revenue",
#                         hovertemplate="Budget: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>"
#                     ))
#                     # Revenue target line
#                     fig_gs.add_hline(
#                         y=gs_target, line_dash="dash",
#                         line_color=PALETTE["accent3"], line_width=2.5,
#                         annotation_text=f"<b>Target: {fmt(gs_target,'$')}</b>",
#                         annotation_font=dict(size=11, color=PALETTE["accent3"]),
#                         annotation_position="top left"
#                     )
#                     # Baseline budget line
#                     fig_gs.add_vline(
#                         x=_baseline_total_lb, line_dash="dot",
#                         line_color=PALETTE["muted"], line_width=1.5,
#                         annotation_text=f"Baseline: {fmt(_baseline_total_lb,'$')}",
#                         annotation_font=dict(size=9, color=PALETTE["muted"]),
#                         annotation_position="bottom right"
#                     )
#                     if _target_met:
#                         fig_gs.add_vline(
#                             x=_min_bgt, line_dash="dash",
#                             line_color=PALETTE["accent2"], line_width=2.5,
#                             annotation_text=f"<b>Min: {fmt(_min_bgt,'$')}</b>",
#                             annotation_font=dict(size=11, color=PALETTE["accent2"]),
#                             annotation_position="top right"
#                         )
#                         # Mark intersection point
#                         fig_gs.add_trace(go.Scatter(
#                             x=[_min_bgt], y=[_min_bgt_rev],
#                             mode='markers',
#                             marker=dict(size=14, color=PALETTE["accent2"],
#                                         symbol='star', line=dict(color='white', width=2)),
#                             name="Minimum budget point",
#                             hovertemplate=(f"Min budget: ${_min_bgt:,.0f}<br>"
#                                            f"Revenue: ${_min_bgt_rev:,.0f}<extra></extra>")
#                         ))

#                     fig_gs.update_layout(
#                         title=dict(text="Revenue vs Budget — Goal-Seek Curve",
#                                    font=dict(size=12, color=PALETTE["text2"])),
#                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
#                         font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
#                         height=400, margin=dict(l=10, r=20, t=45, b=60),
#                         legend=dict(bgcolor="rgba(255,255,255,0.9)",
#                                     bordercolor=PALETTE["border"], borderwidth=1,
#                                     orientation="h", y=-0.18, font=dict(size=10)),
#                         xaxis=dict(title="Total Budget ($)", tickformat="$,.0f",
#                                    gridcolor="#EAECF0", linecolor=PALETTE["border2"],
#                                    tickfont=dict(size=10, color=PALETTE["muted"])),
#                         yaxis=dict(title="Optimized Revenue ($)", tickformat="$,.0f",
#                                    gridcolor="#EAECF0", linecolor=PALETTE["border2"],
#                                    tickfont=dict(size=10, color=PALETTE["muted"]))
#                     )
#                     st.plotly_chart(fig_gs, width='stretch')

#                 # Scan table
#                 st.markdown(f"<div class='section-header'>Budget scan results</div>",
#                             unsafe_allow_html=True)
#                 _gs_disp = _gs_df.copy()
#                 _gs_disp['Meets Target'] = _gs_disp['opt_revenue'].apply(
#                     lambda x: "✅ Yes" if x >= gs_target else "❌ No"
#                 )
#                 _gs_disp['budget']      = _gs_disp['budget'].apply(lambda x: f"${x:,.0f}")
#                 _gs_disp['opt_revenue'] = _gs_disp['opt_revenue'].apply(lambda x: f"${x:,.0f}")
#                 _gs_disp['opt_profit']  = _gs_disp['opt_profit'].apply(lambda x: f"${x:,.0f}")
#                 _gs_disp.columns = ['Budget', 'Revenue', 'Profit', 'Meets Target']
#                 st.dataframe(_gs_disp, width='stretch', hide_index=True)

#                 # Download optimal channel allocation at minimum budget
#                 if _target_met:
#                     _gs_opt_row = _gs_rows[_meets.index[0]]
#                     _gs_spends  = _gs_opt_row["spends"]
#                     _gs_csv_df  = df.copy()
#                     _gs_csv_df['opt_spend'] = _gs_spends
#                     _gs_csv_df['opt_revenue'] = [
#                         revenue_from_spend(_gs_spends[i], _gs_csv_df.iloc[i],
#                                            use_hill_global, use_log_global)
#                         for i in range(len(_gs_csv_df))
#                     ]
#                     _gs_csv_df['opt_profit'] = _gs_csv_df['opt_revenue'] - _gs_csv_df['opt_spend']
#                     _gs_csv_buf = io.StringIO()
#                     _gs_csv_df[['channel', 'opt_spend', 'opt_revenue', 'opt_profit']].to_csv(
#                         _gs_csv_buf, index=False
#                     )
#                     st.download_button(
#                         f"⬇️ Download optimal allocation at {fmt(_min_bgt,'$')} (CSV)",
#                         data=_gs_csv_buf.getvalue(),
#                         file_name="mmm_launch_goalseek.csv",
#                         mime="text/csv"
#                     )

#                 _gs_prompt = (
#                     f"GOAL-SEEK LAUNCH ANALYSIS\n\n"
#                     f"Revenue target: {fmt(gs_target,'$')} | "
#                     f"Target met: {'Yes' if _target_met else 'No'}\n"
#                     f"Minimum budget required: {fmt(_min_bgt,'$')} "
#                     f"({(_min_bgt/_baseline_total_lb-1)*100:+.1f}% vs baseline)\n"
#                     f"Revenue at minimum budget: {fmt(_min_bgt_rev,'$')}\n"
#                     f"Profit at minimum budget: {fmt(_min_bgt_prof,'$')}\n\n"
#                     f"BASELINE PORTFOLIO: Revenue={fmt(tot_rev,'$')}, "
#                     f"Profit={fmt(tot_profit,'$')}, ROI={avg_roas:.2f}x\n\n"
#                     f"Channel descriptions:\n"
#                     + "\n".join([f"  {k}: {v}" for k, v in ch_desc_map.items()])
#                     + "\n\nThis is a GOAL-SEEK analysis. Analyse: "
#                     f"(1) is the minimum budget of {fmt(_min_bgt,'$')} realistic and "
#                     f"how does it compare to industry norms for this launch type; "
#                     f"(2) what is the revenue-to-budget efficiency at the minimum budget point "
#                     f"vs the efficient frontier; "
#                     f"(3) what are the risks of operating at minimum budget "
#                     f"(near saturation vs headroom); "
#                     f"(4) would a 10-20% buffer above minimum be warranted given pharma launch uncertainty?"
#                 )
#                 render_ai_button(_gs_prompt, "ai_gs_btn", "🤖 Interpret Goal-Seek Results")

#             else:
#                 _launch_empty_state("🔍", "Set a revenue target and click Find Minimum Budget",
#                                     "The solver scans budget levels to find the minimum spend needed to hit your target.")



# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — WHAT-IF SCENARIOS                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

with tab_wi:
    st.markdown(f"<div class='section-header'>Interactive What-If Scenario Builder</div>", unsafe_allow_html=True)

    wi_mode_col, wi_chart_col = st.columns([.35, .65])

    with wi_mode_col:
        wi_mode = st.radio("Scenario type", [
            "📊 Uniform Budget Shift",
            "🎯 Channel-by-Channel",
           # "📊 Sensitivity Analysis",
           # "🔀 Mix Comparison",
            "🔍 Goal-Seek Analysis",
        ], label_visibility="collapsed")

    with wi_chart_col:
        # ── UNIFORM BUDGET SHIFT ──────────────────────────────────────────────
        if "Uniform" in wi_mode:
            pct_shift = st.slider("Budget shift (%)", -50, 100, 0, 5,
                                   format="%d%%",
                                   help="Apply a uniform % change across all non-locked channels")

            df_wi = df.copy()
            shift_factor = 1.0 + pct_shift / 100.0
            new_spends = []
            for _, row in df_wi.iterrows():
                base = float(row['total_spend'])
                if int(row.get('lock_spend', 0)) == 1:
                    new_spends.append(base)
                else:
                    lo_ = float(row.get('lower_bound_pct', 0)) * base
                    hi_ = float(row.get('upper_bound_pct', 10)) * base
                    new_spends.append(np.clip(base * shift_factor, lo_, hi_))

            df_wi['wi_spend'] = new_spends
            df_wi['wi_revenue'] = [revenue_from_spend(df_wi.iloc[i]['wi_spend'], df_wi.iloc[i], use_hill_global, use_log_global)
                                    for i in range(len(df_wi))]
            df_wi['wi_profit'] = df_wi['wi_revenue'] - df_wi['wi_spend']

            wi_rev = df_wi['wi_revenue'].sum()
            wi_prof = df_wi['wi_profit'].sum()
            wi_sp = df_wi['wi_spend'].sum()

            wk1, wk2, wk3 = st.columns(3)
            wk1.metric("Revenue", fmt(wi_rev, "$"),
                       delta=f"{fmt(wi_rev - tot_rev, '$', '+' if wi_rev >= tot_rev else '')}")
            wk2.metric("Profit", fmt(wi_prof, "$"),
                       delta=f"{fmt(wi_prof - tot_profit, '$', '+' if wi_prof >= tot_profit else '')}")
            wk3.metric("ROI", f"{wi_rev/max(wi_sp,EPS):.2f}×",
                       delta=f"{wi_rev/max(wi_sp,EPS) - avg_roas:+.2f}×")

            if PLOTLY:
                fig_wi = go.Figure()
                fig_wi.add_trace(go.Bar(
                    name='Baseline', x=df_wi['channel'], y=df_wi['baseline_profit'],
                    marker_color=f"rgba(88,166,255,0.5)",
                    marker_line_color=PALETTE["accent"], marker_line_width=1.5
                ))
                fig_wi.add_trace(go.Bar(
                    name='What-If', x=df_wi['channel'], y=df_wi['wi_profit'],
                    marker_color=[PALETTE["accent2"] if v >= bv else PALETTE["accent3"]
                                  for v, bv in zip(df_wi['wi_profit'], df_wi['baseline_profit'])],
                    marker_line_width=0
                ))
                plotly_dark_layout(fig_wi, height=300, title="Profit · Baseline vs What-If")
                fig_wi.update_layout(barmode='group', margin=dict(b=60),
                                      xaxis=dict(tickangle=-30))
                st.plotly_chart(fig_wi, width='stretch')

        # ── CHANNEL-BY-CHANNEL ────────────────────────────────────────────────
        elif "Channel-by-Channel" in wi_mode:
            wi_spends = {}
            slider_cols = st.columns(2)
            for i, (_, row) in enumerate(df.iterrows()):
                ch = row['channel']
                base = float(row['total_spend'])
                if int(row.get('lock_spend', 0)) == 1:
                    wi_spends[ch] = base
                    with slider_cols[i % 2]:
                        st.markdown(f"<span class='chip chip-gold'>🔒 {ch}: {fmt(base, '$')}</span>",
                                    unsafe_allow_html=True)
                else:
                    lo_ = float(row.get('lower_bound_pct', 0.5)) * base
                    hi_ = float(row.get('upper_bound_pct', 1.5)) * base
                    with slider_cols[i % 2]:
                        wi_spends[ch] = st.slider(
                            f"{ch} ($K)", lo_ / 1000, hi_ / 1000, base / 1000,
                            step=max((hi_ - lo_) / 100 / 1000, 0.1),
                            format="%.0f K",
                            key=f"wi_sl_{ch}"
                        ) * 1000

            df_wi = df.copy()
            df_wi['wi_spend'] = [wi_spends[ch] for ch in df_wi['channel']]
            df_wi['wi_revenue'] = [revenue_from_spend(df_wi.iloc[i]['wi_spend'], df_wi.iloc[i], use_hill_global, use_log_global)
                                    for i in range(len(df_wi))]
            df_wi['wi_profit'] = df_wi['wi_revenue'] - df_wi['wi_spend']

            wi_rev = df_wi['wi_revenue'].sum()
            wi_prof = df_wi['wi_profit'].sum()
            wi_sp = df_wi['wi_spend'].sum()

            wk1, wk2, wk3, wk4 = st.columns(4)
            wk1.metric("Total Spend", fmt(wi_sp, "$"),
                       delta=f"{fmt(wi_sp - tot_spend, '$', '+' if wi_sp >= tot_spend else '')}")
            wk2.metric("Revenue", fmt(wi_rev, "$"),
                       delta=f"{fmt(wi_rev - tot_rev, '$', '+' if wi_rev >= tot_rev else '')}")
            wk3.metric("Profit", fmt(wi_prof, "$"),
                       delta=f"{fmt(wi_prof - tot_profit, '$', '+' if wi_prof >= tot_profit else '')}")
            wk4.metric("ROI", f"{wi_rev/max(wi_sp,EPS):.2f}×",
                       delta=f"{wi_rev/max(wi_sp,EPS) - avg_roas:+.2f}×")

            if PLOTLY:
                fig_ch = make_subplots(rows=1, cols=2,
                                        subplot_titles=["Revenue Breakdown", "Profit Comparison"],
                                        horizontal_spacing=0.08)
                fig_ch.add_trace(go.Bar(
                    name='Baseline', x=df_wi['channel'], y=df_wi['baseline_revenue'],
                    marker_color=f"rgba(88,166,255,0.5)", marker_line_color=PALETTE["accent"],
                    marker_line_width=1.5
                ), row=1, col=1)
                fig_ch.add_trace(go.Bar(
                    name='What-If', x=df_wi['channel'], y=df_wi['wi_revenue'],
                    marker_color=[PALETTE["accent2"] if v >= bv else PALETTE["accent3"]
                                  for v, bv in zip(df_wi['wi_revenue'], df_wi['baseline_revenue'])]
                ), row=1, col=1)
                for i, (_, row) in enumerate(df_wi.iterrows()):
                    col_c = channel_color(i)
                    fig_ch.add_trace(go.Bar(
                        name=row['channel'], x=[row['channel']],
                        y=[row['wi_profit'] - row['baseline_profit']],
                        marker_color=PALETTE["accent2"] if row['wi_profit'] >= row['baseline_profit'] else PALETTE["accent3"],
                        showlegend=False
                    ), row=1, col=2)
                fig_ch.update_layout(
                    barmode='group', paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor=PALETTE["surface"],
                    font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                    height=320, margin=dict(l=5, r=5, t=35, b=60),
                    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.3),
                    xaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                               tickangle=-30, tickfont=dict(size=9, color=PALETTE["muted"])),
                    xaxis2=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                                tickangle=-30, tickfont=dict(size=9, color=PALETTE["muted"])),
                    yaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"]),
                    yaxis2=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"])
                )
                st.plotly_chart(fig_ch, width='stretch')

        # ── BUDGET FRONTIER ───────────────────────────────────────────────────
        elif "Sensitivity" in wi_mode:
            st.markdown(
                f"<span class='chip chip-blue'>Channel Sensitivity — profit impact of ±10/20/30% spend shifts per channel</span>",
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

            # Shifts to test
            shifts_pct = [-30, -20, -10, 10, 20, 30]
            shift_labels = [f"{s:+d}%" for s in shifts_pct]

            # Compute sensitivity: for each channel, shift its spend by each %
            # while keeping ALL other channels at baseline
            sens_rows = []
            for _, row in df.iterrows():
                ch   = row['channel']
                desc = ch_desc_map.get(ch, ch)
                base_sp  = float(row['total_spend'])
                base_rev = revenue_from_spend(base_sp, row, use_hill_global, use_log_global)
                base_pf  = base_rev - base_sp

                ch_row = {'Channel': ch, 'Description': desc, 'Baseline Profit': base_pf}
                for s in shifts_pct:
                    new_sp  = base_sp * (1 + s / 100)
                    new_rev = revenue_from_spend(new_sp, row, use_hill_global, use_log_global)
                    new_pf  = new_rev - new_sp
                    ch_row[f"{s:+d}%"] = new_pf - base_pf   # delta profit
                sens_rows.append(ch_row)

            sens_df = pd.DataFrame(sens_rows)

            if PLOTLY:
                # ── Heatmap-style: channels × shifts, coloured by Δ profit ──────
                delta_cols = [f"{s:+d}%" for s in shifts_pct]
                z_vals  = sens_df[delta_cols].values
                ch_labs = [f"{r['Channel']}" for _, r in sens_df.iterrows()]

                # Normalise colour scale around 0
                _abs_max = max(abs(z_vals.min()), abs(z_vals.max()), 1)

                fig_heat = go.Figure(go.Heatmap(
                    z=z_vals,
                    x=delta_cols,
                    y=ch_labs,
                    colorscale=[
                        [0.0,  PALETTE["accent3"]],   # deep red  → biggest loss
                        [0.45, "#FEE2E2"],             # light red
                        [0.5,  "#F9FAFB"],             # neutral
                        [0.55, "#D1FAE5"],             # light green
                        [1.0,  PALETTE["accent2"]],   # deep green → biggest gain
                    ],
                    zmid=0,
                    zmin=-_abs_max, zmax=_abs_max,
                    text=[[fmt(v, "$", "+" if v >= 0 else "") for v in row]
                          for row in z_vals],
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    colorbar=dict(
                        title="Δ Profit ($)",
                        tickformat="$,.0f",
                        len=0.8
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Spend shift: %{x}<br>"
                        "Δ Profit: %{text}<extra></extra>"
                    )
                ))
                fig_heat.update_layout(
                    title=dict(
                        text="Profit Sensitivity — Δ Profit from shifting each channel's spend independently",
                        font=dict(size=11, color=PALETTE["text2"])
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                    font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                    height=max(280, len(sens_df) * 48 + 80),
                    margin=dict(l=10, r=10, t=50, b=40),
                    xaxis=dict(
                        title="Spend Shift",
                        side="bottom",
                        tickfont=dict(size=11, color=PALETTE["text2"])
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10, color=PALETTE["text2"]),
                        autorange="reversed"
                    )
                )
                st.plotly_chart(fig_heat, width='stretch')

                # ── Bar chart: most sensitive channels ───────────────────────────
                # Sensitivity = total absolute profit swing across all shifts
                sens_df['_total_swing'] = sens_df[delta_cols].abs().sum(axis=1)
                sens_sorted = sens_df.sort_values('_total_swing', ascending=True)

                fig_swing = go.Figure()
                fig_swing.add_trace(go.Bar(
                    y=sens_sorted['Channel'],
                    x=sens_sorted['_total_swing'],
                    orientation='h',
                    marker_color=[channel_color(list(df['channel']).index(c))
                                  for c in sens_sorted['Channel']],
                    text=[fmt(v, "$") for v in sens_sorted['_total_swing']],
                    textposition='outside',
                    textfont=dict(size=9),
                    hovertemplate="<b>%{y}</b><br>Total profit swing: $%{x:,.0f}<extra></extra>"
                ))
                fig_swing.update_layout(
                    title=dict(
                        text="Channel Sensitivity Ranking  (total Δ profit across all shifts)",
                        font=dict(size=11, color=PALETTE["text2"])
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                    font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                    height=max(220, len(sens_df) * 38 + 60),
                    margin=dict(l=10, r=60, t=40, b=10),
                    showlegend=False,
                    xaxis=dict(title="Total |Δ Profit| ($)", tickformat="$,.0f",
                               gridcolor="#EAECF0",
                               tickfont=dict(size=9, color=PALETTE["muted"])),
                    yaxis=dict(gridcolor="#EAECF0", linecolor=PALETTE["border2"],
                               tickfont=dict(size=9, color=PALETTE["muted"]))
                )
                st.plotly_chart(fig_swing, width='stretch')

                # ── Insight callout ───────────────────────────────────────────────
                most_sensitive = sens_sorted.iloc[-1]['Channel']
                least_sensitive = sens_sorted.iloc[0]['Channel']
                _desc_most = ch_desc_map.get(most_sensitive, most_sensitive)
                _desc_least = ch_desc_map.get(least_sensitive, least_sensitive)
                st.markdown(
                    f"<div class='alert-box alert-info'>"
                    f"📊 <b>Most sensitive:</b> {most_sensitive} ({_desc_most}) — "
                    f"largest profit swing from spend changes. "
                    f"<b>Least sensitive:</b> {least_sensitive} ({_desc_least}) — "
                    f"most stable regardless of investment level. "
                    f"High sensitivity = high leverage but also high risk.</div>",
                    unsafe_allow_html=True
                )

                # AI interpretation button
                _sens_ctx = "\n".join([
                    f"  {r['Channel']} ({ch_desc_map.get(r['Channel'],r['Channel'])}): "
                    + ", ".join([f"{c}: {fmt(r[c],'$','+' if r[c]>=0 else '')}" for c in delta_cols])
                    for _, r in sens_df.iterrows()
                ])
                _sens_prompt = (
                    f"CHANNEL SENSITIVITY ANALYSIS\n\n"
                    f"Shows Δ profit when each channel's spend is shifted independently "
                    f"(all others held at baseline):\n{_sens_ctx}\n\n"
                    f"Baseline portfolio: spend={fmt(tot_spend,'$')}, "
                    f"revenue={fmt(tot_rev,'$')}, profit={fmt(tot_profit,'$')}\n\n"
                    f"Channel descriptions:\n"
                    + "\n".join([f"  {k}: {v}" for k,v in ch_desc_map.items()])
                    + "\n\nAnalyse: (1) Which channels are highest leverage — "
                    f"most profit gain per dollar? (2) Which are most asymmetric — "
                    f"bigger downside than upside? (3) Which are safe to test-reduce? "
                    f"(4) Prioritised reallocation recommendation based on sensitivity."
                )
                render_ai_button(_sens_prompt, "ai_sens_btn",
                                  "🤖 Interpret Sensitivity Analysis")

        # ── MIX COMPARISON ────────────────────────────────────────────────────
        elif "Comparison" in wi_mode:
            st.markdown(f"<span class='chip chip-purple'>Compare up to 3 custom budget mixes side by side</span>",
                        unsafe_allow_html=True)

            n_scenarios = st.number_input("Number of scenarios", 2, 3, 2)
            scenario_names = [f"Scenario {i+1}" for i in range(n_scenarios)]
            scenario_colors = [PALETTE["accent"], PALETTE["accent2"], PALETTE["accent4"]]

            scenarios = {}
            sc_cols = st.columns(n_scenarios)
            for si, sc_col in enumerate(sc_cols):
                with sc_col:
                    st.markdown(f"<span class='chip' style='background:{scenario_colors[si]}22;color:{scenario_colors[si]};border-color:{scenario_colors[si]}44'>{scenario_names[si]}</span>",
                                unsafe_allow_html=True)
                    sc_spends = {}
                    for _, row in df.iterrows():
                        ch = row['channel']
                        base = float(row['total_spend'])
                        if int(row.get('lock_spend', 0)) == 1:
                            sc_spends[ch] = base
                        else:
                            sc_spends[ch] = st.number_input(
                                ch, value=base, min_value=0.0, step=max(base * 0.05, 1000.0),
                                format="%.0f", key=f"sc{si}_{ch}", label_visibility="visible"
                            )
                    scenarios[scenario_names[si]] = sc_spends

            # Compute scenario results
            sc_results = {}
            for sc_name, sc_spends in scenarios.items():
                df_sc = df.copy()
                df_sc['sc_spend'] = [sc_spends.get(ch, float(df_sc[df_sc['channel']==ch]['total_spend'].iloc[0]))
                                      for ch in df_sc['channel']]
                df_sc['sc_revenue'] = [revenue_from_spend(df_sc.iloc[i]['sc_spend'], df_sc.iloc[i], use_hill_global, use_log_global)
                                        for i in range(len(df_sc))]
                df_sc['sc_profit'] = df_sc['sc_revenue'] - df_sc['sc_spend']
                sc_results[sc_name] = df_sc

            if PLOTLY:
                fig_cmp = go.Figure()
                metrics_cmp = ['Total Spend', 'Total Revenue', 'Total Profit']
                # Add baseline
                baseline_vals = [tot_spend, tot_rev, tot_profit]
                fig_cmp.add_trace(go.Bar(
                    name='Baseline', x=metrics_cmp, y=baseline_vals,
                    marker_color=f"rgba(139,148,158,0.5)",
                    marker_line_color=PALETTE["muted"], marker_line_width=1.5
                ))
                for si, (sc_name, df_sc) in enumerate(sc_results.items()):
                    vals = [df_sc['sc_spend'].sum(), df_sc['sc_revenue'].sum(), df_sc['sc_profit'].sum()]
                    fig_cmp.add_trace(go.Bar(
                        name=sc_name, x=metrics_cmp, y=vals,
                        marker_color=hex_to_rgba(scenario_colors[si], 0.55),
                        marker_line_color=scenario_colors[si], marker_line_width=1.5,
                        text=[fmt(v, "$") for v in vals],
                        textposition='outside', textfont=dict(size=9)
                    ))
                plotly_dark_layout(fig_cmp, height=370, title="Scenario Comparison · Key Metrics")
                fig_cmp.update_layout(barmode='group',
                                       xaxis=dict(tickfont=dict(size=11)),
                                       legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_cmp, width='stretch')

            # Summary table
            summary_rows = [{'Scenario': 'Baseline',
                              'Total Spend': fmt(tot_spend, "$"),
                              'Total Revenue': fmt(tot_rev, "$"),
                              'Total Profit': fmt(tot_profit, "$"),
                              'ROI': f"{avg_roas:.2f}×"}]
            for sc_name, df_sc in sc_results.items():
                sp_ = df_sc['sc_spend'].sum()
                rv_ = df_sc['sc_revenue'].sum()
                pf_ = df_sc['sc_profit'].sum()
                summary_rows.append({
                    'Scenario': sc_name,
                    'Total Spend': fmt(sp_, "$"),
                    'Total Revenue': fmt(rv_, "$"),
                    'Total Profit': fmt(pf_, "$"),
                    'ROI': f"{rv_/max(sp_, EPS):.2f}×"
                })
            st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)


        # ── GOAL-SEEK ANALYSIS ────────────────────────────────────────────────
        elif "Goal-Seek" in wi_mode:
            st.markdown(
                f"<div class='alert-box alert-info' style='margin-bottom:.75rem;font-size:.8rem'>"
                f"<b>How Goal-Seek works with your model:</b> "
                f"Uses the uploaded model coefficients (alpha, Adj_Factor, cost_per_mention) to "
                f"back-solve spend questions. Choose a target metric and the tool finds the exact "
                f"spend required to hit it — channel-by-channel or portfolio-wide.</div>",
                unsafe_allow_html=True
            )

            gs_wi_mode = st.radio(
                "Goal-seek type",
                [
                    "🎯 Single Channel — hit a revenue target by adjusting one channel",
                    "💰 Portfolio — minimum budget to hit a total revenue target",
                    "📐 Break-Even — find each channel's break-even spend",
                ],
                horizontal=False,
                label_visibility="collapsed",
                key="gs_wi_submode"
            )

            # ── SINGLE CHANNEL GOAL-SEEK ──────────────────────────────────────
            if "Single Channel" in gs_wi_mode:
                gs_c1, gs_c2 = st.columns([0.38, 0.62])

                with gs_c1:
                    gs_channel = st.selectbox(
                        "Channel to adjust",
                        df['channel'].tolist(),
                        key="gs_wi_channel",
                        help="All other channels stay at baseline spend"
                    )
                    gs_target_type = st.radio(
                        "Target metric",
                        ["Total Portfolio Revenue", "Total Portfolio Profit",
                         "This Channel Revenue", "This Channel Profit"],
                        key="gs_wi_target_type"
                    )
                    _ch_row    = df[df['channel'] == gs_channel].iloc[0]
                    _base_sp   = float(_ch_row['total_spend'])
                    _base_rev  = float(_ch_row['baseline_revenue'])
                    _base_prof = float(_ch_row['baseline_profit'])

                    # Sensible default targets
                    if "Portfolio Revenue" in gs_target_type:
                        _default_tgt = tot_rev * 1.10
                        _tgt_label   = "Target portfolio revenue ($)"
                    elif "Portfolio Profit" in gs_target_type:
                        _default_tgt = tot_profit * 1.10
                        _tgt_label   = "Target portfolio profit ($)"
                    elif "Channel Revenue" in gs_target_type:
                        _default_tgt = _base_rev * 1.20
                        _tgt_label   = f"Target revenue from {gs_channel} ($)"
                    else:
                        _default_tgt = max(_base_prof * 1.20, _base_rev * 0.05)
                        _tgt_label   = f"Target profit from {gs_channel} ($)"

                    gs_wi_target = st.number_input(
                        _tgt_label,
                        value=float(round(_default_tgt / 1000) * 1000),
                        min_value=0.0, step=100000.0, format="%.0f",
                        key="gs_wi_target_val"
                    )

                    # Spend search bounds for this channel
                    _lo_sp = _base_sp * float(_ch_row.get('lower_bound_pct', 0.1))
                    _hi_sp = _base_sp * float(_ch_row.get('upper_bound_pct', 5.0))
                    _hi_sp = max(_hi_sp, _base_sp * 5.0)  # allow up to 5× for goal-seek

                    # Other channels contribution (fixed at baseline)
                    _others_rev  = sum(
                        revenue_from_spend(float(df.iloc[i]['total_spend']), df.iloc[i],
                                           use_hill_global, use_log_global)
                        for i in range(len(df)) if df.iloc[i]['channel'] != gs_channel
                    )
                    _others_sp   = tot_spend - _base_sp

                with gs_c2:
                    # Binary search for the required spend
                    def _gs_obj_val(spend_val):
                        rev_ch = revenue_from_spend(spend_val, _ch_row, use_hill_global, use_log_global)
                        if "Portfolio Revenue" in gs_target_type:
                            return _others_rev + rev_ch
                        elif "Portfolio Profit" in gs_target_type:
                            return _others_rev + rev_ch - _others_sp - spend_val
                        elif "Channel Revenue" in gs_target_type:
                            return rev_ch
                        else:  # Channel Profit
                            return rev_ch - spend_val

                    # Check feasibility
                    _val_at_lo = _gs_obj_val(_lo_sp)
                    _val_at_hi = _gs_obj_val(_hi_sp)
                    _target_feasible = _val_at_hi >= gs_wi_target

                    if _target_feasible:
                        # Binary search
                        _lo, _hi = _lo_sp, _hi_sp
                        for _ in range(60):
                            _mid = (_lo + _hi) / 2
                            if _gs_obj_val(_mid) < gs_wi_target:
                                _lo = _mid
                            else:
                                _hi = _mid
                        gs_solved_spend = (_lo + _hi) / 2
                        gs_solved_val   = _gs_obj_val(gs_solved_spend)
                        gs_solved_rev_ch = revenue_from_spend(gs_solved_spend, _ch_row,
                                                               use_hill_global, use_log_global)
                        gs_solved_prof_ch = gs_solved_rev_ch - gs_solved_spend
                        gs_delta_spend    = gs_solved_spend - _base_sp
                        gs_delta_pct      = gs_delta_spend / _base_sp * 100 if _base_sp > 0 else 0

                        # Summary KPIs
                        gk1, gk2, gk3 = st.columns(3)
                        gk1.metric(
                            f"Required {gs_channel} Spend",
                            fmt(gs_solved_spend, "$"),
                            delta=f"{fmt(gs_delta_spend,'$','+' if gs_delta_spend>=0 else '')} "
                                  f"({gs_delta_pct:+.1f}% vs baseline)"
                        )
                        gk2.metric(
                            "Channel Revenue at Target",
                            fmt(gs_solved_rev_ch, "$"),
                            delta=f"{fmt(gs_solved_rev_ch - _base_rev,'$','+' if gs_solved_rev_ch>=_base_rev else '')}"
                        )
                        gk3.metric(
                            "Channel Profit at Target",
                            fmt(gs_solved_prof_ch, "$"),
                            delta=f"{fmt(gs_solved_prof_ch - _base_prof,'$','+' if gs_solved_prof_ch>=_base_prof else '')}"
                        )

                        st.markdown(
                            f"<div class='alert-box alert-success' style='font-size:.8rem'>"
                            f"✅ To achieve <b>{gs_target_type}</b> of <b>{fmt(gs_wi_target,'$')}</b>, "
                            f"<b>{gs_channel}</b> spend needs to move from "
                            f"<b>{fmt(_base_sp,'$')}</b> → <b>{fmt(gs_solved_spend,'$')}</b> "
                            f"(<b>{gs_delta_pct:+.1f}%</b>). "
                            f"All other channels remain at baseline.</div>",
                            unsafe_allow_html=True
                        )

                        # Response curve showing baseline → target
                        if PLOTLY:
                            _n_pts    = 400
                            _sp_range = np.linspace(max(_lo_sp * 0.5, 1), _hi_sp, _n_pts)
                            _obj_vals = [_gs_obj_val(s) for s in _sp_range]

                            fig_gs_wi = go.Figure()
                            fig_gs_wi.add_trace(go.Scatter(
                                x=_sp_range, y=_obj_vals,
                                mode='lines', name=gs_target_type,
                                line=dict(color=PALETTE["accent"], width=3),
                                hovertemplate="Spend: $%{x:,.0f}<br>Value: $%{y:,.0f}<extra></extra>"
                            ))
                            # Target line
                            fig_gs_wi.add_hline(
                                y=gs_wi_target, line_dash="dash",
                                line_color=PALETTE["accent3"], line_width=2,
                                annotation_text=f"Target: {fmt(gs_wi_target,'$')}",
                                annotation_font=dict(size=10, color=PALETTE["accent3"]),
                                annotation_position="top left"
                            )
                            # Baseline point
                            fig_gs_wi.add_trace(go.Scatter(
                                x=[_base_sp], y=[_gs_obj_val(_base_sp)],
                                mode='markers',
                                marker=dict(size=12, color=PALETTE["accent3"],
                                            symbol='circle', line=dict(color='white', width=2)),
                                name='Current Spend',
                                hovertemplate=f"Current: ${_base_sp:,.0f}<br>Value: ${_gs_obj_val(_base_sp):,.0f}<extra></extra>"
                            ))
                            # Solution point
                            fig_gs_wi.add_trace(go.Scatter(
                                x=[gs_solved_spend], y=[gs_solved_val],
                                mode='markers',
                                marker=dict(size=14, color=PALETTE["accent2"],
                                            symbol='star', line=dict(color='white', width=2)),
                                name='Goal-Seek Solution',
                                hovertemplate=f"Solution: ${gs_solved_spend:,.0f}<br>Value: ${gs_solved_val:,.0f}<extra></extra>"
                            ))
                            fig_gs_wi.add_vline(
                                x=gs_solved_spend, line_dash="dot",
                                line_color=PALETTE["accent2"], line_width=1.5
                            )
                            fig_gs_wi.update_layout(
                                title=dict(
                                    text=f"{gs_channel} — {gs_target_type} vs Spend  (Goal-Seek)",
                                    font=dict(size=11, color=PALETTE["text2"])
                                ),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                                font=dict(family="Inter, system-ui, sans-serif",
                                          color=PALETTE["text2"]),
                                height=360,
                                margin=dict(l=10, r=20, t=42, b=50),
                                legend=dict(bgcolor="rgba(255,255,255,0.9)",
                                            bordercolor=PALETTE["border"], borderwidth=1,
                                            orientation="h", y=-0.16, font=dict(size=10)),
                                xaxis=dict(title=f"{gs_channel} Spend ($)", tickformat="$,.0f",
                                           gridcolor="#EAECF0",
                                           tickfont=dict(size=10, color=PALETTE["muted"])),
                                yaxis=dict(title=gs_target_type + " ($)", tickformat="$,.0f",
                                           gridcolor="#EAECF0",
                                           tickfont=dict(size=10, color=PALETTE["muted"]))
                            )
                            st.plotly_chart(fig_gs_wi, width='stretch')

                        # Detailed result table
                        _gs_detail = pd.DataFrame([
                            {'Metric': 'Channel',                     'Baseline': gs_channel,          'Goal-Seek': gs_channel},
                            {'Metric': 'Spend',                       'Baseline': fmt(_base_sp,'$'),   'Goal-Seek': fmt(gs_solved_spend,'$')},
                            {'Metric': 'Δ Spend',                     'Baseline': '—',                 'Goal-Seek': fmt(gs_delta_spend,'$','+' if gs_delta_spend>=0 else '')},
                            {'Metric': 'Δ Spend %',                   'Baseline': '—',                 'Goal-Seek': f"{gs_delta_pct:+.1f}%"},
                            {'Metric': 'Channel Revenue',             'Baseline': fmt(_base_rev,'$'),  'Goal-Seek': fmt(gs_solved_rev_ch,'$')},
                            {'Metric': 'Channel Profit',              'Baseline': fmt(_base_prof,'$'), 'Goal-Seek': fmt(gs_solved_prof_ch,'$')},
                            {'Metric': 'Channel mROI at solution',    'Baseline': f"{float(_ch_row['baseline_mroi']):.2f}×", 'Goal-Seek': f"{mroi(gs_solved_spend, _ch_row, use_hill_global, use_log_global):.2f}×"},
                            {'Metric': gs_target_type + ' (target)',  'Baseline': fmt(_gs_obj_val(_base_sp),'$'), 'Goal-Seek': fmt(gs_solved_val,'$')},
                        ])
                        st.dataframe(_gs_detail, width='stretch', hide_index=True)

                    else:
                        st.error(
                            f"❌ Target {fmt(gs_wi_target,'$')} is not achievable by adjusting "
                            f"{gs_channel} alone within its bounds "
                            f"({fmt(_lo_sp,'$')} – {fmt(_hi_sp,'$')}). "
                            f"Maximum achievable value is {fmt(_val_at_hi,'$')}. "
                            f"Try reducing the target or selecting a different channel."
                        )

                    # AI interpretation
                    _gs_wi_prompt = (
                        f"GOAL-SEEK ANALYSIS — Single Channel\n\n"
                        f"Channel adjusted: {gs_channel} ({ch_desc_map.get(gs_channel,gs_channel)})\n"
                        f"Target metric: {gs_target_type}\n"
                        f"Target value: {fmt(gs_wi_target,'$')}\n"
                        f"Target feasible: {'Yes' if _target_feasible else 'No'}\n"
                        + (f"Required spend: {fmt(gs_solved_spend,'$')} ({gs_delta_pct:+.1f}% vs baseline {fmt(_base_sp,'$')})\n"
                           f"Channel mROI at solution: {mroi(gs_solved_spend, _ch_row, use_hill_global, use_log_global):.2f}x\n"
                           if _target_feasible else f"Max achievable: {fmt(_val_at_hi,'$')}\n")
                        + f"\nChannel model params: alpha={float(_ch_row['alpha']):.3f}, "
                        f"coef={float(_ch_row['coefficient']):.5f}, "
                        f"AdjFactor={float(_ch_row['Adj_Factor']):.3f}, "
                        f"curve={_ch_row.get('type_transformation','power')}\n\n"
                        f"BASELINE PORTFOLIO: revenue={fmt(tot_rev,'$')}, "
                        f"profit={fmt(tot_profit,'$')}, ROI={avg_roas:.2f}x\n\n"
                        f"Analyse: (1) Is the required spend increase realistic for this channel type in pharma? "
                        f"(2) What does the mROI at the solution point tell us about investment efficiency? "
                        f"(3) What are the risks of relying on a single channel to hit the target? "
                        f"(4) Is there a better multi-channel approach to the same target?"
                    )
                    render_ai_button(_gs_wi_prompt, "ai_gs_wi_single", "🤖 Interpret Goal-Seek")

            # ── PORTFOLIO GOAL-SEEK ───────────────────────────────────────────
            elif "Portfolio" in gs_wi_mode:
                # ═══════════════════════════════════════════════════════════════
                # PORTFOLIO GOAL-SEEK
                # Finds the minimum total budget needed to hit a revenue/profit
                # target, with realistic per-tactic spend bounds enforced so the
                # optimizer doesn't produce extreme allocations (e.g. 0% or 3000%).
                # ═══════════════════════════════════════════════════════════════
                pg_c1, pg_c2 = st.columns([0.35, 0.65])

                with pg_c1:
                    # ── Target metric ─────────────────────────────────────────
                    pg_target_type = st.radio(
                        "Target metric",
                        ["Total Revenue", "Total Profit"],
                        key="pg_wi_target_type"
                    )
                    _is_rev_pg   = "Revenue" in pg_target_type
                    _base_val_pg = tot_rev if _is_rev_pg else tot_profit
                    _max_val_pg  = _base_val_pg * 3.0
                    _step_pg     = max(round(_base_val_pg * 0.05 / 100000) * 100000, 100000.0)
                    _default_pg  = _base_val_pg * 1.15

                    st.markdown(
                        f"<div class='alert-box alert-info' style='font-size:.75rem;margin-bottom:.4rem'>"
                        f"Baseline {pg_target_type}: <b>{fmt(_base_val_pg,'$')}</b> · "
                        f"Realistic max (~3×): <b>{fmt(_max_val_pg,'$')}</b></div>",
                        unsafe_allow_html=True
                    )
                    pg_target = st.number_input(
                        f"Target {pg_target_type} ($)",
                        value=float(round(_default_pg / _step_pg) * _step_pg),
                        min_value=float(_step_pg),
                        max_value=float(_max_val_pg * 1.5),
                        step=float(_step_pg),
                        format="%.0f",
                        key="pg_wi_target_val",
                        help=f"Baseline is {fmt(_base_val_pg,'$')}. Max achievable ~{fmt(_max_val_pg,'$')}."
                    )
                    if pg_target > _max_val_pg:
                        st.warning(
                            f"⚠️ Target {fmt(pg_target,'$')} may not be achievable — "
                            f"realistic max is ~{fmt(_max_val_pg,'$')}."
                        )

                    # ── Scan settings ─────────────────────────────────────────
                    pg_n_scan = st.slider(
                        "Scan resolution", 10, 50, 20, 5,
                        key="pg_wi_scan_pts",
                        help="Number of budget levels to scan. More = slower but more precise."
                    )
                    pg_method = st.selectbox(
                        "Solver",
                        ["SLSQP (fast)", "GEKKO / IPOPT (recommended)"],
                        key="pg_wi_solver"
                    )

                    # ── Per-tactic realistic spend bounds ─────────────────────
                    st.markdown(
                        "<div style='font-size:.7rem;font-weight:700;text-transform:uppercase;"
                        "letter-spacing:.06em;color:#6B7280;padding:.5rem 0 .15rem'>"
                        "Per-Tactic Spend Bounds (Goal-Seek)</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(
                        "These bounds apply only to this goal-seek scan. They prevent the optimizer "
                        "from assigning unrealistic spend to any channel. "
                        "Min = floor (e.g. 50% = at least half baseline). "
                        "Max = ceiling (e.g. 200% = at most 2× baseline)."
                    )

                    # Global defaults — user can override per channel
                    _gs_lo_default = st.number_input(
                        "Default Min % (all channels)", value=50, min_value=0,
                        max_value=100, step=10, key="pg_gs_lo_global",
                        help="Minimum spend as % of each channel's baseline. Applied to all unless overridden."
                    )
                    _gs_hi_default = st.number_input(
                        "Default Max % (all channels)", value=300, min_value=100,
                        max_value=500, step=50, key="pg_gs_hi_global",
                        help="Maximum spend as % of each channel's baseline (300% = up to 3× baseline). Applied to all unless overridden."
                    )

                    # Per-channel override expander
                    _gs_ch_bounds = {}  # channel -> (lo_pct, hi_pct) as fractions
                    with st.expander("Override bounds per channel (optional)", expanded=False):
                        st.caption("Leave at defaults unless a specific channel needs tighter constraints.")
                        for _, _row in df.iterrows():
                            _ch   = _row["channel"]
                            _base = float(_row["total_spend"])
                            _is_locked = int(_row.get("lock_spend", 0)) == 1
                            if _is_locked:
                                st.markdown(
                                    f"<span class='chip chip-gold'>🔒 {_ch} — locked at {fmt(_base,'$')}</span>",
                                    unsafe_allow_html=True
                                )
                                _gs_ch_bounds[_ch] = (1.0, 1.0)  # fixed
                                continue
                            _col1, _col2 = st.columns(2)
                            _lo_ch = _col1.number_input(
                                f"{_ch} Min%", value=_gs_lo_default,
                                min_value=0, max_value=200, step=10,
                                key=f"pg_gs_lo_{_ch}"
                            )
                            _hi_ch = _col2.number_input(
                                f"{_ch} Max%", value=_gs_hi_default,
                                min_value=100, max_value=500, step=50,
                                key=f"pg_gs_hi_{_ch}"
                            )
                            _gs_ch_bounds[_ch] = (_lo_ch / 100.0, _hi_ch / 100.0)

                    # Fill any channels not in expander with defaults
                    for _, _row in df.iterrows():
                        _ch = _row["channel"]
                        if _ch not in _gs_ch_bounds:
                            _is_locked = int(_row.get("lock_spend", 0)) == 1
                            _gs_ch_bounds[_ch] = (1.0, 1.0) if _is_locked else (
                                _gs_lo_default / 100.0, _gs_hi_default / 100.0
                            )

                    run_pg = st.button("🔍 Find Minimum Budget", type="primary",
                                       width="stretch", key="run_pg_wi_btn")

                with pg_c2:
                    # ── Baseline check ────────────────────────────────────────
                    if pg_target <= _base_val_pg:
                        st.markdown(
                            f"<div class='alert-box alert-success' style='font-size:.8rem'>"
                            f"✅ Baseline {pg_target_type} <b>{fmt(_base_val_pg,'$')}</b> already meets "
                            f"<b>{fmt(pg_target,'$')}</b>. No extra budget needed.</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        # Build bounds-enforced df for the scan
                        # These bounds stay hard — the auto-scaling in GEKKO only
                        # widens them when the budget is infeasible, but we clamp
                        # back to the user's realistic limits after.
                        _df_gs = df.copy()
                        for _, _row in _df_gs.iterrows():
                            _ch = _row["channel"]
                            if _ch in _gs_ch_bounds:
                                _lo_f, _hi_f = _gs_ch_bounds[_ch]
                                _df_gs.loc[_df_gs["channel"] == _ch, "lower_bound_pct"] = _lo_f
                                _df_gs.loc[_df_gs["channel"] == _ch, "upper_bound_pct"] = _hi_f

                        _pg_cache_key = (
                            f"wi_gs_portfolio|{pg_target_type}|{pg_target:.0f}|"
                            f"{pg_n_scan}|{method_map[pg_method]}|"
                            f"{use_hill_global}|{use_log_global}|"
                            f"{_gs_lo_default}|{_gs_hi_default}"
                        )

                        if run_pg:
                            # Start scan from 80% of baseline — below that,
                            # revenue is unlikely to meet a target > baseline.
                            # Use 20 points focused in the realistic range.
                            _pg_lo_bgt = max(tot_spend * 0.8, _base_val_pg * 0.5 / 3.0)
                            _pg_hi_bgt = tot_spend * 3.0
                            _pg_budgets = np.linspace(_pg_lo_bgt, _pg_hi_bgt, pg_n_scan)
                            _pg_rows = []
                            _pg_prog = st.progress(0, text="Scanning budget levels…")
                            for _bi, _bgt in enumerate(_pg_budgets):
                                _pg_prog.progress(
                                    int((_bi + 1) / len(_pg_budgets) * 100),
                                    text=f"Scanning {fmt(_bgt,'$')}…"
                                )
                                _r = optimize_budget(
                                    _df_gs, _bgt,
                                    objective="profit",   # profit-optimal allocation based on mROI
                                    use_hill=use_hill_global, use_log=use_log_global,
                                    method=method_map[pg_method]
                                )
                                if _r.get("success") and _r.get("spends") is not None:
                                    _rev_i = sum(
                                        revenue_from_spend(_r["spends"][i], _df_gs.iloc[i],
                                                           use_hill_global, use_log_global)
                                        for i in range(len(_df_gs))
                                    )
                                    _prf_i = _rev_i - _bgt
                                    _met_i = _rev_i if _is_rev_pg else _prf_i
                                    _pg_rows.append({
                                        "budget":     _bgt,
                                        "metric_val": _met_i,
                                        "revenue":    _rev_i,
                                        "profit":     _prf_i,
                                        "spends":     _r["spends"]
                                    })
                            _pg_prog.empty()
                            st.session_state[_pg_cache_key] = _pg_rows

                        _pg_rows_cached = st.session_state.get(_pg_cache_key)

                        if _pg_rows_cached:
                            _pg_df = pd.DataFrame([{k: v for k, v in r.items() if k != "spends"}
                                                    for r in _pg_rows_cached])
                            _meets_pg = _pg_df[_pg_df["metric_val"] >= pg_target]

                            if not _meets_pg.empty:
                                _min_bgt_pg     = float(_meets_pg.iloc[0]["budget"])
                                _min_bgt_metric = float(_meets_pg.iloc[0]["metric_val"])
                                _min_bgt_rev    = float(_meets_pg.iloc[0]["revenue"])
                                _min_bgt_prof   = float(_meets_pg.iloc[0]["profit"])
                                _pg_met         = True
                            else:
                                _max_row        = _pg_df.loc[_pg_df["metric_val"].idxmax()]
                                _min_bgt_pg     = float(_max_row["budget"])
                                _min_bgt_metric = float(_max_row["metric_val"])
                                _min_bgt_rev    = float(_max_row["revenue"])
                                _min_bgt_prof   = float(_max_row["profit"])
                                _pg_met         = False

                            # Summary KPIs
                            # Note: revenue at min budget may exceed the target because
                            # the scan finds the FIRST budget where revenue >= target.
                            # The optimizer maximises revenue at that budget level.
                            pk1, pk2 = st.columns(2)
                            pk1.metric(
                                "Min Budget for Target" if _pg_met else "Max Achievable Budget",
                                fmt(_min_bgt_pg, "$"),
                                delta=f"{fmt(_min_bgt_pg - tot_spend,'$','+' if _min_bgt_pg>=tot_spend else '')} vs baseline"
                            )
                            pk2.metric(
                                "Target Revenue",
                                fmt(pg_target, "$"),
                                delta=f"Goal to achieve"
                            )
                            pk3, pk4 = st.columns(2)
                            pk3.metric(
                                "Revenue at Min Budget",
                                fmt(_min_bgt_rev, "$"),
                                delta=f"{fmt(_min_bgt_rev - pg_target,'$','+' if _min_bgt_rev>=pg_target else '')} vs target"
                            )
                            pk4.metric("Profit at Min Budget", fmt(_min_bgt_prof, "$"))

                            if not _pg_met:
                                st.error(
                                    f"❌ Target {fmt(pg_target,'$')} not achievable within scan range "
                                    f"with per-tactic bounds Min={_gs_lo_default}% / Max={_gs_hi_default}%. "
                                    f"Max revenue achievable = {fmt(_min_bgt_rev,'$')} at {fmt(_min_bgt_pg,'$')} budget. "
                                    f"Try: (1) widen Max% bounds above {_gs_hi_default}%, "
                                    f"(2) reduce your target below {fmt(_min_bgt_rev,'$')}, or "
                                    f"(3) increase Scan Resolution to catch a higher budget point."
                                )
                            else:
                                _overshoot = _min_bgt_rev - pg_target
                                st.markdown(
                                    f"<div class='alert-box alert-success' style='font-size:.8rem'>"
                                    f"✅ Minimum budget to achieve {pg_target_type} ≥ {fmt(pg_target,'$')} "
                                    f"is <b>{fmt(_min_bgt_pg,'$')}</b> "
                                    f"({(_min_bgt_pg/tot_spend-1)*100:+.1f}% vs current). "
                                    f"Profit-optimal allocation at this budget yields revenue of "
                                    f"<b>{fmt(_min_bgt_rev,'$')}</b> "
                                    f"({fmt(_overshoot,'$','+' if _overshoot>=0 else '')} vs target). "
                                    f"Tactic bounds: Min={_gs_lo_default}% / Max={_gs_hi_default}%.</div>",
                                    unsafe_allow_html=True
                                )

                            if PLOTLY:
                                fig_pg = go.Figure()
                                fig_pg.add_trace(go.Scatter(
                                    x=_pg_df['budget'], y=_pg_df['metric_val'],
                                    fill='tozeroy', fillcolor="rgba(26,86,219,0.06)",
                                    line=dict(width=0), showlegend=False, hoverinfo='skip'
                                ))
                                fig_pg.add_trace(go.Scatter(
                                    x=_pg_df['budget'], y=_pg_df['metric_val'],
                                    mode='lines+markers',
                                    line=dict(color=PALETTE["accent"], width=3),
                                    marker=dict(size=6, color=PALETTE["accent"],
                                                line=dict(color='white', width=1.5)),
                                    name=pg_target_type,
                                    hovertemplate="Budget: $%{x:,.0f}<br>" + pg_target_type + ": $%{y:,.0f}<extra></extra>"
                                ))
                                fig_pg.add_hline(
                                    y=pg_target, line_dash="dash",
                                    line_color=PALETTE["accent3"], line_width=2.5,
                                    annotation_text=f"Target: {fmt(pg_target,'$')}",
                                    annotation_font=dict(size=10, color=PALETTE["accent3"]),
                                    annotation_position="top left"
                                )
                                fig_pg.add_vline(
                                    x=tot_spend, line_dash="dot",
                                    line_color=PALETTE["muted"], line_width=1.5,
                                    annotation_text=f"Current: {fmt(tot_spend,'$')}",
                                    annotation_font=dict(size=9, color=PALETTE["muted"]),
                                    annotation_position="bottom right"
                                )
                                if _pg_met:
                                    fig_pg.add_vline(
                                        x=_min_bgt_pg, line_dash="dash",
                                        line_color=PALETTE["accent2"], line_width=2,
                                        annotation_text=f"Min: {fmt(_min_bgt_pg,'$')}",
                                        annotation_font=dict(size=10, color=PALETTE["accent2"]),
                                        annotation_position="top right"
                                    )
                                    fig_pg.add_trace(go.Scatter(
                                        x=[_min_bgt_pg], y=[_min_bgt_metric],
                                        mode='markers',
                                        marker=dict(size=14, color=PALETTE["accent2"],
                                                    symbol='star', line=dict(color='white', width=2)),
                                        name='Minimum budget',
                                        hovertemplate=f"Min budget: ${_min_bgt_pg:,.0f}<br>{pg_target_type}: ${_min_bgt_metric:,.0f}<extra></extra>"
                                    ))
                                fig_pg.update_layout(
                                    title=dict(text=f"{pg_target_type} vs Total Budget — Goal-Seek",
                                               font=dict(size=11, color=PALETTE["text2"])),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                                    font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                                    height=380, margin=dict(l=10, r=20, t=42, b=60),
                                    legend=dict(bgcolor="rgba(255,255,255,0.9)",
                                                bordercolor=PALETTE["border"], borderwidth=1,
                                                orientation="h", y=-0.18, font=dict(size=10)),
                                    xaxis=dict(title="Total Budget ($)", tickformat="$,.0f",
                                               gridcolor="#EAECF0",
                                               tickfont=dict(size=10, color=PALETTE["muted"])),
                                    yaxis=dict(title=pg_target_type + " ($)", tickformat="$,.0f",
                                               gridcolor="#EAECF0",
                                               tickfont=dict(size=10, color=PALETTE["muted"]))
                                )
                                st.plotly_chart(fig_pg, width='stretch')

                                # Optimal channel allocation at minimum budget
                            if _pg_met and _pg_rows_cached:
                                _sol_idx  = list(_meets_pg.index)[0]
                                _sol_row  = _pg_rows_cached[_sol_idx]
                                _sol_spds = _sol_row["spends"]
                                df_pg_sol = _df_gs.copy()
                                df_pg_sol["gs_spend"]    = _sol_spds
                                df_pg_sol["gs_revenue"]  = [
                                    revenue_from_spend(_sol_spds[i], df_pg_sol.iloc[i],
                                                       use_hill_global, use_log_global)
                                    for i in range(len(df_pg_sol))
                                ]
                                df_pg_sol["gs_profit"]   = df_pg_sol["gs_revenue"] - df_pg_sol["gs_spend"]
                                df_pg_sol["gs_delta_sp"] = df_pg_sol["gs_spend"] - df_pg_sol["total_spend"]
                                df_pg_sol["gs_delta_pct"] = (
                                    df_pg_sol["gs_delta_sp"] /
                                    df_pg_sol["total_spend"].replace(0, np.nan) * 100
                                )
                                # Enforce bounds for display: cap Δ% to bounds
                                df_pg_sol["bound_min_pct"] = [
                                    _gs_ch_bounds.get(ch,(0.5,2.0))[0]*100
                                    for ch in df_pg_sol["channel"]
                                ]
                                df_pg_sol["bound_max_pct"] = [
                                    _gs_ch_bounds.get(ch,(0.5,2.0))[1]*100
                                    for ch in df_pg_sol["channel"]
                                ]
                                df_pg_sol["gs_roi"] = (
                                    df_pg_sol["gs_revenue"] /
                                    df_pg_sol["gs_spend"].replace(0, np.nan)
                                )

                                st.markdown(
                                    f"<div class='section-header'>"
                                    f"Profit-optimal channel allocation at {fmt(_min_bgt_pg,'$')} "
                                    f"(tactic bounds: Min={_gs_lo_default}% / Max={_gs_hi_default}%)"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"<div class='alert-box alert-info' style='font-size:.76rem'>"
                                    f"The optimizer allocates {fmt(_min_bgt_pg,'$')} budget across channels "
                                    f"to maximise <b>profit</b> (based on each channel's mROI and response curve). "
                                    f"Channels with higher mROI receive more budget. "
                                    f"Each tactic stays within its Min/Max bounds. "
                                    f"Δ% shows change vs each channel's baseline spend.</div>",
                                    unsafe_allow_html=True
                                )
                                _pg_disp = df_pg_sol[[
                                    "channel","total_spend","gs_spend",
                                    "gs_delta_sp","gs_delta_pct",
                                    "bound_min_pct","bound_max_pct",
                                    "gs_revenue","gs_profit","gs_roi"
                                ]].copy()
                                _pg_disp.columns = [
                                    "Channel","Baseline Spend","Allocated Spend",
                                    "Δ Spend","Δ%",
                                    "Min Bound%","Max Bound%",
                                    "Revenue","Profit","ROI"
                                ]
                                for c in ["Baseline Spend","Allocated Spend","Δ Spend",
                                          "Revenue","Profit"]:
                                    _pg_disp[c] = _pg_disp[c].apply(lambda x: f"${x:,.0f}")
                                _pg_disp["Δ%"] = _pg_disp["Δ%"].apply(
                                    lambda x: f"{x:+.1f}%" if pd.notnull(x) else "—")
                                _pg_disp["Min Bound%"] = _pg_disp["Min Bound%"].apply(
                                    lambda x: f"{x:.0f}%")
                                _pg_disp["Max Bound%"] = _pg_disp["Max Bound%"].apply(
                                    lambda x: f"{x:.0f}%")
                                _pg_disp["ROI"] = _pg_disp["ROI"].apply(
                                    lambda x: f"{x:.2f}×" if pd.notnull(x) else "—")
                                st.dataframe(_pg_disp, width="stretch", hide_index=True)

                                # Download
                                _pg_csv = io.StringIO()
                                df_pg_sol[[
                                    "channel","total_spend","gs_spend","gs_delta_sp",
                                    "gs_delta_pct","bound_min_pct","bound_max_pct",
                                    "gs_revenue","gs_profit","gs_roi"
                                ]].to_csv(_pg_csv, index=False)
                                st.download_button(
                                    "⬇️ Download allocation (CSV)",
                                    data=_pg_csv.getvalue(),
                                    file_name="mmm_goalseeked_allocation.csv",
                                    mime="text/csv"
                                )

                            _pg_prompt = (
                                f"PORTFOLIO GOAL-SEEK ANALYSIS\n\n"
                                f"Target: {pg_target_type} = {fmt(pg_target,'$')}\n"
                                f"Minimum budget required: {fmt(_min_bgt_pg,'$') if _pg_met else 'Not achievable'}\n"
                                f"Current baseline budget: {fmt(tot_spend,'$')} | "
                                f"Revenue: {fmt(tot_rev,'$')} | Profit: {fmt(tot_profit,'$')}\n\n"
                                f"Channel model context:\n"
                                + "\n".join([
                                    f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
                                    f"mROI={float(r['baseline_mroi']):.2f}x, alpha={float(r['alpha']):.3f}"
                                    for _, r in df.iterrows()
                                ])
                                + f"\n\nAnalyse: (1) Is the minimum budget realistic for this brand type? "
                                f"(2) How does the optimal allocation at minimum budget differ from current — "
                                f"which channels gain/lose and why? "
                                f"(3) What is the efficiency of the minimum budget vs current spend? "
                                f"(4) What incremental investment beyond minimum would yield the next meaningful gain?"
                            )
                            render_ai_button(_pg_prompt, "ai_gs_wi_portfolio", "🤖 Interpret Portfolio Goal-Seek")
                        else:
                            _s = PALETTE["surface"]
                            _b = PALETTE["border"]
                            _t = PALETTE["text"]
                            st.markdown(
                                f"<div style='margin-top:2rem;text-align:center;padding:2rem;"
                                f"background:{_s};border-radius:14px;"
                                f"border:2px dashed {_b}'>"
                                f"<div style='font-size:1.8rem;margin-bottom:.5rem'>💰</div>"
                                f"<div style='font-size:.9rem;font-weight:600;color:{_t}'>Set a revenue or profit target and click Find Minimum Budget</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

            # ── BREAK-EVEN ANALYSIS ───────────────────────────────────────────
            elif "Break-Even" in gs_wi_mode:
                st.markdown(
                    f"<div class='alert-box alert-info' style='margin-bottom:.5rem;font-size:.8rem'>"
                    f"Break-even spend = the spend at which channel profit = 0 (revenue = spend). "
                    f"Channels spending below break-even are generating negative profit. "
                    f"The <b>optimal spend</b> is where mROI = 1 (profit is maximised).</div>",
                    unsafe_allow_html=True
                )

                be_rows = []
                for _, row in df.iterrows():
                    ch       = row['channel']
                    base_sp  = float(row['total_spend'])
                    base_rev = float(row['baseline_revenue'])
                    base_pf  = float(row['baseline_profit'])
                    base_mroi = float(row['baseline_mroi'])

                    # Find break-even spend via binary search (revenue = spend)
                    _lo, _hi = 0.01, base_sp * 20
                    _be_found = False
                    if revenue_from_spend(_hi, row, use_hill_global, use_log_global) >= _hi:
                        # Revenue always > spend in this range — no break-even (channel always profitable)
                        be_sp = None
                    elif revenue_from_spend(0.01, row, use_hill_global, use_log_global) <= 0.01:
                        be_sp = 0.01  # break-even near zero
                    else:
                        for _ in range(60):
                            _mid = (_lo + _hi) / 2
                            rev_mid = revenue_from_spend(_mid, row, use_hill_global, use_log_global)
                            if rev_mid > _mid:
                                _lo = _mid
                            else:
                                _hi = _mid
                        be_sp = (_lo + _hi) / 2

                    # Find mROI=1 optimal spend (profit-maximising) via binary search on mROI
                    _lo2, _hi2 = 0.01, base_sp * 10
                    for _ in range(60):
                        _mid2 = (_lo2 + _hi2) / 2
                        if mroi(_mid2, row, use_hill_global, use_log_global) > 1.0:
                            _lo2 = _mid2
                        else:
                            _hi2 = _mid2
                    opt_sp = (_lo2 + _hi2) / 2
                    opt_rev  = revenue_from_spend(opt_sp, row, use_hill_global, use_log_global)
                    opt_prof = opt_rev - opt_sp

                    be_rows.append({
                        'Channel':          ch,
                        'Description':      ch_desc_map.get(ch, ch),
                        'Baseline Spend':   base_sp,
                        'Baseline Revenue': base_rev,
                        'Baseline Profit':  base_pf,
                        'Baseline mROI':    base_mroi,
                        'Break-Even Spend': be_sp,
                        'Profit-Max Spend': opt_sp,
                        'Profit-Max Revenue': opt_rev,
                        'Profit-Max Profit':  opt_prof,
                        'vs Baseline (opt)':  opt_sp - base_sp,
                        'Status': (
                            '✅ Profitable' if base_pf > 0 else '❌ Loss-making'
                        ),
                        'vs Optimal': (
                            '📈 Under-invested' if base_sp < opt_sp * 0.9
                            else ('📉 Over-invested' if base_sp > opt_sp * 1.1
                                  else '✅ Near-optimal')
                        )
                    })

                be_df = pd.DataFrame(be_rows)

                # KPI summary
                bek1, bek2, bek3, bek4 = st.columns(4)
                bek1.metric("Profitable Channels",
                             f"{(be_df['Baseline Profit'] > 0).sum()} / {len(be_df)}")
                bek2.metric("Under-invested",
                             f"{(be_df['vs Optimal'] == '📈 Under-invested').sum()} channels")
                bek3.metric("Over-invested",
                             f"{(be_df['vs Optimal'] == '📉 Over-invested').sum()} channels")
                bek4.metric("Total Profit-Max Spend",
                             fmt(be_df['Profit-Max Spend'].sum(), "$"),
                             delta=f"{fmt(be_df['Profit-Max Spend'].sum()-tot_spend,'$','+' if be_df['Profit-Max Spend'].sum()>=tot_spend else '')} vs baseline")

                if PLOTLY:
                    # Grouped bar: Baseline vs Break-Even vs Profit-Max spend
                    _be_ch = be_df['Channel'].tolist()
                    fig_be = go.Figure()
                    fig_be.add_trace(go.Bar(
                        name='Baseline Spend', x=_be_ch, y=be_df['Baseline Spend'],
                        marker_color=hex_to_rgba(PALETTE["accent"], 0.5),
                        marker_line_color=PALETTE["accent"], marker_line_width=1.5,
                        text=[fmt(v,"$") for v in be_df['Baseline Spend']],
                        textposition='outside', textfont=dict(size=8)
                    ))
                    fig_be.add_trace(go.Bar(
                        name='Break-Even Spend', x=_be_ch,
                        y=[v if v is not None else 0 for v in be_df['Break-Even Spend']],
                        marker_color=hex_to_rgba(PALETTE["accent3"], 0.5),
                        marker_line_color=PALETTE["accent3"], marker_line_width=1.5,
                        text=[fmt(v,"$") if v is not None else "Always+" for v in be_df['Break-Even Spend']],
                        textposition='outside', textfont=dict(size=8)
                    ))
                    fig_be.add_trace(go.Bar(
                        name='Profit-Max Spend (mROI=1)', x=_be_ch, y=be_df['Profit-Max Spend'],
                        marker_color=hex_to_rgba(PALETTE["accent2"], 0.5),
                        marker_line_color=PALETTE["accent2"], marker_line_width=1.5,
                        text=[fmt(v,"$") for v in be_df['Profit-Max Spend']],
                        textposition='outside', textfont=dict(size=8)
                    ))
                    fig_be.update_layout(
                        barmode='group',
                        title=dict(text="Break-Even vs Profit-Maximising Spend by Channel",
                                   font=dict(size=11, color=PALETTE["text2"])),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFC",
                        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                        height=360, margin=dict(l=5, r=10, t=42, b=80),
                        legend=dict(bgcolor="rgba(255,255,255,0.9)",
                                    bordercolor=PALETTE["border"], borderwidth=1,
                                    orientation="h", y=-0.24, font=dict(size=10)),
                        xaxis=dict(gridcolor="#EAECF0", tickangle=-30,
                                   tickfont=dict(size=9, color=PALETTE["muted"])),
                        yaxis=dict(title="Spend ($)", tickformat="$,.0f",
                                   gridcolor="#EAECF0", zeroline=False,
                                   tickfont=dict(size=9, color=PALETTE["muted"]))
                    )
                    st.plotly_chart(fig_be, width='stretch')

                # Results table
                st.markdown(f"<div class='section-header'>Break-even & profit-maximising detail</div>",
                            unsafe_allow_html=True)
                _be_disp = be_df[['Channel','Baseline Spend','Baseline Profit','Baseline mROI',
                                   'Break-Even Spend','Profit-Max Spend','vs Baseline (opt)',
                                   'Status','vs Optimal']].copy()
                for c in ['Baseline Spend','Baseline Profit','Break-Even Spend',
                          'Profit-Max Spend','vs Baseline (opt)']:
                    _be_disp[c] = _be_disp[c].apply(
                        lambda x: (fmt(x,"$","+") if x>=0 else fmt(x,"$"))
                        if x is not None and pd.notnull(x) else "Always profitable"
                    )
                _be_disp['Baseline mROI'] = _be_disp['Baseline mROI'].apply(lambda x: f"{x:.2f}×")
                st.dataframe(_be_disp, width='stretch', hide_index=True)

                _be_prompt = (
                    f"BREAK-EVEN & PROFIT-MAXIMISING ANALYSIS\n\n"
                    + "\n".join([
                        f"  {r['Channel']} ({ch_desc_map.get(r['Channel'],r['Channel'])}): "
                        f"baseline spend={fmt(r['Baseline Spend'],'$')}, "
                        f"profit={fmt(r['Baseline Profit'],'$')}, mROI={r['Baseline mROI']}, "
                        f"break-even={fmt(r['Break-Even Spend'],'$') if r['Break-Even Spend'] is not None else 'always profitable'}, "
                        f"profit-max spend={fmt(r['Profit-Max Spend'],'$')}, "
                        f"status={r['Status']}, position={r['vs Optimal']}"
                        for _, r in be_df.iterrows()
                    ])
                    + f"\n\nPortfolio: spend={fmt(tot_spend,'$')}, revenue={fmt(tot_rev,'$')}, "
                    f"profit={fmt(tot_profit,'$')}\n\n"
                    f"Analyse: (1) Which channels are loss-making and should spend be reduced or cut? "
                    f"(2) Which are furthest from profit-maximising spend and by how much? "
                    f"(3) What is the total profit uplift if all channels moved to mROI=1? "
                    f"(4) Prioritised reallocation roadmap: which moves first and why?"
                )
                render_ai_button(_be_prompt, "ai_gs_wi_breakeven", "🤖 Interpret Break-Even Analysis")


        # ── AI Interpretation — What-If ───────────────────────────────────────────
    _wi_mode_txt = wi_mode.replace("📊 ","").replace("🎯 ","").replace("📐 ","").replace("🔀 ","")
    _wi_portfolio = "\n".join([
        f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
        f"mROI={float(r['baseline_mroi']):.2f}x, ROI={float(r['baseline_roi']):.2f}x, "
        f"spend=${float(r['total_spend']):,.0f}, alpha={float(r['alpha']):.3f}"
        for _, r in df.iterrows()
    ])
    _wi_prompt = (
        f"WHAT-IF SCENARIO ANALYSIS — Mode: {_wi_mode_txt}\n\n"
        f"BASELINE PORTFOLIO:\n"
        f"  Total spend: {fmt(tot_spend,'$')} | Revenue: {fmt(tot_rev,'$')} | "
        f"Profit: {fmt(tot_profit,'$')} | ROI: {avg_roas:.2f}x\n\n"
        f"CHANNEL DETAILS:\n{_wi_portfolio}\n\n"
        f"Channel descriptions:\n"
        + "\n".join([f"  {k}: {v}" for k,v in ch_desc_map.items()])
        + "\n\nAnalyse using the four-section structure. Focus on: "
        f"(1) what this scenario reveals about portfolio sensitivity and channel interdependencies; "
        f"(2) which channel types ({'/ '.join(ch_desc_map.values())}) respond best to investment changes; "
        f"(3) risks of this allocation vs baseline; "
        f"(4) pharma best-practice channel mix for this type of brand."
    )
    render_ai_button(_wi_prompt, "ai_wi_btn", "🤖 Interpret What-If Scenarios")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — DATA & DIAGNOSTICS                                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

with tab_data:
    diag1, diag2 = st.columns([.55, .45])

    with diag1:
        st.markdown(f"<div class='section-header'>Input data + derived metrics</div>", unsafe_allow_html=True)
        view_cols = [
            'channel', 'total_activity', 'total_spend', 'total_sales',
            'type_transformation', 'coefficient', 'alpha', 'total_segments', 'net_per_unit',
            'activity_per_segment', 'cost_per_mention', 'Unadjusted_impact', 'Adj_Factor',
            'baseline_revenue', 'baseline_profit', 'baseline_roi', 'baseline_mroi',
            'hill_ec50', 'hill_slope', 'hill_max_response'
        ]
        display_data = df[[c for c in view_cols if c in df.columns]].copy()
        num_disp = ['total_spend','total_sales','Unadjusted_impact','baseline_revenue','baseline_profit',
                    'hill_ec50','hill_max_response']
        for c in num_disp:
            if c in display_data.columns:
                display_data[c] = display_data[c].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "—")
        for c in ['coefficient','alpha','Adj_Factor','cost_per_mention','baseline_roi','baseline_mroi',
                  'hill_slope']:
            if c in display_data.columns:
                display_data[c] = display_data[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
        st.dataframe(display_data, width='stretch', hide_index=True)

        buf_data = io.StringIO()
        df[[c for c in view_cols if c in df.columns]].to_csv(buf_data, index=False)
        st.download_button("⬇️ Download full diagnostics (CSV)",
                            data=buf_data.getvalue(),
                            file_name="mmm_diagnostics.csv",
                            mime="text/csv")

    with diag2:
        st.markdown(f"<div class='section-header'>Channel diagnostics</div>", unsafe_allow_html=True)

        if PLOTLY:
            # mROI heatmap bar
            fig_mroi = go.Figure()
            mroi_vals_list = df['baseline_mroi'].tolist()
            colors_mroi = [PALETTE["accent2"] if v > 1.1 else
                            (PALETTE["accent3"] if v < 0.9 else PALETTE["accent"])
                            for v in mroi_vals_list]
            fig_mroi.add_trace(go.Bar(
                x=df['channel'], y=mroi_vals_list,
                marker_color=colors_mroi,
                text=[f"{v:.2f}×" for v in mroi_vals_list],
                textposition='outside', textfont=dict(size=9)
            ))
            fig_mroi.add_hline(y=1.0, line_dash="dash",
                                 line_color=PALETTE["muted"], annotation_text="mROI=1",
                                 annotation_font_size=9)
            plotly_dark_layout(fig_mroi, height=240,
                                title="Marginal ROI by Channel")
            fig_mroi.update_layout(showlegend=False, margin=dict(b=60),
                                     xaxis=dict(tickangle=-30, tickfont=dict(size=9, color=PALETTE["muted"])))
            st.plotly_chart(fig_mroi, width='stretch')

            # Adj Factor bar
            fig_adj = go.Figure()
            fig_adj.add_trace(go.Bar(
                x=df['channel'], y=df['Adj_Factor'],
                marker_color=[channel_color(i) for i in range(len(df))],
                text=[f"{v:.3f}" for v in df['Adj_Factor']],
                textposition='outside', textfont=dict(size=9)
            ))
            plotly_dark_layout(fig_adj, height=220, title="Calibration Adjustment Factors")
            fig_adj.update_layout(showlegend=False, margin=dict(b=60),
                                   xaxis=dict(tickangle=-30, tickfont=dict(size=9, color=PALETTE["muted"])))
            st.plotly_chart(fig_adj, width='stretch')

            # Revenue mix donut
            fig_pie = go.Figure(go.Pie(
                labels=df['channel'],
                values=df['baseline_revenue'],
                marker=dict(colors=[channel_color(i) for i in range(len(df))],
                            line=dict(color=PALETTE["bg"], width=2)),
                hole=0.55,
                textinfo='label+percent',
                textfont=dict(size=9)
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text2"]),
                height=280, margin=dict(l=5, r=5, t=35, b=5),
                showlegend=False,
                title=dict(text="Revenue Mix (baseline)", font=dict(size=11))
            )
            st.plotly_chart(fig_pie, width='stretch')

    # ── AI Interpretation — Data & Diagnostics ────────────────────────────────
    _diag_rows = "\n".join([
        f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
        f"alpha={float(r['alpha']):.3f}, coef={float(r['coefficient']):.5f}, "
        f"AdjFactor={float(r['Adj_Factor']):.3f}, mROI={float(r['baseline_mroi']):.2f}x, "
        f"ROI={float(r['baseline_roi']):.2f}x, profit={fmt(float(r['baseline_profit']),'$')}"
        for _, r in df.iterrows()
    ])
    _diag_prompt = (
        f"MMM MODEL DIAGNOSTICS & CALIBRATION AUDIT\n\n"
        f"PORTFOLIO:\n"
        f"  Total spend: {fmt(tot_spend,'$')} | Revenue: {fmt(tot_rev,'$')} | "
        f"Profit: {fmt(tot_profit,'$')} | Avg ROI: {avg_roas:.2f}x\n\n"
        f"CHANNEL METRICS:\n{_diag_rows}\n\n"
        f"Analyse using the four-section structure. Focus on: "
        f"(1) model calibration quality — are AdjFactors reasonable for each channel type? "
        f"Flag any that are unusually high (>10) or low (<0.5); "
        f"(2) alpha values — which channels show strongest diminishing returns and what does that mean "
        f"for investment headroom; "
        f"(3) portfolio concentration risk — revenue mix and dependency on any single channel; "
        f"(4) pharma industry benchmarks for these channel types and what good looks like."
    )
    render_ai_button(_diag_prompt, "ai_data_btn", "🤖 Interpret Data & Diagnostics")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 — AI INTERPRETATION                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

with tab_ai:
    st.markdown(f"<div class='section-header'>AI-powered channel & portfolio interpretation</div>",
                unsafe_allow_html=True)


    

    # ── Sidebar: provider + key ───────────────────────────────────────────────
    with st.sidebar:
        _smu = PALETTE["sidebar_mu"]
        st.markdown(
            f"<div style='font-size:.65rem;font-weight:700;color:{_smu};"
            "text-transform:uppercase;letter-spacing:.1em;padding:.75rem 0 .25rem'>AI Settings</div>",
            unsafe_allow_html=True)
        
        if st.session_state.get("ai_provider") == "Google Gemini (free)":
            ai_model_sel = st.selectbox("Model", _PREFERRED_MODELS,
                                         key="ai_model_gemini",
                                         help="gemini-1.5-flash is fastest on free tier")
        else:
            ai_model_sel = st.selectbox("Model",
                                         ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
                                         key="ai_model_groq")

    # ── Gemini SDK check ──────────────────────────────────────────────────────
    try:
        import google.generativeai as _genai
        _GENAI_OK = True
    except ImportError:
        _GENAI_OK = False



    
    

    # ── Mode selector ─────────────────────────────────────────────────────────
    ai_mode = st.radio(
        "Analysis type",
        [
            "📈 Response Curve",
            "💼 Portfolio Summary",
            "⚡ Optimization Results",
            "🔁 What-If Scenario",
            "🔬 Model Diagnostics",
        ],
        horizontal=True, label_visibility="collapsed"
    )

    # ── Provider badge ────────────────────────────────────────────────────────
    _badge_cls = "chip-green" if "Gemini" in st.session_state.get("ai_provider","") else "chip-purple"
    _sdk_note  = "" if _GENAI_OK else "  ⚠️ SDK missing (pip install google-generativeai)"
    st.markdown(
        f"<span class='chip {_badge_cls}'>{st.session_state.get('ai_provider','—')}</span> &nbsp;"
        f"<span class='chip chip-blue'>{ai_model_sel}</span> &nbsp;"
        f"<span class='chip chip-gold'>{curve_model_choice}</span>",
        unsafe_allow_html=True)
    if _sdk_note and st.session_state.get("ai_provider") == "Google Gemini (free)":
        st.warning(_sdk_note)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Shared channel context builder ────────────────────────────────────────
    def _portfolio_ctx():
        return "\n".join([
            f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
            f"spend=${float(r['total_spend']):,.0f}, "
            f"revenue=${float(r['baseline_revenue']):,.0f}, "
            f"profit=${float(r['baseline_profit']):,.0f}, "
            f"ROI={float(r['baseline_roi']):.2f}x, "
            f"mROI={float(r['baseline_mroi']):.2f}x, "
            f"alpha={float(r['alpha']):.3f}, "
            f"curve={r.get('type_transformation','power')}"
            for _, r in df.iterrows()
        ])

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 1 — RESPONSE CURVE
    # ═══════════════════════════════════════════════════════════════════════════
    if "Response Curve" in ai_mode:
        ai_ch = st.selectbox("Select channel", df['channel'].tolist(), key="ai_channel_sel")
        rr    = df[df['channel'] == ai_ch].iloc[0]
        _desc = ch_desc_map.get(ai_ch, ai_ch)

        opt_df_ai  = st.session_state.get('opt_result_df')
        has_opt_ai = opt_df_ai is not None and ai_ch in opt_df_ai['channel'].values
        opt_info   = ""
        if has_opt_ai:
            or_ = opt_df_ai[opt_df_ai['channel'] == ai_ch].iloc[0]
            opt_info = (
                f"  Optimized: spend=${float(or_['opt_spend']):,.0f} "
                f"(Δ{float(or_['delta_pct_spend']):+.1f}%), "
                f"profit=${float(or_['opt_profit']):,.0f} "
                f"(Δ{float(or_['delta_profit']):+,.0f})\n"
            )

        st.caption(f"**{ai_ch}** — {_desc}")
        _tab6_rc_prompt = (
            f"RESPONSE CURVE ANALYSIS\n"
            f"Channel: {ai_ch} | Type: {_desc}\n"
            f"Curve model: {curve_model_choice} | "
            f"Transformation: {rr['type_transformation']} | "
            f"Alpha: {float(rr['alpha']):.3f} | "
            f"Coefficient: {float(rr['coefficient']):.5f}\n"
            f"Adj Factor: {float(rr['Adj_Factor']):.4f} | "
            f"Segments: {int(rr['total_segments'])} | "
            f"Activity: {float(rr['total_activity']):,.0f}\n\n"
            f"PERFORMANCE:\n"
            f"  Spend: ${float(rr['total_spend']):,.0f} | "
            f"Revenue: ${float(rr['baseline_revenue']):,.0f} | "
            f"Profit: ${float(rr['baseline_profit']):,.0f}\n"
            f"  ROI: {float(rr['baseline_roi']):.2f}x | "
            f"mROI: {float(rr['baseline_mroi']):.2f}x | "
            f"Locked: {'Yes' if int(rr['lock_spend']) else 'No'}\n"
            + (f"OPTIMIZER RESULT:\n{opt_info}" if has_opt_ai else "")
            + f"\nFULL PORTFOLIO:\n{_portfolio_ctx()}\n\n"
            f"This is a RESPONSE CURVE analysis. Focus on:\n"
            f"1. What does this curve shape (alpha={float(rr['alpha']):.3f}, {curve_model_choice}) "
            f"tell us about {_desc} — where is it on the saturation curve?\n"
            f"2. Is mROI={float(rr['baseline_mroi']):.2f}x signalling under/over-investment "
            f"for this channel type in pharma? What is the industry benchmark for {_desc}?\n"
            f"3. What is the optimal investment range and how much uplift could be captured?\n"
            f"4. How does this channel's curve compare to others in the portfolio — "
            f"which has the most headroom?"
        )
        render_ai_button(_tab6_rc_prompt, f"tab6_rc_{ai_ch}", "Interpret Response Curve")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 2 — PORTFOLIO SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    elif "Portfolio" in ai_mode:
        _tab6_port_prompt = (
            f"PORTFOLIO SUMMARY ANALYSIS\n\n"
            f"CHANNEL MIX:\n{_portfolio_ctx()}\n\n"
            f"TOTALS: spend=${tot_spend:,.0f}, revenue=${tot_rev:,.0f}, "
            f"profit=${tot_profit:,.0f}, avg ROI={avg_roas:.2f}x\n"
            f"Active curve model: {curve_model_choice}\n\n"
            f"This is a PORTFOLIO SUMMARY analysis. Focus on:\n"
            f"1. Which channel types (field force, digital HCP, DTC, social) are "
            f"over/under-invested relative to pharma norms — cite mROI for each.\n"
            f"2. What is the overall channel mix vs best-in-class pharma brands of similar size?\n"
            f"3. Where is the biggest reallocation opportunity — which specific channels "
            f"should gain/lose budget and by roughly how much?\n"
            f"4. What is the revenue concentration risk — which channels does the portfolio "
            f"depend on most, and what happens if they underperform?\n"
            f"5. Brand director summary: 4 bullet exec actions."
        )
        render_ai_button(_tab6_port_prompt, "tab6_portfolio", "Interpret Portfolio")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 3 — OPTIMIZATION RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    elif "Optimization" in ai_mode:
        opt_df_ai = st.session_state.get('opt_result_df')
        if opt_df_ai is None:
            st.info("Run Budget Optimization first, then return here.")
        else:
            obj_used    = st.session_state.get('opt_objective_used', 'Maximize Profit')
            method_used = st.session_state.get('opt_method_used', 'GEKKO')
            budget_used = st.session_state.get('opt_budget_used', tot_spend)
            new_rev_ai  = opt_df_ai['opt_revenue'].sum()
            new_prof_ai = opt_df_ai['opt_profit'].sum()
            new_sp_ai   = opt_df_ai['opt_spend'].sum()

            _opt_ctx = "\n".join([
                f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
                f"${float(r['total_spend']):,.0f}→${float(r['opt_spend']):,.0f} "
                f"(Δ{float(r['delta_pct_spend']):+.1f}%), "
                f"profit Δ=${float(r['delta_profit']):+,.0f}, "
                f"ROI {float(r['baseline_roi']):.2f}x→{float(r['opt_roi']):.2f}x"
                for _, r in opt_df_ai.iterrows()
            ])

            _tab6_opt_prompt = (
                f"OPTIMIZATION RESULTS ANALYSIS\n"
                f"Objective: {obj_used} | Solver: {method_used} | Budget: ${budget_used:,.0f}\n\n"
                f"CHANNEL REALLOCATION:\n{_opt_ctx}\n\n"
                f"PORTFOLIO IMPACT:\n"
                f"  Profit: ${tot_profit:,.0f}→${new_prof_ai:,.0f} "
                f"(Δ{new_prof_ai-tot_profit:+,.0f})\n"
                f"  Revenue: ${tot_rev:,.0f}→${new_rev_ai:,.0f}\n"
                f"  ROI: {avg_roas:.2f}x→{new_rev_ai/max(new_sp_ai,1):.2f}x\n\n"
                f"This is an OPTIMIZATION RESULTS analysis. Focus on:\n"
                f"1. What is the single most important strategic shift — "
                f"which channel types gain/lose and does this align with pharma best practice?\n"
                f"2. For each significant shift (>20%), explain the economic logic: "
                f"why does moving budget from/to that channel type make sense given its mROI?\n"
                f"3. Are channels converging toward mROI=1 (profit-optimal)? "
                f"Which are furthest from equilibrium?\n"
                f"4. Implementation sequencing: which shifts need phased rollout "
                f"(field force headcount takes 6+ months) vs which can be immediate (digital spend)?\n"
                f"5. Five prioritised exec action steps with owner and timeline."
            )
            render_ai_button(_tab6_opt_prompt, "tab6_optimization", "Interpret Optimization Results")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 4 — WHAT-IF SCENARIO
    # ═══════════════════════════════════════════════════════════════════════════
    elif "What-If" in ai_mode:
        st.info("Run a What-If scenario in the 🎛 tab, then interpret it here.")
        _wi_scenario = st.text_area(
            "Describe your what-if scenario",
            placeholder="e.g. I cut HCP PLD by 20% and shifted to DSE. What does this mean?",
            height=100,
            key="tab6_wi_textarea"
        )
        if _wi_scenario:
            _tab6_wi_prompt = (
                f"WHAT-IF SCENARIO ANALYSIS\n\n"
                f"SCENARIO DESCRIBED BY USER:\n{_wi_scenario}\n\n"
                f"BASELINE PORTFOLIO:\n{_portfolio_ctx()}\n\n"
                f"PORTFOLIO TOTALS: spend=${tot_spend:,.0f}, revenue=${tot_rev:,.0f}, "
                f"profit=${tot_profit:,.0f}, avg ROI={avg_roas:.2f}x\n\n"
                f"This is a WHAT-IF SCENARIO analysis. Focus on:\n"
                f"1. What would happen to revenue and profit if this scenario were implemented — "
                f"estimate the impact using the mROI and response curve data provided.\n"
                f"2. Which channels would benefit most / be most hurt by this reallocation, "
                f"and why — considering their channel type and saturation level?\n"
                f"3. What is the risk of this scenario — what could go wrong?\n"
                f"4. Is there a better alternative that achieves the same goal with less risk?\n"
                f"5. Clear recommendation: proceed, modify, or reject?"
            )
            render_ai_button(_tab6_wi_prompt, "tab6_whatif", "Interpret What-If")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 5 — MODEL DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        _diag_ctx = "\n".join([
            f"  {r['channel']} ({ch_desc_map.get(r['channel'],r['channel'])}): "
            f"alpha={float(r['alpha']):.3f}, coef={float(r['coefficient']):.5f}, "
            f"AdjFactor={float(r['Adj_Factor']):.3f}, "
            f"curve={r.get('type_transformation','power')}, "
            f"mROI={float(r['baseline_mroi']):.2f}x, ROI={float(r['baseline_roi']):.2f}x"
            for _, r in df.iterrows()
        ])
        _tab6_diag_prompt = (
            f"MODEL DIAGNOSTICS ANALYSIS\n"
            f"Active curve model: {curve_model_choice}\n\n"
            f"CHANNEL PARAMETERS:\n{_diag_ctx}\n\n"
            f"PORTFOLIO: spend=${tot_spend:,.0f}, revenue=${tot_rev:,.0f}, "
            f"profit=${tot_profit:,.0f}, avg ROI={avg_roas:.2f}x\n\n"
            f"This is a MODEL DIAGNOSTICS analysis. Focus on:\n"
            f"1. Calibration quality: are the Adjustment Factors (AdjFactor) credible "
            f"for each channel type? Flag any >10 or <0.5 as suspect.\n"
            f"2. Alpha values: which channels have the steepest diminishing returns "
            f"(low alpha) and which have the most linear response (alpha near 1)? "
            f"Are these values consistent with how these channels actually work in pharma?\n"
            f"3. Curve model assessment: is {curve_model_choice} the right choice for "
            f"this channel mix? When would you switch to Log or Hill?\n"
            f"4. Model risks: any channels where the model may be unreliable "
            f"(e.g. very high AdjFactor, near-zero coefficient, suspicious mROI)?\n"
            f"5. Recommendations to improve model quality."
        )
        render_ai_button(_tab6_diag_prompt, "tab6_diagnostics", "Interpret Model Quality")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"<hr style='margin:2rem 0 1rem;border-color:{PALETTE['border']}'>", unsafe_allow_html=True)
_footer_mu  = PALETTE["muted"]; _footer_txt = PALETTE["text"]
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem'>"
    f"<div style='font-size:.7rem;color:{_footer_mu}'>"
    f"⬡ <b style='color:{_footer_txt}'>MMM Budget Optimizer</b> &nbsp;·&nbsp;"
    f"Power &amp; Hill Response Curves &nbsp;·&nbsp; Multi-Objective Optimization"
    f"</div>"
    f"<div style='display:flex;gap:.4rem;flex-wrap:wrap'>"
    f"<span class='chip chip-blue'>GEKKO / IPOPT</span>"
    f"<span class='chip chip-purple'>SLSQP · DE</span>"
    f"<span class='chip chip-green'>R-equivalent Hill Fit</span>"
    f"<span class='chip chip-gold'>Pharma Commercial</span>"
    f"<span class='chip chip-purple'>Google/Groq AI</span>"
    f"</div></div>",
    unsafe_allow_html=True
)

# ===================== CHANNEL-LEVEL BOUNDS FEASIBILITY (ADDED) =====================

def check_channel_bounds(df, spend_col):
    """Check optimized spends against channel-level bounds.
    Returns (violations_df, feasible_bool)
    """
    records = []
    for _, r in df.iterrows():
        base = float(r.get('total_spend', 0))
        lo = base * float(r.get('lower_bound_pct', 0.0))
        hi = base * float(r.get('upper_bound_pct', float('inf')))
        spend = float(r.get(spend_col, 0))
        if spend < lo - 1e-6 or spend > hi + 1e-6:
            records.append({
                'channel': r.get('channel'),
                'optimized_spend': spend,
                'lower_bound': lo,
                'upper_bound': hi
            })
    vdf = pd.DataFrame(records)
    return vdf, vdf.empty

# Render global warning if bounds infeasible (safe no-op if not applicable)
if 'opt_result' in st.session_state:
    try:
        _df_out = st.session_state.get('opt_result_df') or None
        if _df_out is not None:
            _viol, _ok = check_channel_bounds(_df_out, 'opt_spend')
            st.session_state['bounds_feasible'] = _ok
            st.session_state['bounds_violations'] = _viol
            if not _ok:
                st.error(
                    '⚠️ Channel-level bounds are NOT feasible for this total budget. '
                    'One or more channels exceed their upper bounds to satisfy the budget constraint.',
                    icon='⚠️'
                )
    except Exception:
        pass
# ==============================================================================
