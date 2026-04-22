"""WaferGuard ML — Streamlit App."""

import base64
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from app.config import (
    BASE_PATTERN_SCENARIO_DEFAULTS,
    DATASET_PATH,
    DATASET_SLUG,
    ENABLE_RUNTIME_DATASET_BOOTSTRAP,
    FINANCIAL_CONFIG_DEFAULTS,
    MAX_BATCH_SIZE,
    MODEL_REGISTRY,
    TOP_N_DEFAULT,
)
from app.financial_ui import BASE_PATTERNS, analyze_results, is_pattern_name_related
from app.labels import get_description
from app.model_utils import load_model, predict_batch, predict_single
from app.preprocessing import parse_upload, prepare_for_model
from app.visualization import (
    build_results_dataframe,
    render_action_table,
    render_base_anomaly_chart,
    render_confidence_chart,
    render_kpi_cards,
    render_pattern_card,
    render_wafer_map,
)
from data.download_kaggle_data import ensure_wafer_dataset

st.set_page_config(
    page_title="WaferGuard ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ───────────────────────────────────────────────────────────
t = {
    "bg": "#0E1117",
    "bg2": "#262730",
    "text": "#FAFAFA",
    "text_muted": "rgba(250,250,250,0.6)",
    "logo_color": "#FAFAFA",
    "card": "#0a0c12",
}

st.markdown(
    f"""<style>
    @property --sidebar-glow {{
        syntax: '<color>';
        inherits: false;
        initial-value: #002776;
    }}
    @keyframes sidebar-glow-cycle {{
        0%, 100% {{ --sidebar-glow: #002776; }}
        33%      {{ --sidebar-glow: #92d400; }}
        66%      {{ --sidebar-glow: #00a1de; }}
    }}

    /* Main area */
    .stApp, .stApp > header {{
        background-color: {t["bg"]} !important;
    }}
    .stApp .main .block-container {{
        color: {t["text"]};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {t["bg2"]} !important;
        animation: sidebar-glow-cycle 6s ease-in-out infinite;
        border-right: 2px solid var(--sidebar-glow) !important;
        box-shadow: 2px 0 20px color-mix(in srgb, var(--sidebar-glow) 25%, transparent),
                    4px 0 40px color-mix(in srgb, var(--sidebar-glow) 10%, transparent);
        transition: background-color 0.3s ease;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] span {{
        color: {t["text"]};
    }}
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {t["text"]} !important;
    }}

    /* Metric labels */
    [data-testid="stMetricLabel"] {{
        color: {t["text_muted"]} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {t["text"]} !important;
    }}

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {t["text_muted"]} !important;
    }}
    </style>""",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────
SIDEBAR_HERO_HTML = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: transparent; overflow: hidden;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

  @property --sb-glow {
    syntax: '<color>';
    inherits: false;
    initial-value: #002776;
  }
  @keyframes sb-cycle {
    0%, 100% { --sb-glow: #002776; }
    33%      { --sb-glow: #92d400; }
    66%      { --sb-glow: #00a1de; }
  }

  .sb-card {
    position: relative;
    width: 100%;
    padding: 14px 16px 12px;
    border-radius: 10px;
    animation: sb-cycle 6s ease-in-out infinite;
    border: 1px solid var(--sb-glow);
    box-shadow: 0 0 12px color-mix(in srgb, var(--sb-glow) 20%, transparent),
                inset 0 0 20px color-mix(in srgb, var(--sb-glow) 6%, transparent);
    background: SIDEBAR_CARD_BG;
    transition: box-shadow 0.3s ease;
    cursor: default;
  }
  .sb-card:hover {
    box-shadow: 0 0 20px color-mix(in srgb, var(--sb-glow) 40%, transparent),
                0 0 40px color-mix(in srgb, var(--sb-glow) 12%, transparent),
                inset 0 0 30px color-mix(in srgb, var(--sb-glow) 10%, transparent);
  }

  .sb-logo {
    font-size: 1.1rem;
    font-weight: 900;
    color: SIDEBAR_LOGO_COLOR;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }
  .sb-logo .dot { color: #92d400; }

  .sb-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: SIDEBAR_TEXT_COLOR;
    margin-bottom: 2px;
  }
  .sb-caption {
    font-size: 0.75rem;
    color: SIDEBAR_MUTED_COLOR;
  }
</style>
<div class="sb-card">
  <div class="sb-logo">Deloitte<span class="dot">.</span></div>
  <div class="sb-title">WaferGuard ML</div>
  <div class="sb-caption">Wafer Map Defect Classifier</div>
</div>
"""

with st.sidebar:
    # Sidebar hero card with glow effect
    _sb_html = (
        SIDEBAR_HERO_HTML
        .replace("SIDEBAR_CARD_BG", t["bg2"])
        .replace("SIDEBAR_LOGO_COLOR", t["logo_color"])
        .replace("SIDEBAR_TEXT_COLOR", t["text"])
        .replace("SIDEBAR_MUTED_COLOR", t["text_muted"])
    )
    components.html(_sb_html, height=105)
    st.divider()

    model_choice = st.radio(
        "Select Model",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
        help=(
            "Custom CNN is lighter (4.3 MB) and scores higher (F1=0.974). "
            "MobileNetV2 uses transfer learning from ImageNet (28 MB, F1=0.967)."
        ),
    )

    top_n = st.slider("Top-N Predictions", min_value=3, max_value=38, value=TOP_N_DEFAULT)

    st.divider()
    st.markdown(
        '<span style="color: #92d400; font-weight: 700; font-size: 1rem;">Financial Assumptions</span>',
        unsafe_allow_html=True,
    )
    st.slider(
        "Wafers per hour",
        min_value=10,
        max_value=500,
        value=int(FINANCIAL_CONFIG_DEFAULTS["WPH"]),
        step=5,
        key="fin_wph",
    )
    st.slider(
        "Value per wafer ($)",
        min_value=500,
        max_value=20_000,
        value=int(FINANCIAL_CONFIG_DEFAULTS["VALUE_PER_WAFER"]),
        step=100,
        key="fin_value_per_wafer",
    )
    st.slider(
        "Repair time (hours)",
        min_value=1,
        max_value=48,
        value=int(FINANCIAL_CONFIG_DEFAULTS["REPAIR_HOURS"]),
        key="fin_repair_hours",
    )
    st.slider(
        "Planning horizon (days)",
        min_value=7,
        max_value=90,
        value=int(FINANCIAL_CONFIG_DEFAULTS["PLANNING_HORIZON"]),
        key="fin_planning_horizon",
    )
    st.slider(
        "Low-confidence threshold",
        min_value=0.50,
        max_value=0.95,
        value=float(FINANCIAL_CONFIG_DEFAULTS["CONFIDENCE_THRESHOLD"]),
        step=0.01,
        key="fin_conf_threshold",
    )

    st.divider()

    # Model info
    info = MODEL_REGISTRY[model_choice]
    st.markdown(
        f'<span style="color: #92d400; font-weight: 600;">Model:</span> {model_choice}',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="color: #92d400; font-weight: 600; font-size: 0.9rem;">Size:</span> {info["size"]}'
        f' &nbsp;|&nbsp; <span style="color: #92d400; font-weight: 600; font-size: 0.9rem;">Params:</span>'
        f" {info['params']}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="color: #92d400; font-weight: 600; font-size: 0.9rem;">Macro F1:</span> {info["macro_f1"]}',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="color: #92d400; font-weight: 600; font-size: 0.9rem;">Weighted F1:</span> {info["weighted_f1"]}',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="color: #92d400; font-weight: 600; font-size: 0.9rem;">Input:</span> 52 x 52 x 3'
        ' &nbsp;|&nbsp; <span style="color: #92d400; font-weight: 600; font-size: 0.9rem;">Output:</span>'
        " 38 classes",
        unsafe_allow_html=True,
    )
    st.caption(info["description"])

    st.divider()
    st.markdown(
        '<span style="color: #92d400; font-weight: 700; font-size: 1rem;">About</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "ML-powered defect pattern classification for semiconductor wafer maps. "
        "Upload a wafer map image or .npz file to classify defect patterns "
        "across 38 categories."
    )

    st.divider()
    st.toggle("Show hero banner", value=True, key="show_hero")

# ── Load model ───────────────────────────────────────────────────────
with st.spinner(f"Loading {model_choice} model..."):
    model = load_model(model_choice)


# ── Clipboard paste: JS injected into parent document ────────────────
PASTE_JS = """
<script>
(function() {
    // Find the hidden textarea by placeholder text
    function findTextarea() {
        const els = window.parent.document.querySelectorAll('input[placeholder="paste_target"]');
        return els.length > 0 ? els[0] : null;
    }

    // Listen for paste on the entire parent document
    window.parent.document.addEventListener('paste', function(e) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const reader = new FileReader();
                reader.onload = function(ev) {
                    const b64 = ev.target.result.split(',')[1];
                    const input = findTextarea();
                    if (input) {
                        const nativeSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value'
                        ).set;
                        nativeSetter.call(input, b64);
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        // Trigger Streamlit rerun by pressing Enter
                        input.dispatchEvent(new KeyboardEvent('keydown', {
                            key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true
                        }));
                    }
                };
                reader.readAsDataURL(blob);
                e.preventDefault();
                return;
            }
        }
    });
})();
</script>
"""

_HERO_TEMPLATE = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: THEME_BG; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

  @property --glow-color {
    syntax: '<color>';
    inherits: false;
    initial-value: #002776;
  }
  @property --glow-color-2 {
    syntax: '<color>';
    inherits: false;
    initial-value: #00a1de;
  }
  @property --rotate {
    syntax: '<number>';
    inherits: true;
    initial-value: 0;
  }
  @property --bg-x {
    syntax: '<number>';
    inherits: true;
    initial-value: 0;
  }
  @property --bg-y {
    syntax: '<number>';
    inherits: true;
    initial-value: 0;
  }
  @property --bg-size {
    syntax: '<number>';
    inherits: true;
    initial-value: 1;
  }
  @property --glow-blur {
    syntax: '<number>';
    inherits: true;
    initial-value: 6;
  }
  @property --glow-opacity {
    syntax: '<number>';
    inherits: true;
    initial-value: 1;
  }
  @property --glow-scale {
    syntax: '<number>';
    inherits: true;
    initial-value: 1.5;
  }
  @property --glow-radius {
    syntax: '<number>';
    inherits: true;
    initial-value: 100;
  }
  @property --glow-translate-y {
    syntax: '<number>';
    inherits: true;
    initial-value: 0;
  }
  @property --white-shadow {
    syntax: '<number>';
    inherits: true;
    initial-value: 0;
  }

  @keyframes palette-cycle {
    0%, 100% { --glow-color: #002776; --glow-color-2: #00a1de; }
    33%      { --glow-color: #92d400; --glow-color-2: #002776; }
    66%      { --glow-color: #00a1de; --glow-color-2: #92d400; }
  }

  @keyframes rotate-bg {
    0%   { --bg-x: 0;   --bg-y: 0; }
    25%  { --bg-x: 100; --bg-y: 0; }
    50%  { --bg-x: 100; --bg-y: 100; }
    75%  { --bg-x: 0;   --bg-y: 100; }
    100% { --bg-x: 0;   --bg-y: 0; }
  }

  @keyframes orbit {
    from { --rotate: -70; --glow-translate-y: -65; }
    25%  { --glow-translate-y: -65; }
    50%  { --glow-translate-y: -65; }
    75%  { --glow-translate-y: -65; }
    to   { --rotate: 290; --glow-translate-y: -65; }
  }

  @keyframes shadow-pulse {
    0%, 24%, 46%, 73%, 96% { --white-shadow: 0.5; }
    12%, 28%, 41%, 63%, 75%, 82%, 98% { --white-shadow: 2.5; }
    6%, 32%, 57% { --white-shadow: 1.3; }
    18%, 52%, 88% { --white-shadow: 3.5; }
  }

  .glow-container {
    --card-color: THEME_CARD;
    --card-radius: 20px;
    --border-width: 2px;
    --animation-speed: 6s;
    --interaction-speed: 0.55s;
    --glow-rotate-unit: 1deg;

    width: calc(100% - 16px);
    height: 300px;
    margin: 8px auto;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 2;
    border-radius: var(--card-radius);
    cursor: pointer;
    animation: palette-cycle 6s ease-in-out infinite;
  }

  .glow-container:before,
  .glow-container:after {
    content: "";
    display: block;
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: var(--card-radius);
  }

  .glow-content {
    position: absolute;
    inset: 0;
    background: var(--card-color);
    border-radius: calc(var(--card-radius) * 0.85);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    text-align: center;
    gap: 12px;
  }

  .glow-content:before {
    content: "";
    display: block;
    position: absolute;
    width: calc(100% + var(--border-width));
    height: calc(100% + var(--border-width));
    border-radius: calc(var(--card-radius) * 0.85);
    box-shadow: 0 0 20px rgba(0,0,0,0.8);
    mix-blend-mode: color-burn;
    z-index: -1;
    background: hsl(0deg 0% 12%) radial-gradient(
      30% 40% at calc(var(--bg-x) * 1%) calc(var(--bg-y) * 1%),
      var(--glow-color) 0%,
      color-mix(in srgb, var(--glow-color) 60%, transparent) calc(30% * var(--bg-size)),
      color-mix(in srgb, var(--glow-color-2) 40%, transparent) calc(60% * var(--bg-size)),
      transparent 100%
    );
    animation: rotate-bg var(--animation-speed) linear infinite;
    transition: --bg-size var(--interaction-speed) ease;
  }

  .glow {
    --glow-translate-y: 0;
    display: block;
    position: absolute;
    width: 80px;
    height: 80px;
    animation: orbit var(--animation-speed) linear infinite;
    transform: rotateZ(calc(var(--rotate) * var(--glow-rotate-unit)));
    transform-origin: center;
    border-radius: calc(var(--glow-radius) * 10vw);
  }

  .glow:after {
    content: "";
    display: block;
    z-index: -2;
    filter: blur(calc(var(--glow-blur) * 10px));
    width: 130%;
    height: 130%;
    left: -15%;
    top: -15%;
    background: var(--glow-color);
    position: relative;
    border-radius: calc(var(--glow-radius) * 10vw);
    transform: scaleY(calc(var(--glow-scale) / 1.1))
               scaleX(calc(var(--glow-scale) * 1.2))
               translateY(calc(var(--glow-translate-y) * 1%));
    opacity: var(--glow-opacity);
  }

  .glow-container:hover .glow-content {
    mix-blend-mode: darken;
    box-shadow: 0 0 calc(var(--white-shadow) * 1vw) calc(var(--white-shadow) * 0.15vw) rgba(255, 255, 255, 0.2);
    animation: shadow-pulse calc(var(--animation-speed) * 2) linear infinite;
  }

  .glow-container:hover .glow-content:before {
    --bg-size: 15;
    animation-play-state: paused;
    transition: --bg-size var(--interaction-speed) ease;
  }

  .glow-container:hover .glow {
    --glow-blur: 1.5;
    --glow-opacity: 0.6;
    --glow-scale: 2.5;
    --glow-radius: 0;
    --rotate: 900;
    --glow-rotate-unit: 0;
    animation-play-state: paused;
  }

  .glow-container:hover .glow:after {
    --glow-translate-y: 0;
    animation-play-state: paused;
    transition: --glow-translate-y 0s ease, --glow-blur 0.05s ease,
                --glow-opacity 0.05s ease, --glow-scale 0.05s ease,
                --glow-radius 0.05s ease;
  }

  .hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: THEME_TEXT;
    letter-spacing: 0.04em;
    line-height: 1.1;
    z-index: 1;
  }

  .hero-title .accent {
    color: #92d400;
  }

  .hero-subtitle {
    font-size: 1.1rem;
    color: #00a1de;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    z-index: 1;
  }

  .hero-tagline {
    font-size: 0.85rem;
    color: THEME_TEXT_MUTED;
    z-index: 1;
    margin-top: 4px;
  }
</style>

<div class="glow-container">
  <span class="glow"></span>
  <div class="glow-content">
    <div class="hero-title">Wafer<span class="accent">Guard</span> ML</div>
    <div class="hero-subtitle">ML-Powered Wafer Defect Classification</div>
    <div class="hero-tagline">Semiconductor wafer map analysis across 38 defect patterns</div>
  </div>
</div>
"""


def build_hero_html(theme):
    """Build hero HTML string adapted to the current theme."""
    return (
        _HERO_TEMPLATE
        .replace("THEME_BG", theme["bg"])
        .replace("THEME_CARD", theme["card"])
        .replace("THEME_TEXT_MUTED", theme["text_muted"])
        .replace("THEME_TEXT", theme["text"])
    )


def _financial_config_from_state(batch_id: str) -> dict:
    """Build financial config from sidebar controls."""
    return {
        "BATCH_ID": batch_id,
        "WPH": st.session_state["fin_wph"],
        "VALUE_PER_WAFER": st.session_state["fin_value_per_wafer"],
        "REPAIR_HOURS": st.session_state["fin_repair_hours"],
        "PLANNING_HORIZON": st.session_state["fin_planning_horizon"],
        "CONFIDENCE_THRESHOLD": st.session_state["fin_conf_threshold"],
    }


def render_leadership_panel(results: list[dict], batch_id: str, section_title: str, lot_level: bool = True) -> None:
    """Render financial KPI, selected-pattern insights, and action recommendations."""
    if not results:
        return

    base_config = _financial_config_from_state(batch_id)
    threshold = float(st.session_state["fin_conf_threshold"])
    focus_pattern = st.session_state.get("fin_focus_pattern", "Donut")
    analysis_view = analyze_results(
        results,
        base_config,
        threshold,
        use_pattern_scenario=False,
        focus_pattern=focus_pattern,
    )

    st.subheader(section_title)
    if not lot_level:
        st.caption("Single-wafer values are indicative estimates. Leadership KPIs should be interpreted at lot level.")

    render_kpi_cards(analysis_view["summary_payload"])
    st.markdown(f"**{focus_pattern}-Related Insights**")
    render_pattern_card(focus_pattern, analysis_view["pattern_metrics"])

    st.markdown("**Immediate Recommendation**")
    st.success(analysis_view["executive_recommendation"])

    with st.expander("Action Prioritization", expanded=True):
        render_action_table(analysis_view["df_actions"], top_n=5)

    with st.expander("Base Anomaly Breakdown", expanded=False):
        render_base_anomaly_chart(analysis_view["df_anomaly"])

    with st.expander("Top Financially Impactful Patterns", expanded=False):
        if analysis_view["df_financial"].empty:
            st.info("No defect financial patterns to display.")
        else:
            st.dataframe(
                analysis_view["df_financial"]
                .head(10)[
                    [
                        "pattern_name",
                        "count",
                        "batch_pct",
                        "risk_level",
                        "weighted_daily_loss",
                        "break_even_days",
                    ]
                ]
                .rename(
                    columns={
                        "pattern_name": "Pattern",
                        "count": "Count",
                        "batch_pct": "Batch %",
                        "risk_level": "Risk",
                        "weighted_daily_loss": "Daily Loss",
                        "break_even_days": "Break-even (days)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


# ── Display prediction results ───────────────────────────────────────
def display_prediction(raw, result, elapsed, top_n):
    """Display prediction results for a single wafer."""
    st.caption(f"Inference time: {elapsed:.2f}s")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Wafer Map")
        fig_wafer = render_wafer_map(raw[0])
        st.pyplot(fig_wafer)
        plt.close(fig_wafer)

    with col2:
        st.subheader("Prediction")
        m1, m2 = st.columns(2)
        m1.metric("Defect Pattern", result["pattern_name"])
        m2.metric("Confidence", f"{result['confidence']:.1%}")

        st.subheader("Top Predictions")
        fig_chart = render_confidence_chart(result["probabilities"], top_n)
        st.pyplot(fig_chart)
        plt.close(fig_chart)

    st.subheader("Defect Description")
    st.info(get_description(result["pattern_name"]))


# ── Hero banner ─────────────────────────────────────────────────────
if st.session_state.get("show_hero", True):
    components.html(build_hero_html(t), height=340)

# ── Navigation cards ─────────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "single"

NAV_CARDS = [
    ("single", "Single Wafer", "Upload and classify a single wafer map."),
    ("batch", "Batch Upload", "Process multiple wafer maps at once."),
    ("sample", "Sample Data", "Generate and analyze random samples from the dataset."),
]

# CSS for card-style nav buttons
st.markdown(
    """<style>
    /* Card base — shared by both active and inactive */
    .nav-cards button {
        background: #1a1d26 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #FAFAFA !important;
        text-align: left !important;
        padding: 12px 20px 28px !important;
        transition: all 0.25s ease !important;
        position: relative !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        overflow: visible !important;
    }

    /* Hover */
    .nav-cards button:hover {
        border-color: rgba(255,255,255,0.5) !important;
        background: #22252e !important;
    }

    /* Corner brackets on hover */
    .nav-cards button:hover::before {
        content: '';
        position: absolute;
        top: 6px; left: 8px;
        width: 12px; height: 12px;
        border-top: 2px solid rgba(255,255,255,0.6);
        border-left: 2px solid rgba(255,255,255,0.6);
        pointer-events: none;
    }

    /* Arrow on hover */
    .nav-cards button:hover > span::after {
        content: ' \\2192';
        position: absolute;
        right: 16px; top: 14px;
        font-size: 1.2rem;
        font-weight: 400;
        color: rgba(255,255,255,0.6);
    }

    /* Bottom glow line */
    .nav-cards div[data-testid="stButton"]::after {
        content: '';
        display: block;
        height: 2px;
        margin-top: -2px;
        border-radius: 0 0 8px 8px;
        background: linear-gradient(90deg, transparent, #92d400, transparent);
        opacity: 0;
        transition: opacity 0.25s ease;
    }
    .nav-cards div[data-testid="stButton"]:hover::after {
        opacity: 1;
    }

    /* Active card */
    .nav-cards button[kind="primary"] {
        background: #1e2230 !important;
        border-color: #92d400 !important;
    }
    .nav-cards button[kind="primary"]:hover {
        border-color: #92d400 !important;
        background: #252a38 !important;
    }
    .nav-cards div:has(button[kind="primary"])::after {
        opacity: 1;
    }

    /* Description text via ::after on each button */
    .nav-cards div[data-testid="stButton"]:nth-of-type(1) button::after {
        content: 'Upload and classify a single wafer map.';
    }
    .nav-cards div[data-testid="stButton"]:nth-of-type(2) button::after {
        content: 'Process multiple wafer maps at once.';
    }
    .nav-cards div[data-testid="stButton"]:nth-of-type(3) button::after {
        content: 'Generate and analyze random samples from the dataset.';
    }
    .nav-cards button::after {
        display: block;
        font-size: 0.78rem;
        font-weight: 400;
        color: rgba(250,250,250,0.4);
        margin-top: 3px;
    }

    /* Tighten gap between cards */
    .nav-cards > div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
    </style>""",
    unsafe_allow_html=True,
)


def _set_tab(tab_key):
    st.session_state.active_tab = tab_key


with st.container():
    st.markdown('<div class="nav-cards">', unsafe_allow_html=True)
    for key, title, _desc in NAV_CARDS:
        is_active = st.session_state.active_tab == key
        st.button(
            title,
            key=f"nav_{key}",
            on_click=_set_tab,
            args=(key,),
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Main content ─────────────────────────────────────────────────────

# ── Single Wafer ────────────────────────────────────────────────────
if st.session_state.active_tab == "single":
    input_method = st.radio(
        "Input method",
        ["Upload file", "Paste from clipboard"],
        horizontal=True,
        key="input_method",
    )

    image_bytes = None
    image_name = None

    if input_method == "Upload file":
        uploaded = st.file_uploader(
            "Upload a wafer map",
            type=["npz", "png", "jpg", "jpeg"],
            key="single_upload",
            help="PNG/JPG image or .npz file containing 52x52 wafer map data.",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            image_name = uploaded.name

    else:
        st.info("Press **Ctrl+V** / **Cmd+V** anywhere on this page to paste an image from your clipboard.")

        # Inject JS listener into parent document
        components.html(PASTE_JS, height=0)

        # Hidden input that JS writes base64 data into
        paste_data = st.text_input(
            "paste_target",
            key="paste_b64_input",
            label_visibility="collapsed",
            placeholder="paste_target",
        )

        if paste_data:
            try:
                image_bytes = base64.b64decode(paste_data)
                image_name = "clipboard.png"
                st.success("Image captured from clipboard!")
            except Exception:
                st.error("Failed to decode clipboard data. Try uploading the file instead.")

    # ── Process input ────────────────────────────────────────────────
    if image_bytes is not None and image_name is not None:
        try:
            t0 = time.perf_counter()
            raw = parse_upload(image_name, image_bytes)
            prepared = prepare_for_model(raw)
            result = predict_single(model, prepared[:1])
            elapsed = time.perf_counter() - t0
        except (ValueError, Exception) as e:
            st.error(f"Failed to process file: {e}")
        else:
            display_prediction(raw, result, elapsed, top_n)
            render_leadership_panel(
                [result],
                batch_id=f"SINGLE_{int(time.time())}",
                section_title="Financial Snapshot",
                lot_level=False,
            )

# ── Batch Upload ────────────────────────────────────────────────────
elif st.session_state.active_tab == "batch":
    uploaded_files = st.file_uploader(
        "Upload wafer maps",
        type=["npz", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
        help=f"Upload multiple files. NPZ files with multiple wafers are expanded. Max {MAX_BATCH_SIZE} wafers.",
    )

    if uploaded_files:
        import io

        import numpy as np

        # Parse all files
        all_raw = []
        errors = []
        for f in uploaded_files:
            try:
                raw = parse_upload(f.name, f.getvalue())
                all_raw.append(raw)
            except (ValueError, Exception) as e:
                errors.append(f"{f.name}: {e}")

        if errors:
            for err in errors:
                st.warning(err)

        if all_raw:
            combined = np.concatenate(all_raw, axis=0)

            if len(combined) > MAX_BATCH_SIZE:
                st.warning(f"Batch truncated to {MAX_BATCH_SIZE} wafers (uploaded {len(combined)}).")
                combined = combined[:MAX_BATCH_SIZE]

            with st.spinner(f"Classifying {len(combined)} wafer(s)..."):
                t0 = time.perf_counter()
                prepared = prepare_for_model(combined)
                results = predict_batch(model, prepared)
                elapsed = time.perf_counter() - t0

            # Download uploaded batch as NPZ
            buf = io.BytesIO()
            np.savez_compressed(buf, arr_0=combined)
            st.download_button(
                label=f"Download batch ({len(combined)} wafers, .npz)",
                data=buf.getvalue(),
                file_name="wafer_batch.npz",
                mime="application/octet-stream",
            )

            # Summary
            st.subheader(f"Results ({len(results)} wafers)")
            st.caption(f"Batch inference time: {elapsed:.2f}s ({elapsed / len(results):.3f}s per wafer)")
            df = build_results_dataframe(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            render_leadership_panel(
                results,
                batch_id=f"BATCH_{int(time.time())}",
                section_title="Leadership Decision Support",
                lot_level=True,
            )

            # Expandable details per wafer
            st.subheader("Details")
            for r in results:
                with st.expander(f"Wafer #{r['index'] + 1} — {r['pattern_name']} ({r['confidence']:.1%})"):
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        fig_w = render_wafer_map(combined[r["index"]])
                        st.pyplot(fig_w)
                        plt.close(fig_w)
                    with c2:
                        fig_c = render_confidence_chart(r["probabilities"], top_n)
                        st.pyplot(fig_c)
                        plt.close(fig_c)
                    st.info(get_description(r["pattern_name"]))

# ── Sample Data ─────────────────────────────────────────────────────
elif st.session_state.active_tab == "sample":
    import io

    import numpy as np

    from app.labels import ID_TO_PATTERN

    st.subheader("Random Sample Generator")
    st.caption("Generate random wafer map samples from the dataset for demo and testing purposes.")

    @st.cache_resource
    def _ensure_runtime_dataset() -> Path:
        """Download dataset on first run in Streamlit Cloud if file is not present."""
        username = st.secrets.get("KAGGLE_USERNAME") if "KAGGLE_USERNAME" in st.secrets else None
        key = st.secrets.get("KAGGLE_KEY") if "KAGGLE_KEY" in st.secrets else None

        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key

        return ensure_wafer_dataset(
            dataset_file=DATASET_PATH,
            dataset_slug=DATASET_SLUG,
            username=username,
            key=key,
        )

    dataset_error = None
    dataset_file = DATASET_PATH
    if ENABLE_RUNTIME_DATASET_BOOTSTRAP:
        try:
            with st.spinner("Preparing wafer dataset..."):
                dataset_file = _ensure_runtime_dataset()
        except Exception as exc:
            dataset_error = str(exc)
    elif not DATASET_PATH.exists():
        dataset_error = (
            "Dataset not found and runtime bootstrap is disabled. "
            "Set ENABLE_RUNTIME_DATASET_BOOTSTRAP=1 or download it locally."
        )

    if dataset_error:
        st.error("Unable to load Wafer_Map_Datasets.npz.")
        st.code(dataset_error)
        st.info(
            "For Streamlit Cloud, set KAGGLE_USERNAME and KAGGLE_KEY in Secrets. "
            "For local runs, execute: poetry run python data/download_kaggle_data.py"
        )
    else:

        @st.cache_data
        def load_dataset(dataset_npz_path: str):
            """Load the full wafer map dataset and compute pattern labels."""
            data = np.load(dataset_npz_path, allow_pickle=True)
            images = data["arr_0"]  # (N, 52, 52)
            labels = data["arr_1"]  # (N, 8) multi-label

            # Map multi-label rows to pattern IDs (0-37)
            label_tuples = [tuple(row) for row in labels]
            unique_patterns = sorted(set(label_tuples))
            pattern_map = {p: i for i, p in enumerate(unique_patterns)}
            pattern_ids = np.array([pattern_map[t] for t in label_tuples])

            return images, pattern_ids

        images, pattern_ids = load_dataset(str(dataset_file))

        selected_patterns = None

        # Batch size
        sample_size = st.slider("Number of samples", min_value=800, max_value=6000, value=1000)
        sample_mode = st.radio(
            "Sample mode",
            ["Random samples", "Base-pattern demo scenario"],
            horizontal=True,
            help="Use base-pattern mode to simulate a realistic lot with mostly Normal wafers and elevated selected-pattern defects.",
        )

        focus_pattern = st.session_state.get("fin_focus_pattern", "Donut")

        if sample_mode == "Base-pattern demo scenario":
            current_pattern = st.session_state.get("fin_focus_pattern", "Donut")
            if current_pattern not in BASE_PATTERNS:
                current_pattern = "Donut"
            selected_index = list(BASE_PATTERNS).index(current_pattern)

            focus_pattern = st.selectbox(
                "Focus base defect pattern",
                options=list(BASE_PATTERNS),
                index=selected_index,
                key="fin_focus_pattern",
                help="Used for targeted insights and demo-scenario composition.",
            )
            st.caption(
                f"Target composition: {BASE_PATTERN_SCENARIO_DEFAULTS['NORMAL_SHARE']:.0%} Normal, "
                f"{BASE_PATTERN_SCENARIO_DEFAULTS['TARGET_SHARE']:.0%} {focus_pattern}-related, "
                f"{1 - BASE_PATTERN_SCENARIO_DEFAULTS['NORMAL_SHARE'] - BASE_PATTERN_SCENARIO_DEFAULTS['TARGET_SHARE']:.0%} Other"
            )
        else:
            # Pattern filter is only relevant for random sampling mode.
            all_pattern_names = [ID_TO_PATTERN[i] for i in range(len(ID_TO_PATTERN))]
            selected_patterns = st.multiselect(
                "Select defect patterns",
                options=all_pattern_names,
                default=None,
                help="Leave empty to sample from all patterns.",
                placeholder="All patterns",
            )

        if st.button("Generate Samples", type="primary"):
            # Filter by selected patterns for Random samples mode.
            if sample_mode == "Random samples" and selected_patterns:
                from app.labels import PATTERN_TO_ID

                selected_ids = [PATTERN_TO_ID[p] for p in selected_patterns]
                mask = np.isin(pattern_ids, selected_ids)
                pool = images[mask]
            else:
                pool = images

            if len(pool) == 0:
                st.error("No samples found for the selected patterns.")
            else:
                actual_size = min(sample_size, len(pool))
                rng = np.random.default_rng()

                if sample_mode == "Base-pattern demo scenario":
                    # Build defect families from full dataset to create a controlled scenario lot.
                    pattern_names = np.array([ID_TO_PATTERN[int(pid)] for pid in pattern_ids])
                    normal_idx = np.where(pattern_names == "Normal")[0]
                    target_mask = np.array([is_pattern_name_related(name, focus_pattern) for name in pattern_names])
                    target_idx = np.where(target_mask)[0]
                    other_idx = np.where((pattern_names != "Normal") & (~target_mask))[0]

                    n_normal = round(actual_size * BASE_PATTERN_SCENARIO_DEFAULTS["NORMAL_SHARE"])
                    n_target = round(actual_size * BASE_PATTERN_SCENARIO_DEFAULTS["TARGET_SHARE"])
                    n_other = max(0, actual_size - n_normal - n_target)
                    n_normal += actual_size - (n_normal + n_target + n_other)

                    def _draw(source_idx, n):
                        if n <= 0:
                            return np.array([], dtype=int)
                        base = source_idx if len(source_idx) > 0 else np.arange(len(images))
                        return rng.choice(base, size=n, replace=len(base) < n)

                    selected_idx = np.concatenate([
                        _draw(normal_idx, n_normal),
                        _draw(target_idx, n_target),
                        _draw(other_idx, n_other),
                    ])
                    rng.shuffle(selected_idx)
                    sampled = images[selected_idx]
                    st.session_state["sample_demo_mix"] = {
                        "normal": int(n_normal),
                        "target": int(n_target),
                        "other": int(n_other),
                        "focus_pattern": focus_pattern,
                    }
                else:
                    indices = rng.choice(len(pool), size=actual_size, replace=False)
                    sampled = pool[indices]
                    st.session_state["sample_demo_mix"] = None

                # Store in session for display
                st.session_state["generated_samples"] = sampled
                st.session_state["sample_mode"] = sample_mode

                if sample_mode == "Base-pattern demo scenario":
                    st.success(f"Generated {actual_size} {focus_pattern}-dominant demo wafers.")
                else:
                    st.success(f"Generated {actual_size} random wafer map samples.")

        # Display and download generated samples
        if "generated_samples" in st.session_state:
            sampled = st.session_state["generated_samples"]

            # Download button
            buf = io.BytesIO()
            np.savez_compressed(buf, arr_0=sampled)
            st.download_button(
                label=f"Download samples ({len(sampled)} wafers, .npz)",
                data=buf.getvalue(),
                file_name="wafer_samples.npz",
                mime="application/octet-stream",
            )

            # Classify samples
            with st.spinner(f"Classifying {len(sampled)} sample(s)..."):
                t0 = time.perf_counter()
                prepared = prepare_for_model(sampled)
                results = predict_batch(model, prepared)
                elapsed = time.perf_counter() - t0

            st.subheader(f"Results ({len(results)} wafers)")
            st.caption(f"Inference time: {elapsed:.2f}s ({elapsed / len(results):.3f}s per wafer)")
            df = build_results_dataframe(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            if st.session_state.get("sample_mode") == "Base-pattern demo scenario":
                mix = st.session_state.get("sample_demo_mix") or {
                    "normal": 0,
                    "target": 0,
                    "other": 0,
                    "focus_pattern": focus_pattern,
                }
                st.info(
                    f"{mix['focus_pattern']} demo mode: N/T/O wafers = {mix['normal']}/{mix['target']}/{mix['other']}"
                )

            render_leadership_panel(
                results,
                batch_id=f"SAMPLE_{int(time.time())}",
                section_title="Leadership Decision Support",
                lot_level=True,
            )

            # Expandable details
            st.subheader("Details")
            for r in results:
                with st.expander(f"Wafer #{r['index'] + 1} — {r['pattern_name']} ({r['confidence']:.1%})"):
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        fig_w = render_wafer_map(sampled[r["index"]])
                        st.pyplot(fig_w)
                        plt.close(fig_w)
                    with c2:
                        fig_c = render_confidence_chart(r["probabilities"], top_n)
                        st.pyplot(fig_c)
                        plt.close(fig_c)
                    st.info(get_description(r["pattern_name"]))
