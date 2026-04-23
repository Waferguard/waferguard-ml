"""WaferGuard ML — Streamlit App."""

import base64
import logging
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
    render_all_anomaly_treemap,
    render_combinations_sunburst,
    render_confidence_chart,
    render_kpi_cards,
    render_pattern_card,
    render_wafer_map,
)
from data.download_kaggle_data import ensure_wafer_dataset

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

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

  .sb-card {
    position: relative;
    width: 100%;
    padding: 8px 0;
  }

  .sb-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: SIDEBAR_TEXT_COLOR;
    margin-bottom: 2px;
  }
  .sb-caption {
    font-size: 0.75rem;
    color: SIDEBAR_MUTED_COLOR;
  }
</style>
<div class="sb-card">
  <div class="sb-title">WaferGuard ML</div>
  <div class="sb-caption">Wafer Map Defect Classifier</div>
</div>
"""

with st.sidebar:
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

# ── Load model ───────────────────────────────────────────────────────
with st.spinner(f"Loading {model_choice} model..."):
    model = load_model(model_choice)


def _streamlit_secret_value(secret_name: str) -> str | None:
    """Return a Streamlit secret value if present without exposing it in logs."""
    try:
        if secret_name in st.secrets:
            value = st.secrets.get(secret_name)
            return str(value) if value is not None else None
    except Exception:
        logger.exception("Unable to read Streamlit secret %s", secret_name)
    return None


@st.cache_resource
def _bootstrap_dataset_at_startup() -> Path:
    """Ensure the wafer dataset is present as soon as the app starts."""
    username = _streamlit_secret_value("KAGGLE_USERNAME")
    key = _streamlit_secret_value("KAGGLE_KEY")

    logger.info(
        "App startup dataset bootstrap (enabled=%s, path=%s, slug=%s, has_username=%s, has_key=%s, cwd=%s)",
        ENABLE_RUNTIME_DATASET_BOOTSTRAP,
        DATASET_PATH,
        DATASET_SLUG,
        bool(username),
        bool(key),
        Path.cwd(),
    )

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        logger.info("Kaggle credentials loaded from Streamlit secrets.")
    else:
        logger.warning("Kaggle credentials are missing from Streamlit secrets.")

    return ensure_wafer_dataset(
        dataset_file=DATASET_PATH,
        dataset_slug=DATASET_SLUG,
        username=username,
        key=key,
    )


DATASET_BOOTSTRAP_ERROR: str | None = None
DATASET_READY_PATH: Path = DATASET_PATH

if ENABLE_RUNTIME_DATASET_BOOTSTRAP:
    try:
        logger.info("Starting eager dataset bootstrap at app startup.")
        DATASET_READY_PATH = _bootstrap_dataset_at_startup()
        logger.info("Eager dataset bootstrap complete: %s", DATASET_READY_PATH)
    except Exception:
        logger.exception("Eager dataset bootstrap failed")
        DATASET_BOOTSTRAP_ERROR = f"Dataset bootstrap failed for {DATASET_PATH}. See deployment logs for details."
elif not DATASET_PATH.exists():
    logger.warning("Runtime dataset bootstrap is disabled and the dataset file is missing.")
    DATASET_BOOTSTRAP_ERROR = (
        "Dataset not found and runtime bootstrap is disabled. "
        "Set ENABLE_RUNTIME_DATASET_BOOTSTRAP=1 or download it locally."
    )


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

  .header-container {
    width: 100%;
    margin: 8px auto;
    padding: 16px 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 6px;
  }

  .hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: THEME_TEXT;
    letter-spacing: 0.02em;
    line-height: 1.1;
  }

  .hero-title .accent {
    color: #92d400;
  }

  .hero-subtitle {
    font-size: 0.95rem;
    color: #00a1de;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .hero-tagline {
    font-size: 0.85rem;
    color: THEME_TEXT_MUTED;
    margin-top: 2px;
  }

  .main-logo {
    font-size: 1.2rem;
    font-weight: 800;
    color: THEME_LOGO_COLOR;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
  }
  .main-logo .dot { color: #92d400; }
</style>

<div class="header-container">
  <div class="main-logo">Deloitte<span class="dot">.</span></div>
  <div class="hero-title">Wafer<span class="accent">Guard</span> ML</div>
  <div class="hero-subtitle">ML-Powered Wafer Defect Classification</div>
  <div class="hero-tagline">Semiconductor wafer map analysis across 38 defect patterns</div>
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
        .replace("THEME_LOGO_COLOR", theme["logo_color"])
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

    with st.expander("All Anomalies Treemap", expanded=False):
        render_all_anomaly_treemap(analysis_view["df_anomaly"])

    with st.expander("Defect Combinations Sunburst", expanded=False):
        render_combinations_sunburst(analysis_view["df_batch"])

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
    components.html(build_hero_html(t), height=180)



# ── Main content ─────────────────────────────────────────────────────
tab_single, tab_batch, tab_sample = st.tabs(["Single Wafer", "Batch Upload", "Sample Data"])

# ── Single Wafer ────────────────────────────────────────────────────
with tab_single:
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
with tab_batch:
    uploaded_files = st.file_uploader(
        "Upload wafer maps",
        type=["npz", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
        help="Upload multiple files. NPZ files with multiple wafers are expanded.",
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

            render_leadership_panel(
                results,
                batch_id=f"BATCH_{int(time.time())}",
                section_title="Leadership Decision Support",
                lot_level=True,
            )

            # Expandable details per pattern
            st.subheader("Details")
            st.caption(f"Batch inference time: {elapsed:.2f}s ({elapsed / len(results):.3f}s per wafer)")
            
            from collections import defaultdict
            pattern_groups = defaultdict(list)
            for r in results:
                pattern_groups[r['pattern_name']].append(r)
                
            for pattern_name, items in sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True):
                count = len(items)
                r = items[0]  # Show the first wafer as representative
                with st.expander(f"{pattern_name} ({count} found) — Example Wafer #{r['index'] + 1}"):
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        fig_w = render_wafer_map(combined[r["index"]])
                        st.pyplot(fig_w)
                        plt.close(fig_w)
                    with c2:
                        fig_c = render_confidence_chart(r["probabilities"], top_n)
                        st.pyplot(fig_c)
                        plt.close(fig_c)
                    st.info(get_description(pattern_name))

# ── Sample Data ─────────────────────────────────────────────────────
with tab_sample:
    import io

    import numpy as np

    from app.labels import ID_TO_PATTERN

    st.subheader("Random Sample Generator")
    st.caption("Generate random wafer map samples from the dataset for demo and testing purposes.")
    dataset_error = DATASET_BOOTSTRAP_ERROR
    dataset_file = DATASET_READY_PATH

    if dataset_error:
        st.error("Unable to load Wafer_Map_Datasets.npz.")
        st.code(dataset_error)
        st.caption(f"Dataset path checked: {DATASET_PATH}")
        st.info(
            "For Streamlit Cloud, set KAGGLE_USERNAME and KAGGLE_KEY in Secrets. "
            "For local runs, execute: poetry run python data/download_kaggle_data.py"
        )
    else:
        if dataset_file != DATASET_PATH:
            logger.info("Sample tab using bootstrapped dataset path %s", dataset_file)

        @st.cache_data
        def load_dataset(dataset_npz_path: str):
            """Load the full wafer map dataset and compute pattern labels."""
            logger.info("Loading wafer dataset from %s", dataset_npz_path)
            data = np.load(dataset_npz_path, allow_pickle=True)
            images = data["arr_0"]  # (N, 52, 52)
            labels = data["arr_1"]  # (N, 8) multi-label
            logger.info("Loaded dataset arrays: images=%s labels=%s", images.shape, labels.shape)

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
            ["Base-pattern demo scenario", "Random samples"],
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

            render_leadership_panel(
                results,
                batch_id=f"SAMPLE_{int(time.time())}",
                section_title="Leadership Decision Support",
                lot_level=True,
            )

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

            # Expandable details per pattern
            st.subheader("Details")
            st.caption(f"Inference time: {elapsed:.2f}s ({elapsed / len(results):.3f}s per wafer)")
            
            from collections import defaultdict
            pattern_groups = defaultdict(list)
            for r in results:
                pattern_groups[r['pattern_name']].append(r)
                
            for pattern_name, items in sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True):
                count = len(items)
                r = items[0]  # Show the first wafer as representative
                with st.expander(f"{pattern_name} ({count} found) — Example Wafer #{r['index'] + 1}"):
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        fig_w = render_wafer_map(sampled[r["index"]])
                        st.pyplot(fig_w)
                        plt.close(fig_w)
                    with c2:
                        fig_c = render_confidence_chart(r["probabilities"], top_n)
                        st.pyplot(fig_c)
                        plt.close(fig_c)
                    st.info(get_description(pattern_name))
