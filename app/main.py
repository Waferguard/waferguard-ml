"""WaferGuard ML — Streamlit App."""

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from app.config import MAX_BATCH_SIZE, MODEL_REGISTRY, TOP_N_DEFAULT
from app.labels import get_description
from app.model_utils import load_model, predict_batch, predict_single
from app.preprocessing import parse_upload, prepare_for_model
from app.visualization import build_results_dataframe, render_confidence_chart, render_wafer_map

st.set_page_config(
    page_title="WaferGuard ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("WaferGuard ML")
    st.caption("Wafer Map Defect Classifier")
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

    # Model info
    info = MODEL_REGISTRY[model_choice]
    st.markdown(f"**Model:** {model_choice}")
    st.markdown(f"**Size:** {info['size']}  |  **Params:** {info['params']}")
    st.markdown(f"**Macro F1:** {info['macro_f1']}  |  **Weighted F1:** {info['weighted_f1']}")
    st.markdown("**Input:** 52 x 52 x 3  |  **Output:** 38 classes")
    st.caption(info["description"])

    st.divider()
    st.markdown("**About**")
    st.caption(
        "AI-powered defect pattern classification for semiconductor wafer maps. "
        "Upload a wafer map image or .npz file to classify defect patterns "
        "across 38 categories."
    )

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


# ── Main content ─────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Wafer", "Batch Upload"])

# ── Single Wafer Tab ─────────────────────────────────────────────────
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

# ── Batch Upload Tab ─────────────────────────────────────────────────
with tab_batch:
    uploaded_files = st.file_uploader(
        "Upload wafer maps",
        type=["npz", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
        help=f"Upload multiple files. NPZ files with multiple wafers are expanded. Max {MAX_BATCH_SIZE} wafers.",
    )

    if uploaded_files:
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

            # Summary
            st.subheader(f"Results ({len(results)} wafers)")
            st.caption(f"Batch inference time: {elapsed:.2f}s ({elapsed / len(results):.3f}s per wafer)")
            df = build_results_dataframe(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

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
