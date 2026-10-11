import os
import html
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from image_utils import validate_and_load_image
from detector import load_yolo_model, run_inference, highlight_specific_object, image_to_png_bytes

# --------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Aero Sentinel — Drone Obstacle Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# YOLOv8 Model Resource Caching
# --------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "yolov8n.pt"

@st.cache_resource(show_spinner=False)
def get_cached_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load and cache the YOLOv8 model instance across Streamlit reruns.
    Downloads the model weights automatically on first call if not present.
    """
    try:
        model = load_yolo_model(model_name)
        return model, "Ready", None
    except Exception as e:
        return None, "Error", str(e)

# --------------------------------------------------------------------------
# Load Custom Stylesheet (style.css)
# --------------------------------------------------------------------------
def load_stylesheet(css_path="style.css"):
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_stylesheet()

# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 40

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "detection_result" not in st.session_state:
    st.session_state.detection_result = None

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

def set_page(page_name):
    st.session_state.active_page = page_name

# --------------------------------------------------------------------------
# Top Avionics Bar
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-avionics-bar">
        <div>
            <span class="avionics-badge">
                <span class="avionics-indicator"></span> AVIONICS SYSTEM: READY
            </span>
            &nbsp;|&nbsp; UAV IDENT: SENTINEL-ALPHA-01 &nbsp;|&nbsp; FREQ: 5.8 GHz
        </div>
        <div>STATION: GROUND COMMAND // PROTOCOL: STANAG-4586</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Navigation Header
# --------------------------------------------------------------------------
col_brand, col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([4, 1.3, 1.8, 1.3, 1.1])

with col_brand:
    st.markdown(
        """
        <div class="brand-wrapper">
            <div class="brand-radar-icon">⌖</div>
            <div>
                <div class="brand-title-text">AERO SENTINEL</div>
                <div class="brand-sub-tag">DRONE OBSTACLE DETECTION // YOLOv8</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav1:
    btn_type = "primary" if st.session_state.active_page == "Home" else "secondary"
    if st.button("Home", key="nav_home", type=btn_type):
        set_page("Home")
        st.rerun()

with col_nav2:
    btn_type = "primary" if st.session_state.active_page == "Detection Workspace" else "secondary"
    if st.button("Detection Workspace", key="nav_workspace", type=btn_type):
        set_page("Detection Workspace")
        st.rerun()

with col_nav3:
    btn_type = "primary" if st.session_state.active_page == "Technology" else "secondary"
    if st.button("Technology", key="nav_tech", type=btn_type):
        set_page("Technology")
        st.rerun()

with col_nav4:
    btn_type = "primary" if st.session_state.active_page == "About" else "secondary"
    if st.button("About", key="nav_about", type=btn_type):
        set_page("About")
        st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE: HOME
# ==========================================================================
if st.session_state.active_page == "Home":

    # --- Hero Section ---
    hero_col_text, hero_col_visual = st.columns([1.1, 1], gap="large")

    with hero_col_text:
        st.markdown(
            """
            <div class="hero-box">
                <div class="hero-tag">AUTONOMOUS UAV VISION PLATFORM</div>
                <div class="hero-main-title">
                    See the Unseen.<br><span>Fly with Confidence.</span>
                </div>
                <div class="hero-description">
                    Analyze aerial imagery using AI-powered object detection. Identify visible objects 
                    and inspect model predictions through an intuitive computer-vision workspace.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cta_col1, cta_col2, cta_col3 = st.columns([1.3, 1.5, 1])
        with cta_col1:
            if st.button("Analyze Image", type="primary", key="cta_analyze"):
                set_page("Detection Workspace")
                st.rerun()
        with cta_col2:
            if st.button("Explore Technology", key="cta_explore"):
                set_page("Technology")
                st.rerun()

    with hero_col_visual:
        hero_img_path = "assets/hero_drone_vision.png"
        if os.path.exists(hero_img_path):
            st.markdown(
                """
                <div class="hero-visual-card">
                    <div class="hero-visual-header">
                        <div>OPTICAL SENSOR // <span>4K HUD TELEMETRY</span></div>
                        <div>GRID: 37°N 122°W</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(hero_img_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="hero-visual-card" style="padding: 40px; text-align: center;">
                    <div style="font-size: 2.2rem; color: #00E5FF; margin-bottom: 8px;">⌖</div>
                    <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.2rem; color: #F8FAFC;">
                        AERIAL COMPUTER VISION FEED
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">
                        Sensor Telemetry & Bounding Box Overlay Ready
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Feature Cards ---
    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-label">CORE CAPABILITIES</div>
            <div class="section-title">Architected for Aerial Precision</div>
            <div class="section-subtitle">
                Designed specifically for low-altitude flight safety, utility inspection, and autonomous navigation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-icon-wrap">⌖</div>
                <div class="panel-title">AI Object Detection</div>
                <p class="panel-body">
                    Neural network architecture tailored for aerial perspectives. Identifies ground obstacles, 
                    transmission pylons, buildings, and nearby aerial traffic with high spatial localization.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feat_col2:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-icon-wrap">◈</div>
                <div class="panel-title">Confidence Scoring</div>
                <p class="panel-body">
                    Calibrated probabilistic scoring per detected bounding box. Enables transparent, 
                    adjustable safety thresholds before triggering autonomous collision-avoidance maneuvers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feat_col3:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-icon-wrap">▦</div>
                <div class="panel-title">Visual Image Analysis</div>
                <p class="panel-body">
                    Detailed frame-by-frame inspection. Examine raw sensor dimensions, coordinate geometries, 
                    and multi-channel visuals in a high-fidelity command console.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- How It Works ---
    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-label">SYSTEM WORKFLOW</div>
            <div class="section-title">How It Works</div>
            <div class="section-subtitle">
                A streamlined three-stage pipeline connecting optical drone sensors to actionable obstacle intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">01</div>
                <div class="step-title">Upload a Drone Image</div>
                <p class="step-text">
                    Ingest high-resolution aerial snapshots or reconnaissance camera frames into the 
                    secure local sensor buffer.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">02</div>
                <div class="step-title">Analyze with YOLOv8</div>
                <p class="step-text">
                    Execute forward-pass inference on the pretrained Ultralytics YOLOv8 computer vision 
                    backbone to extract multi-scale feature maps.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">03</div>
                <div class="step-title">Inspect Labels & Boxes</div>
                <p class="step-text">
                    Review pinpointed bounding boxes, classification labels, and confidence metrics to evaluate 
                    flight pathway clearance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================================================
# PAGE: DETECTION WORKSPACE
# ==========================================================================
elif st.session_state.active_page == "Detection Workspace":

    st.markdown(
        """
        <div class="workspace-container">
            <div class="workspace-title">⌖ Aerial Sensor Ingestion Workspace</div>
            <div class="workspace-desc">
                Deploy high-resolution drone imagery into the avionics workspace. Inspect optical properties, 
                configure neural detection confidence thresholds, and execute real YOLOv8 obstacle classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Load cached YOLOv8 model instance
    model_instance, model_state, model_err = get_cached_model(DEFAULT_MODEL_NAME)

    # Responsive Two-Column Layout (Desktop side-by-side, Mobile vertically stacked)
    col_controls, col_preview = st.columns([1, 1.25], gap="large")

    with col_controls:
        # File Uploader with dynamic key for clean resetting
        uploader_key = f"drone_uploader_{st.session_state.uploader_key}"
        uploaded_file = st.file_uploader(
            "Select or drop aerial image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key=uploader_key,
            help="Maximum upload limit: 10 MB. Supported formats: JPG, JPEG, PNG, WEBP.",
        )

        valid_image = None
        metadata = None

        if uploaded_file is not None:
            valid_image, metadata, error_msg = validate_and_load_image(
                uploaded_file, max_size_bytes=10 * 1024 * 1024
            )

            if error_msg:
                st.error(f"⚠️ Validation Error: {error_msg}")
            elif metadata:
                clean_filename = html.escape(metadata["filename"])
                clean_format = html.escape(metadata["format"])
                clean_mode = html.escape(metadata["mode"])
                clean_size = html.escape(metadata["size_readable"])

                st.markdown(
                    f"""
                    <div class="panel-card" style="margin-top: 14px;">
                        <div class="panel-title" style="font-size: 1.15rem; margin-bottom: 12px;">
                            Sensor Frame Telemetry
                        </div>
                        <div class="meta-spec-row">
                            <span class="meta-spec-label">Filename</span>
                            <span class="meta-spec-val">{clean_filename}</span>
                        </div>
                        <div class="meta-spec-row">
                            <span class="meta-spec-label">Resolution</span>
                            <span class="meta-spec-val">{metadata['width']} &times; {metadata['height']} px</span>
                        </div>
                        <div class="meta-spec-row">
                            <span class="meta-spec-label">Aspect Ratio</span>
                            <span class="meta-spec-val">{metadata['aspect_ratio']}</span>
                        </div>
                        <div class="meta-spec-row">
                            <span class="meta-spec-label">Color Profile</span>
                            <span class="meta-spec-val">{clean_format} / {clean_mode}</span>
                        </div>
                        <div class="meta-spec-row">
                            <span class="meta-spec-label">Payload Size</span>
                            <span class="meta-spec-val">{clean_size}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Detection Controls Panel (Persistent across runs)
        # Dynamic Model Status Badge
        if model_state == "Ready":
            model_badge_html = f"""
            <div class="model-status-pill">
                <span class="model-status-dot-ready"></span>
                STATUS: YOLOv8 READY ({DEFAULT_MODEL_NAME})
            </div>
            """
        elif model_state == "Error":
            clean_err = html.escape(str(model_err or "Load failed"))
            model_badge_html = f"""
            <div class="model-status-pill" style="border-color: rgba(239, 68, 68, 0.4); color: #EF4444;">
                <span class="model-status-dot-error"></span>
                STATUS: MODEL ERROR ({clean_err})
            </div>
            """
        else:
            model_badge_html = """
            <div class="model-status-pill" style="color: #F59E0B;">
                <span class="model-status-dot-loading"></span>
                STATUS: LOADING YOLOv8 MODEL...
            </div>
            """

        st.markdown(
            f"""
            <div class="control-panel-box">
                <div class="control-panel-heading">Detection Controls & Parameters</div>
                {model_badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence Threshold Slider (10% to 90%, default 40%)
        conf_slider = st.slider(
            "Confidence Threshold",
            min_value=10,
            max_value=90,
            value=st.session_state.confidence_threshold,
            step=5,
            format="%d%%",
            help="Filter detections below this confidence threshold (10% - 90%).",
            key="conf_slider_widget",
        )
        st.session_state.confidence_threshold = conf_slider

        # Detection and Image Actions Row
        col_act1, col_act2 = st.columns([1.3, 1])

        with col_act1:
            run_detection = st.button(
                "Run Detection",
                type="primary",
                key="btn_run_detection",
                help="Execute real YOLOv8 obstacle detection pipeline on the ingested frame.",
            )

        with col_act2:
            remove_image = st.button(
                "Remove Image",
                type="secondary",
                key="btn_remove_image",
                disabled=(uploaded_file is None),
                help="Clear the current image from the buffer.",
            )

        if run_detection:
            if valid_image is not None and metadata is not None:
                if model_instance is None:
                    st.error(f"⚠️ Inference engine offline: {model_err}")
                else:
                    conf_fraction = st.session_state.confidence_threshold / 100.0
                    with st.spinner("Executing YOLOv8 neural network inference..."):
                        annotated_img, detections, duration_ms = run_inference(
                            model_instance, valid_image, conf_threshold=conf_fraction
                        )
                    st.session_state.detection_result = {
                        "annotated_image": annotated_img,
                        "detections": detections,
                        "duration_ms": duration_ms,
                        "filename": metadata["filename"],
                        "conf_pct": st.session_state.confidence_threshold,
                    }
                    # Record entry in session-only detection history (deduplicated per run)
                    st.session_state.detection_history.insert(
                        0,
                        {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Filename": metadata["filename"],
                            "Objects Detected": len(detections),
                            "Unique Classes": len(set(d["class"] for d in detections)),
                            "Confidence Floor": f"{st.session_state.confidence_threshold}%",
                        },
                    )
            else:
                st.warning("⚠️ Ingest a valid aerial drone frame before initiating detection.")

        if remove_image:
            st.session_state.detection_result = None
            st.session_state.uploader_key += 1
            st.rerun()

    with col_preview:
        if valid_image is not None and metadata is not None:
            clean_filename = html.escape(metadata["filename"])

            # Check if detection results exist for the currently loaded image
            det_res = st.session_state.get("detection_result")
            has_matching_results = (
                det_res is not None and det_res.get("filename") == metadata["filename"]
            )

            if has_matching_results:
                channel_choice = st.radio(
                    "Display Channel",
                    ["Annotated Detection Feed", "Raw Optical Sensor"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="channel_toggle",
                )

                all_dets = det_res["detections"]
                num_objects = len(all_dets)

                if channel_choice == "Annotated Detection Feed":
                    # Object Target Highlighting Control
                    if num_objects > 0:
                        target_options = ["None (All Targets)"] + [
                            f"Target #{i+1}: {d['class'].upper()} ({d['confidence_pct']}%)"
                            for i, d in enumerate(all_dets)
                        ]
                        sel_target = st.selectbox(
                            "Highlight Specific Target",
                            options=target_options,
                            index=0,
                            key="target_highlight_dropdown",
                            help="Renders an aerospace cyan targeting reticle around the chosen object.",
                        )
                        if sel_target != "None (All Targets)":
                            target_idx = (
                                int(sel_target.split(":")[0].replace("Target #", "").strip()) - 1
                            )
                            chosen_obj = all_dets[target_idx]
                            viewer_img = highlight_specific_object(
                                det_res["annotated_image"],
                                chosen_obj["box"],
                                chosen_obj["class"],
                                chosen_obj["confidence_pct"],
                            )
                        else:
                            viewer_img = det_res["annotated_image"]
                    else:
                        viewer_img = det_res["annotated_image"]

                    st.markdown(
                        f"""
                        <div class="hero-visual-card">
                            <div class="hero-visual-header">
                                <div>FRAME: <span>{clean_filename}</span></div>
                                <div>MODE: <span>ANNOTATED FEED</span></div>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(viewer_img, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # In-Memory PNG Download Button
                    png_bytes = image_to_png_bytes(viewer_img)
                    base_fn = clean_filename.rsplit(".", 1)[0]
                    st.download_button(
                        label="⬇ Download Annotated Image (PNG)",
                        data=png_bytes,
                        file_name=f"aero_sentinel_{base_fn}_annotated.png",
                        mime="image/png",
                        key="btn_download_annotated_png",
                    )

                else:
                    st.markdown(
                        f"""
                        <div class="hero-visual-card">
                            <div class="hero-visual-header">
                                <div>FRAME: <span>{clean_filename}</span></div>
                                <div>MODE: <span>RAW OPTICAL SENSOR</span></div>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(valid_image, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Telemetry Summary Metrics Grid
                num_classes = len(set(d["class"] for d in all_dets))
                latency_str = f"{det_res['duration_ms']:.1f} ms"
                conf_used_str = f"{det_res['conf_pct']}%"

                st.markdown(
                    f"""
                    <div class="telemetry-grid">
                        <div class="telemetry-card">
                            <div class="telemetry-val">{num_objects}</div>
                            <div class="telemetry-lbl">Objects Found</div>
                        </div>
                        <div class="telemetry-card">
                            <div class="telemetry-val">{num_classes}</div>
                            <div class="telemetry-lbl">Unique Classes</div>
                        </div>
                        <div class="telemetry-card">
                            <div class="telemetry-val">{latency_str}</div>
                            <div class="telemetry-lbl">Inference Latency</div>
                        </div>
                        <div class="telemetry-card">
                            <div class="telemetry-val">{conf_used_str}</div>
                            <div class="telemetry-lbl">Confidence Floor</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Professional Results Dashboard Tabs
                tab_table, tab_chart, tab_hist = st.tabs([
                    "📋 Object Table & Confidence",
                    "📊 Class Breakdown Chart",
                    "📜 Session Mission History",
                ])

                with tab_table:
                    if num_objects > 0:
                        unique_classes = sorted(list(set(d["class"] for d in all_dets)))
                        filter_classes = st.multiselect(
                            "Filter by Detected Class",
                            options=unique_classes,
                            default=unique_classes,
                            key="class_table_filter",
                        )
                        filtered_dets = [d for d in all_dets if d["class"] in filter_classes]

                        if len(filtered_dets) > 0:
                            table_rows = []
                            for idx, d in enumerate(filtered_dets):
                                table_rows.append({
                                    "Object #": f"#{idx+1}",
                                    "Detected Class": d["class"].upper(),
                                    "Confidence %": f"{d['confidence_pct']}%",
                                    "Bounding Box (x1, y1, x2, y2)": f"[{d['box'][0]}, {d['box'][1]}, {d['box'][2]}, {d['box'][3]}]",
                                })
                            st.dataframe(
                                pd.DataFrame(table_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Confidence Progress Bars Visualization
                            st.markdown(
                                "<div style='font-family: Rajdhani, sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin: 16px 0 8px 0;'>Confidence Rating per Object</div>",
                                unsafe_allow_html=True,
                            )
                            for idx, d in enumerate(filtered_dets):
                                conf_val = max(0.0, min(1.0, float(d["confidence"])))
                                st.progress(
                                    conf_val,
                                    text=f"Object #{idx+1} [{d['class'].upper()}] — {d['confidence_pct']}% Confidence",
                                )
                        else:
                            st.info("No objects match the selected class filter.")
                    else:
                        st.markdown(
                            f"""
                            <div class="workspace-notice-box" style="margin-top: 14px;">
                                <strong>Radar Clear:</strong> No obstacles or objects were detected above the 
                                <strong>{conf_used_str}</strong> confidence floor. 
                                Adjust the confidence slider to capture lower-certainty or distant hazards.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with tab_chart:
                    if num_objects > 0:
                        class_counts = (
                            pd.Series([d["class"].upper() for d in all_dets])
                            .value_counts()
                            .rename("Count")
                        )
                        st.bar_chart(class_counts, color="#00E5FF")
                    else:
                        st.info("No objects detected. Class distribution chart is inactive.")

                with tab_hist:
                    if len(st.session_state.detection_history) > 0:
                        hist_df = pd.DataFrame(st.session_state.detection_history)
                        st.dataframe(hist_df, use_container_width=True, hide_index=True)
                        if st.button("Clear Detection History", type="secondary", key="btn_clear_history"):
                            st.session_state.detection_history = []
                            st.rerun()
                    else:
                        st.info("No detection runs recorded in this session yet.")

            else:
                # Image loaded, but detection not run yet
                st.markdown(
                    f"""
                    <div class="hero-visual-card">
                        <div class="hero-visual-header">
                            <div>FRAME: <span>{clean_filename}</span></div>
                            <div>OPTICAL SENSOR // {metadata['width']}&times;{metadata['height']}</div>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(valid_image, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="workspace-notice-box" style="margin-top: 14px;">
                        <strong>Optical Sensor Feed Buffered:</strong> Frame validated and staged in memory. 
                        Threshold configured to <strong>{st.session_state.confidence_threshold}%</strong>. 
                        Click <strong>Run Detection</strong> to execute forward-pass YOLOv8 inference.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # Empty state when no image uploaded
            st.markdown(
                """
                <div class="panel-card" style="text-align: center; padding: 72px 24px; border-style: dashed;">
                    <div style="font-size: 2.6rem; color: #00E5FF; margin-bottom: 12px;">⌖</div>
                    <div class="panel-title">Selected Image Preview</div>
                    <p class="panel-body" style="max-width: 440px; margin: 0 auto;">
                        No drone frame loaded. Use the left panel to browse or drop an aerial sensor capture 
                        (JPG, PNG, or WEBP under 10 MB). A benchmark sample is available in 
                        <code>assets/sample_drone_recon.jpg</code>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )




# ==========================================================================
# PAGE: TECHNOLOGY
# ==========================================================================
elif st.session_state.active_page == "Technology":

    st.markdown(
        """
        <div class="section-header-block" style="margin-top: 10px;">
            <div class="section-label">ENGINEERING STACK</div>
            <div class="section-title">Planned Technology Architecture</div>
            <div class="section-subtitle">
                Built with a streamlined Python-native framework designed for edge UAV deployment, 
                high numerical throughput, and zero frontend bloat.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tech_c1, tech_c2, tech_c3, tech_c4 = st.columns(4)

    with tech_c1:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-header">
                    <div class="tech-name">Python</div>
                    <div class="tech-badge">CORE RUNTIME</div>
                </div>
                <div class="tech-desc">
                    High-level scripting backbone providing native tensor handling, memory safety, 
                    and seamless integration across scientific computing libraries.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech_c2:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-header">
                    <div class="tech-name">Streamlit</div>
                    <div class="tech-badge">TELEMETRY UI</div>
                </div>
                <div class="tech-desc">
                    Reactive web framework powering the mission-control dashboard. Enables rapid UI 
                    state transitions without requiring Node.js or JavaScript build steps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech_c3:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-header">
                    <div class="tech-name">Ultralytics YOLOv8</div>
                    <div class="tech-badge">DETECTION ENGINE</div>
                </div>
                <div class="tech-desc">
                    State-of-the-art pretrained computer vision model engineered for real-time aerial 
                    object detection, multi-label bounding boxes, and low inference latency.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech_c4:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-header">
                    <div class="tech-name">OpenCV</div>
                    <div class="tech-badge">VISION UTILITIES</div>
                </div>
                <div class="tech-desc">
                    Industry-standard image processing library responsible for matrix manipulation, 
                    color space conversions, and high-performance bounding box rendering.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Architecture Pipeline Diagram
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title" style="margin-bottom: 14px;">Inference Pipeline Architecture</div>
            <div class="steps-container" style="margin: 0; grid-template-columns: repeat(4, 1fr);">
                <div class="step-card" style="padding: 16px;">
                    <div class="step-number" style="font-size: 1.4rem;">STAGE 1</div>
                    <div class="step-title" style="font-size: 1.05rem;">Optical Sensor Ingestion</div>
                    <p class="step-text">Ingest raw camera frame into memory; validate format and color profile.</p>
                </div>
                <div class="step-card" style="padding: 16px;">
                    <div class="step-number" style="font-size: 1.4rem;">STAGE 2</div>
                    <div class="step-title" style="font-size: 1.05rem;">Pre-processing (OpenCV)</div>
                    <p class="step-text">Resize, normalize pixel intensities, and construct batched tensors.</p>
                </div>
                <div class="step-card" style="padding: 16px;">
                    <div class="step-number" style="font-size: 1.4rem;">STAGE 3</div>
                    <div class="step-title" style="font-size: 1.05rem;">YOLOv8 Inference</div>
                    <p class="step-text">Forward-pass feature extraction, anchor assignment, and bounding-box regression.</p>
                </div>
                <div class="step-card" style="padding: 16px;">
                    <div class="step-number" style="font-size: 1.4rem;">STAGE 4</div>
                    <div class="step-title" style="font-size: 1.05rem;">Spatial Risk Assessment</div>
                    <p class="step-text">Confidence threshold filtering, distance vector calculation, and HUD alerting.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
elif st.session_state.active_page == "About":

    st.markdown(
        """
        <div class="section-header-block" style="margin-top: 10px;">
            <div class="section-label">PROJECT PROFILE</div>
            <div class="section-title">About Aero Sentinel</div>
            <div class="section-subtitle">
                Pioneering reliable, real-time aerial computer vision intelligence for autonomous flight systems.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    about_col1, about_col2 = st.columns([1.2, 1], gap="large")

    with about_col1:
        st.markdown(
            """
            <div class="panel-card" style="margin-bottom: 20px;">
                <div class="panel-title">Mission Overview</div>
                <p class="panel-body" style="margin-bottom: 12px;">
                    Autonomous unmanned aerial vehicles (UAVs) increasingly operate in complex environments — 
                    from urban canyons and power line corridors to dense forestry and construction zones. 
                    Traditional GPS waypoint systems cannot anticipate dynamic or uncharted physical obstructions.
                </p>
                <p class="panel-body">
                    <strong>Aero Sentinel</strong> provides an intelligent visual perception layer by combining 
                    lightweight edge computer vision with deep learning object detection, safeguarding UAVs 
                    against costly collisions and mission failures.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with about_col2:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Development Roadmap</div>
                <div class="meta-spec-row">
                    <span class="meta-spec-label">Phase 1: Environment & Setup</span>
                    <span class="meta-spec-val" style="color: #10B981;">COMPLETED</span>
                </div>
                <div class="meta-spec-row">
                    <span class="meta-spec-label">Phase 2: Premium Aerospace UI</span>
                    <span class="meta-spec-val" style="color: #00E5FF;">ACTIVE</span>
                </div>
                <div class="meta-spec-row">
                    <span class="meta-spec-label">Phase 3: YOLOv8 Weight Loading</span>
                    <span class="meta-spec-val" style="color: #94A3B8;">UPCOMING</span>
                </div>
                <div class="meta-spec-row">
                    <span class="meta-spec-label">Phase 4: Telemetry & Video Feeds</span>
                    <span class="meta-spec-val" style="color: #94A3B8;">PLANNED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================================================
# FOOTER (All Pages)
# ==========================================================================
st.markdown(
    """
    <div class="footer-wrapper">
        <div class="footer-brand">AERO SENTINEL — DRONE OBSTACLE DETECTION USING YOLOV8</div>
        <div class="footer-disclaimer">
            Prototype software for autonomous UAV computer vision research. Ultralytics YOLOv8 neural network weights 
            and live bounding-box inference will be integrated in subsequent development phases. 
            Not certified for autonomous live flight navigation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
