
import io
import hashlib
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Aero Sentinel | AI Vision",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MODEL_PATH = "yolov8n.pt"


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #080D16;
    color: #F8FAFC;
}
[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #263448;
}
h1, h2, h3 {
    color: #F8FAFC !important;
}
p, label, span {
    line-height: 1.6;
}
div.stButton > button {
    background: #00E5FF;
    color: #080D16;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    min-height: 42px;
}
div.stButton > button:hover {
    background: #7AF2FF;
    color: #080D16;
}
div[data-testid="stMetric"] {
    background: #101827;
    border: 1px solid #263448;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stFileUploader"] {
    background: #101827;
    border-radius: 12px;
}
hr {
    border-color: #263448;
}
.small-note {
    color: #94A3B8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pretrained model once per Streamlit cache."""
    return YOLO(MODEL_PATH)


# ---------------- SESSION STATE ----------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "history" not in st.session_state:
    st.session_state.history = []

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False


# ---------------- HEADER ----------------

st.markdown("# ◈ AERO SENTINEL")
st.markdown(
    "### AI-powered drone image analysis"
)
st.markdown(
    '<p class="small-note">'
    'Computer vision powered by a pretrained YOLOv8 model'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## VISION CONTROL")

    confidence = st.slider(
        "Confidence threshold",
        min_value=10,
        max_value=90,
        value=40,
        step=5,
        help="Only predictions at or above this confidence are retained.",
    )

    st.caption(f"Current threshold: {confidence}%")

    if st.session_state.model_ready:
        st.success("Model loaded in this session")
    else:
        st.info("Model loads when detection starts")

    st.divider()
    st.markdown("### Supported formats")
    st.write("JPG · JPEG · PNG · WEBP")

    st.caption("Maximum upload size: 10 MB")

    if st.button("Clear detection history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------------- IMAGE UPLOAD ----------------

st.markdown("## 01 / Upload Image")

uploaded_file = st.file_uploader(
    "Choose a drone image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Select an image containing visible objects.",
)

image_rgb = None
upload_bytes = None
upload_hash = None
upload_error = None

if uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    upload_hash = hashlib.sha256(upload_bytes).hexdigest()

    if len(upload_bytes) > MAX_FILE_SIZE:
        upload_error = "Image exceeds the 10 MB upload limit."

    else:
        try:
            # Verify image structure before decoding it.
            with Image.open(io.BytesIO(upload_bytes)) as check_image:
                check_image.verify()

            # Reopen after verify() and convert to RGB.
            with Image.open(io.BytesIO(upload_bytes)) as source:
                width, height = source.size

                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        "Image dimensions exceed the permitted limit."
                    )

                image_rgb = source.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            ValueError,
        ) as error:
            upload_error = f"Cannot process this image: {error}"

if upload_error:
    st.error(upload_error)

# Remove a previous image's results when a different image is selected.
previous_result = st.session_state.analysis_result

if previous_result is not None:
    if upload_hash != previous_result["image_hash"]:
        st.session_state.analysis_result = None

if uploaded_file is None:
    st.session_state.analysis_result = None


# ---------------- ORIGINAL IMAGE PREVIEW ----------------

if image_rgb is not None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Uploaded Image")
        st.image(image_rgb, use_container_width=True)

    with right:
        st.markdown("### Image Information")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Dimensions:** {image_rgb.width} × {image_rgb.height}")
        st.write(f"**File size:** {len(upload_bytes) / 1024:.1f} KB")

        st.info(
            "The image is ready for inference. "
            "Detected objects will appear after analysis."
        )

    st.divider()


# ---------------- RUN REAL YOLO INFERENCE ----------------

st.markdown("## 02 / Run AI Detection")

run_detection = st.button(
    "Run Detection",
    type="primary",
    use_container_width=True,
    disabled=(image_rgb is None or upload_error is not None),
)

if run_detection and image_rgb is not None:
    try:
        with st.spinner(
            "Loading YOLOv8 and analyzing your image..."
        ):
            model = load_model()
            st.session_state.model_ready = True

            # Ultralytics accepts OpenCV-style BGR NumPy images.
            image_bgr = cv2.cvtColor(
                np.array(image_rgb),
                cv2.COLOR_RGB2BGR,
            )

            start_time = time.perf_counter()

            predictions = model.predict(
                source=image_bgr,
                conf=confidence / 100.0,
                device="cpu",
                verbose=False,
            )

            inference_seconds = time.perf_counter() - start_time
            prediction = predictions[0]

            rows = []

            if prediction.boxes is not None:
                xyxy = prediction.boxes.xyxy.cpu().numpy()
                scores = prediction.boxes.conf.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy()

                for index, (box, score, class_id) in enumerate(
                    zip(xyxy, scores, classes),
                    start=1,
                ):
                    x1, y1, x2, y2 = box.tolist()
                    class_name = prediction.names[int(class_id)]

                    rows.append({
                        "Object": index,
                        "Class": class_name,
                        "Confidence (%)": round(float(score) * 100, 2),
                        "X1": round(float(x1), 1),
                        "Y1": round(float(y1), 1),
                        "X2": round(float(x2), 1),
                        "Y2": round(float(y2), 1),
                    })

            # Ultralytics plot() returns a BGR NumPy image.
            annotated_bgr = prediction.plot()
            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB,
            )

            # Convert both images to PNG bytes for display/download.
            annotated_buffer = io.BytesIO()
            Image.fromarray(annotated_rgb).save(
                annotated_buffer,
                format="PNG",
            )

            original_buffer = io.BytesIO()
            image_rgb.save(original_buffer, format="PNG")

            result_data = {
                "image_hash": upload_hash,
                "filename": uploaded_file.name,
                "rows": rows,
                "annotated_png": annotated_buffer.getvalue(),
                "original_png": original_buffer.getvalue(),
                "inference_seconds": inference_seconds,
                "threshold": confidence,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

            st.session_state.analysis_result = result_data

            # Add history once for each successful analysis.
            st.session_state.history.insert(0, {
                "Time": result_data["timestamp"],
                "Filename": result_data["filename"],
                "Objects": len(rows),
                "Unique classes": len({
                    row["Class"] for row in rows
                }),
                "Threshold (%)": confidence,
            })

            # Keep only the most recent 10 entries.
            st.session_state.history = (
                st.session_state.history[:10]
            )

    except Exception as error:
        st.error(
            "Detection failed. Check your model installation, "
            "internet connection on first use, and terminal output."
        )
        st.caption(f"Error details: {error}")


# ---------------- RESULTS DASHBOARD ----------------

result_data = st.session_state.analysis_result

if result_data is not None:
    rows = result_data["rows"]
    results_df = pd.DataFrame(
        rows,
        columns=[
            "Object",
            "Class",
            "Confidence (%)",
            "X1",
            "Y1",
            "X2",
            "Y2",
        ],
    )

    total_objects = len(rows)
    unique_classes = len({row["Class"] for row in rows})

    st.divider()
    st.markdown("## 03 / Detection Report")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Objects detected", total_objects)
    metric2.metric("Unique classes", unique_classes)
    metric3.metric(
        "Inference time",
        f"{result_data['inference_seconds']:.3f} s",
    )

    st.caption(
        f"Threshold used for this result: "
        f"{result_data['threshold']}%"
    )

    image_view = st.radio(
        "Image view",
        ["Annotated image", "Original image"],
        horizontal=True,
    )

    if image_view == "Annotated image":
        display_bytes = result_data["annotated_png"]
    else:
        display_bytes = result_data["original_png"]

    st.image(display_bytes, use_container_width=True)

    st.download_button(
        "Download annotated image (PNG)",
        data=result_data["annotated_png"],
        file_name="aero_sentinel_annotated.png",
        mime="image/png",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Detected Objects")

    if total_objects == 0:
        st.info(
            "No objects met the selected confidence threshold. "
            "This does not prove the image is obstacle-free."
        )
    else:
        class_options = ["All classes"] + sorted({
            row["Class"] for row in rows
        })

        selected_class = st.selectbox(
            "Filter by object class",
            class_options,
        )

        if selected_class != "All classes":
            filtered_df = results_df[
                results_df["Class"] == selected_class
            ]
        else:
            filtered_df = results_df

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Confidence Scores")

        for row in filtered_df.to_dict("records"):
            st.write(
                f"**{row['Class']}** — "
                f"{row['Confidence (%)']:.2f}%"
            )
            st.progress(
                min(100, max(0, int(row["Confidence (%)"]))),
                text="Model confidence",
            )

        st.markdown("### Object Class Breakdown")

        class_counts = Counter(
            row["Class"] for row in rows
        )

        chart_df = pd.DataFrame(
            list(class_counts.items()),
            columns=["Class", "Count"],
        ).set_index("Class")

        st.bar_chart(chart_df)

    st.caption(
        "Detection confidence is not a guarantee of correctness. "
        "This prototype does not determine depth, collision probability, "
        "or whether an object is in the drone's flight path."
    )


# ---------------- SESSION HISTORY ----------------

st.divider()
st.markdown("## 04 / Session History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "History is stored in the current Streamlit session. "
        "It is not a permanent database."
    )
else:
    st.info(
        "No analyses yet. Upload an image and run detection "
        "to create your first history entry."
    )

st.divider()
st.caption(
    "AERO SENTINEL | Python · Streamlit · YOLOv8 · OpenCV"
)
