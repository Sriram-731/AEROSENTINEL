import time
from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO


def load_yolo_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    Instantiate and return a pretrained YOLOv8 model.
    Downloads the model checkpoint weights automatically on first call if not present locally.
    """
    model = YOLO(model_name)
    return model


def run_inference(
    model: YOLO,
    image: Image.Image,
    conf_threshold: float = 0.40,
) -> Tuple[Image.Image, List[Dict[str, Any]], float]:
    """
    Run forward-pass YOLOv8 inference on a PIL Image.
    
    Args:
        model: Preloaded YOLO instance
        image: Validated PIL Image
        conf_threshold: Float confidence cutoff between 0.10 and 0.90
        
    Returns:
        (annotated_image, detections_list, inference_duration_ms)
    """
    # 1. Normalize image to RGB mode
    if image.mode != "RGB":
        image_rgb = image.convert("RGB")
    else:
        image_rgb = image

    # 2. Measure actual model execution time
    start_time = time.perf_counter()
    results = model.predict(
        source=image_rgb,
        conf=conf_threshold,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 3. Parse detections
    detections: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = result.names.get(cls_id, f"Class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [int(round(coord)) for coord in box.xyxy[0].tolist()]

                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "confidence_pct": round(conf * 100.0, 1),
                    "box": xyxy,  # [x1, y1, x2, y2] in pixels
                })

        # 4. Generate visual annotated frame using actual model detections
        if len(detections) > 0:
            annotated_bgr = result.plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(annotated_rgb)
        else:
            annotated_image = image_rgb.copy()
    else:
        annotated_image = image_rgb.copy()

    return annotated_image, detections, elapsed_ms


def highlight_specific_object(
    base_image: Image.Image,
    box: List[int],
    class_name: str,
    confidence_pct: float,
) -> Image.Image:
    """
    Render a high-visibility aerospace cyan targeting reticle with corner brackets
    around a specific detected object, without modifying original coordinates.
    """
    img_np = np.array(base_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # Cyan color in BGR: (255, 229, 0)
    cyan_bgr = (255, 229, 0)
    white_bgr = (255, 255, 255)

    # 1. Main highlight box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), cyan_bgr, 2)

    # 2. Tactical corner brackets
    corner_len = max(6, min(22, w // 4, h // 4))
    # Top-left
    cv2.line(img_bgr, (x1, y1), (x1 + corner_len, y1), white_bgr, 3)
    cv2.line(img_bgr, (x1, y1), (x1, y1 + corner_len), white_bgr, 3)
    # Top-right
    cv2.line(img_bgr, (x2, y1), (x2 - corner_len, y1), white_bgr, 3)
    cv2.line(img_bgr, (x2, y1), (x2, y1 + corner_len), white_bgr, 3)
    # Bottom-left
    cv2.line(img_bgr, (x1, y2), (x1 + corner_len, y2), white_bgr, 3)
    cv2.line(img_bgr, (x1, y2), (x1, y2 - corner_len), white_bgr, 3)
    # Bottom-right
    cv2.line(img_bgr, (x2, y2), (x2 - corner_len, y2), white_bgr, 3)
    cv2.line(img_bgr, (x2, y2), (x2, y2 - corner_len), white_bgr, 3)

    # 3. Label tag banner
    label_text = f"TARGET // {class_name.upper()} {confidence_pct:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    tag_y1 = max(0, y1 - text_h - 10)
    tag_y2 = y1
    tag_x2 = min(img_bgr.shape[1], x1 + text_w + 12)

    cv2.rectangle(img_bgr, (x1, tag_y1), (tag_x2, tag_y2), cyan_bgr, -1)
    cv2.putText(
        img_bgr,
        label_text,
        (x1 + 6, max(text_h + 2, tag_y2 - 6)),
        font,
        font_scale,
        (16, 13, 8),  # Dark contrast text
        thickness,
        cv2.LINE_AA,
    )

    out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert a PIL Image to raw PNG bytes in-memory for download."""
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

