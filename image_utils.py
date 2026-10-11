import io
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def extract_metadata(image: Image.Image, filename: str, size_bytes: int) -> dict:
    """Extract descriptive metadata from a loaded PIL Image."""
    width, height = image.size
    aspect_ratio = f"{round(width / max(1, height), 2)}:1"

    if size_bytes < 1024 * 1024:
        size_str = f"{round(size_bytes / 1024, 1)} KB"
    else:
        size_str = f"{round(size_bytes / (1024 * 1024), 2)} MB"

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "format": image.format or "RAW",
        "mode": image.mode,
        "size_bytes": size_bytes,
        "size_readable": size_str,
    }


def validate_and_load_image(uploaded_file, max_size_bytes=MAX_UPLOAD_SIZE_BYTES):
    """
    Safely validate and decode an uploaded image file using Pillow.
    
    Returns:
        (image: Image.Image | None, metadata: dict | None, error_message: str | None)
    """
    if uploaded_file is None:
        return None, None, "No image file provided."

    # 1. Size constraint check (10 MB limit)
    file_size = getattr(uploaded_file, "size", 0)
    if file_size > max_size_bytes:
        max_mb = round(max_size_bytes / (1024 * 1024))
        actual_mb = round(file_size / (1024 * 1024), 2)
        return (
            None,
            None,
            f"File exceeds maximum upload limit of {max_mb} MB (received {actual_mb} MB).",
        )

    # 2. Extract in-memory bytes
    try:
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        else:
            file_bytes = uploaded_file.read()
    except Exception as e:
        return None, None, f"Failed to read upload data: {e}"

    if not file_bytes or len(file_bytes) == 0:
        return None, None, "Uploaded file is empty (0 bytes)."

    # 3. Decode & verify using Pillow (safe in-memory stream)
    try:
        # Initial verify pass
        verify_stream = io.BytesIO(file_bytes)
        img_verify = Image.open(verify_stream)
        img_verify.verify()

        # Re-open stream for pixel loading (verify() invalidates image object)
        decode_stream = io.BytesIO(file_bytes)
        image = Image.open(decode_stream)
        image.load()  # Force decode to detect truncated or corrupted pixel blocks

        # Verify format is supported
        detected_format = (image.format or "").upper()
        if detected_format not in ALLOWED_FORMATS:
            return (
                None,
                None,
                f"Unsupported image format '{detected_format}'. Allowed formats: JPG, JPEG, PNG, WEBP.",
            )

        filename = getattr(uploaded_file, "name", "sensor_frame.raw")
        metadata = extract_metadata(image, filename, len(file_bytes))
        return image, metadata, None

    except UnidentifiedImageError:
        return (
            None,
            None,
            "Unsupported or unrecognized file format. Please upload a valid JPG, JPEG, PNG, or WEBP image.",
        )
    except Exception as e:
        return (
            None,
            None,
            f"Corrupted or unreadable image file: {e}",
        )
