
# Aero Sentinel
## Drone Obstacle Detection Using YOLOv8

A Python-based computer vision prototype that analyzes uploaded
images using a pretrained YOLOv8 model.
## Features

- Image upload and preview
- Pretrained YOLOv8 object detection
- Annotated images with bounding boxes
- Object names and confidence scores
- Detection counts and class breakdown
- Download annotated images
- Session-only detection history

## Technology Stack

- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- Pillow
- NumPy
- Pandas

## Requirements

- Python version compatible with the installed dependencies
- Internet access for the first model download
- Sufficient RAM for CPU inference

## Installation on Windows

Open PowerShell in the project folder.

Create a virtual environment:

    python -m venv .venv

Install dependencies:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Start the application:

    .\.venv\Scripts\python.exe -m streamlit run app.py

Open the local URL displayed in the terminal.

## How to Use

1. Upload a JPG, PNG, or WEBP image.
2. Set the confidence threshold.
3. Click Run Detection.
4. Review object labels, confidence scores, and bounding boxes.
5. Download the annotated PNG.

## Model

The application uses the pretrained Ultralytics YOLOv8n model.
The model weights may be downloaded automatically on first use.

Review the applicable Ultralytics and pretrained-weight licensing
terms before redistribution or commercial deployment.

## Limitations

The model detects common object categories but may miss specialized
drone hazards, including thin wires and cables.

A still image does not establish object distance, collision probability,
or whether an object lies in the drone's flight path.

This project is a prototype, not a certified collision-avoidance system.

