# YOLO ROS2 Cone Detection

## Overview

This project implements a real-time cone detection system for autonomous rover navigation using YOLO and ROS2. The system detects cones of specific colors, extracts their image coordinates, and publishes them as a ROS2 topic for downstream tasks such as path planning and control.

---

## Features

* Real-time object detection using YOLO (Ultralytics)
* Color-specific cone filtering (blue, yellow, orange, etc.)
* Extraction of bounding box centers for each detected cone
* ROS2 publisher node for streaming cone coordinates
* Webcam-based inference pipeline

---

## System Pipeline

1. Capture frame from webcam
2. Run YOLO inference on the frame
3. Filter detections by target cone color
4. Compute center coordinates of bounding boxes
5. Publish results to ROS2 topic

---

## ROS2 Integration

**Node Name:**
`cone_publisher`

**Published Topic:**
`/cone_coordinates`

**Message Type:**
`Float32MultiArray`

**Format:**
[cx1, cy1, class1, cx2, cy2, class2, ...]

Where:

* `cx`, `cy` are pixel coordinates of cone centers
* `class` corresponds to cone type

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure ROS2 is installed and sourced in your environment.

---

## Usage

Run the inference script:

```bash
python src/main_inference.py
```

Press `q` to exit the live detection window.

---

## Configuration

Inside the script, you can modify:

* `TARGET_COLOR_NAME` to select cone type
* `conf_threshold` to adjust detection confidence
* `USE_WEBCAM` to toggle between live and static input
* Camera index in `cv2.VideoCapture()`

---

## Model

The project uses a trained YOLO model (`best.pt`).

Note:

* Model weights are not included in the repository
* Replace the path with your own trained model

---





## Repository Structure

```
detection_ws/
    src/
        main_inference.py
    .gitignore
    README.md
    requirements.txt
    LICENSE
```

---


