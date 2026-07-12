import streamlit as st
import cv2
import os
import numpy as np
from detector import Detector
from utils.fps import FPSCounter
from utils.counter import ObjectCounter
from datetime import datetime

os.makedirs("output", exist_ok=True)

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="◆",
    layout="wide"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #d4d7dd;
        }

        .stApp {
            background: #0e1116;
        }

        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        section[data-testid="stSidebar"] {
            background: #14171e;
            border-right: 1px solid #262b35;
        }

        .block-container {
            padding-top: 2.5rem;
            max-width: 1150px;
        }

        .eyebrow {
            color: #6e7681;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .page-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #f0f2f5;
            margin: 0 0 8px 0;
            letter-spacing: -0.01em;
        }

        .page-sub {
            color: #8b929e;
            font-size: 0.92rem;
            margin: 0 0 28px 0;
            max-width: 680px;
            line-height: 1.6;
        }

        .sidebar-label {
            color: #6e7681;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-top: 20px;
            margin-bottom: 8px;
        }

        .panel {
            background: #14171e;
            border: 1px solid #262b35;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }

        .panel-title {
            color: #d4d7dd;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .steps {
            display: flex;
            border: 1px solid #262b35;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 24px;
        }
        .step {
            flex: 1;
            padding: 14px 18px;
            border-right: 1px solid #262b35;
        }
        .step:last-child { border-right: none; }
        .step-num {
            color: #565d68;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            margin-bottom: 6px;
        }
        .step-title {
            color: #e6e8ec;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .step-text {
            color: #7d8590;
            font-size: 0.8rem;
            line-height: 1.5;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #14171e !important;
            border: 1px dashed #2b303a !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #a8afba !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: #1c2230 !important;
            border: 1px solid #2b3140 !important;
            color: #e6e8ec !important;
            border-radius: 6px !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #232a3a !important;
            border-color: #3d4658 !important;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            fill: #6e7681 !important;
        }

        .placeholder-box {
            min-height: 420px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #565d68;
            font-size: 0.9rem;
            border: 1px dashed #2b303a;
            border-radius: 8px;
        }

        .chip {
            display: inline-block;
            margin: 0 6px 6px 0;
            padding: 4px 10px;
            border-radius: 5px;
            background: #1c2028;
            color: #a8afba;
            font-size: 0.78rem;
            border: 1px solid #262b35;
        }

        .status-line {
            font-size: 0.85rem;
            color: #8b929e;
            line-height: 1.6;
        }
        .status-line b {
            color: #e6e8ec;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }

        div.stButton > button {
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            min-height: 38px;
            border: 1px solid #2b3140;
            background: #1c2230;
            color: #e6e8ec;
        }
        div.stButton > button:hover {
            border-color: #3d4658;
            background: #232a3a;
        }
        div.stButton > button:first-child {
            background: #3562e8;
            border-color: #3562e8;
            color: #ffffff;
        }
        div.stButton > button:first-child:hover {
            background: #2e56d1;
        }

        img { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="eyebrow">CodeAlpha AI Internship · Task 4</div>'
    '<div class="page-title">Object Detection & Tracking</div>'
    '<div class="page-sub">Real-time object detection and multi-object tracking using YOLOv8 and ByteTrack, with live statistics and exportable output.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-label">Source</div>', unsafe_allow_html=True)
    source_type = st.radio(
        "Source",
        ["Webcam", "Upload Video", "Upload Image"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Confidence threshold</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Run", key="start_btn", use_container_width=True)
    with col2:
        st.button("Reset", key="stop_btn", use_container_width=True)

    video_file = None
    image_file = None

    if source_type == "Upload Video":
        st.markdown('<div class="sidebar-label">Video file</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

    if source_type == "Upload Image":
        st.markdown('<div class="sidebar-label">Image file</div>', unsafe_allow_html=True)
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">Status</div>', unsafe_allow_html=True)
    stats_placeholder = st.empty()
    stats_placeholder.markdown(
        '<div class="panel"><div class="status-line">Idle — waiting for a run.</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="steps">'
    '<div class="step"><div class="step-num">01</div><div class="step-title">Detect</div>'
    '<div class="step-text">YOLOv8 identifies objects in each frame at the chosen confidence threshold.</div></div>'
    '<div class="step"><div class="step-num">02</div><div class="step-title">Track</div>'
    '<div class="step-text">ByteTrack maintains a consistent ID for each object across frames.</div></div>'
    '<div class="step"><div class="step-num">03</div><div class="step-title">Export</div>'
    '<div class="step-text">Annotated output is saved and available to download.</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

video_col, info_col = st.columns([3, 1], gap="large")

with video_col:
    frame_placeholder = st.empty()
    frame_placeholder.markdown(
        '<div class="placeholder-box">Choose a source in the sidebar, then click Run.</div>',
        unsafe_allow_html=True,
    )

with info_col:
    st.markdown('<div class="panel-title">Detected classes</div>', unsafe_allow_html=True)
    classes_placeholder = st.empty()
    classes_placeholder.markdown(
        "<div class='status-line'>No objects detected yet.</div>",
        unsafe_allow_html=True,
    )

DISPLAY_WIDTH = 900
SKIP = 2
detector = Detector("models/yolov8n.pt")
fps_counter = FPSCounter()
obj_counter = ObjectCounter()

if start_button and source_type == "Upload Image":
    if image_file is None:
        st.warning("Please upload an image first.")
        st.stop()

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = detector.track(frame, conf=conf_threshold, imgsz=384)
    annotated = result.plot()

    h, w = annotated.shape[:2]
    scale = DISPLAY_WIDTH / w
    annotated = cv2.resize(annotated, (DISPLAY_WIDTH, int(h * scale)))

    frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/detection_{timestamp}.jpg"
    cv2.imwrite(output_path, annotated)

    stats_placeholder.markdown(
        f'<div class="panel"><div class="status-line"><b>{len(result.boxes)}</b> objects detected.</div></div>',
        unsafe_allow_html=True,
    )

    if len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy()
        unique_classes = sorted(set(detector.class_names[int(c)] for c in cls_ids))
        classes_html = "".join([f"<span class='chip'>{c.title()}</span>" for c in unique_classes])
        classes_placeholder.markdown(classes_html, unsafe_allow_html=True)
    else:
        classes_placeholder.markdown(
            "<div class='status-line'>No objects detected.</div>",
            unsafe_allow_html=True,
        )

    st.success(f"Detection complete. Output saved to {output_path}")
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download output image",
            data=f,
            file_name=os.path.basename(output_path),
            mime="image/jpeg",
        )

if start_button and source_type in ["Webcam", "Upload Video"]:
    if source_type == "Webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Could not access the webcam. Check camera permissions and availability.")
            st.stop()
    else:
        if video_file is None:
            st.warning("Please upload a video first.")
            st.stop()
        with open("input/temp_upload.mp4", "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture("input/temp_upload.mp4")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/detection_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    writer_initialized = False
    out = None

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if source_type == "Webcam":
            frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % SKIP != 0:
            continue

        result = detector.track(frame, conf=conf_threshold, imgsz=384)
        annotated = result.plot()

        h, w = annotated.shape[:2]
        scale = DISPLAY_WIDTH / w
        annotated = cv2.resize(annotated, (DISPLAY_WIDTH, int(h * scale)))

        if not writer_initialized:
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                input_fps / SKIP,
                (annotated.shape[1], annotated.shape[0]),
            )
            writer_initialized = True

        out.write(annotated)

        fps = fps_counter.update()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy()
            total_tracked = obj_counter.update(track_ids)
        else:
            total_tracked = 0

        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        stats_placeholder.markdown(
            f'<div class="panel"><div class="status-line">'
            f'<b>{len(result.boxes)}</b> objects in frame<br>'
            f'<b>{total_tracked}</b> tracked IDs<br>'
            f'<b>{fps}</b> FPS'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy()
            unique_classes = sorted(set(detector.class_names[int(c)] for c in cls_ids))
            classes_html = "".join([f"<span class='chip'>{c.title()}</span>" for c in unique_classes])
            classes_placeholder.markdown(classes_html, unsafe_allow_html=True)
        else:
            classes_placeholder.markdown(
                "<div class='status-line'>No objects detected yet.</div>",
                unsafe_allow_html=True,
            )

    cap.release()
    if out is not None:
        out.release()

    st.success(f"Detection complete. Output saved to {output_path}")
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download output video",
                data=f,
                file_name=os.path.basename(output_path),
                mime="video/mp4",
            )