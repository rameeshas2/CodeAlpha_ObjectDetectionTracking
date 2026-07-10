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
    page_icon="🎯",
    layout="wide"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #111c31 45%, #0b1326 100%);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101a2f 0%, #0b1424 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .hero-card {
            padding: 26px 28px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(14, 21, 38, 0.96), rgba(8, 15, 29, 0.92));
            border: 1px solid rgba(255,255,255,0.09);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
            margin-bottom: 18px;
            backdrop-filter: blur(14px);
        }

        .eyebrow {
            color: #8fa8ff;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f7fbff;
            margin: 0 0 8px 0;
            letter-spacing: -0.02em;
        }

        .hero-sub {
            color: #9ca8c3;
            font-size: 1rem;
            margin: 0;
            max-width: 820px;
            line-height: 1.6;
        }

        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .hero-pill {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(134, 168, 255, 0.16);
            color: #cfdcff;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid rgba(134, 168, 255, 0.28);
        }

        .panel-card {
            background: linear-gradient(145deg, rgba(10, 16, 29, 0.95), rgba(8, 13, 24, 0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 16px 16px 14px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.24);
            margin-bottom: 14px;
        }

        .preview-card {
            padding: 12px;
        }

        .section-title {
            color: #dce7ff;
            font-size: 0.92rem;
            font-weight: 700;
            margin: 6px 0 8px;
        }

        .sidebar-stack {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .workflow-wrap {
            margin: 4px 0 18px 0;
        }

        .workflow-header {
            color: #e2ebff;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: 0.03em;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .workflow-card {
            background: linear-gradient(145deg, rgba(17, 25, 43, 0.95), rgba(8, 15, 28, 0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 15px;
            min-height: 128px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .workflow-step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: linear-gradient(135deg, #7b8cff, #51c4ff);
            color: white;
            font-size: 0.8rem;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .workflow-title {
            color: #f7fbff;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .workflow-text {
            color: #8da0c5;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .placeholder-box {
            min-height: 440px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #7d8baa;
            font-size: 1.04rem;
            font-weight: 600;
            border: 1px dashed rgba(134, 168, 255, 0.25);
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(134,168,255,0.04));
        }

        .chip {
            display: inline-block;
            margin: 5px 8px 0 0;
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(134, 168, 255, 0.16);
            color: #dce8ff;
            font-size: 0.86rem;
            border: 1px solid rgba(134, 168, 255, 0.22);
        }

        div.stButton > button {
            border-radius: 12px;
            font-weight: 700;
            min-height: 44px;
            border: none;
            background: linear-gradient(90deg, #7b8cff 0%, #51c4ff 100%);
            color: white;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(81, 196, 255, 0.26);
        }

        div.stButton > button:focus {
            box-shadow: 0 0 0 2px rgba(255,255,255,0.16);
        }

        .status-box {
            border-radius: 14px;
            padding: 12px 13px;
            background: rgba(255,255,255,0.03);
            color: #b7c2da;
            border: 1px solid rgba(255,255,255,0.06);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .status-box strong {
            color: #f7fbff;
        }

        img {
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Computer Vision Studio</div>
        <h1 class="hero-title">Object Detection & Tracking</h1>
        <p class="hero-sub">A refined interface for running YOLOv8-based detection and tracking with a clear workflow, responsive controls, and polished output presentation.</p>
        <div class="hero-pills">
            <span class="hero-pill">Live preview</span>
            <span class="hero-pill">Tracked objects</span>
            <span class="hero-pill">Export-ready results</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Session controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stack">', unsafe_allow_html=True)

    source_type = st.radio(
        "Source",
        ["Webcam", "Upload Video", "Upload Image"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-title">Confidence</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Run detection", key="start_btn", use_container_width=True)
    with col2:
        st.button("Reset", key="stop_btn", use_container_width=True)

    video_file = None
    image_file = None

    if source_type == "Upload Video":
        st.markdown('<div class="section-title">Upload video</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

    if source_type == "Upload Image":
        st.markdown('<div class="section-title">Upload image</div>', unsafe_allow_html=True)
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown("<div class='section-title'>Workflow guide</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="status-box">
            Select a source, adjust the confidence slider, and launch detection. The preview updates instantly and the final output is saved for download.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Live status</div>", unsafe_allow_html=True)
    stats_placeholder = st.empty()
    stats_placeholder.markdown(
        """
        <div class="status-box">
            Waiting for a run to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="workflow-wrap">
        <div class="workflow-header">Detection workflow</div>
        <div class="workflow-grid">
            <div class="workflow-card">
                <div class="workflow-step">01</div>
                <div class="workflow-title">Detect</div>
                <div class="workflow-text">YOLOv8 identifies objects in each frame using your selected confidence threshold.</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">02</div>
                <div class="workflow-title">Track</div>
                <div class="workflow-text">Each detected object is followed across frames with consistent tracking IDs.</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">03</div>
                <div class="workflow-title">Export</div>
                <div class="workflow-text">Annotated results are saved and ready to download as image or video output.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

video_col, info_col = st.columns([3, 1], gap="large")

with video_col:
    st.markdown('<div class="panel-card preview-card">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    frame_placeholder.markdown(
        """
        <div class="placeholder-box">
            Choose a source and run detection to generate a live preview of the annotated output.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detected classes</div>', unsafe_allow_html=True)
    classes_placeholder = st.empty()
    classes_placeholder.markdown(
        "<div class='status-box'>No objects detected yet.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
        f"""
        <div class="status-box">
            <strong>{len(result.boxes)}</strong> objects detected in the uploaded image.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy()
        unique_classes = sorted(set(detector.class_names[int(c)] for c in cls_ids))
        classes_html = "".join([f"<span class='chip'>{c.title()}</span>" for c in unique_classes])
        classes_placeholder.markdown(
            f"<div class='status-box'>{classes_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        classes_placeholder.markdown(
            "<div class='status-box'>No objects detected.</div>",
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
            st.error("Could not access the webcam. Please check camera permissions and availability.")
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
            f"""
            <div class="status-box">
                <strong>{len(result.boxes)}</strong> objects in the current frame • <strong>{total_tracked}</strong> tracked IDs • <strong>{fps}</strong> FPS
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy()
            unique_classes = sorted(set(detector.class_names[int(c)] for c in cls_ids))
            classes_html = "".join([f"<span class='chip'>{c.title()}</span>" for c in unique_classes])
            classes_placeholder.markdown(
                f"<div class='status-box'>{classes_html}</div>",
                unsafe_allow_html=True,
            )
        else:
            classes_placeholder.markdown(
                "<div class='status-box'>No objects detected yet.</div>",
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