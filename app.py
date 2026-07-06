import streamlit as st
import cv2
import os
import numpy as np
from detector import Detector
from utils.fps import FPSCounter
from utils.counter import ObjectCounter
from datetime import datetime
import os

os.makedirs("output", exist_ok=True)

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

# ---------- Custom styling ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b 0%, #0f0f23 45%, #0a0a17 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171335 0%, #100e28 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .hero-emoji {
        font-size: 2.6rem;
        filter: drop-shadow(0 0 12px rgba(123,97,255,0.6));
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6ec7, #7b61ff 50%, #4fd1ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #a1a1c2;
        font-size: 0.95rem;
        margin-left: 54px;
        margin-top: -6px;
    }
    .badge {
        display: inline-block;
        background: rgba(123,97,255,0.15);
        border: 1px solid rgba(123,97,255,0.4);
        color: #c4b5fd;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 54px;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .video-frame {
        border-radius: 20px;
        padding: 10px;
        background: linear-gradient(135deg, rgba(123,97,255,0.15), rgba(255,110,199,0.08));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    .placeholder-box {
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b6b8f;
        font-size: 1.1rem;
        font-weight: 600;
        border: 2px dashed rgba(123,97,255,0.3);
        border-radius: 14px;
        text-align: center;
    }

    .stat-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        border-color: rgba(123,97,255,0.5);
        transform: translateY(-2px);
    }
    .stat-icon {
        font-size: 1.3rem;
        margin-bottom: 4px;
    }
    .stat-label {
        color: #8b8ba7;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .stat-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4fd1ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
        border: none;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #7b61ff, #ff6ec7);
        color: white;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 20px rgba(123,97,255,0.5);
    }

    .sidebar-heading {
        color: #c4b5fd;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    img { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="hero-emoji">🎯</div>
    <div class="hero-title">Object Detection & Tracking</div>
</div>
<div class="hero-sub">Real-time computer vision powered by YOLOv8 + ByteTrack</div>
<div class="badge">✨ CodeAlpha AI Internship — Task 4</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="sidebar-heading">⚙️ Input Settings</div>', unsafe_allow_html=True)
    source_type = st.radio(
        "Source",
        ["Webcam", "Upload Video", "Upload Image"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-heading">🎚️ Confidence Threshold</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶ Start", key="start_btn", use_container_width=True)
    with col2:
        stop_button = st.button("⏹ Stop", key="stop_btn", use_container_width=True)

    video_file = None
    image_file = None

    if source_type == "Upload Video":
        st.markdown('<div class="sidebar-heading">📁 Upload</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

    if source_type == "Upload Image":
        st.markdown('<div class="sidebar-heading">🖼️ Upload</div>', unsafe_allow_html=True)
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-heading">📊 Live Statistics</div>', unsafe_allow_html=True)
    stats_placeholder = st.empty()
    stats_placeholder.markdown("""
        <div class="stat-card" style="color:#6b6b8f; text-align:center;">
            Waiting to start...
        </div>
    """, unsafe_allow_html=True)

# ---------- Main layout ----------
video_col, info_col = st.columns([3, 1], gap="large")

with video_col:
    st.markdown('<div class="video-frame">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    frame_placeholder.markdown("""
        <div class="placeholder-box">
            🎬 Press Start to begin detection
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with info_col:
    st.markdown("#### 🖼️ Detected Classes")
    classes_placeholder = st.empty()
    classes_placeholder.markdown(
        "<div class='stat-card' style='color:#6b6b8f;'>No objects detected yet</div>",
        unsafe_allow_html=True
    )

DISPLAY_WIDTH = 900
SKIP = 2
detector = Detector("models/yolov8n.pt")
fps_counter = FPSCounter()
obj_counter = ObjectCounter()

# =====================================================
# IMAGE MODE
# =====================================================
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

    # Save output image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/detection_{timestamp}.jpg"
    cv2.imwrite(output_path, annotated)

    stats_placeholder.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📦</div>
        <div class="stat-label">Objects Detected</div>
        <div class="stat-value">{len(result.boxes)}</div>
    </div>
    """, unsafe_allow_html=True)

    if len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy()
        unique_classes = sorted(set(detector.class_names[int(c)] for c in cls_ids))
        classes_placeholder.markdown(
            "".join([f"<div class='stat-card'>🔹 {c.title()}</div>" for c in unique_classes]),
            unsafe_allow_html=True
        )
    else:
        classes_placeholder.markdown(
            "<div class='stat-card' style='color:#6b6b8f;'>No objects detected</div>",
            unsafe_allow_html=True
        )

    st.success(f"✅ Detection complete! Saved to `{output_path}`")
    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Output Image",
            data=f,
            file_name=os.path.basename(output_path),
            mime="image/jpeg"
        )

# =====================================================
# VIDEO / WEBCAM MODE
# =====================================================
if start_button and source_type in ["Webcam", "Upload Video"]:
    if source_type == "Webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Check permissions or if another app is using it.")
            st.stop()
    else:
        if video_file is None:
            st.warning("Please upload a video first.")
            st.stop()
        with open("input/temp_upload.mp4", "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture("input/temp_upload.mp4")

    # ---- Set up output video writer ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/detection_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 20  # fallback if webcam reports 0
    writer_initialized = False
    out = None

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if source_type == "Webcam":
            frame = cv2.flip(frame, 1)  # mirror webcam only

        frame_count += 1
        if frame_count % SKIP != 0:
            continue

        result = detector.track(frame, conf=conf_threshold, imgsz=384)
        annotated = result.plot()

        h, w = annotated.shape[:2]
        scale = DISPLAY_WIDTH / w
        annotated = cv2.resize(annotated, (DISPLAY_WIDTH, int(h * scale)))

        # Initialize writer with correct frame size (first frame only)
        if not writer_initialized:
            out = cv2.VideoWriter(output_path, fourcc, input_fps / SKIP,
                                   (annotated.shape[1], annotated.shape[0]))
            writer_initialized = True

        out.write(annotated)  # save annotated frame to output file

        fps = fps_counter.update()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy()
            total_tracked = obj_counter.update(track_ids)
        else:
            total_tracked = 0

        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        stats_placeholder.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">📦</div>
            <div class="stat-label">Objects in Frame</div>
            <div class="stat-value">{len(result.boxes)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🆔</div>
            <div class="stat-label">Unique Tracked IDs</div>
            <div class="stat-value">{total_tracked}</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-label">FPS</div>
            <div class="stat-value">{fps}</div>
        </div>
        """, unsafe_allow_html=True)

        if len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy()
            unique_classes = sorted(set(detector.class_names[int(c)] for c in cls_ids))
            classes_placeholder.markdown(
                "".join([f"<div class='stat-card'>🔹 {c.title()}</div>" for c in unique_classes]),
                unsafe_allow_html=True
            )
        else:
            classes_placeholder.markdown(
                "<div class='stat-card' style='color:#6b6b8f;'>No objects detected yet</div>",
                unsafe_allow_html=True
            )

    cap.release()
    if out is not None:
        out.release()

    st.success(f"✅ Detection complete! Saved to `{output_path}`")
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Output Video",
                data=f,
                file_name=os.path.basename(output_path),
                mime="video/mp4"
            )