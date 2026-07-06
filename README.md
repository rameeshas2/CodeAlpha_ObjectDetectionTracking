# 🎯 Real-Time Object Detection & Tracking

A real-time computer vision application that detects and tracks multiple objects in a video stream using **YOLOv8** for detection and **ByteTrack** for multi-object tracking. Built with **Streamlit** for an interactive, browser-based UI.

> Developed as **Task 4** of the **CodeAlpha Artificial Intelligence Internship**.

---

## 📌 Overview

This project takes live webcam footage, an uploaded video, or a single image, runs it through a pre-trained YOLOv8 model to detect objects (from the 80 COCO classes), and — for video/webcam input — assigns each detected object a persistent tracking ID using ByteTrack so it can be followed across frames even through movement and partial occlusion.

The processed output is displayed live in the browser with bounding boxes, class labels, confidence scores, and tracking IDs, and is automatically saved to the `output/` folder for download.

---

## ✨ Features

- **Three input modes**: Webcam (live), Upload Video, Upload Image
- **Real-time object detection** using YOLOv8 (pre-trained on the COCO dataset, 80 object classes)
- **Multi-object tracking** with persistent IDs via ByteTrack, so the same object keeps the same ID across frames
- **Live statistics panel**: objects currently in frame, total unique tracked IDs, live FPS
- **Detected classes panel**: shows which object categories are currently visible
- **Adjustable confidence threshold** via a slider, to control detection sensitivity
- **Automatic output saving**: annotated video/image is saved to `output/` with a timestamped filename
- **Download button** to retrieve the processed output directly from the browser
- **Mirrored webcam view** for a natural, selfie-style display (uploaded videos are left unmirrored)
- **Custom, modern dark-themed UI** built with Streamlit + custom CSS

---

## 🧠 How It Works

```
Input Source (Webcam / Video / Image)
              │
              ▼
        OpenCV captures frame
              │
              ▼
   YOLOv8 detects objects in the frame
              │
              ▼
  ByteTrack assigns/updates tracking IDs
     (video & webcam modes only)
              │
              ▼
  Bounding boxes, labels, confidence,
       and IDs drawn on frame
              │
              ▼
   Frame displayed live in Streamlit
              │
              ▼
  Annotated frame written to output file
              │
              ▼
   Statistics panel updated (objects,
      unique IDs, FPS, detected classes)
```

### Detection

The `Detector` class (`detector.py`) loads a pre-trained YOLOv8 model (`yolov8n.pt`, trained on COCO) and exposes a single `track()` method. Each frame is passed through `model.track()`, which performs detection and tracking in one call using Ultralytics' built-in ByteTrack integration (`tracker="bytetrack.yaml"`). Frames are resized to a smaller inference size (`imgsz=384`) to keep detection fast enough for near real-time performance on CPU.

### Tracking

Unlike single-frame detection, tracking requires knowing which detected object in the current frame corresponds to which object in the previous frame. ByteTrack solves this by matching detections across frames based on position and motion, assigning each object a stable ID (`result.boxes.id`) that persists as long as the object remains visible or briefly occluded.

### Frame Skipping

To keep the UI responsive, only every *n*th frame (`SKIP = 2`) is actually run through YOLO — the rest are skipped. This trades a small amount of temporal smoothness for a meaningful boost in perceived frame rate, since YOLO inference is the main performance bottleneck on CPU.

### Output Saving

For video/webcam mode, each processed frame is written to an `mp4` file via `cv2.VideoWriter`, using a timestamped filename (`output/detection_YYYYMMDD_HHMMSS.mp4`) so previous runs are never overwritten. For image mode, the single annotated frame is saved as a `.jpg` in the same folder. Once processing finishes, a download button lets the user retrieve the file directly from the browser.

---

## 🛠️ Tech Stack

| Component         | Technology                          |
|--------------------|--------------------------------------|
| Language           | Python 3.10                         |
| Object Detection   | YOLOv8 (Ultralytics)                |
| Tracking           | ByteTrack (built into Ultralytics)  |
| Computer Vision    | OpenCV                              |
| Web UI             | Streamlit                           |
| Numerical ops      | NumPy                               |

---

## 📁 Project Structure

```
CodeAlpha_ObjectDetectionTracking/
│
├── app.py                  # Main Streamlit application (UI + processing loop)
├── detector.py              # YOLOv8 + ByteTrack wrapper class
├── requirements.txt          # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── models/
│   └── yolov8n.pt            # Pre-trained YOLOv8 nano weights (COCO)
│
├── input/
│   └── (temporary uploaded files land here)
│
├── output/
│   ├── detection_*.mp4        # Saved annotated videos (timestamped)
│   ├── detection_*.jpg         # Saved annotated images (timestamped)
│   └── screenshots/            # UI screenshots for documentation
│
└── utils/
    ├── fps.py                  # FPS counter helper class
    └── counter.py                # Unique object ID counter helper class
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
   cd CodeAlpha_ObjectDetectionTracking
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv yolov8_env
   yolov8_env\Scripts\activate      # Windows
   # source yolov8_env/bin/activate # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model weights** — `yolov8n.pt` will auto-download on first run if not already present in `models/`.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

**Using the app:**
1. Choose an input source in the sidebar: **Webcam**, **Upload Video**, or **Upload Image**
2. Adjust the confidence threshold slider if needed (default `0.50`)
3. Click **▶ Start** to begin detection
4. Watch live bounding boxes, tracking IDs, and statistics update in real time
5. Click **⏹ Stop**, or let the video finish, to end the session
6. Download the processed output using the **⬇️ Download Output** button

<img width="700" height="375" alt="image" src="https://github.com/user-attachments/assets/794c1fc0-68ae-468a-95ce-d7ae7f069242" />

<img width="700" height="375" alt="image" src="https://github.com/user-attachments/assets/bd0ce578-d681-470a-bab3-7034e0d2b639" />

---

## 📊 Live Statistics Explained

| Stat                  | Meaning                                                             |
|------------------------|----------------------------------------------------------------------|
| **Objects in Frame**    | Number of objects detected in the current frame                     |
| **Unique Tracked IDs**  | Total number of distinct objects tracked so far in the session       |
| **FPS**                 | Frames processed per second (live performance indicator)            |
| **Detected Classes**    | List of object categories currently visible (e.g. person, car, dog) |

---

## 🖼️ Screenshots

<img width="500" height="375" alt="detection_20260706_171841" src="https://github.com/user-attachments/assets/2d865435-a674-42e6-b6ff-63dccc6140cb" />



<img width="700" height="375" alt="detection_20260706_171653" src="https://github.com/user-attachments/assets/fb0554da-7e0c-4fb8-8d5f-333df2c421cd" />



<img width="700" height="375" alt="detection_20260706_171741" src="https://github.com/user-attachments/assets/4b3d3f6b-2721-4b78-83e3-d27e00a8bd0a" />


<img width="700" height="375" alt="image" src="https://github.com/user-attachments/assets/6a7ab923-8e69-4bad-99ae-7b1d57d3763d" />


---

## 🚀 Possible Improvements

- GPU acceleration (CUDA) for significantly higher FPS
- Object counting across a defined line/zone (entry/exit counting)
- Support for multiple simultaneous video streams
- Model selection dropdown (YOLOv8n / YOLOv8s / YOLOv8m or YOLOv11)
- Export detection statistics as a CSV/log file

---

## 📄 License

This project was built for educational purposes as part of the CodeAlpha AI Internship program.
