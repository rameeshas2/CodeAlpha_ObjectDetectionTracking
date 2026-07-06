from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # dict, auto-loaded from COCO

    def track(self, frame, conf=0.5, imgsz=384):
        """Runs detection + ByteTrack in one call. Returns the results object."""
        results = self.model.track(frame, persist=True, conf=conf,
                                    tracker="bytetrack.yaml", verbose=False,
                                    imgsz=imgsz)
        return results[0]