import cv2

def draw_boxes(frame, boxes, class_names, track_ids=None):
    """Draw bounding boxes, class labels, confidence, and optional track IDs."""
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        label = class_names[cls_id]

        if track_ids is not None:
            label = f"{label} ID:{track_ids[i]}"
        label += f" {conf:.2f}"

        color = get_color(track_ids[i] if track_ids is not None else cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def get_color(idx):
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    return colors[int(idx) % len(colors)]