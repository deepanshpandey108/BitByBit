import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================
# 🔹 HARD CODE YOUR VIDEO PATH HERE
# =========================================
VIDEO_PATH = "test_vid.mp4"     # <-- change this
OUTPUT_PATH = "output_fight_detection.mp4"

# =========================================
# PARAMETERS
# =========================================
WINDOW_SIZE = 25
DIST_THRESHOLD = 80
IOU_THRESHOLD = 0.15
ACC_THRESHOLD = 10
FIGHT_SCORE_THRESHOLD = 6

# =========================================
# LOAD MODEL
# =========================================
model = YOLO("yolov8n.pt")

# =========================================
# STORAGE
# =========================================
track_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
velocity_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
acc_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# =========================================
# HELPER FUNCTIONS
# =========================================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def center(box):
    return ((box[0]+box[2])//2, (box[1]+box[3])//2)

# =========================================
# OPEN VIDEO
# =========================================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("Processing video...")

# =========================================
# MAIN LOOP
# =========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])

    fight_score = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        # -----------------------
        # PER PERSON FEATURES
        # -----------------------
        for box, track_id in zip(boxes, ids):
            box = box.astype(int)
            c = center(box)

            track_history[track_id].append(c)

            if len(track_history[track_id]) >= 2:
                v = np.linalg.norm(
                    np.array(track_history[track_id][-1]) -
                    np.array(track_history[track_id][-2])
                )
                velocity_history[track_id].append(v)

                if len(velocity_history[track_id]) >= 2:
                    a = abs(
                        velocity_history[track_id][-1] -
                        velocity_history[track_id][-2]
                    )
                    acc_history[track_id].append(a)

                    if a > ACC_THRESHOLD:
                        fight_score += 1

        # -----------------------
        # PAIRWISE FEATURES
        # -----------------------
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                boxA = boxes[i]
                boxB = boxes[j]

                cA = center(boxA)
                cB = center(boxB)

                dist = np.linalg.norm(np.array(cA) - np.array(cB))
                iou = compute_iou(boxA, boxB)

                if dist < DIST_THRESHOLD:
                    fight_score += 1

                if iou > IOU_THRESHOLD:
                    fight_score += 1

        # -----------------------
        # GLOBAL MOTION ENERGY
        # -----------------------
        global_energy = 0
        for tid in velocity_history:
            if len(velocity_history[tid]) > 0:
                global_energy += velocity_history[tid][-1]

        if global_energy > 200:
            fight_score += 1

        # -----------------------
        # DRAW BOXES
        # -----------------------
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # -----------------------
    # FIGHT ALERT
    # -----------------------
    if fight_score > FIGHT_SCORE_THRESHOLD:
        cv2.putText(frame, "FIGHT DETECTED!", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)

    out.write(frame)

cap.release()
out.release()

print("Done.")
print("Saved output to:", OUTPUT_PATH)
