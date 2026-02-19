from ultralytics import YOLO
import cv2

MODEL_PATH = "weapon_detector.pt"
CONF = 0.90
CAM_ID = 0

def main():
    model = YOLO(MODEL_PATH)
    names = model.names

    cap = cv2.VideoCapture(CAM_ID)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✅ Weapon Detection Started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=CONF, verbose=False)
        r = results[0]
        boxes = r.boxes

        weapon_count = 0

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())

                label = names.get(cls_id, str(cls_id))

                weapon_count += 1

                # Red bounding box for weapon
                color = (0, 0, 255)
                text = f"Weapon Detected {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Filled label background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - th - 12),
                              (x1 + tw + 10, y1), color, -1)

                # Label text
                cv2.putText(frame, text,
                            (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2)

        # Global Alert
        if weapon_count > 0:
            alert_text = f"🚨 ALERT: {weapon_count} Weapon(s) Detected!"
            cv2.putText(frame, alert_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3)

        cv2.imshow("Weapon Detection (Live)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
