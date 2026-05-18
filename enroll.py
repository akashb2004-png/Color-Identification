import cv2
import numpy as np
import os
import json

FACES_DIR = "faces"
MODEL_PATH = os.path.join(FACES_DIR, "model.yml")
LABELS_PATH = os.path.join(FACES_DIR, "labels.json")
FACE_SIZE = (100, 100)
SAMPLES_NEEDED = 40

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def train_model():
    faces, labels = [], []
    label_map = {}
    label_id = 0

    for person_name in sorted(os.listdir(FACES_DIR)):
        person_dir = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[label_id] = person_name
        for img_file in sorted(os.listdir(person_dir)):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)
        label_id += 1

    if not faces:
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f)

    print(f"Model trained with {len(label_map)} user(s): {list(label_map.values())}")
    return True


def enroll():
    name = input("Enter your name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Clear old samples for a clean re-enroll
    for f in os.listdir(person_dir):
        os.remove(os.path.join(person_dir, f))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    print(f"Collecting {SAMPLES_NEEDED} face samples for '{name}'...")

    while count < SAMPLES_NEEDED:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        display = frame.copy()

        for (x, y, w, h) in detected[:1]:
            if count < SAMPLES_NEEDED:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                cv2.imwrite(os.path.join(person_dir, f"{count:04d}.jpg"), face_roi)
                count += 1
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(display, "Face detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)

        if not len(detected):
            cv2.putText(display, "No face detected — move closer", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1, cv2.LINE_AA)

        # Header
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (20, 20, 20), -1)
        cv2.putText(display, f"Enrolling: {name}   {count}/{SAMPLES_NEEDED}", (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Progress bar
        h_frame = display.shape[0]
        bw = display.shape[1] - 40
        cv2.rectangle(display, (20, h_frame - 30), (20 + bw, h_frame - 12), (50, 50, 50), -1)
        filled = int(bw * count / SAMPLES_NEEDED)
        cv2.rectangle(display, (20, h_frame - 30), (20 + filled, h_frame - 12), (0, 200, 0), -1)
        cv2.putText(display, "Q to cancel", (20, h_frame - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1, cv2.LINE_AA)

        cv2.imshow("Enroll Face", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    print("Training model...")
    if train_model():
        print(f"Done! Run:  python3 color_identifier.py")
    else:
        print("Error: could not train model.")


if __name__ == "__main__":
    os.makedirs(FACES_DIR, exist_ok=True)
    enroll()
