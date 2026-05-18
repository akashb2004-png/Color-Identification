import cv2
import numpy as np
import os
import json
import time

FACES_DIR = "faces"
MODEL_PATH = os.path.join(FACES_DIR, "model.yml")
LABELS_PATH = os.path.join(FACES_DIR, "labels.json")
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 70   # LBPH: lower = better match
DETECT_EVERY_N = 5          # run face detection every N frames

STATE_LOGIN = "login"
STATE_WELCOME = "welcome"
STATE_COLOR_ID = "color_id"

mouse_x, mouse_y = -1, -1

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Mouse callback ─────────────────────────────────────────────────────────────

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


# ── Face model ─────────────────────────────────────────────────────────────────

def load_face_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return recognizer, labels


def detect_and_recognize(gray, recognizer, labels):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    results = []
    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
        label_id, confidence = recognizer.predict(roi)
        name = labels.get(label_id, "Unknown") if confidence < CONFIDENCE_THRESHOLD else "Unknown"
        results.append((x, y, w, h, name, confidence))
    return results


# ── Color detection ────────────────────────────────────────────────────────────

def get_color_name(h, s, v):
    if v < 30:
        return "Black"
    if v < 60 and s < 50:
        return "Dark Gray"
    if s < 35:
        if v > 220:
            return "White"
        if v > 185:
            return "Light Gray"
        if v > 110:
            return "Gray"
        return "Dark Gray"

    if 10 <= h <= 25 and s > 60 and v < 80:
        return "Dark Brown"
    if 8 <= h <= 25 and 60 < s <= 160 and 80 <= v < 160:
        return "Brown"
    if 10 <= h <= 30 and 30 < s <= 100 and v >= 150:
        return "Beige"
    if 8 <= h <= 25 and s > 160 and 80 <= v < 160:
        return "Chocolate"

    red_hue = h <= 10 or h >= 165
    if red_hue:
        if s > 180 and v < 90:
            return "Maroon"
        if s > 150 and v > 160:
            return "Red"
        if s > 100 and v > 100:
            return "Dark Red"
        if s <= 150 and v > 160:
            return "Pink"
        if s <= 100 and v > 140:
            return "Light Pink"
        return "Red"

    if 10 < h <= 14:
        if s > 180 and v > 150:
            return "Orange Red"
        if s > 100 and v > 120:
            return "Coral"
        if 80 < s <= 180 and v > 150:
            return "Salmon"

    if 14 < h <= 25:
        if s > 200 and v > 180:
            return "Orange"
        if s > 150 and v > 140:
            return "Dark Orange"
        if 80 < s <= 200 and v > 160:
            return "Peach"

    if 25 < h <= 33:
        if s > 180 and v > 180:
            return "Yellow"
        if s > 120 and 100 <= v <= 200:
            return "Gold"
        if s <= 120 and v > 160:
            return "Cream"

    if 33 < h <= 48:
        if v < 120:
            return "Olive"
        if s > 150:
            return "Yellow Green"
        return "Light Olive"

    if 48 < h <= 85:
        if s > 180 and v > 180:
            return "Lime Green"
        if v < 80:
            return "Dark Green"
        if v < 140 and s > 100:
            return "Forest Green"
        if s < 100 and v > 160:
            return "Mint"
        return "Green"

    if 85 < h <= 100:
        if v < 120:
            return "Teal"
        if s > 150:
            return "Cyan"
        return "Light Cyan"

    if 100 < h <= 115:
        if s < 120 and v > 180:
            return "Sky Blue"
        if v < 100:
            return "Navy Blue"
        if s > 180:
            return "Azure"
        return "Light Blue"

    if 115 < h <= 130:
        if v < 100:
            return "Dark Blue"
        if s > 180 and v > 150:
            return "Blue"
        if s > 120:
            return "Royal Blue"
        return "Steel Blue"

    if 130 < h <= 145:
        if v < 100:
            return "Dark Indigo"
        return "Indigo"

    if 145 < h <= 158:
        if s < 120 and v > 160:
            return "Lavender"
        if v < 100:
            return "Dark Purple"
        return "Purple"

    if 158 < h <= 168:
        if s > 180 and v > 160:
            return "Magenta"
        if s > 100 and v > 140:
            return "Violet"
        if s < 120:
            return "Lavender"
        return "Purple"

    if 168 < h <= 179:
        if s > 180 and v > 180:
            return "Hot Pink"
        if s > 100 and v > 150:
            return "Deep Pink"
        return "Pink"

    return "Unknown"


def hsv_to_bgr(h, s, v):
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)
    return int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2])


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_login_screen(frame, face_results, status_text, status_color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "Face Login", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    for (x, y, w, h, name, conf) in face_results:
        color = (0, 220, 0) if name != "Unknown" else (0, 100, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{name}  ({100 - int(conf):.0f}% match)" if name != "Unknown" else "Not recognized"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Status bar at bottom
    h_frame = frame.shape[0]
    cv2.rectangle(frame, (0, h_frame - 45), (frame.shape[1], h_frame), (20, 20, 20), -1)
    cv2.putText(frame, status_text, (20, h_frame - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q to quit", (frame.shape[1] - 100, h_frame - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)


def draw_welcome_screen(frame, name):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text = f"Welcome, {name}!"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    tx = (frame.shape[1] - size[0]) // 2
    ty = frame.shape[0] // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "Starting color identifier...", (tx + 20, ty + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)


def draw_color_overlay(frame, cx, cy, color_name, hex_str, r, g, b, swatch_bgr, username):
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (330, 140), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

    sw_b, sw_g, sw_r = swatch_bgr
    cv2.rectangle(frame, (22, 22), (90, 90), (sw_b, sw_g, sw_r), -1)
    cv2.rectangle(frame, (22, 22), (90, 90), (220, 220, 220), 1)

    cv2.putText(frame, color_name, (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, hex_str, (100, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"RGB  {r}, {g}, {b}", (100, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    hint = f"Logged in: {username}  |  Move mouse to sample  |  Q to quit"
    cv2.putText(frame, hint, (12, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1, cv2.LINE_AA)

    # Crosshair
    for dx1, dx2, dy1, dy2 in [(-14, -4, 0, 0), (4, 14, 0, 0), (0, 0, -14, -4), (0, 0, 4, 14)]:
        cv2.line(frame, (cx+dx1, cy+dy1), (cx+dx2, cy+dy2), (0, 0, 0), 3)
        cv2.line(frame, (cx+dx1, cy+dy1), (cx+dx2, cy+dy2), (255, 255, 255), 1)

    cv2.circle(frame, (cx, cy), 3, (sw_b, sw_g, sw_r), -1)
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    recognizer, labels = load_face_model()
    skip_login = recognizer is None
    if skip_login:
        print("No face model found. Run enroll.py to register. Starting without login.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    win = "Color Identifier"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    state = STATE_COLOR_ID if skip_login else STATE_LOGIN
    logged_in_user = "Guest" if skip_login else None
    welcome_start = None
    frame_count = 0
    last_face_results = []
    status_text = "Look at the camera to log in"
    status_color = (180, 180, 180)
    sample_radius = 6

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]
        frame_count += 1

        # ── Login state ──────────────────────────────────────────────────────
        if state == STATE_LOGIN:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % DETECT_EVERY_N == 0:
                last_face_results = detect_and_recognize(gray, recognizer, labels)
                recognized = [(x, y, w, h, n, c) for (x, y, w, h, n, c) in last_face_results if n != "Unknown"]
                if recognized:
                    _, _, _, _, name, _ = recognized[0]
                    logged_in_user = name
                    state = STATE_WELCOME
                    welcome_start = time.time()
                elif last_face_results:
                    status_text = "Face detected — not recognized"
                    status_color = (0, 100, 255)
                else:
                    status_text = "Look at the camera to log in"
                    status_color = (180, 180, 180)

            draw_login_screen(frame, last_face_results, status_text, status_color)

        # ── Welcome state ────────────────────────────────────────────────────
        elif state == STATE_WELCOME:
            draw_welcome_screen(frame, logged_in_user)
            if time.time() - welcome_start > 2.5:
                state = STATE_COLOR_ID

        # ── Color ID state ───────────────────────────────────────────────────
        elif state == STATE_COLOR_ID:
            cx = mouse_x if 0 <= mouse_x < w_frame else w_frame // 2
            cy = mouse_y if 0 <= mouse_y < h_frame else h_frame // 2

            y1 = max(0, cy - sample_radius)
            y2 = min(h_frame, cy + sample_radius)
            x1 = max(0, cx - sample_radius)
            x2 = min(w_frame, cx + sample_radius)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_med = int(np.median(hsv_roi[:, :, 0]))
            s_med = int(np.median(hsv_roi[:, :, 1]))
            v_med = int(np.median(hsv_roi[:, :, 2]))

            color_name = get_color_name(h_med, s_med, v_med)
            swatch_bgr = hsv_to_bgr(h_med, s_med, v_med)

            bgr_median = np.median(roi.reshape(-1, 3), axis=0).astype(int)
            b_val, g_val, r_val = int(bgr_median[0]), int(bgr_median[1]), int(bgr_median[2])
            hex_str = f"#{r_val:02X}{g_val:02X}{b_val:02X}"

            draw_color_overlay(frame, cx, cy, color_name, hex_str,
                               r_val, g_val, b_val, swatch_bgr, logged_in_user)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
