from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json
import os

app = Flask(__name__)

FACES_DIR = "faces"
MODEL_PATH = os.path.join(FACES_DIR, "model.yml")
LABELS_PATH = os.path.join(FACES_DIR, "labels.json")
FACE_SIZE = (100, 100)
CONF_THRESH = 70
SAMPLES_NEEDED = 40

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = None
labels = {}
enroll_buf = {}  # name -> [face_arrays]


def reload_model():
    global recognizer, labels
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        r = cv2.face.LBPHFaceRecognizer_create()
        r.read(MODEL_PATH)
        with open(LABELS_PATH) as f:
            labels = {int(k): v for k, v in json.load(f).items()}
        recognizer = r


def decode_img(data_url):
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def retrain():
    faces, lbls, lmap = [], [], {}
    for person in sorted(
        d for d in os.listdir(FACES_DIR)
        if os.path.isdir(os.path.join(FACES_DIR, d))
    ):
        lid = len(lmap)
        lmap[lid] = person
        for f in os.listdir(os.path.join(FACES_DIR, person)):
            img = cv2.imread(os.path.join(FACES_DIR, person, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                lbls.append(lid)
    if not faces:
        return False
    r = cv2.face.LBPHFaceRecognizer_create()
    r.train(faces, np.array(lbls))
    r.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump({str(k): v for k, v in lmap.items()}, f)
    reload_model()
    return True


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


@app.route("/api/status")
def status():
    users = (
        [d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))]
        if os.path.exists(FACES_DIR) else []
    )
    return jsonify({"has_model": recognizer is not None, "users": users})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if not recognizer:
        return jsonify({"status": "no_model", "name": None, "faces": []})

    img = decode_img(request.json["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if not len(detected):
        return jsonify({"status": "no_face", "name": None, "faces": []})

    out, best_name, best_conf = [], None, float("inf")
    for x, y, w, h in detected:
        roi = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
        lid, conf = recognizer.predict(roi)
        name = labels.get(lid, "Unknown") if conf < CONF_THRESH else "Unknown"
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "name": name, "conf": float(conf)})
        if name != "Unknown" and conf < best_conf:
            best_name, best_conf = name, conf

    return jsonify({
        "status": "recognized" if best_name else "detected",
        "name": best_name,
        "faces": out,
    })


@app.route("/api/enroll/frame", methods=["POST"])
def enroll_frame():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name required"})

    img = decode_img(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    count = len(enroll_buf.get(name, []))

    if not len(detected):
        return jsonify({"ok": False, "count": count, "bbox": None})

    x, y, w, h = detected[0]
    roi = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)

    enroll_buf.setdefault(name, [])
    if count < SAMPLES_NEEDED:
        enroll_buf[name].append(roi)
        count += 1

    done = count >= SAMPLES_NEEDED
    if done:
        person_dir = os.path.join(FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        for f in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, f))
        for i, face in enumerate(enroll_buf.pop(name)):
            cv2.imwrite(os.path.join(person_dir, f"{i:04d}.jpg"), face)
        retrain()

    return jsonify({"ok": True, "count": count, "done": done,
                    "bbox": [int(x), int(y), int(w), int(h)]})


if __name__ == "__main__":
    os.makedirs(FACES_DIR, exist_ok=True)
    reload_model()
    app.run(debug=True, port=5000)
