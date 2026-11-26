import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request
import numpy as np

from main import SmartGenderFaceSystem


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
camera = cv2.VideoCapture(0)
detection_system = SmartGenderFaceSystem(enable_recognition=False)
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def generate_frames():
    """Continuously read webcam frames and stream them as JPEG images."""
    previous_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 0.0001)
        previous_time = current_time

        frame = detection_system.process_frame(frame, fps=fps, save_captures=True)
        encoded, buffer = cv2.imencode(".jpg", frame)

        if not encoded:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    """Main web page."""
    return render_template("index.html")


@app.route("/video")
def video():
    """Live webcam video stream route."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/predict-image", methods=["POST"])
def predict_image():
    """Classify gender from one uploaded image using the same CNN pipeline."""
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Please choose an image first."}), 400

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not read this image file."}), 400

    faces = detection_system.detect_faces(image)
    if len(faces) == 0:
        face_crop = image
    else:
        # Use the largest face if the uploaded image contains multiple people.
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_crop = detection_system.crop_face_with_padding(image, x, y, w, h)

    gender, confidence = detection_system.predict_gender(face_crop)

    return jsonify(
        {
            "gender": gender,
            "confidence": round(confidence * 100, 1),
            "facesFound": int(len(faces)),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
