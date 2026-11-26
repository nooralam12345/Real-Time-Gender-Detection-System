from pathlib import Path
import re
import time

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CAPTURED_DIR = BASE_DIR / "static" / "captured"
DATASET_DIR = BASE_DIR / "dataset"

HAAR_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
GENDER_PROTO_PATH = MODELS_DIR / "gender_deploy.prototxt"
GENDER_MODEL_PATH = MODELS_DIR / "gender_net.caffemodel"

GENDER_LABELS = ["Male", "Female"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class SmartGenderFaceSystem:
    """Reusable webcam processing system for CLI and Flask streaming."""

    def __init__(
        self,
        dataset_dir=DATASET_DIR,
        captured_dir=CAPTURED_DIR,
        capture_delay_seconds=5,
        enable_recognition=True,
    ):
        self.captured_dir = Path(captured_dir)
        self.captured_dir.mkdir(parents=True, exist_ok=True)
        self.capture_delay_seconds = capture_delay_seconds
        self.last_capture_times = {}

        self.face_detector = self._load_face_detector()
        self.gender_net = self._load_gender_model()
        if enable_recognition:
            from face_recognizer import FaceRecognizer

            self.face_recognizer = FaceRecognizer(dataset_dir)
        else:
            self.face_recognizer = None

    def _load_face_detector(self):
        """Load Haar Cascade face detector from the models folder."""
        if not HAAR_CASCADE_PATH.exists():
            raise FileNotFoundError(f"Missing Haar cascade: {HAAR_CASCADE_PATH}")

        detector = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
        if detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {HAAR_CASCADE_PATH}")

        return detector

    def _load_gender_model(self):
        """Load the pre-trained Caffe CNN gender model."""
        if not GENDER_PROTO_PATH.exists() or not GENDER_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Missing gender model files. Run: python download_models.py"
            )

        return cv2.dnn.readNet(str(GENDER_MODEL_PATH), str(GENDER_PROTO_PATH))

    def detect_faces(self, frame):
        """Detect all visible faces in a frame using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    def predict_gender(self, face_image):
        """Predict gender and confidence from a cropped face image."""
        blob = cv2.dnn.blobFromImage(
            face_image,
            scalefactor=1.0,
            size=(227, 227),
            mean=MODEL_MEAN_VALUES,
            swapRB=False,
            crop=False,
        )

        self.gender_net.setInput(blob)
        predictions = self.gender_net.forward()[0]
        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])

        return GENDER_LABELS[class_id], confidence

    @staticmethod
    def crop_face_with_padding(frame, x, y, w, h, padding_ratio=0.2):
        """Crop a face with a little surrounding context for better CNN results."""
        frame_height, frame_width = frame.shape[:2]
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)

        x1 = max(x - padding_x, 0)
        y1 = max(y - padding_y, 0)
        x2 = min(x + w + padding_x, frame_width)
        y2 = min(y + h + padding_y, frame_height)

        return frame[y1:y2, x1:x2]

    def save_face_if_needed(self, face_image, name, gender):
        """Save a detected face every few seconds to avoid rapid duplicates."""
        now = time.time()
        capture_key = f"{name}_{gender}"
        last_capture_time = self.last_capture_times.get(capture_key, 0)

        if now - last_capture_time < self.capture_delay_seconds:
            return None

        safe_name = self._safe_filename_part(name if name else "Unknown")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{gender}_{timestamp}.jpg"
        output_path = self.captured_dir / filename

        cv2.imwrite(str(output_path), face_image)
        self.last_capture_times[capture_key] = now
        return output_path

    @staticmethod
    def _safe_filename_part(value):
        """Make a name safe for Windows/macOS/Linux filenames."""
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        return cleaned or "Unknown"

    @staticmethod
    def draw_label(frame, text, x, y, color):
        """Draw a filled label background with readable text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        top = max(y - text_height - baseline - 8, 0)
        bottom = top + text_height + baseline + 8

        cv2.rectangle(frame, (x, top), (x + text_width + 10, bottom), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 5, bottom - baseline - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_fps(frame, fps):
        """Display FPS in the top-left corner."""
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def process_frame(self, frame, fps=None, save_captures=True):
        """Detect, recognize, classify, capture, and annotate every face."""
        faces = self.detect_faces(frame)

        for x, y, w, h in faces:
            face_image = self.crop_face_with_padding(frame, x, y, w, h)
            if face_image.size == 0:
                continue

            if self.face_recognizer is None:
                match = {"name": "Unknown"}
            else:
                match = self.face_recognizer.match_face(frame, (x, y, w, h))
            name = match["name"]
            ai_gender, confidence = self.predict_gender(face_image)

            if save_captures:
                self.save_face_if_needed(face_image, name, ai_gender)

            if name != "Unknown":
                label = f"{name} | {ai_gender}: {confidence * 100:.1f}%"
            else:
                label = f"Unknown | {ai_gender}: {confidence * 100:.1f}%"

            color = (0, 180, 0) if name != "Unknown" else (0, 120, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            self.draw_label(frame, label, x, y, color)

        if fps is not None:
            self.draw_fps(frame, fps)

        return frame


def run_webcam():
    """Run the desktop OpenCV webcam window."""
    system = SmartGenderFaceSystem()
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    previous_time = time.time()
    print("Webcam started. Press 'q' to quit.")

    while True:
        success, frame = webcam.read()
        if not success:
            print("Could not read frame from webcam.")
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 0.0001)
        previous_time = current_time

        frame = system.process_frame(frame, fps=fps, save_captures=True)
        cv2.imshow("Smart Real-Time Gender Detection and Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
