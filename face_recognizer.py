from pathlib import Path
import importlib.util

import cv2
import numpy as np

if importlib.util.find_spec("face_recognition_models") is None:
    face_recognition = None
    FACE_RECOGNITION_IMPORT_ERROR = "Missing package: face_recognition_models"
else:
    try:
        import face_recognition_models
        import face_recognition
        FACE_RECOGNITION_IMPORT_ERROR = None
    except BaseException as error:
        face_recognition = None
        FACE_RECOGNITION_IMPORT_ERROR = error


class FaceRecognizer:
    """Load known faces from dataset folders and match webcam faces."""

    def __init__(self, dataset_dir="dataset", tolerance=0.42, min_distance_gap=0.05):
        self.dataset_dir = Path(dataset_dir)
        self.tolerance = tolerance
        self.min_distance_gap = min_distance_gap
        self.known_encodings = []
        self.known_names = []
        self.known_genders = []

        if face_recognition is None:
            print(
                "face_recognition is not ready. Recognition is disabled; "
                "all faces will be marked as Unknown."
            )
            print(f"Reason: {FACE_RECOGNITION_IMPORT_ERROR}")
            return

        self.load_dataset()

    def load_dataset(self):
        """Read dataset/men and dataset/women images and create face encodings."""
        if not self.dataset_dir.exists():
            print(f"Dataset folder not found: {self.dataset_dir}")
            return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            path
            for path in self.dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print("No dataset images found. Add images inside dataset/men or dataset/women.")
            return

        loaded_counts = {"Male": 0, "Female": 0, "Unknown": 0}

        for image_path in image_paths:
            name = self._name_from_file(image_path)
            gender = self._gender_from_folder(image_path)
            encoding = self._encode_dataset_image(image_path)

            if encoding is None:
                print(f"Skipping {image_path}: no face encoding found.")
                continue

            self.known_encodings.append(encoding)
            self.known_names.append(name)
            self.known_genders.append(gender)
            loaded_counts[gender] += 1

        print(
            "Loaded "
            f"{len(self.known_encodings)} known face(s): "
            f"{loaded_counts['Male']} male, "
            f"{loaded_counts['Female']} female, "
            f"{loaded_counts['Unknown']} unknown folder."
        )

    def _encode_dataset_image(self, image_path):
        """Return the first face encoding found in one dataset image."""
        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            return None

        return encodings[0]

    @staticmethod
    def _name_from_file(image_path):
        """Use filename as the display name: dataset/men/john_doe.jpg -> John Doe."""
        return image_path.stem.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _gender_from_folder(image_path):
        """Use the dataset parent folder as the known gender label."""
        folder_name = image_path.parent.name.lower()

        if folder_name in {"men", "man", "male"}:
            return "Male"

        if folder_name in {"women", "woman", "female"}:
            return "Female"

        return "Unknown"

    def match_face(self, frame_bgr, face_box):
        """Match one detected webcam face against the loaded dataset."""
        if face_recognition is None or not self.known_encodings:
            return {
                "name": "Unknown",
                "dataset_gender": None,
                "distance": None,
                "status": "recognition_not_ready",
            }

        x, y, w, h = face_box
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # face_recognition expects boxes as (top, right, bottom, left).
        location = (y, x + w, y + h, x)
        encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=[location])

        if not encodings:
            return {
                "name": "Unknown",
                "dataset_gender": None,
                "distance": None,
                "status": "face_encoding_failed",
            }

        face_encoding = encodings[0]
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)

        if len(distances) == 0:
            return {
                "name": "Unknown",
                "dataset_gender": None,
                "distance": None,
                "status": "no_known_faces",
            }

        best_match_index = int(np.argmin(distances))
        best_distance = float(distances[best_match_index])
        sorted_distances = np.sort(distances)
        second_best_distance = (
            float(sorted_distances[1]) if len(sorted_distances) > 1 else 1.0
        )
        distance_gap = second_best_distance - best_distance

        if best_distance <= self.tolerance and distance_gap >= self.min_distance_gap:
            return {
                "name": self.known_names[best_match_index],
                "dataset_gender": self.known_genders[best_match_index],
                "distance": best_distance,
                "status": "matched",
            }

        return {
            "name": "Unknown",
            "dataset_gender": None,
            "distance": best_distance,
            "status": "not_matched",
        }
