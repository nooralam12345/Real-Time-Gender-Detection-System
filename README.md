# Smart Real-Time Gender Detection and Face Recognition System with Web UI

This project detects faces from a webcam, predicts gender with a pre-trained Caffe CNN model, recognizes known people from a manual dataset, saves detected face crops automatically, and streams the processed video through a Flask web UI.

## Project structure

```text
.
├── app.py
├── main.py
├── face_recognizer.py
├── download_models.py
├── requirements.txt
├── models/
│   ├── haarcascade_frontalface_default.xml
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── dataset/
│   ├── men/
│   └── women/
├── static/
│   └── captured/
└── templates/
    └── index.html
```

## Features

- Live webcam capture with OpenCV
- Haar Cascade real-time face detection
- Pre-trained Caffe CNN gender classification
- Multiple face support
- Face recognition from `dataset/men` and `dataset/women`
- Label format: recognized name + gender + confidence score
- Automatic face image capture every 5 seconds
- Saved image format: `name_gender_timestamp.jpg`
- FPS counter on video
- Desktop OpenCV mode with `q` to quit
- Flask web UI at `/` with live stream at `/video`

## Setup

Use PowerShell from the project folder:

```powershell
cd "D:\final-project face analizer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Important Windows note: `face-recognition` depends on `dlib` and `face-recognition-models`. If installation fails on Python 3.13, use Python 3.10 or 3.11, or install a compatible prebuilt `dlib` wheel first.

If the app asks for `face_recognition_models`, install it with:

```powershell
python -m pip install face-recognition-models
```

If that does not work and Git is installed, use:

```powershell
python -m pip install git+https://github.com/ageitgey/face_recognition_models
```

## Download model files

Run:

```powershell
python download_models.py
```

The downloader places these files in `models/`:

- Haar cascade: https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
- Gender prototxt: https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt
- Gender Caffe model: https://www.dropbox.com/scl/fi/j7gcbj1l3ur6r8jdf3k7k/gender_net.caffemodel?rlkey=g7us6rvnw8ji7z64d5zy002h4&dl=1

The Caffe model is from the LearnOpenCV age/gender example: https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/

## Add known faces

Put clear face photos in:

```text
dataset/
├── men/
│   ├── john_doe.jpg
│   └── rahul.png
└── women/
    ├── jane_doe.jpg
    └── priya.png
```

The filename becomes the displayed person name. For example, `john_doe.jpg` displays as `John Doe`.

The folder is only used to organize your manual dataset:

- Images inside `dataset/men` are treated as `Male`
- Images inside `dataset/women` are treated as `Female`

The live label uses the CNN gender prediction. Face recognition is used only to display the person's name when a strong dataset match is found. This avoids a bad dataset match forcing every face to show as `Male` or `Female`.

Use one face per dataset image for best results. If a dataset image has no usable face encoding, the app skips it and prints a message.

Good dataset image rules:

- Use clear, front-facing face photos.
- Use one person per image.
- Avoid sunglasses, side faces, heavy blur, and very dark lighting.
- Restart `python app.py` after adding new dataset images.

## Run desktop webcam app

```powershell
python main.py
```

Press `q` in the OpenCV window to quit safely.

## Run Flask web UI

```powershell
python app.py
```

Open this in your browser:

```text
http://127.0.0.1:5001
```

The live video stream route is:

```text
http://127.0.0.1:5001/video
```

The Flask web app starts with face-name recognition disabled so the browser opens quickly and gender detection is not affected by a noisy dataset. The desktop `main.py` mode still loads dataset recognition.

## Captured images

Detected faces are saved automatically in:

```text
static/captured/
```

The capture delay is set to 5 seconds in `SmartGenderFaceSystem`. You can change it in `main.py` by passing a different `capture_delay_seconds` value.
