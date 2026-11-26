from pathlib import Path
from urllib.request import urlretrieve


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FILES = {
    MODELS_DIR
    / "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
    MODELS_DIR
    / "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    MODELS_DIR
    / "gender_net.caffemodel": "https://www.dropbox.com/scl/fi/j7gcbj1l3ur6r8jdf3k7k/gender_net.caffemodel?rlkey=g7us6rvnw8ji7z64d5zy002h4&dl=1",
}


def download_file(destination, url):
    """Download one file unless it already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {destination.name}...")
    urlretrieve(url, destination)
    print(f"Saved to: {destination}")


def main():
    for destination, url in FILES.items():
        download_file(destination, url)

    print("\nAll model files are ready.")
    print("Desktop app: python main.py")
    print("Web UI:      python app.py")


if __name__ == "__main__":
    main()
