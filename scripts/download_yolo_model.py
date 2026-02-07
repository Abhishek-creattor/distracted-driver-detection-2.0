from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
OUT = Path("models/yolov8n.pt")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Model already exists: {OUT}")
        return
    print("Downloading YOLOv8 model...")
    urlretrieve(URL, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
