import os
from ultralytics import YOLO


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(script_dir, "assets", "models", "yolo26n.pt"))

    model.export(
        format="engine",
        imgsz=640,
        half=True,
        batch=1,
        dynamic=False,
        device=0,
    )


if __name__ == "__main__":
    main()
