from ultralytics import YOLO
import shutil
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"#can it solve oom problem?

if __name__ == '__main__':
    from ultralytics import YOLO

    print(f"Train dataset")
    model = YOLO('yolov11n-extra-p2.yaml')
    model.load('yolo11n.pt')#load yolov11n.pt to blank 'yolov11n-extra-p2.yaml
    results = model.train(
        data=f"tennis-ball-detection-6/data.yaml",
        epochs=400,
        imgsz=1280,
        batch=16,
        workers=0,
        device=0, #use gpu
    )