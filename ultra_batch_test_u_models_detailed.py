import os
from ultralytics import YOLO
from pathlib import Path
import time

# Define model names to test
models = [
    'yolov5nu.pt',
    'yolov5su.pt',
    'yolov5mu.pt',
    'yolov5lu.pt',
    'yolov5xu.pt',
    'yolov5n6u.pt',
    'yolov5s6u.pt',
    'yolov5m6u.pt',
    'yolov5l6u.pt',
    'yolov5x6u.pt',
]

# Define test images with full paths
images = [
    ('448x320', '/home/raspberry/new_tests/test_448x320.jpg'),
    ('640x640', '/home/raspberry/new_tests/test_640x640.jpg'),
    ('800x608', '/home/raspberry/new_tests/test_800x608.jpg'),
    ('1024x768', '/home/raspberry/new_tests/test_1024x768.jpg'),
    ('1280x1280', '/home/raspberry/new_tests/test_1280x1280.jpg'),
    ('1920x1088', '/home/raspberry/new_tests/test_1920x1088.jpg'),
]

# Output root
base_output = Path('ultra_results')
base_output.mkdir(exist_ok=True)

# Loop through models and images
for model_name in models:
    print(f"\n=== Running model: {model_name} ===")
    model = YOLO(model_name)
    model_output_dir = base_output / Path(model_name).stem

    for label, image_path in images:
        print(f"\n--- Processing image: {image_path} at {label} ---")
        t1 = time.time()
        results = model.predict(
            source=image_path,
            save=True,
            project=str(model_output_dir),
            name=label,
            exist_ok=True,
            conf=0.25,
            imgsz=640
        )
        t2 = time.time()
        result = results[0]
        num_objects = len(result.boxes)
        avg_conf = (
            sum(result.boxes.conf.cpu().numpy()) / num_objects
            if num_objects > 0 else 0
        )
        print(f"Detected {num_objects} objects, avg conf: {avg_conf:.2f}")
        print(f"Inference time: {(t2 - t1):.3f} seconds")
        print(f"FPS (simulated): {1 / (t2 - t1):.2f}")
        print(f"Saved result to {model_output_dir}/{label}")
