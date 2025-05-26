from ultralytics import YOLO
from pathlib import Path
import time

# Define model-task pairs
models_tasks = {
    'yolov8n-seg.pt': 'segment',
    'yolov8n-pose.pt': 'pose',
    'yolov8n-cls.pt': 'classify'
}

# Define test images
images = [
    ('448x320', '/home/raspberry/new_tests/test_448x320.jpg'),
    ('640x640', '/home/raspberry/new_tests/test_640x640.jpg'),
    ('800x608', '/home/raspberry/new_tests/test_800x608.jpg'),
    ('1024x768', '/home/raspberry/new_tests/test_1024x768.jpg'),
    ('1280x1280', '/home/raspberry/new_tests/test_1280x1280.jpg'),
    ('1920x1088', '/home/raspberry/new_tests/test_1920x1088.jpg'),
]

# Output root
base_output = Path('yolov8_special_results')
base_output.mkdir(exist_ok=True)

for model_name, task in models_tasks.items():
    print(f"\n=== Running model: {model_name} (task: {task}) ===")
    model = YOLO(model_name)
    model_output_dir = base_output / Path(model_name).stem

    for label, image_path in images:
        print(f"\n--- Processing image: {image_path} at {label} ---")
        t1 = time.time()
        results = model.predict(
            source=image_path,
            save=True,
            save_txt=False,
            project=str(model_output_dir),
            name=label,
            exist_ok=True,
            conf=0.25,
            imgsz=640,
            task=task
        )
        t2 = time.time()
        print(f"Inference time: {(t2 - t1):.3f} seconds")
        print(f"Saved result to {model_output_dir}/{label}")
