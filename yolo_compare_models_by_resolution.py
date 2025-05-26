import sys
sys.path.append('/home/raspberry/yolov5')

import time
import torch
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.dataloaders import LoadImages
from utils.plots import Annotator
from utils.torch_utils import select_device

# Set model weight path (change this to test different YOLOv5 variants)
weights_path = 'yolov5n.pt'  # Replace with yolov5s.pt, yolov5m.pt etc. as needed

# Define test images with specific resolution
images = [
    ('640x640', 'test_640x640.jpg', 640),
    ('1280x1280', 'test_1280x1280.jpg', 1280),
    ('1920x1088', 'test_1920x1088.jpg', 1920),
    ('2560x1920', 'test_2560x1920.jpg', 2560),
    ('1024x768', 'test_1024x768.jpg', 1024),
    ('800x608', 'test_800x608.jpg', 800),
]

# Load YOLO model
device = select_device('cpu')
model = DetectMultiBackend(weights_path, device=device, dnn=False)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, 640, 640))

# Get model name for saving
model_name = Path(weights_path).stem

# Process each image
for label, img_path, img_size in images:
    print(f"\n--- Processing image: {img_path} at {label} ---")
    dataset = LoadImages(img_path, img_size=img_size, stride=stride, auto=True)

    for path, img, im0s, _, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time.time()
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)
        t2 = time.time()
        fps = 1 / (t2 - t1)

        for i, det in enumerate(pred):
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, example=names)

            if len(det):
                total_conf = sum(conf.item() for *_, conf, _ in det)
                avg_conf = total_conf / len(det)
                for *xyxy, conf, cls in det:
                    label_text = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label_text)
                print(f"Detected {len(det)} objects, avg conf: {avg_conf:.2f}")
            else:
                print("No objects detected.")

            print(f"Inference time: {(t2 - t1):.3f} seconds")
            print(f"FPS (simulated): {fps:.2f}")

            # Save results by model and resolution
            save_dir = Path(f"res_compare/{model_name}/{label}")
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / Path(img_path).name
            cv2.imwrite(str(out_path), annotator.result())
            print(f"Saved result to {out_path}")
