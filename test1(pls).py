from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data="/home/kavin/Downloads/arrows/data/data.yaml",epochs=3)