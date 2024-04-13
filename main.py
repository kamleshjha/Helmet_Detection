from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="data_config.yaml", epochs=50)