from ultralytics import YOLO

model = YOLO('C:/Users/ACER/Desktop/Helmet_Detection/runs/detect/train2/weights/best.pt')


results = model(source='Video2.mp4',show=True,conf=0.4,save=True)
