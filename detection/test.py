from ultralytics import YOLO
import os

if __name__ == '__main__':
        
    model = YOLO('D:/IntelligentSystem/YOLOV8/runs/detect/train37/weights/best.pt')  # load a custom model
    metrics = model.val(data="D:/IntelligentSystem/YOLOV8/data_custom.yaml", save=True)  # no arguments needed, dataset and settings remembered
    