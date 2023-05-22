from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/model8/weights/best.pt')
    metrics = model.val(data="data_custom.yaml", save=True)