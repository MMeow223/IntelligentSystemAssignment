from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train44/weights/best.pt')
    metrics = model.val(data="data_custom.yaml", save=True)