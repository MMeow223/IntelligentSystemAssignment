from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/classify/train18/weights/best.pt')  # load a custom model

    # Validate the model
    metrics = model.val(data="data_custom.yaml")  # no arguments needed, dataset and settings remembered
    metrics.top1   # top1 accuracy
    metrics.top5   # top5 accuracy
