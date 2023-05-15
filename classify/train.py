from ultralytics import YOLO



if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='D:/IntelligentSystem/classify/dataset/', epochs=200,save_period=5, imgsz=512, batch=16, verbose=True,lr0=0.01,lrf=0.01)
