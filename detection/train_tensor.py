from ultralytics import YOLO
import os
# import matplotlib.pyplot as plt


# def on_train_epoch_end(model):
    
    # model.val(data="D:/IntelligentSystem/YOLOV8/data_custom.yaml")  # no arguments needed, dataset and settings remembered
    # Update the chart with the latest metrics
    # plt.plot(epoch, loss, label='Training Loss')
    # plt.plot(epoch, accuracy, label='Training Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Metric Value')
    # plt.legend()
    # plt.show()
    
    
    
    

if __name__ == '__main__':
            
    # Create a YOLO model instance
    model = YOLO('yolov8n.pt')
    
    # add tensorboard callback for the model

    # Add the custom callback to the model
    # model.add_callback("on_train_epoch_end", on_train_epoch_end(model))

    # # Iterate through the results and frames
    # for (result, frame) in model.track/predict():
    #     pass

    model.train(data="data_custom.yaml",
                save=True,
                epochs=100,
                batch=4,
                imgsz=640,
                model="yolov8m.pt",
                val=True,
                seed=888,
                pretrained=True,
        )