from ultralytics import YOLO
import os

# for files name "train" in folder "D:/AI/runs/detect"

if __name__ == '__main__':
        
        # Load a model
        # model = YOLO()
        
        model = YOLO('D:/IntelligentSystem/YOLOV8/yolov8m.pt')  # load a custom model
        
        
        # export model 
        # model.export()  # creates onnx, coreml, and torchscript models
        
        
        # let the model scramble the dataset
        model.train(data="data_custom.yaml",
                    save=True,
                    epochs=150,
                    batch=4,
                    imgsz=640,
                    model="yolov8m.yaml",
                    lrf=0.0005,
                    lr0=0.0005,
                    save_period=5
            )
        
        # plot the training results in real time
        # model.plot_results()
        

        # Validate the model with test data
        # metrics = model.val(data="D:/IntelligentSystem/YOLOV8/data_custom.yaml",save=True)  # no arguments needed, dataset and settings remembered


        # use tensorboard to show the results
        # result = model.tensorboard()  # plot results to val/train2020-12-07_14-06-01
        
        