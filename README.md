# Automated Fall Detection System
This is a repository for the  ***COS30018 Intelligent System*** Assignment. The project is about building an automated fall detection system using two approach, One-Step and Two-Step approach.

# **Instruction for Custom CNN Model**
1. The Custom CNN Classification model can be trained and tested in the google colab.

|                 | URL                                                                                                                                                           |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Google Drive    | [https://drive.google.com/drive/u/0/folders/1Eo_hgHU0jV9p8iKSjPvWsNU_0mIQlgED](#https://drive.google.com/drive/u/0/folders/1Eo_hgHU0jV9p8iKSjPvWsNU_0mIQlgED) |
| Python Notebook | [https://colab.research.google.com/drive/1ZQpeDb0ca8KJH77eLzCfHTPm6OrhYhL1](#https://colab.research.google.com/drive/1ZQpeDb0ca8KJH77eLzCfHTPm6OrhYhL1)       |


# **Installation Instruction for YOLOv5**


##### **Train Model**
[1. Train Object Detection Model](#train-object-detection-modelyolov5) <br>

##### **Evaluate Model**
[1. Evaluate Object Detection Model](#evaluate-object-detection-modelyolov5) <br>

##### **Live Predict with Model**
[1. Live Predict One-Step Model](#live-predict-one-step-modelyolov5) <br>

## Setup
Before starting to run train, test, live prediction, you need to perform the step below.
### 1. Clone this repository
You can clone this repository by using the command below.
###### Open a terminal and paste the command below
`git clone https://github.com/MMeow223/IntelligentSystemAssignment.git`

### 2. Install packages
Assuming you have anaconda installed on your device. You will need to create an new anaconda environment and install required packages with the following command.
*Make sure to change the directory to yolov5 in the terminal
`cd yolov5`
###### Open a terminal and paste the command below
`conda create -n is_env python=3.9`
###### Activate the anaconda environment
`conda activate is_env`
###### Install required packages
`pip install -r requirements.txt`

### 3. Update the dataset directory
The dataset directory has already been set to `./dataset` by default for train, val and test.

If changes were to be made to the directory of the dataset, you can change it by opening the file and change the path to the dataset directory in your device.

###### In data/coco128.yaml
1. Change the line `train: ./dataset/train`, `val: ./dataset/val` or `test: ./dataset/test`to the changed path of your own device

Now, your dataset are ready.

## **Train Object Detection Model(YOLOv5)**
The example below uses yolov5m pre-trained check point, but you can change it to other v5 model such as `yolov5n`,`yolov5s` and etc.
To change the Hyperparameters such as Learning rate, it can be change in `data/hyps/hyp.scratch-low` file along with with other parameters.
###### Open the terminal,make sure the directory is yolov5 and paste the commands below. 
```
python train.py --img 640 --epochs 100 --data coco128.yaml --weights yolov5m.pt
```

## **Evaluate Object Detection Model(YOLOv5)**
To evaluate different model, you can change the `exp` in `--weights runs/train/exp/weights/best.pt` to other `exp` file.
###### Open the terminal,make sure the directory is yolov5 and paste the commands below. 
```
python val.py --weights runs/train/exp/weights/best.pt --data coco128.yaml --img 640 --half --task test
```

## **Live Predict One-Step Model(YOLOv5)**
To do live predict with different model, you can change the `exp` in `--weights runs/train/exp/weights/best.pt` to other `exp` file. `--source 0` indicates using web-cam.
###### Open the terminal,make sure the directory is yolov5 and paste the commands below. 
```
python detect.py --weights runs/train/exp/weights/best.pt --source 0
```

# **Installation Instruction for YOLOv8**

##### **Train Model**
[1. Train Object Detection Model](#train-object-detection-model) <br>
[2. Train Image Classification Model](#train-image-classification-model)

##### **Evaluate Model**
[1. Evaluate Object Detection Model](#evaluate-object-detection-model) <br>
[2. Evaluate Image Classification Model](#evaluate-image-classification-model)

##### **Live Predict with Model**
[1. Live Predict One-Step Model](#live-predict-one-step-model) <br>
[2. Live Predict Two-Step Model](#live-predict-two-step-model)

##### **Run Flask Web Application**
[1. Run Web Application](#run-web-application)

## Setup
Before starting to run train, test, live prediction, and web application, you need to perform the step below.
### 1. Clone this repository
You can clone this repository by using the command below.
###### Open a terminal and paste the command below
`git clone https://github.com/MMeow223/IntelligentSystemAssignment.git`


### 2. Install packages
Assuming you have anaconda installed on your device. You will need to create an new anaconda environment with the following command.
###### Open a terminal and paste the command below
`conda create --name is_env --file requirements.txt`
###### Activate the anaconda environment
`conda activate is_env`

Now, your environment is ready.
### 3. Update the dataset directory
Once you pulling this repository, you will need to update the dataset directory. You should see a directory named "dataset" in the "detection" and "classification" directory. You can also see "data_custom.yaml" files in both of these directory.

You will need to update the path of the dataset directory in the "data_custom.yaml" file. You can do it by opening the file and change the path to the dataset directory in your device.

###### In detection/data_custom.yaml
1. Change the line `path: D:\IntelligentSystem\IntelligentSystemAssignment\detection\dataset` to the absolute path of `detection\dataset` in your device.

###### In classify/data_custom.yaml
1. Change the line `path: D:\IntelligentSystem\IntelligentSystemAssignment\classify\dataset` to the absolute path of `classify\dataset` in your device.


Now, your dataset are ready.



## **Train Object Detection Model**
The file for training Object Detection Model of One-Step Model is ready in `/detection/train.py`. By default, it will use the yolov8m pre-trained check point, but you can change it to any other yolo models as you like.
###### Open the terminal from the repository directory and paste the commands below. 
```
cd detection
python train.py
```

## **Train Image Classification Model**
The file for training Object Detection Model of One-Step Model is ready in ```/classify/train.py```. By default, it will use the yolov8m-cls pre-trained check point, but you can change it to any other yolo models as you like.
###### Open the terminal from the repository directory and paste the commands below. 
```
cd classify
python train.py
```

## **Evaluate Object Detection Model**
The file for training Object Detection Model of One-Step Model is ready in ```/detection/test.py```. By default, it will use the yolov8m-cls pre-trained check point, but you can change it to any other yolo models as you like.
###### Open the terminal from the repository directory and paste the commands below. 
```
cd detection
python test.py
```

## **Evaluate Image Classification Model**
The file for training Object Detection Model of One-Step Model is ready in ```/classify/test.py```. By default, it will use the yolov8m-cls pre-trained check point, but you can change it to any other yolo models as you like.
###### Open the terminal from the repository directory and paste the commands below. 
```
cd classify
python test.py
```

## **Live Predict One-Step Model**
The file for live predict with One-Step model is ready in ```/detection/live_predict.py```.
###### Open the terminal from the repository directory and paste the commands below. 
```
cd detection
python live_predict.py
```

## **Live Predict Two-Step Model**
The file for live predict with Two-Step model is ready in ```/step2/predict.py```.
###### Open the terminal from the repository directory and paste the commands below. 
```
cd step2
python predict.py
```
## **Run Web Application**
###### Open the terminal from the repository directory and paste the commands below. 
```
cd web
flask --app flaskr --debug run
```
