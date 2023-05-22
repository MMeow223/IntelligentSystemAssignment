#**Instruction**
<hr>
#####**Train Model**
[1. Train Object Detection Model](#train-object-detection-model)
[2. Train Image Classification Model](#train-image-classification-model)

#####**Evaluate Model**
[1. Evaluate Object Detection Model](#evaluate-object-detection-model)
[2. Evaluate Image Classification Model](#evaluate-image-classification-model)

#####**Live Predict with Model**
[1. Live Predict One-Step Model](#live-predict-one-step-model)
[2. Live Predict Two-Step Model](#live-predict-two-step-model)

##Setup
Before starting to run train, test, live prediction, and web application, you need to perform the step below.
###1.Install packages
Assuming you have anaconda installed on your device. You will need to create an new anaconda environment with the following command.
######Open a terminal from the repository directory and paste the command below
`conda create --name is_env --file requirements.txt`
######Activate the anaconda environment
`conda activate is_env
Your environment is ready now.
###2. Download Dataset from Google Drive
Here are the link of the dataset. You can manually download them and put them your repository directory with directory name of "train", "test", "val"
| Dataset    | URL                                                                                                                                                                            |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Train      | ss[https://drive.google.com/drive/folders/1CNJJwzwxkamFhMaQQc1-7_TqkQBXOTKA?usp=sharing](https://drive.google.com/drive/folders/1CNJJwzwxkamFhMaQQc1-7_TqkQBXOTKA?usp=sharing) |
| Validation | [https://drive.google.com/drive/folders/1Vq4WsQn791qBicQB5tOs6B6tFLCIj6n8?usp=sharing](https://drive.google.com/drive/folders/1Vq4WsQn791qBicQB5tOs6B6tFLCIj6n8?usp=sharing)   |
| Test       | [https://drive.google.com/drive/folders/1qjfFZK62GaxEr2aYsM0IWPNb3nSuGTSI?usp=sharing](https://drive.google.com/drive/folders/1qjfFZK62GaxEr2aYsM0IWPNb3nSuGTSI?usp=sharing)   |
##**Train Object Detection Model**
The file for training Object Detection Model of One-Step Model is ready in ```/detection/train.py```. By default, it will use the yolov8m pre-trained check point, but you can change it to any other yolo models as you like.
######Open the terminal from the repository directory and paste the commands below. 
```
cd detection
python train.py
```

##**Train Image Classification Model**
The file for training Object Detection Model of One-Step Model is ready in ```/classify/train.py```. By default, it will use the yolov8m-cls pre-trained check point, but you can change it to any other yolo models as you like.
######Open the terminal from the repository directory and paste the commands below. 
```
cd classify
python train.py
```

##**Evaluate Object Detection Model**
The file for training Object Detection Model of One-Step Model is ready in ```/detection/test.py```. By default, it will use the yolov8m-cls pre-trained check point, but you can change it to any other yolo models as you like.
######Open the terminal from the repository directory and paste the commands below. 
```
cd detection
python train.py
```

##**Evaluate Image Classification Model**
```
cd detection
python train.py
```

##**Live Predict One-Step Model**
```
cd detection
python train.py
```

##**Live Predict Two-Step Model**
```
cd detection
python train.py
```
