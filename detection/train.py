from ultralytics import YOLO

if __name__ == '__main__':
	model = YOLO('yolov8m.pt')  # load a custom model

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
