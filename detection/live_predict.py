import cv2
from ultralytics import YOLO

# Load the pre-trained classifier for object detection
# classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the camera
camera = cv2.VideoCapture(0)

# model = YOLO('D:/IntelligentSystem/YOLOV8/runs/detect/train29/weights/best.pt')
model = YOLO('D:/IntelligentSystem/YOLOV8/runs/detect/train37/weights/best.pt')

# model.val()
# model.val(data='coco128.yaml')

# Continuously capture frames from the camera
while True:
    ret, frame = camera.read()

    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect
    # results = model.predict (source="0",conf=0.5,show=True, save=True, save_txt=True)
    results = model(source="0",
                    conf=0.4,
                    show=True,
                    stream=True,
                    iou=0.85,
                    line_thickness=2,
                    )  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
    #     masks = r.masks  # Masks object for segment masks outputs
    #     probs = r.probs  # Class probabilities for classification outputs
    

    # Exit the program if the user presses the "q" key
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()



