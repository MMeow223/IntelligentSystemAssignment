import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0)

model = YOLO('runs/detect/train37/weights/best.pt')

while True:
    ret, frame = camera.read()

    results = model(source="0",
                    conf=0.4,
                    show=True,
                    stream=True,
                    iou=0.85,
                    line_thickness=2,
                    )  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs

    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()



