import cv2
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image, ImageDraw,ImageFont

# Initialize the camera
# camera = cv2.VideoCapture(0)
model = YOLO('D:/IntelligentSystem/YOLOV8/yolov8m.pt')
cls_model = YOLO('D:/IntelligentSystem/classify/runs/classify/train18/weights/best.pt')

class_names = ['fall', 'sit', 'walk']

def main():
    # Continuously capture frames from the camera
    while True:
        results = model(source="0",
                        conf=0.4,
                        show=False,
                        stream=True,
                        iou=0.9,
                        line_thickness=2,
                        classes=0
                        )  # generator of Results objects
        
        for r in results:
            img = r.orig_img.copy()  # make a copy of the original image
            boxes = r.boxes.data  # Boxes object for bbox outputs

            # crop the objects
            for box in boxes:
                # Get the coordinates of the bounding box
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw the bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Crop the object using array slicing
                cropped_img = img[y1:y2, x1:x2]
                predicted_class = detect(cropped_img)
                
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = draw_text(img,predicted_class)
            cv2.imshow('Object Detection', img)
            
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


def detect(frame):
    
    # frame to image
    frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image_processed = frame.resize((512,512))
    results = cls_model(image_processed)  # predict on an image
    probability = results[0].probs
    max_index = probability.argmax().item()
    
    text = class_names[int(max_index)]
    
    return text
    
    

def draw_text(frame,text):
    
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image_draw = ImageDraw.Draw(image)
    
    myFont = ImageFont.truetype("C:/Windows/Fonts/Calibri.ttf",30)
    image_draw.text((28, 36), text, font=myFont,fill=(255, 0, 0))
    frame = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    
    return frame

if __name__ == '__main__':
    main()