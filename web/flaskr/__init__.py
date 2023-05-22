from flask import Flask, render_template, Response, request,redirect,url_for,jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw,ImageFont


step2_model = YOLO('yolov8m.pt')
cls_model = YOLO('../classification.pt')

class_names = ['fall', 'sit', 'walk']
    

confident = 0.25
iou = 0.7
line_thickness = 3
show_label = True

step_1_alert = False
step_2_alert = False



def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    camera = cv2.VideoCapture(0)
    
    model = YOLO("../detection.pt")
    

    
    @app.route('/check')
    def returnsetting():
        global confident, iou, line_thickness, show_label
        string = "confident: " + str(confident) + " iou: " + str(iou) + " line_thickness: " + str(line_thickness) + " show_label: " + str(show_label)
        return string
        
    
    
    @app.route('/settings', methods=['POST'])
    def change_settings():
        global confident, iou, line_thickness, show_label
        
        if request.method == 'POST':
            confident = float(request.form['conf'])
            # iou = float(request.form['iou'])
            line_thickness = int(request.form['line_thickness'])
            
        return redirect(url_for('one_step'))

    @app.route('/')
    def index():
        return redirect(url_for('one_step'))
    

    @app.route('/one-step')
    def one_step():
        
        # list of parameters to pass to the template
        params = {
            "confident": float(confident),
            "iou": float(iou),
            "line_thickness": float(line_thickness),
            "show_label": "checked" if (show_label==True)  else "",
            "links": get_top_links(),
            "video": video_feed_1()
        }
        
        # return "OK"
        return render_template('one-step.html', **params)
    
        
    @app.route('/two-step')
    def two_step():
        # list of parameters to pass to the template
        params = {
            "confident": float(confident),
            "iou": float(iou),
            "line_thickness": float(line_thickness),
            "show_label": "checked" if (show_label==True)  else "",
            "links": get_top_links(),
            "video": video_feed_2()
        }
        
        # return "OK"
        return render_template('two-step.html', **params)
    
    
    def get_top_links():
        
        links = [
            {"icon":"bi bi-1-square-fill", "url": "/one-step", "name": "One-Step Model"},
            {"icon":"bi bi-2-square", "url": "/two-step", "name": "Two-Step Model"},
        ]
            
        return links
    
    @app.route('/video_feed_1')
    def video_feed_1():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
    @app.route('/video_feed_2')
    def video_feed_2():
        return Response(gen_frames_2(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def gen_frames():  
        global confident, iou, line_thickness, show_label, step_1_alert
        
        print("condifent: ", confident, " iou: ", iou, " line_thickness: ", line_thickness)

        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                results = model(frame)
                # ('boxes', 'masks', 'probs', 'keypoints')
                   # get the ploted box text and position
                text = results[0].boxes.cls
                
                # convert all the element in the list to int
                text = [int(i) for i in text]
                
                # if list contain element 0, then it is fall
                if 0 in text:
                    step_1_alert = True
                else:
                    step_1_alert = False
                
                annotated_frame = results[0].plot(line_width=line_thickness, conf=confident)
                
             
                
                
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                
    
    def gen_frames_2():  
        global confident, iou, line_thickness, show_label
        
        
        print("condifent: ", confident, " iou: ", iou, " line_thickness: ", line_thickness)

        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                
                results = step_two_predict(frame)
                # results = model(frame)
                
                # annotated_frame = results[0].plot(line_width=line_thickness, conf=confident)
                
                ret, buffer = cv2.imencode('.jpg', results)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
              

       

    # Initialize the camera
    # camera = cv2.VideoCapture(0)
    

    def step_two_predict(frame):
        global step_2_alert
        # Continuously capture frames from the camera
        # while True:
        
        # frame to image
        img = Image.fromarray(frame)
        
        results = step2_model(source=img,
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
                if box is None:
                    continue
                # Get the coordinates of the bounding box
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw the bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Crop the object using array slicing
                cropped_img = img[y1:y2, x1:x2]
                predicted_class = detect(cropped_img)
                
                if predicted_class == "fall":
                    step_2_alert = True
                else:
                    step_2_alert = False
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = draw_text(img,predicted_class)
            
        
        return img


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
        frame = cv2.cvtColor(np.asarray(frame),cv2.COLOR_BGR2RGB)
        
        return frame

    @app.route('/get_step_1_alert', methods=['GET'])
    def get_step_1_alert():
        global step_1_alert
        results = {'alert': str(step_1_alert)}
        return jsonify(results)
    
    @app.route('/get_step_2_alert', methods=['GET'])
    def get_step_2_alert():
        global step_2_alert
        results = {'alert': str(step_2_alert)}
        return jsonify(results)

    return app