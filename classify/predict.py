from ultralytics import YOLO
from PIL import Image, ImageDraw,ImageFont

class_names = ['fall', 'sit', 'walk']

if __name__ == '__main__':

    # get an images from the web
    image = Image.open('D:/IntelligentSystem/classify/test/fall.png')

    image_processed = image.resize((512,512))
    
    # image.show()

    model = YOLO('D:/IntelligentSystem/classify/runs/classify/train18/weights/best.pt')  # load a custom model

    # Predict with the model
    results = model(image_processed)  # predict on an image
    
    # probs: tensor([6.1829e-05, 2.8721e-04, 9.9965e-01], device='cuda:0')
    probability = results[0].probs
    
    # get the index of the max probability
    max_index = probability.argmax().item()
    
    text = class_names[int(max_index)]
    
    # Call draw Method to add 2D graphics in an image
    image_draw = ImageDraw.Draw(image)
    
    myFont = ImageFont.truetype("C:/Windows/Fonts/Calibri.ttf",30)
    
    # Add Text to an image
    image_draw.text((28, 36), text, font=myFont,fill=(255, 0, 0))
    
    image.show()