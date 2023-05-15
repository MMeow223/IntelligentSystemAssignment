import os
import random
import uuid
# specify the path of the folder containing the files to be renamed
images_folder = "D:/IntelligentSystem/YOLOV8/train/shuffle+day/images"
label_folder = "D:/IntelligentSystem/YOLOV8/train/shuffle+day/labels"

# file_name_start_with = "night_train_fall"
new_file_name = "train"

# specify the starting number for the files
num = 1

exist_nums = []
# loop through all the files in the folder
for filename in os.listdir(images_folder):
    
     # extract the file extension
     ext = os.path.splitext(filename)[1]

     label_old_name = os.path.splitext(filename)[0] + ".txt"

     # num = random.randint(1, 100000000)

     # unique = str(uuid.uuid4())
     unique = num

     # while(num in exist_nums):
     #     num = random.randint(1, 100000000)

     # specify the new name and format
     image_new_name = new_file_name + str(unique) + ext
     label_new_name = new_file_name + str(unique) + ".txt"

     # also rename the label file located in the labels folder
     # if the label file exists
     # if (os.path.exists(label_folder +"/"+ label_old_name) == False):               

     os.rename(label_folder +"/"+ label_old_name, label_folder +"/"+  label_new_name)

     # rename the image file
     os.rename(os.path.join(images_folder, filename), os.path.join(images_folder, image_new_name))

     print("{0} --> {1}".format(label_old_name,label_new_name) )
     
     num +=1

