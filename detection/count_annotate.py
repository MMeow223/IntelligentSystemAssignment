import os

classes_file = "D:/IntelligentSystem/YOLOV8/classes.txt"

# specify the path of the folder containing the text files
folder_path = "D:/IntelligentSystem/YOLOV8/train/train5/labels"

# initialize a variable to keep track of the total number of lines
total_lines = 0


# get the list of classes
with open(classes_file) as f:
    # make it into dictionary

    classes = f.read().splitlines()

dictionary = dict.fromkeys(classes, 0)

# loop through all the files in the folder
for filename in os.listdir(folder_path):
    # check if the file is a text file
    if filename.endswith(".txt"):
        # open the file and count the number of lines
        with open(os.path.join(folder_path, filename)) as f:

            for line in f:
                if line[0].isdigit():
                    dictionary[classes[int(line[0])]] += 1


# total line is the sum of value of dictionary
total_lines = sum(dictionary.values())
print(dictionary)
print("Total number of annotations: ", total_lines)
