import os
import cv2
import numpy as np
import pickle


#prepare data
input_dir = os.path.join('.', 'Resource', 'data_raw')
output_dir = os.path.join('.', 'Resource', 'data')
categories = ['empty', 'not_empty']



for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = cv2.imread(img_path)                                            # Read each image
        img = cv2.resize(img, (32, 32))                                        # Resize each image
        cv2.imwrite(os.path.join(output_dir, category, file), img)
