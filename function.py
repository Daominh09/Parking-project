import cv2
import tensorflow as tf
import keras.models
import numpy as np
from space import MODEL

Space_Classifier = keras.models.load_model("nn.h5")
def get_parking_space(connectedComponent):
    (num_label, labels, stats, centroids) = connectedComponent
    
    space = []
    
    for i in range(1, num_label):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        space.append([x,y,w,h])
        
    return space

def emty_check(space_img_list):
    space_check_list = []
    flat_data = []
    for space_img in space_img_list:
        img_resized = cv2.resize(space_img, (32, 32))
        flat_data.append(img_resized)
    flat_data = np.array(flat_data)
    y_output = Space_Classifier.predict(flat_data)
    y_output = tf.nn.sigmoid(y_output).numpy()
    for i in range (len(space_img_list)):
        if y_output[i,0] <= 0.5:
            space_check_list.append(1)
        else:
            space_check_list.append(0)
    return space_check_list



