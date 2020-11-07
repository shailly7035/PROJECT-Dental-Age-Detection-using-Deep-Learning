import tensorflow as tf
import cv2
import numpy as np
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_age(gender):
    HEIGHT = 200
    WIDTH = 300
    NUM_CHANNELS = 3
    path = os.path.join(os.getcwd(),'output_test','input_data.jpg')
    img_array = cv2.imread(path)
    new_array =  cv2.resize(img_array,(300,200))
    test_X = np.array(img_array).reshape(-1,200,300,NUM_CHANNELS)
    test_X = test_X/255.0
    path = os.path.join(os.getcwd(),'saved_models','person.h5')   
    new_model = tf.keras.models.load_model(path)
    #print("enterd2")
    group = new_model.predict(test_X)
    group = group[:, 0]
    if group<=0.5:
        group = 0
    else:
        group = 1
    
    img_array = cv2.imread(os.path.join(os.getcwd(),"output_test","input_data.jpg"))
    new_array =  cv2.resize(img_array,(300,200))
    test_X = np.array(img_array).reshape(-1,300,200,NUM_CHANNELS)
    test_X = test_X/255.0
    
    if group == 0:
        new_model = tf.keras.models.load_model( os.path.join(os.getcwd(),'saved_models','female-4-14.h5'))
    elif group == 1:
        new_model = tf.keras.models.load_model(os.path.join(os.getcwd(),'saved_models','male-15-19.h5'))
        
    age = new_model.predict(test_X)
    return float(age)


