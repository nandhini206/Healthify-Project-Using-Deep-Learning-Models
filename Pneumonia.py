import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt

import seaborn as sns
import random
# Deep learning libraries
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image

loaded_model=tf.keras.models.load_model('my_model.h5')



#st.set_option('depreciation.showfileUploaderEncoding',False)
#@st.cache(allow_output_mutation)
st.title("Pneumonia Detection Machine")

file=st.sidebar.file_uploader("Please upload your X-Ray image and Nothing Else", type= ["png","JPG","jpeg"])

def predict(image_path):
    image1 = image.load_img(image_path, target_size=(150, 150))
    image1 = image.img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    #st.write(image1.shape)
    img_array= image1/255
    prediction = loaded_model.predict(img_array)
    if prediction[0][0]>.6:
        string="You have a high chance of having Pneoumonia, Please consult a doctor"
    else:
        string="You have a low chance of having Pneoumonia, Nothing to panic!"
        
    st.success(string)
        
    
    
    
    
    

if file is not None:
    img=Image.open(file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    predict(file)
    
    #label = predict(img)
    
    
    
