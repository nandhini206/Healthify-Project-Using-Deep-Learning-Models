import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
st.title('Brain Tumor Classifier')
st.write('')
st.write('This is a python app which classifies a brain MRI into one of the four classes : ')
st.write(' No tumor, Pituitary tumor,Meningioma tumor or Glioma tumor')
file = st.file_uploader(label='Upload image', type=['jpg','jpeg','png'], accept_multiple_files=False, key=None)
IMAGE_SIZE = 150

from tensorflow.keras.applications import EfficientNetB0
effnet = EfficientNetB0(weights = None,include_top=False,input_shape=(IMAGE_SIZE,IMAGE_SIZE, 3))
model1 = effnet.output
model1 = tf.keras.layers.GlobalAveragePooling2D()(model1)
model1 = tf.keras.layers.Dropout(0.5)(model1)
model1 = tf.keras.layers.Dense(4, activation = 'softmax')(model1)
model1 = tf.keras.models.Model(inputs = effnet.input, outputs = model1)
model1.load_weights('effnet.h5')


if file is not None:
    image = Image.open(file)
    image = np.array(image)
    # image = image[:,:,::-1].copy()
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # plt.imshow(image)
    st.image(image)
    images = image.reshape(1,150,150,3)
    predictions1 = model1.predict(images)
    predictions1 = np.argmax(predictions1, axis=1)
    labels = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']
    st.write('Prediction over the uploaded image:')
    st.title(labels[predictions1[0]])

