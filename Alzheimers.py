import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import img_to_array,load_img

# load the saved model
model = load_model('best_model.h5')

def main():
    st.title('Alzheimers Detection App')
    menu=["Home", "About"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Upload the MRI scan image")
        file = st.file_uploader("")
        show_file = st.empty()
		
        if not file:
            show_file.info("Please upload the file in .png or .jpg format")
            return
        show_file.image(file)
        img = load_img(file, target_size=(224,224) )
        i = img_to_array(img)/255.0
        input_arr = np.array([i])
        pred = np.argmax(model.predict(input_arr))
        
        if pred == 0 :
            st.subheader("Alzheimer is Mild Demented")
        elif pred == 1:
            st.subheader("Alzheimer is Moderate Demented")
        elif pred == 2:
            st.subheader("Alzheimer is Non Demented")
        else:
            st.subheader("Alzheimer is Very Mild Demented")

    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")
        st.write("This model aims in detecteing the Alzheimersfrom the MRI scan image. The accurate diagnosis of Alzheimer's disease (AD) plays an important role in patient treatment, especially at the disease's early stages, because risk awareness allows the patients to undergo preventive measures even before the occurrence of irreversible brain damage. AD can be diagnosed-but not predicted-at its early stages, as prediction is only applicable before the disease manifests itself. Deep Learning (DL) has become a common technique for the early diagnosis of AD.")
        st.caption('Created by: Shamika Shetty')

if __name__ == "__main__":
    main()