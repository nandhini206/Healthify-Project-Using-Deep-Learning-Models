import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# CSS for custom styling
def local_css():
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #4a4a4a;
    }
    .highlight {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stApp {
        background-color: #f0f6fc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .disease-title {
        color: #2c3e50;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

class DiseaseDetector:
    @staticmethod
    def detect_alzheimers(file):
        model = load_model('ALZ.h5')
        img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
        i = img_to_array(img)
        i = i / 255.0
        input_arr = np.expand_dims(i, axis=0)
        pred = np.argmax(model.predict(input_arr), axis=-1)
        
        labels = {
            0: "Mild Demented",
            1: "Moderate Demented", 
            2: "Non Demented", 
            3: "Very Mild Demented"
        }
        return labels.get(pred[0], "Unknown")

    @staticmethod
    def detect_brain_tumor(file):
        model = load_model('BR.h5')
        IMAGE_SIZE = 150
        image_obj = Image.open(file)
        image_array = np.array(image_obj)
        image_resized = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
        images = image_resized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        predictions = model.predict(images)
        labels = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']
        return labels[np.argmax(predictions, axis=1)[0]]

    @staticmethod
    def detect_pneumonia(file):
        loaded_model = load_model('PN.h5')
        image1 = tf.keras.preprocessing.image.load_img(file, target_size=(150, 150))
        image1 = img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        img_array = image1 / 255.0 
        prediction = loaded_model.predict(img_array)
        if prediction[0][0] > 0.3:
            return "Pneumonia Detected"
        else:
            return "No Pneumonia Detected"

    @staticmethod
    def detect_skin_cancer(file):
        model = load_model('SK.h5')
        image_obj = Image.open(file)
        
        def preprocess_image(uploaded_image):
            resized_image = uploaded_image.resize((256, 256))
            image_array = img_to_array(resized_image)
            image_array /= 255.
            return image_array
        
        def prediction(image_array):
            pred = model.predict(np.expand_dims(image_array, axis=0))
            return pred
        
        inp = preprocess_image(image_obj)
        ans = prediction(inp)
        classes = ['Benign', 'Malignant']
        
        pred_class = np.argmax(ans)
        confidence = ans[0][pred_class]
        confidence_percentage = round(confidence * 100, 2)
        
        data = {
            'Predicted Class': [classes[pred_class]],
            'Confidence (%)': [confidence_percentage]
        }
        df = pd.DataFrame(data)
        
        return f"{classes[pred_class]} (Confidence: {confidence_percentage}%)", df

    @staticmethod
    def detect_malaria(file):
        model = load_model('malaria_detector.h5')
        img = tf.keras.preprocessing.image.load_img(file, target_size=(130, 130))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        return "Uninfected" if prediction[0][0] > 0.5 else "Infected/Parasitized"

def home_page():
    st.title("🏥 Healthify: Advanced Disease Detection Platform")
    
    st.markdown("""
    <div class="highlight">
    <h2 class="disease-title">Welcome to Healthify</h2>
    <p class="big-font">
    Healthify is an advanced medical image analysis platform that leverages 
    cutting-edge machine learning to detect various diseases quickly and accurately.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Our Detection Capabilities:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🧠 Alzheimer's")
        st.markdown("Detect early stages of dementia")
    
    with col2:
        st.markdown("#### 🧬 Brain Tumor")
        st.markdown("Identify potential brain abnormalities")
    
    with col3:
        st.markdown("#### 🫁 Pneumonia")
        st.markdown("Analyze chest X-rays for lung infections")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("#### 🌞 Skin Cancer")
        st.markdown("Detect potential skin malignancies")
    
    with col5:
        st.markdown("#### 🦠 Malaria")
        st.markdown("Identify parasitic infections")
    
    with col6:
        st.markdown("#### 🛡️ Early Detection")
        st.markdown("Empowering health through technology")

def brain_tumor_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🧠 Brain Tumor Detection")
    with col2:
        st.image("brainimg.png", width=200)
    
    st.markdown("""
    <div class="highlight">
    <h3>Brain Tumor Detection using Machine Learning</h3>
    <p class="big-font">
    Our advanced neural network analyzes brain MRI scans to detect potential tumors.
    Supports detection of No Tumor, Pituitary, Meningioma, and Glioma Tumors.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        
        if st.button("Detect Brain Tumor"):
            with st.spinner('Analyzing MRI...'):
                result = DiseaseDetector.detect_brain_tumor(uploaded_file)
                st.success(f"Brain Tumor Prediction: {result}")

def pneumonia_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🫁 Pneumonia Detection")
    with col2:
        st.image("pne.png", width=200)
    
    st.markdown("""
    <div class="highlight">
    <h3>Pneumonia Detection from Chest X-Rays</h3>
    <p class="big-font">
    Our AI model can detect pneumonia by analyzing chest radiographs 
    with high accuracy and speed.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-Ray", width=300)
        
        if st.button("Detect Pneumonia"):
            with st.spinner('Analyzing X-Ray...'):
                result = DiseaseDetector.detect_pneumonia(uploaded_file)
                st.success(result)

def skin_cancer_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🌞 Skin Cancer Detection")
    with col2:
        st.image("skincare.png", width=200)
    
    st.markdown("""
    <div class="highlight">
    <h3>Skin Cancer Classification</h3>
    <p class="big-font">
    Advanced AI to classify skin lesions as Benign or Malignant 
    with confidence percentage.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Skin Image", width=300)
        
        if st.button("Detect Skin Cancer"):
            with st.spinner('Analyzing Skin Lesion...'):
                result, dataframe = DiseaseDetector.detect_skin_cancer(uploaded_file)
                st.success(f"Skin Cancer Prediction: {result}")
                st.dataframe(dataframe)

def alzheimer_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🧠 Alzheimer's Detection")
    with col2:
        st.image("az.png", width=200)
    
    st.markdown("""
    <div class="highlight">
    <h3>Early Alzheimer's Stage Detection</h3>
    <p class="big-font">
    Machine learning model to detect different stages of Alzheimer's 
    from brain MRI scans.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Brain MRI", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        
        if st.button("Detect Alzheimer's Stage"):
            with st.spinner('Analyzing Brain Scan...'):
                result = DiseaseDetector.detect_alzheimers(uploaded_file)
                st.success(f"Alzheimer's Status: {result}")

def malaria_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🦠 Malaria Detection")
    with col2:
        st.image("mos.png", width=200)
    
    st.markdown("""
    <div class="highlight">
    <h3>Malaria Parasite Detection</h3>
    <p class="big-font">
    AI-powered microscopic blood cell analysis to detect 
    malaria parasites with high precision.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Blood Smear Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Blood Smear", width=300)
        
        if st.button("Detect Malaria"):
            with st.spinner('Analyzing Blood Smear...'):
                result = DiseaseDetector.detect_malaria(uploaded_file)
                st.success(result)

def exit_page():
    st.title("👋 Thank You for Using Healthify")
    st.markdown("""
    <div class="highlight">
    <h3>Your Health, Our Priority</h3>
    <p class="big-font">
    Healthify is committed to using advanced AI technologies 
    to support early disease detection and healthcare diagnostics.
    
    Remember: Our AI assists medical professionals, 
    but does not replace professional medical advice.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Exit Button with Balloons
    if st.button("Exit Healthify"):
        st.balloons()
        st.stop()

def main():
    st.set_page_config(page_title="Healthify", page_icon="🏥", layout="wide")  # Move this line here
    local_css()

    st.sidebar.title("Healthify")

    # Add an image in the sidebar
    st.sidebar.image("images.jpeg", width=250)  # Replace with your image URL or path
    
    # Pages dictionary
    pages = {
        "Home": home_page,
        "Brain Tumor Detection": brain_tumor_page,
        "Pneumonia Detection": pneumonia_page,
        "Skin Cancer Detection": skin_cancer_page,
        "Alzheimer's Detection": alzheimer_page,
        "Malaria Detection": malaria_page,
        "Exit": exit_page
    }
    # Sidebar navigation
    page = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[page]()

    # Update sidebar with developer info and college
    st.sidebar.markdown("### 👤 About")
    st.sidebar.info("""
    **Developers:**
    - Students Name

    **College:** College Name 🎓

    This application is built using Streamlit ✨
    """)

# Run the app
if __name__ == "__main__":
    main()