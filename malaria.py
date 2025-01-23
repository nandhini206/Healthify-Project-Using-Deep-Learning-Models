import streamlit as st
from tensorflow.keras.models import load_model
from files_upload import FilesUpload

def main():

    st.title("Malaria Detector Wep App")
    acitvity = ['Little Description', 'Prediction', 'About']
    choice = st.sidebar.selectbox('Chose An Activity', acitvity)

    if choice == 'Little Description':
        st.subheader("ORIGINAL DATA SOURCE")
        st.text("The dataset contains 2 folders - Infected - Uninfected")
        st.text("Acknowledgements This Dataset is taken from the official NIH Website: ")
        st.markdown("https://ceb.nlm.nih.gov/repositories/malaria-datasets/")
        #print(st.__version__)


    if choice == 'Prediction':

        #image_shape = (130, 130, 3)
        #model = load_model('./models/malaria_detector.h5')
        # Taking Instance of Class 'FilesUpload' by calling "from files_upload import FilesUpload"
        files_upload = FilesUpload()
        img = files_upload.run()
        if st.button("Predict"):
            st.text('Wait...Model is being loaded!')
            model = load_model('./models/malaria_detector.h5')
            st.success("Model Loaded")
            st.text('Wait...')
            if model.predict(img)[0][0] > 0.5:
                st.text("Uninfected")
                st.text("Probability: {}".format(model.predict(img)[0][0]))
            else:
                st.text("If irrelevent image is uploaded then model will assume it is infected")
                st.text("Infected/Parasitized")
                st.text("Probability: {}".format(model.predict(img)[0][0]))



    if choice == 'About':
        st.subheader("Malaria Dectection Wep App made with Streamlit by: Mazhar")
        st.info("mazqoty.01@gmail.com")


if __name__ == '__main__':
    main()
