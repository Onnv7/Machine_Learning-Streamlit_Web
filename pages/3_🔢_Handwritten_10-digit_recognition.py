import streamlit as st
import cv2
from HandwrittenDigitRecognition import digit_recognize


st.markdown("""
    <h1 style='text-align: center;'>Handwritten 10 digit recognition</h1>
""", unsafe_allow_html=True)

st.text("Step 1: Click to choose \"Create Image\" to create and load the image")
st.text("Step 1: Step 2: Click to \"Recognize Image\" to recognize")

if "created_image" not in st.session_state:
    st.session_state.created_image = False

create_image_col, recognize_col = st.columns(2)

btn_create_image = create_image_col.button("Create Image")
btn_recognize = recognize_col.button("Recognize Image")
# IMAGE = st.image("./HandwrittenDigitRecognition/digit_random.jpg")

image = cv2.imread("./HandwrittenDigitRecognition/digit_random.jpg")

if st.session_state.created_image == True:
    create_image_col.image(image, use_column_width=True)


if btn_create_image:
    # if st.session_state.created_image == False:
    image = digit_recognize.create_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # IMAGE = create_image_col.image(image)
    st.session_state.created_image = True
    st.experimental_rerun()

if btn_recognize:
    if st.session_state.created_image == False:
        st.error("Photo must be created before identification")
    else:
        result = digit_recognize.recognize()
        # create_image_col.image(image)
        recognize_col.markdown(
            f"**Result:<br>** {result.replace('@#', '<br>')}", unsafe_allow_html=True)
        # st.session_state.created_image = False
    # st.experimental_rerun()
