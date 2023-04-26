import streamlit as st
import cv2
from YoloDetection import detection

css = '''
<style>
.img-container {
    max-width: 300px; /* chiều rộng tối đa của container chứa ảnh */
    margin: 0 auto; /* căn giữa container chứa ảnh */
}
.img-container img {
    width: 100%; /* độ rộng của ảnh trong container */
    object-fit: contain; /* tỷ lệ giữa chiều rộng và chiều cao của ảnh */
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)
st.markdown("""
    <h1 style='text-align: center;'>Yolo detection</h1>
""", unsafe_allow_html=True)
if "model" not in st.session_state:
    st.session_state.model = ""


turn_on_col, turn_off_col = st.columns(2)

model = None

cap = cv2.VideoCapture(0)


def on_mode_change(value):
    if value == "Smoke":
        path = "smoke"
    elif value == "Mask or No Mask":
        path = "mask"

    cap.release()
    cv2.destroyAllWindows()
    # FRAME_CAMERA.image([])
    # detection.load_model(path)


mode_selected = st.selectbox(
    "Choose mode", ["Mask or No Mask", "Smoke"], index=0)
on_mode_change(mode_selected)

btn_turn_on = turn_on_col.button("Turn On Camera")
btn_turn_off = turn_off_col.button("Turn Off Camera")

FRAME_CAMERA = st.image([])

if btn_turn_on:
    cap = cv2.VideoCapture(0)
    print(model)
    while True:
        ret, frame = cap.read()
        frame = detection.process(frame, mode_selected)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_CAMERA.image(frame, channels='RGB')

        if btn_turn_off:
            break
    cap.release()
    cv2.destroyAllWindows()
    FRAME_CAMERA.image([])
