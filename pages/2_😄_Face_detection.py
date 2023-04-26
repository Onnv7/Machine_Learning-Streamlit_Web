import streamlit as st
import cv2
from streamlit_option_menu import option_menu
import time
from FaceDetection import get_data, train_model, predict


def check_name(name, id):
    ids, names = get_data.get_id_name_list("./FaceDetection/data")
    id = "{:06d}".format(int(id))
    if id in ids:
        return False, "Id already exists"

    if name.strip() == "":
        return False, "Name is blank"
    return True, ""


st.markdown("""
    <h1 style='text-align: center;'>Face detection</h1>
""", unsafe_allow_html=True)

selected_procedure = option_menu(
    menu_title="Procedures",
    options=["Get data", "Train model", "Predict"],
    default_index=0,
    orientation="horizontal",
)

if selected_procedure == "Get data":
    with st.expander("Thông báo lỗi"):
        error_message = None
    if 'isSubmited' not in st.session_state:
        st.session_state.isSubmited = False
    if 'get_random_id' not in st.session_state:
        st.session_state.random_id_checked = False

    id_col, name_col = st.columns(2)
    if st.session_state.random_id_checked == False:
        id_input = id_col.number_input(
            "What is your id?", min_value=1, step=1, format="%d")
    name_input = name_col.text_input("What is your name?")

    # random_col, submit_col = st.columns(2)
    # btn_random = random_col.button("Random ID")
    btn_submit = st.button("Submit")

    names = get_data.get_folder_name("./FaceDetection/data")
    # if btn_random:
    #     random_number = random.randint(1, 999999)
    #     id_input = id_col.number_input(
    #         "What is your id?", value=random_number, min_value=1, step=1, format="%d")
    #     st.session_state.random_id_checked = True
    # st.experimental_rerun()
    if btn_submit:
        if st.session_state.isSubmited == True:
            st.session_state.isSubmited = False
        else:
            ret, msg = check_name(name_input, str(id_input))
            if ret == False:
                error_message = msg
                st.warning(error_message)
            elif ret == True:
                id = str(id_input)
                create_folder = get_data.make_folder(id, name_input)
                # names.append(id + " - " + name_input)
                # st.success("Created successfully")
                st.experimental_rerun()

    name_selectbox = st.selectbox("Name list", names, index=0)
    btn_delete_data = None
    btn_delete_all = None

    # quantity_input = st.number_input(
    #     "How many images do you want to train?", min_value=20, max_value=300, step=1, format="%d")
    quantity_input = st.slider(
        "How many images do you want to train?", min_value=20, max_value=300, value=50, step=1)
    if name_selectbox:
        delete_col, delete_all_col = st.columns(2)
        btn_delete_data = delete_col.button("Delete: " + name_selectbox)
        btn_delete_all = delete_all_col.button("Delete all", type="primary")

    btn_create_data = st.button("Create data")
    FRAME_CAMERA = st.image([])

    if btn_delete_data:
        get_data.remove_folder(name_selectbox)
        st.experimental_rerun()

    if btn_delete_all:
        folder_names = get_data.get_folder_name("./FaceDetection/data/")
        for fn in folder_names:
            get_data.remove_folder(fn)
        st.experimental_rerun()

    if btn_create_data:
        get_data.clear_data(name_selectbox)
        count = quantity_input
        cap = cv2.VideoCapture(0)
        i = 0
        while True:
            ret, frame = cap.read()

            # file_name = './FaceDetection/data/'+"000000 - Unknown" + \
            #     '/' + "000000 - Unknown"+'%04d.bmp' % i
            # i += 1
            # if i == 100:
            #     break
            # print("aA")
            # cv2.imwrite(file_name, frame)
            folder_name = name_selectbox
            frame, success = get_data.create_data(folder_name, i, frame)
            if success:
                i += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_CAMERA.image(frame, channels='RGB')
            if i == count:
                break
        cap.release()
        cv2.destroyAllWindows()
        FRAME_CAMERA.image([])

if selected_procedure == "Train model":
    ids, name_list, quantity = get_data.get_id_name_quantity_list(
        "./FaceDetection/data")
    data = {
        "ID": ids,
        "Name": name_list,
        "Quantity": quantity
    }
    table_class = st.table(data)
    btn_train = st.button("Let train")
    if btn_train:
        get_data.clear_file_content("./FaceDetection/model/svc.pkl")
        if len(name_list) <= 1:
            st.error("Must have 2 classes to train")
        else:
            progress_bar = st.progress(0)
            for i in range(1, 101):
                progress_bar.progress(i)
                if i == 80:
                    train_model.train()
                    time.sleep(3)
                time.sleep(0.001)

            st.success("The process is complete!")
        # progress_bar = st.progress(0)
        # for i in range(1, 101):
        #     progress_bar.progress(i)
        #     if i == 80:
        #         train_model.train()
        #         time.sleep(3)
        #     time.sleep(0.001)

        # st.success("The process is complete!")
if selected_procedure == "Predict":
    start_col, stop_col = st.columns(2)
    btn_start = start_col.button("Start")
    btn_stop = stop_col.button("Stop")
    FRAME_CAMERA = st.image([])

    cap = cv2.VideoCapture(0)
    if btn_start:
        while True:
            ret, frame = cap.read()
            frame = predict.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_CAMERA.image(frame, channels='RGB')
            if btn_stop:
                break
        cap.release()
        cv2.destroyAllWindows()
        FRAME_CAMERA.image([])
