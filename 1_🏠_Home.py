import streamlit as st

st.set_page_config(page_title="Machine learning",
                   page_icon="👻", layout="wide")

st.title("Home")


st.image("./Image/hcmute.jpg",
         caption="Ho Chi Minh City University of Technical Education", width=100)

st.markdown("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thông tin sinh viên</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            
            .container {
                margin: 100px auto;
                max-width: 400px;
                padding: 20px;
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                text-align: center;
            }
            
            .container h2 {
                margin-bottom: 20px;
            }
            
            .container p {
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Thông tin sinh viên</h2>
            <p>Giảng viên hướng dẫn: Trần Tiến Đức</p>
            <p>Sinh viên thực hiện:</p>
            <p>Nguyễn Văn An - 20110434</p>
            <p>Nguyễn Minh Đức - 20110461</p>
        </div>
    </body>
    </html>

""", unsafe_allow_html=True)
st.markdown("### 20110434 Nguyễn Văn An")
st.markdown("# 20110434 Nguyễn Minh Đức")
