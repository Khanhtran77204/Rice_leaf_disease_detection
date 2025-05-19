# giao diện 

import numpy as np
import streamlit as st
from rice_leaf.predict import predict_image
from PIL import Image, ImageOps
from auth import register_user, login_user
from rice_leaf.gradcam import generate_gradcam

st.set_page_config(page_title="🌾 Rice Leaf Scanner Framework", layout="centered")
# lưu trạng thái đăng nhập
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    
# giao diện đăng ký, đăng nhập
def show_login_page():
    st.title("📝 Rice Leaf Disease Recognition Using Deep Learning ")
    choice = st.radio("Bạn đã có tài khoản chưa?", ["Đăng nhập", "Đăng ký"])

    if choice == "Đăng ký":
        st.title("🔐 REGISTER")
        username = st.text_input("Username - Tên người dùng")
        password = st.text_input("Password - Mật khẩu", type="password")
        if st.button("Đăng ký"):
            if register_user(username, password):
                st.success("✅ Đăng ký thành công! Hãy đăng nhập.")
            else:
                st.error("⚠️ Tên người dùng đã tồn tại.")

    else:
        st.title("🔐 LOGIN")
        username = st.text_input("Username - Tên người dùng")
        password = st.text_input("Password - Mật khẩu", type="password")
        if st.button("Đăng nhập"):
            if login_user(username, password):
                st.title("🔓 LOGIN")
                st.success("✅ Đăng nhập thành công!")
                st.session_state.logged_in = True
            else:
                st.error("❌ Sai tên đăng nhập hoặc mật khẩu.")
                     
def show_main_page():
    import tempfile
    import warnings
    from PIL import Image
    from rice_leaf.predict import predict_image
    st.title("🌾 Chẩn Đoán Bệnh Lá Lúa")
    st.write("Vui lòng tải lên 01 hình ảnh lá lúa.")
    uploaded_file = st.file_uploader("📷 CHỌN ẢNH:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Ảnh đã được tải", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file.name)

            if st.button("🔍 Bắt đầu chẩn đoán"):
                #label = predict_image(tmp_file.name) # ignore the warning
                result = predict_image(tmp_file.name) # sử dụng cái mới, xóa đi nếu xuất hiện lỗi
                
                predicted_class = max(result, key=result.get)
                
                warnings.filterwarnings("ignore")
                st.write("**Kết quả chẩn đoán:**")
                st.success(f"📋 Bệnh được chẩn đoán: -- {predicted_class}") # trước đây là label
                st.write("\n")
                
                st.subheader("📊 Xác suất bệnh:")
                st.dataframe(result.items(), use_container_width=True)
                heatmap_img = generate_gradcam(tmp_file.name)
                
                # đây là biểu đồ
                st.subheader("📈 Biểu đồ xác suất bệnh:")
                st.bar_chart(result)
                
                st.subheader("💡 Biểu đồ trực quan hóa:")
                st.image(heatmap_img, caption="🔥 Vùng mô hình tập trung (Grad-CAM)", use_container_width=True)
                
                
    if st.button("🔓 Đăng xuất"):
        st.session_state.logged_in = False
        st.rerun()
       
# Điều hướng
if st.session_state.logged_in:
    show_main_page()
else:
    show_login_page()