import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Nhận dạng Đối tượng với YOLOv5",
    page_icon="🤖",
    layout="wide"
)

# --- Hàm tải mô hình ---
@st.cache_resource
def load_model(model_path):
    """
    Tải mô hình YOLOv5 từ file trọng số.
    """
# ---doc:https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/before-you-start---  

# --- model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True) ---
 
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')
return model
    

# --- Giao diện chính ---
st.title("🚀 Ứng dụng nhận dạng đối tượng với YOLOv5")
st.write("Tải lên một hình ảnh và mô hình sẽ phát hiện các đối tượng trong đó.")

model_path = 'best.pt'
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

# Thanh trượt để chọn ngưỡng tin cậy
confidence_threshold = st.slider("Chọn ngưỡng tin cậy (Confidence Threshold)", 0.0, 1.0, 0.45, 0.05)

# Nơi để người dùng tải file lên
uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh người dùng tải lên
    image = Image.open(uploaded_file)
    
    # Tạo hai cột để hiển thị
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_column_width=True)

    # Đưa ảnh vào mô hình để xử lý
    model.conf = confidence_threshold
    results = model(image)

    # Vẽ kết quả lên ảnh và hiển thị
    with col2:
        st.subheader("Ảnh kết quả")
        # results.render() trả về ảnh đã được vẽ bounding box
        st.image(results.render(), use_column_width=True)

    # Hiển thị chi tiết kết quả dưới dạng bảng
    st.subheader("Chi tiết các đối tượng được phát hiện")
    st.dataframe(results.pandas().xyxy[0])
else:
    st.info("Vui lòng tải lên một ảnh để bắt đầu.")