import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model đã lưu
model = pickle.load(open('model_dien.pkl', 'rb'))

st.set_page_config(page_title="Dự báo tiền điện", page_icon="⚡")
st.title("🔌 Dự báo Tiền điện Hàng tháng")
st.write("Dự án ML: Data -> Clean -> Train -> Deploy")

with st.sidebar:
    st.header("Thông tin đầu vào")
    people = st.number_input("Số người", 1, 20, 3)
    area = st.number_input("Diện tích (m2)", 5, 200, 25)

col1, col2 = st.columns(2)
with col1:
    ac_units = st.number_input("Số máy lạnh", 0, 10, 1)
    ac_hours = st.slider("Giờ dùng máy lạnh/ngày", 0, 24, 6)
with col2:
    fans = st.number_input("Số quạt", 0, 20, 2)
    fan_hours = st.slider("Giờ dùng quạt/ngày", 0, 24, 10)
    fridges = st.number_input("Số tủ lạnh", 0, 5, 1)

if st.button("Dự đoán Realtime"):
    # Tạo mảng input đúng thứ tự: People, AC_Units, AC_Hours, Fans, Fan_Hours, Fridges, Area
    features = np.array([[people, ac_units, ac_hours, fans, fan_hours, fridges, area]])
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    st.subheader("Kết quả dự đoán:")
    st.success(f"💰 Số tiền điện ước tính: **{prediction:,.0f} VNĐ**")
    
    # Progress bar giả lập độ tin cậy
    st.write(f"Độ tin cậy mô hình ($R^2$): 0.92") # Thay bằng chỉ số thực của bạn
