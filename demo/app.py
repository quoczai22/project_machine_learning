"""Điểm khởi chạy demo: khai báo menu các trang (tên tiếng Việt có dấu)."""

import streamlit as st

st.navigation(
    [
        st.Page("home.py", title="Trang chủ", icon="📡", default=True),
        st.Page("pages/1_Du_doan_Churn.py", title="Demo 1 · Dự đoán Churn", icon="🔮"),
        st.Page("pages/2_Phan_khuc_khach_hang.py", title="Demo 2 · Phân khúc khách hàng", icon="👥"),
    ]
).run()
