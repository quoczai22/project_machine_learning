"""Trang chủ demo Telco Customer Churn — Nhóm 13 (chạy qua app.py)."""

import streamlit as st

from common import fmt_pct, load_artifacts, setup_page

setup_page("Trang chủ", "📡")
art = load_artifacts()
summary = art["summary"]
rf = summary["classification"]["Random Forest"]

st.title("📡 Telco Customer Churn — Demo")
st.caption("Bài tập lớn Học máy · Nhóm 13 · GVHD: Nguyễn Thị Huyền Trang")

st.markdown(
    "Demo chạy **local** trên các mô hình của nhóm, được huấn luyện lại đúng quy trình "
    "và khớp số liệu đã báo cáo. Chọn một trong hai demo bên dưới hoặc ở thanh bên trái."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Khách hàng", f"{summary['rows']:,}".replace(",", "."))
c2.metric("Tỷ lệ Churn", fmt_pct(summary["churn_rate"]))
c3.metric("Random Forest · F1", f"{rf['F1']:.4f}")
c4.metric("K-Means k = 4 · Silhouette", f"{summary['silhouette_k4']:.4f}")

st.divider()
left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("🔮 Demo 1 · Dự đoán Churn")
    st.write(
        "Nhập thông tin một khách hàng (hợp đồng, dịch vụ, cước phí...) hoặc lấy khách "
        "thật từ tập Test, mô hình **Random Forest** dự đoán xác suất khách rời bỏ."
    )
    st.page_link("pages/1_Du_doan_Churn.py", label="Mở demo dự đoán", icon="➡️")
with right, st.container(border=True):
    st.subheader("👥 Demo 2 · Phân khúc khách hàng")
    st.write(
        "Nhập thời gian gắn bó, cước tháng và các dịch vụ đang dùng, mô hình **K-Means "
        "k = 4** xếp khách vào một trong bốn phân khúc."
    )
    st.page_link("pages/2_Phan_khuc_khach_hang.py", label="Mở demo phân khúc", icon="➡️")

train_rows = f"{summary['train_rows']:,}".replace(",", ".")
test_rows = f"{summary['test_rows']:,}".replace(",", ".")
st.caption(
    f"Dữ liệu: Telco Customer Churn (IBM, Kaggle) · Train/Test {train_rows}/{test_rows} khách, "
    "chia 80/20, stratify, random_state = 42."
)
