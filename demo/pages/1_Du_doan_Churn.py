"""Demo 1: dự đoán khách hàng có rời bỏ dịch vụ (Churn) hay không."""

import altair as alt
import pandas as pd
import streamlit as st

from common import fmt_pct, load_artifacts, setup_page

setup_page("Dự đoán Churn", "🔮")
art = load_artifacts()
prep = art["preprocessor"]
models = art["classifiers"]
samples = art["test_samples"]
metrics = art["summary"]["classification"]

YES_NO = ["No", "Yes"]
INTERNET_ADDONS = {
    "OnlineSecurity": "Bảo mật trực tuyến",
    "OnlineBackup": "Sao lưu trực tuyến",
    "DeviceProtection": "Bảo vệ thiết bị",
    "TechSupport": "Hỗ trợ kỹ thuật",
    "StreamingTV": "Truyền hình trực tuyến",
    "StreamingMovies": "Phim trực tuyến",
}
DEFAULTS = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
    "TotalCharges": 255.0, "auto_total": True,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("loaded_sample", None)


def load_sample(churn: int | None) -> None:
    pool = samples if churn is None else samples[samples["Churn"] == churn]
    row = pool.sample(1).iloc[0]
    for key in DEFAULTS:
        if key in ("auto_total",):
            continue
        value = row[key]
        if key == "SeniorCitizen":
            value = "Yes" if int(value) == 1 else "No"
        elif key == "tenure":
            value = int(value)
        elif key in ("MonthlyCharges", "TotalCharges"):
            value = 0.0 if pd.isna(value) else float(value)
        if key in INTERNET_ADDONS and value == "No internet service":
            value = "No"
        if key == "MultipleLines" and value == "No phone service":
            value = "No"
        st.session_state[key] = value
    st.session_state["auto_total"] = False
    st.session_state["loaded_sample"] = {"customerID": row["customerID"], "Churn": int(row["Churn"])}


def reset_form() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["loaded_sample"] = None


st.title("🔮 Demo 1 · Dự đoán khách hàng rời bỏ")
st.caption(
    "Nhập thông tin khách hàng, mô hình dự đoán xác suất rời bỏ. Tiền xử lý (điền thiếu, "
    "chuẩn hóa, mã hóa) dùng đúng bộ tham số đã fit trên tập Train."
)

b1, b2, b3, b4 = st.columns(4)
b1.button("🎲 Khách ngẫu nhiên (tập Test)", on_click=load_sample, args=(None,), width="stretch")
b2.button("🔴 Khách thực tế đã rời bỏ", on_click=load_sample, args=(1,), width="stretch")
b3.button("🟢 Khách thực tế ở lại", on_click=load_sample, args=(0,), width="stretch")
b4.button("↺ Nhập lại từ đầu", on_click=reset_form, width="stretch")

form_col, result_col = st.columns([3, 2], gap="large")

with form_col:
    with st.container(border=True):
        st.markdown("**👤 Thông tin khách hàng**")
        c1, c2, c3, c4 = st.columns(4)
        c1.selectbox("Giới tính", ["Female", "Male"], key="gender")
        c2.selectbox("Cao tuổi", YES_NO, key="SeniorCitizen")
        c3.selectbox("Có vợ/chồng", YES_NO, key="Partner")
        c4.selectbox("Có người phụ thuộc", YES_NO, key="Dependents")

    with st.container(border=True):
        st.markdown("**📄 Hợp đồng và thanh toán**")
        c1, c2 = st.columns(2)
        c1.slider("Thời gian gắn bó (tenure, tháng)", 0, 72, key="tenure")
        c2.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"], key="Contract")
        c1, c2 = st.columns(2)
        c1.selectbox(
            "Phương thức thanh toán",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="PaymentMethod",
        )
        c2.selectbox("Hóa đơn điện tử", YES_NO, key="PaperlessBilling")
        c1, c2, c3 = st.columns([2, 2, 1.4])
        c1.number_input("Cước tháng (MonthlyCharges)", 18.0, 120.0, step=1.0, key="MonthlyCharges")
        c3.checkbox("Tự tính tổng", key="auto_total", help="TotalCharges ≈ tenure × MonthlyCharges")
        if st.session_state["auto_total"]:
            st.session_state["TotalCharges"] = round(st.session_state["tenure"] * st.session_state["MonthlyCharges"], 2)
        c2.number_input(
            "Tổng cước (TotalCharges)", 0.0, 10000.0, step=10.0, key="TotalCharges",
            disabled=st.session_state["auto_total"],
        )

    with st.container(border=True):
        st.markdown("**📶 Dịch vụ đang dùng**")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Điện thoại", YES_NO, key="PhoneService")
        has_phone = st.session_state["PhoneService"] == "Yes"
        c2.selectbox("Nhiều đường dây", YES_NO, key="MultipleLines", disabled=not has_phone)
        c3.selectbox("Internet", ["DSL", "Fiber optic", "No"], key="InternetService")
        has_internet = st.session_state["InternetService"] != "No"
        cols = st.columns(3)
        for i, (key, label) in enumerate(INTERNET_ADDONS.items()):
            cols[i % 3].selectbox(label, YES_NO, key=key, disabled=not has_internet)
        if not has_internet:
            st.caption("Khách không dùng Internet nên các dịch vụ Internet được ghi là *No internet service*.")


def current_customer() -> pd.DataFrame:
    s = st.session_state
    row = {key: s[key] for key in DEFAULTS if key != "auto_total"}
    row["SeniorCitizen"] = 1 if s["SeniorCitizen"] == "Yes" else 0
    if s["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    if s["InternetService"] == "No":
        for key in INTERNET_ADDONS:
            row[key] = "No internet service"
    # Khách mới (tenure = 0) chưa có tổng cước, giống 11 ô trống trong dữ liệu gốc.
    if s["tenure"] == 0:
        row["TotalCharges"] = float("nan")
    return pd.DataFrame([row])


with result_col:
    model_name = st.radio(
        "Mô hình", ["Random Forest", "Logistic Regression"], horizontal=True,
        captions=["Mô hình chính (F1 cao nhất)", "Recall cao hơn"],
    )
    customer = current_customer()
    features = prep.transform(customer)
    proba = float(models[model_name].predict_proba(features)[0, 1])
    churn = proba >= 0.5

    with st.container(border=True):
        st.markdown("**Kết quả dự đoán**")
        st.metric("Xác suất rời bỏ", fmt_pct(proba))
        st.progress(proba)
        if churn:
            st.error("⚠️ **Có nguy cơ rời bỏ** (xác suất ≥ 50%). Nên ưu tiên chăm sóc khách này.")
        else:
            st.success("✅ **Khả năng ở lại** (xác suất < 50%).")

        loaded = st.session_state["loaded_sample"]
        if loaded:
            actual = "Đã rời bỏ (Yes)" if loaded["Churn"] else "Ở lại (No)"
            correct = bool(loaded["Churn"]) == churn
            st.markdown(f"Khách **{loaded['customerID']}** trong tập Test · Thực tế: **{actual}**")
            (st.success if correct else st.warning)(
                "Mô hình dự đoán **đúng**." if correct else "Mô hình dự đoán **sai** với khách này."
            )
            st.caption("Nếu chỉnh thông tin sau khi lấy khách mẫu, kết quả thực tế ở trên không còn áp dụng.")

    m = metrics[model_name]
    with st.expander(f"📊 Hiệu năng của {model_name} trên tập Test (1.409 khách)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                {"Độ đo": ["Accuracy", "Precision", "Recall", "F1-score"],
                 "Giá trị": [m["Accuracy"], m["Precision"], m["Recall"], m["F1"]]}
            ),
            hide_index=True, width="stretch",
        )
        st.markdown(
            f"Trong **{m['TP'] + m['FN']}** khách thực sự rời bỏ, mô hình bắt đúng **{m['TP']}**, "
            f"bỏ sót **{m['FN']}**. Có **{m['FP']}** khách bị cảnh báo nhầm."
        )

    if model_name == "Random Forest":
        with st.expander("🌲 Đặc trưng quan trọng nhất (Random Forest)", expanded=False):
            importance = (
                pd.DataFrame({"Đặc trưng": prep.feature_columns,
                              "Mức quan trọng": models["Random Forest"].feature_importances_})
                .sort_values("Mức quan trọng", ascending=False)
                .head(10)
            )
            chart = (
                alt.Chart(importance)
                .mark_bar(color="#0E7076")
                .encode(x="Mức quan trọng:Q", y=alt.Y("Đặc trưng:N", sort="-x"))
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")
            st.caption("Mức quan trọng cho biết đặc trưng được mô hình dùng nhiều, không phải quan hệ nhân quả.")
