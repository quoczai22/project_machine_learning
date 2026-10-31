"""Demo 2: xếp khách hàng vào một trong bốn phân khúc K-Means (k = 4)."""

import altair as alt
import pandas as pd
import streamlit as st

from common import SERVICE_COLUMNS, fmt_pct, load_artifacts, setup_page

setup_page("Phân khúc khách hàng", "👥")
art = load_artifacts()
scaler = art["clustering"]["scaler"]
kmeans = art["clustering"]["kmeans"]
points = art["cluster_points"]
profile = art["cluster_profile"].set_index("cluster")
silhouette = art["summary"]["silhouette_k4"]

SERVICE_LABELS = {
    "PhoneService": "Điện thoại",
    "MultipleLines": "Nhiều đường dây",
    "OnlineSecurity": "Bảo mật trực tuyến",
    "OnlineBackup": "Sao lưu trực tuyến",
    "DeviceProtection": "Bảo vệ thiết bị",
    "TechSupport": "Hỗ trợ kỹ thuật",
    "StreamingTV": "Truyền hình trực tuyến",
    "StreamingMovies": "Phim trực tuyến",
}
COLORS = {0: "#3C7BC8", 1: "#0E7076", 2: "#E8590C", 3: "#7A8CA3"}
SUGGESTIONS = {
    0: "Khách mới, dùng ít dịch vụ: gửi ưu đãi chào mừng, giới thiệu thêm gói dịch vụ phù hợp.",
    1: "Khách giá trị cao, gắn bó lâu: chương trình khách hàng thân thiết, chăm sóc ưu tiên.",
    2: "Khách mới nhưng trả phí khá cao: ưu tiên giữ chân, ví dụ ưu đãi chuyển sang hợp đồng dài hạn, hỗ trợ kỹ thuật sớm.",
    3: "Khách trung thành, chi phí thấp: duy trì chất lượng, ít cần can thiệp.",
}

st.session_state.setdefault("seg_tenure", 6)
st.session_state.setdefault("seg_charges", 80.0)
for column in SERVICE_COLUMNS:
    st.session_state.setdefault(f"svc_{column}", column in ("PhoneService", "StreamingTV"))


def load_example(cluster: int) -> None:
    row = points[points["cluster"] == cluster].sample(1).iloc[0]
    st.session_state["seg_tenure"] = int(row["tenure"])
    st.session_state["seg_charges"] = float(row["MonthlyCharges"])
    count = int(row["service_count"])
    for i, column in enumerate(SERVICE_COLUMNS):
        st.session_state[f"svc_{column}"] = i < count


st.title("👥 Demo 2 · Phân khúc khách hàng")
st.caption(
    f"K-Means k = 4 trên ba đặc trưng tenure, MonthlyCharges, service_count (đã chuẩn hóa) · "
    f"Silhouette = {silhouette:.4f}. Nhãn Churn **không** dùng để gom cụm."
)

st.write("Lấy nhanh một khách mẫu thuộc phân khúc:")
cols = st.columns(4)
for cluster in range(4):
    cols[cluster].button(
        f"Cụm {cluster}: {profile.loc[cluster, 'segment']}", on_click=load_example, args=(cluster,),
        width="stretch",
    )

input_col, result_col = st.columns([2, 3], gap="large")

with input_col, st.container(border=True):
    st.markdown("**Thông tin khách hàng**")
    st.slider("Thời gian gắn bó (tenure, tháng)", 0, 72, key="seg_tenure")
    st.slider("Cước tháng (MonthlyCharges)", 18.0, 120.0, step=0.5, key="seg_charges")
    st.markdown("Dịch vụ đang dùng:")
    svc_cols = st.columns(2)
    for i, column in enumerate(SERVICE_COLUMNS):
        svc_cols[i % 2].checkbox(SERVICE_LABELS[column], key=f"svc_{column}")
    n_services = sum(st.session_state[f"svc_{c}"] for c in SERVICE_COLUMNS)
    st.metric("service_count (số dịch vụ)", n_services)

customer = pd.DataFrame(
    [{"tenure": st.session_state["seg_tenure"], "MonthlyCharges": st.session_state["seg_charges"],
      "service_count": n_services}]
)
cluster = int(kmeans.predict(scaler.transform(customer))[0])
info = profile.loc[cluster]

with result_col:
    with st.container(border=True):
        st.markdown(f"### Thuộc **Cụm {cluster}: {info['segment']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Số khách trong cụm", f"{int(info['count']):,}".replace(",", "."))
        c2.metric("Tenure TB", f"{info['tenure_mean']:.1f} tháng")
        c3.metric("Cước tháng TB", f"{info['MonthlyCharges_mean']:.2f}")
        st.metric(
            "Tỷ lệ Churn thực tế trong cụm", fmt_pct(info["churn_rate"]),
            help="Tính sau khi gom cụm để diễn giải, không dùng làm đặc trưng.",
        )
        st.info("💡 Gợi ý: " + SUGGESTIONS[cluster])

    view = points.sample(2500, random_state=42).copy()
    view["Phân khúc"] = view["cluster"].map(lambda c: f"Cụm {c}: {profile.loc[c, 'segment']}")
    domain = [f"Cụm {c}: {profile.loc[c, 'segment']}" for c in range(4)]
    base = (
        alt.Chart(view)
        .mark_circle(size=22, opacity=0.45)
        .encode(
            x=alt.X("tenure:Q", title="Tenure (tháng)"),
            y=alt.Y("MonthlyCharges:Q", title="MonthlyCharges"),
            color=alt.Color("Phân khúc:N", scale=alt.Scale(domain=domain, range=[COLORS[c] for c in range(4)]),
                            legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=["customerID", "tenure", "MonthlyCharges", "service_count", "Phân khúc"],
        )
    )
    star = (
        alt.Chart(customer)
        .mark_point(shape="diamond", size=420, filled=True, color="#111111", stroke="white", strokeWidth=2)
        .encode(x="tenure:Q", y="MonthlyCharges:Q")
    )
    st.altair_chart((base + star).properties(height=380), width="stretch")
    st.caption("◆ màu đen là khách vừa nhập. Biểu đồ vẽ 2.500 khách ngẫu nhiên; màu theo cụm K-Means.")

st.subheader("Hồ sơ bốn phân khúc")
table = profile.reset_index().rename(columns={
    "cluster": "Cụm", "segment": "Phân khúc", "count": "Số khách", "tenure_mean": "Tenure TB",
    "MonthlyCharges_mean": "Cước tháng TB", "service_count_mean": "Số dịch vụ TB", "churn_rate": "Tỷ lệ Churn",
})[["Cụm", "Phân khúc", "Số khách", "Tenure TB", "Cước tháng TB", "Số dịch vụ TB", "Tỷ lệ Churn"]]
table["Tỷ lệ Churn"] = table["Tỷ lệ Churn"].map(fmt_pct)
st.dataframe(
    table.style.format({"Tenure TB": "{:.2f}", "Cước tháng TB": "{:.2f}", "Số dịch vụ TB": "{:.2f}"})
    .apply(lambda r: ["background-color: #FFF1E6" if r["Cụm"] == cluster else "" for _ in r], axis=1),
    hide_index=True, width="stretch",
)
