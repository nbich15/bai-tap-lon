import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Cấu hình trang web
st.set_page_config(
    page_title="Sentinels of Solvency | Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ĐỂ NÂNG CẤP GIAO DIỆN ---
st.markdown("""
    <style>
    /* Nền và font chữ */
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1f77b4;
    }
    /* Header chuyên nghiệp */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3d59;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
        border-radius: 15px;
        margin-bottom: 25px;
    }
    /* Tùy chỉnh sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e3d59;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header"><h1>🛡️ SENTINELS OF SOLVENCY</h1><p>Hệ thống Phân tích & Phát hiện Gian lận Tài chính Cao cấp</p></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=100) # Icon minh họa
    st.title("Bảng Điều Khiển")
    uploaded_file = st.file_uploader("📂 Tải lên dữ liệu sổ cái (CSV)", type=["csv"])
    st.info("Nguồn dữ liệu tham chiếu: Financial Anomaly Data [1]")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- KHỐI CHỈ SỐ (METRICS) ĐẲNG CẤP ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Giao Dịch", f"{len(df):,}")
    with col2:
        avg_val = df.select_dtypes(include=['number']).iloc[:, 0].mean()
        st.metric("Giá Trị Trung Bình", f"${avg_val:,.2f}")
    with col3:
        max_val = df.select_dtypes(include=['number']).iloc[:, 0].max()
        st.metric("Giao Dịch Lớn Nhất", f"${max_val:,.2f}")
    with col4:
        st.metric("Trạng Thái Hệ Thống", "Sẵn sàng", delta="Ổn định")

    # --- TẠO TAB VỚI UI HIỆN ĐẠI ---
    tab1, tab2 = st.tabs(["📈 Phân Tích Xu Hướng", "🔍 Phát Hiện Bất Thường"])

    with tab1:
        st.subheader("Trực quan hóa Dòng tiền")
        # Sử dụng template 'plotly_white' hoặc 'ggplot2' để nhìn chuyên nghiệp hơn
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            fig = px.area(df, y=numeric_cols, 
                          title="Biểu đồ Biến động Giao dịch",
                          line_shape="spline", 
                          color_discrete_sequence=['#1e3d59'])
            fig.update_layout(hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_settings, col_viz = st.columns(2)
        
        with col_settings:
            st.write("### Cài đặt Mô hình AI")
            features = st.multiselect("Đặc trưng phân tích:", numeric_cols, default=numeric_cols[:2])
            contamination = st.select_slider("Mức độ nhạy cảm (Tỷ lệ lỗi):", 
                                            options=[0.01, 0.02, 0.05, 0.1, 0.15], 
                                            value=0.05)
            run_btn = st.button("🚀 Bắt đầu Quét Gian lận")

        if run_btn:
            # Huấn luyện mô hình Isolation Forest
            model = IsolationForest(contamination=contamination, random_state=42)
            data_model = df[features].fillna(0)
            df['anomaly'] = model.fit_predict(data_model)
            df['status'] = df['anomaly'].map({1: 'An toàn', -1: 'Rủi ro cao'})

            with col_viz:
                st.write("### Kết quả Phân tích Trực quan")
                fig_res = px.scatter(df, x=features[0], y=features[1] if len(features)>1 else features[0],
                                     color='status',
                                     color_discrete_map={'An toàn': '#2ecc71', 'Rủi ro cao': '#e74c3c'},
                                     symbol='status',
                                     title="Bản đồ Phân cụm Rủi ro")
                fig_res.update_layout(template="plotly_white")
                st.plotly_chart(fig_res, use_container_width=True)

            # Danh sách đen (Blacklist)
            st.subheader("⚠️ Danh sách Giao dịch cần Kiểm tra Ngay")
            frauds = df[df['status'] == 'Rủi ro cao']
            st.table(frauds.head(10)) # Hiển thị dạng bảng tĩnh cho chuyên nghiệp

            # Nút tải báo cáo thiết kế bắt mắt
            csv = frauds.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải Báo cáo Chi tiết (CSV)",
                data=csv,
                file_name='fraud_report.csv',
                mime='text/csv',
            )
else:
    # Màn hình chờ khi chưa có dữ liệu
    st.warning("Vui lòng tải lên tệp CSV để kích hoạt hệ thống giám sát.")
    st.image("https://i.imgur.com/8W9pX9m.png", caption="Hệ thống đang chờ dữ liệu đầu vào...")
