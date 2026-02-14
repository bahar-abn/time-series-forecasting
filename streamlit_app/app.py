"""
🚀 برنامه اصلی Streamlit برای پیش‌بینی سری‌های زمانی
این برنامه شامل صفحات مختلف برای تحلیل، تجزیه و پیش‌بینی سری‌های زمانی است
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import TimeSeriesGenerator, load_or_generate_time_series
from src.preprocessor import TimeSeriesPreprocessor
from src.decomposition.seasonal_decompose import SeasonalDecomposer
from src.decomposition.stl_decompose import STLDecomposer
from src.models.arima import ARIMAModel
from src.models.sarima import SARIMAModel
from src.models.prophet import ProphetModel
from src.models.baseline import BaselineModels
from src.evaluation.metrics import TimeSeriesMetrics
from src.evaluation.backtesting import TimeSeriesBacktester
from src.visualizer import TimeSeriesVisualizer
from src.utils import format_date, format_number

# ============================================
# تنظیمات صفحه
# ============================================

st.set_page_config(
    page_title="پیش‌بینی سری‌های زمانی",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# استایل CSS - پالت آبی تیره حرفه‌ای
# ============================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap');
    
    * {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    
    /* ========== رنگ‌بندی اصلی ========== */
    /* پس‌زمینه اصلی - آبی تیره بسیار تیره */
    .stApp {
        background-color: #0a1929 !important;
    }
    
    /* ========== سایدبار - آبی تیره روشن‌تر ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #132f4c 0%, #0a1929 100%) !important;
        border-right: 1px solid #1e3a5f !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2b4c7c !important;
        padding-bottom: 0.5rem !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
        background: #1e3a5f !important;
        padding: 0.5rem !important;
        border-radius: 0.5rem !important;
        margin: 0.2rem 0 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #2b4c7c !important;
    }
    
    section[data-testid="stSidebar"] .stSlider label {
        color: #e0e0e0 !important;
    }
    
    /* ========== کارت‌های اصلی ========== */
    .card {
        background: #132f4c !important;
        padding: 1.5rem !important;
        border-radius: 1rem !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3) !important;
        margin-bottom: 1rem !important;
        border: 1px solid #2b4c7c !important;
        color: #ffffff !important;
    }
    
    .card h1, .card h2, .card h3, .card p {
        color: #ffffff !important;
    }
    
    /* ========== کارت‌های متریک ========== */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0a1929 100%) !important;
        padding: 1.5rem !important;
        border-radius: 0.75rem !important;
        text-align: center !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4) !important;
        border: 1px solid #2b4c7c !important;
        color: #ffffff !important;
    }
    
    .metric-card h1, .metric-card h2, .metric-card h3, .metric-card p {
        color: #ffffff !important;
    }
    
    .metric-card .metric-value {
        color: #4fc3f7 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    /* ========== باکس بینش ========== */
    .insight-box {
        background: #1e3a5f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border-right: 4px solid #4fc3f7 !important;
        margin: 1rem 0 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .insight-box h4, .insight-box p {
        color: #ffffff !important;
    }
    
    .insight-box h4 {
        color: #4fc3f7 !important;
        font-weight: bold !important;
    }
    
    /* ========== هدر اصلی ========== */
    .main-header {
        background: linear-gradient(135deg, #132f4c 0%, #0a1929 100%) !important;
        padding: 2rem !important;
        border-radius: 1rem !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
        color: #ffffff !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4) !important;
        border: 1px solid #2b4c7c !important;
    }
    
    .main-header h1, .main-header p {
        color: #ffffff !important;
    }
    
    .main-header h1 {
        color: #4fc3f7 !important;
        font-size: 2.5rem !important;
    }
    
    /* ========== تب‌ها ========== */
    .stTabs [data-baseweb="tab-list"] {
        background: #132f4c !important;
        padding: 0.5rem !important;
        border-radius: 0.75rem !important;
        gap: 0.5rem !important;
        border: 1px solid #2b4c7c !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #1e3a5f !important;
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2b4c7c 0%, #1e3a5f 100%) !important;
        color: #ffffff !important;
        border: 1px solid #4fc3f7 !important;
    }
    
    /* ========== دکمه‌ها ========== */
    .stButton button {
        background: linear-gradient(135deg, #2b4c7c 0%, #1e3a5f 100%) !important;
        color: white !important;
        border: 1px solid #4fc3f7 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #3a6ea5 0%, #2b4c7c 100%) !important;
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(79, 195, 247, 0.3) !important;
        border-color: #4fc3f7 !important;
    }
    
    /* ========== سلکت‌باکس و اینپوت ========== */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background: #1e3a5f !important;
        border: 1px solid #2b4c7c !important;
        border-radius: 0.5rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] * {
        color: #ffffff !important;
    }
    
    /* ========== چک‌باکس ========== */
    .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    .stCheckbox div[data-baseweb="checkbox"] {
        background: #1e3a5f !important;
        border-color: #2b4c7c !important;
    }
    
    /* ========== جدول‌ها ========== */
    .dataframe {
        background: #132f4c !important;
        color: #ffffff !important;
        border-radius: 0.75rem !important;
        border: 1px solid #2b4c7c !important;
    }
    
    .dataframe th {
        background: #1e3a5f !important;
        color: #4fc3f7 !important;
        padding: 0.75rem !important;
        text-align: center !important;
        font-weight: bold !important;
        border-bottom: 2px solid #2b4c7c !important;
    }
    
    .dataframe td {
        color: #e0e0e0 !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid #2b4c7c !important;
    }
    
    .dataframe tr:hover {
        background: #1e3a5f !important;
    }
    
    /* ========== نوتیفیکیشن‌ها ========== */
    .stAlert {
        background: #1e3a5f !important;
        color: #ffffff !important;
        border-radius: 0.5rem !important;
        border-right: 4px solid #4fc3f7 !important;
        border-left: none !important;
    }
    
    .stAlert p {
        color: #ffffff !important;
    }
    
    .stAlert .stAlert-success {
        background: #1e3a5f !important;
        border-right-color: #4fc3f7 !important;
    }
    
    .stAlert .stAlert-error {
        background: #1e3a5f !important;
        border-right-color: #ff6b6b !important;
    }
    
    .stAlert .stAlert-warning {
        background: #1e3a5f !important;
        border-right-color: #ffd93d !important;
    }
    
    .stAlert .stAlert-info {
        background: #1e3a5f !important;
        border-right-color: #4fc3f7 !important;
    }
    
    /* ========== اکسپندر ========== */
    .streamlit-expanderHeader {
        background: #1e3a5f !important;
        color: #ffffff !important;
        border-radius: 0.5rem !important;
        border: 1px solid #2b4c7c !important;
    }
    
    .streamlit-expanderContent {
        background: #132f4c !important;
        border: 1px solid #2b4c7c !important;
        border-top: none !important;
        border-radius: 0 0 0.5rem 0.5rem !important;
    }
    
    /* ========== فوتر ========== */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #e0e0e0 !important;
        font-size: 0.875rem;
        margin-top: 3rem;
        border-top: 1px solid #2b4c7c !important;
        background: #0a1929 !important;
    }
    
    .footer p {
        color: #e0e0e0 !important;
    }
    
    .footer a {
        color: #4fc3f7 !important;
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* ========== متریک‌های استریملیت ========== */
    .stMetric {
        background: #1e3a5f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        border: 1px solid #2b4c7c !important;
    }
    
    .stMetric label {
        color: #e0e0e0 !important;
        font-size: 0.875rem;
    }
    
    .stMetric .metric-value {
        color: #4fc3f7 !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* ========== پروگرس بار ========== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4fc3f7, #2b4c7c) !important;
    }
    
    /* ========== رادیو باتن‌ها ========== */
    div[role="radiogroup"] label {
        background: #1e3a5f !important;
        color: #e0e0e0 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.5rem !important;
        margin: 0.25rem !important;
        border: 1px solid #2b4c7c !important;
    }
    
    div[role="radiogroup"] label:hover {
        background: #2b4c7c !important;
        color: #ffffff !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #2b4c7c 0%, #1e3a5f 100%) !important;
        border: 1px solid #4fc3f7 !important;
        color: #ffffff !important;
    }
    
    /* ========== مولتی سلکت ========== */
    .stMultiSelect div[data-baseweb="select"] {
        background: #1e3a5f !important;
        border: 1px solid #2b4c7c !important;
    }
    
    .stMultiSelect div[data-baseweb="select"] * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# مقداردهی اولیه Session State
# ============================================

def init_session_state():
    """
    🎯 مقداردهی اولیه متغیرهای session state
    """
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'date_col' not in st.session_state:
        st.session_state.date_col = None
    if 'value_col' not in st.session_state:
        st.session_state.value_col = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = TimeSeriesVisualizer()
    if 'metrics' not in st.session_state:
        st.session_state.metrics = TimeSeriesMetrics()
    if 'backtester' not in st.session_state:
        st.session_state.backtester = TimeSeriesBacktester()
    if 'decomposition' not in st.session_state:
        st.session_state.decomposition = None
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = {}

init_session_state()

# ============================================
# هدر اصلی
# ============================================

st.markdown("""
<div class="main-header">
    <h1>📈 پیش‌بینی سری‌های زمانی</h1>
    <p style="font-size: 18px; margin-top: 10px;">
        تحلیل، تجزیه و پیش‌بینی فروش و مصرف انرژی با ARIMA، SARIMA و Prophet
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# سایدبار
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/time-machine.png", width=100)
    st.markdown("<h2 style='color: #ffffff; text-align: center;'>⚙️ کنترل پنل</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ========================================
    # بخش داده
    # ========================================
    
    st.markdown("<h3 style='color: #ffffff;'>📂 داده سری زمانی</h3>", unsafe_allow_html=True)
    
    data_type = st.selectbox(
        "نوع داده:",
        ["فروش (Sales)", "مصرف انرژی (Energy)", "آپلود فایل CSV"],
        index=0,
        key="data_type"
    )
    
    if data_type == "آپلود فایل CSV":
        uploaded_file = st.file_uploader(
            "فایل CSV را انتخاب کنید",
            type=['csv'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ {len(df):,} رکورد بارگذاری شد")
            except Exception as e:
                st.error(f"❌ خطا: {e}")
    else:
        n_points = st.slider(
            "تعداد نقاط:",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            key="n_points"
        )
        
        if st.button("🚀 تولید داده جدید", type="primary", use_container_width=True):
            with st.spinner("در حال تولید داده..."):
                generator = TimeSeriesGenerator()
                
                if data_type == "فروش (Sales)":
                    df = generator.generate_sales_data(save=True)
                else:
                    df = generator.generate_energy_data(save=True)
                
                st.session_state.df = df
                st.success(f"✅ {len(df):,} رکورد تولید شد")
                st.balloons()
    
    st.markdown("---")
    
    # ========================================
# بخش انتخاب ستون‌ها - اصلاح شده با تشخیص خودکار
# ========================================

if st.session_state.df is not None:
    st.markdown("<h3 style='color: #ffffff;'>📌 انتخاب ستون‌ها</h3>", unsafe_allow_html=True)
    
    # تشخیص خودکار ستون تاریخ
    date_keywords = ['date', 'time', 'timestamp', 'ds', 'day', 'month', 'year']
    date_col_options = []
    
    for col in st.session_state.df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            date_col_options.append(col)
    
    # اگر ستون تاریخ تشخیص داده نشد، همه ستون‌ها را نمایش بده
    if not date_col_options:
        date_col_options = st.session_state.df.columns.tolist()
    
    # اولویت با ستون‌های رایج
    priority_date_cols = ['date', 'timestamp', 'ds', 'transaction_time']
    default_date_idx = 0
    for i, col in enumerate(date_col_options):
        if col in priority_date_cols:
            default_date_idx = i
            break
    
    date_col = st.selectbox(
        "ستون تاریخ:",
        options=date_col_options,
        index=default_date_idx if default_date_idx < len(date_col_options) else 0,
        key="date_col_select",
        help="ستونی که شامل تاریخ است (مثلاً date, timestamp, transaction_time)"
    )
    
    # تشخیص خودکار ستون مقدار (فقط ستون‌های عددی)
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    
    # کلمات کلیدی برای ستون مقدار
    value_keywords = ['sales', 'energy', 'consumption', 'value', 'amount', 'price', 'count', 'quantity', 'y']
    
    value_col_options = []
    for col in numeric_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in value_keywords):
            value_col_options.append(col)
    
    # اگر ستون مقدار تشخیص داده نشد، همه ستون‌های عددی را نمایش بده
    if not value_col_options:
        value_col_options = numeric_cols
    
    if not value_col_options:
        st.error("⚠️ هیچ ستون عددی در داده‌ها یافت نشد! لطفاً فایل CSV با ستون‌های عددی آپلود کنید.")
        value_col_options = ["هیچ ستون عددی یافت نشد"]
    
    # اولویت با ستون‌های رایج
    priority_value_cols = ['sales', 'energy_consumption', 'y', 'value', 'amount']
    default_value_idx = 0
    for i, col in enumerate(value_col_options):
        if col in priority_value_cols:
            default_value_idx = i
            break
    
    value_col = st.selectbox(
        "ستون مقدار:",
        options=value_col_options,
        index=default_value_idx if default_value_idx < len(value_col_options) else 0,
        key="value_col_select",
        help="ستونی که شامل مقدار عددی است (مثلاً sales, energy_consumption, amount)"
    )
    
    st.session_state.date_col = date_col
    st.session_state.value_col = value_col
    
    # نمایش نمونه‌ای از داده
    with st.expander("📋 نمونه داده", expanded=False):
        st.write(f"**ستون تاریخ:** {date_col}")
        st.write(f"**ستون مقدار:** {value_col}")
        st.write(f"**نوع داده:** {st.session_state.df[value_col].dtype}")
        
        sample_df = st.session_state.df[[date_col, value_col]].head(10)
        st.dataframe(sample_df)
        
        # نمایش آمار سریع
        if value_col in numeric_cols:
            st.write("**آمار توصیفی:**")
            st.write(f"- حداقل: {st.session_state.df[value_col].min():.2f}")
            st.write(f"- حداکثر: {st.session_state.df[value_col].max():.2f}")
            st.write(f"- میانگین: {st.session_state.df[value_col].mean():.2f}")
            st.write(f"- میانه: {st.session_state.df[value_col].median():.2f}")
        
        # ========================================
        # بخش پیش‌پردازش
        # ========================================
        
        st.markdown("<h3 style='color: #ffffff;'>🛠 پیش‌پردازش</h3>", unsafe_allow_html=True)
        
        handle_missing = st.checkbox("مدیریت مقادیر گمشده", value=True)
        handle_outliers = st.checkbox("مدیریت داده‌های پرت", value=True)
        
        if st.button("🔄 اعمال پیش‌پردازش", use_container_width=True):
            with st.spinner("در حال پیش‌پردازش..."):
                preprocessor = TimeSeriesPreprocessor()
                processed = preprocessor.preprocess(
                    st.session_state.df,
                    date_col=st.session_state.date_col,
                    value_col=st.session_state.value_col
                )
                st.session_state.preprocessor = preprocessor
                st.session_state.processed_data = processed
                st.success("✅ پیش‌پردازش با موفقیت انجام شد")
        
        st.markdown("---")
        
        # ========================================
        # بخش مدل
        # ========================================
        
        st.markdown("<h3 style='color: #ffffff;'>🤖 انتخاب مدل</h3>", unsafe_allow_html=True)
        
        model_type = st.multiselect(
            "مدل‌های پیش‌بینی:",
            ["ARIMA", "SARIMA", "Prophet", "Baseline"],
            default=["ARIMA", "Prophet"],
            key="model_type"
        )
        
        forecast_horizon = st.slider(
            "افق پیش‌بینی (روز):",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            key="forecast_horizon"
        )
        
        seasonal_period = st.selectbox(
            "دوره فصلی:",
            [7, 30, 365],
            index=0,
            format_func=lambda x: f"{x} روز" if x < 100 else f"{x} روز (سالانه)",
            key="seasonal_period"
        )
        
        if st.button("🚀 اجرای پیش‌بینی", type="primary", use_container_width=True):
            st.session_state.run_forecast = True

# ============================================
# صفحه اصلی
# ============================================

if st.session_state.df is None:
    # صفحه خوش‌آمدگویی
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2 style="color: #4fc3f7;">📊 1</h2>
            <h3>داده سری زمانی</h3>
            <p>تولید داده فروش یا مصرف انرژی</p>
            <p style="color: #4fc3f7;">✓ ۵۰۰+ رکورد</p>
            <p style="color: #4fc3f7;">✓ الگوهای فصلی</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h2 style="color: #4fc3f7;">🔍 2</h2>
            <h3>تجزیه و تحلیل</h3>
            <p>تحلیل روند و فصلی بودن</p>
            <p style="color: #4fc3f7;">✓ Seasonal Decompose</p>
            <p style="color: #4fc3f7;">✓ STL Decomposition</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h2 style="color: #4fc3f7;">🤖 3</h2>
            <h3>پیش‌بینی</h3>
            <p>مدل‌های ARIMA, SARIMA, Prophet</p>
            <p style="color: #4fc3f7;">✓ RMSE, MAE, MAPE</p>
            <p style="color: #4fc3f7;">✓ مقایسه مدل‌ها</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h3 style="color: #4fc3f7;">✨ برای شروع، از منوی کناری داده تولید یا آپلود کنید</h3>
    </div>
    """, unsafe_allow_html=True)

else:
    df = st.session_state.df
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col
    
    # ========================================
    # نمایش اطلاعات داده
    # ========================================
    
    st.subheader("📋 پیش‌نمایش داده سری زمانی")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:16px;">تعداد رکوردها</h3>
            <p class="metric-value" style="margin:10px 0; font-size:32px; font-weight:bold;">{len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        min_date = pd.to_datetime(df[date_col]).min()
        max_date = pd.to_datetime(df[date_col]).max()
        date_range = (max_date - min_date).days
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:16px;">بازه زمانی</h3>
            <p class="metric-value" style="margin:10px 0; font-size:24px; font-weight:bold;">{date_range} روز</p>
            <p style="margin:0;">{min_date.date()} تا {max_date.date()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mean_value = df[value_col].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:16px;">میانگین</h3>
            <p class="metric-value" style="margin:10px 0; font-size:32px; font-weight:bold;">{format_number(mean_value)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        std_value = df[value_col].std()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:16px;">انحراف معیار</h3>
            <p class="metric-value" style="margin:10px 0; font-size:32px; font-weight:bold;">{format_number(std_value)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📊 مشاهده داده‌ها", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    
    # ========================================
    # ایجاد تب‌ها
    # ========================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 تحلیل داده",
        "🔍 تجزیه سری زمانی",
        "🤖 آموزش مدل‌ها",
        "📈 پیش‌بینی",
        "📋 گزارش"
    ])
    
    # ========================================
    # تب 1: تحلیل داده
    # ========================================
    
    with tab1:
        st.header("📊 تحلیل داده‌های سری زمانی")
        
        # نمودار اصلی
        fig_main = st.session_state.visualizer.plot_time_series(
            df, date_col, value_col,
            title="سری زمانی اصلی"
        )
        st.plotly_chart(fig_main, use_container_width=True)
        
        # الگوهای فصلی
        col1, col2 = st.columns(2)
        
        with col1:
            fig_monthly = st.session_state.visualizer.plot_seasonal_patterns(
                df, date_col, value_col, period='month'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            fig_weekly = st.session_state.visualizer.plot_seasonal_patterns(
                df, date_col, value_col, period='dayofweek'
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        # آمار توصیفی
        st.subheader("📋 آمار توصیفی")
        st.dataframe(df[value_col].describe(), use_container_width=True)
    
    # ========================================
    # تب 2: تجزیه سری زمانی
    # ========================================
    
    with tab2:
        st.header("🔍 تجزیه سری زمانی")
        
        col1, col2 = st.columns(2)
        
        with col1:
            decomp_method = st.selectbox(
                "روش تجزیه:",
                ["Seasonal Decompose", "STL Decomposition"],
                key="decomp_method"
            )
        
        with col2:
            period = st.number_input(
                "دوره فصلی:",
                min_value=2,
                max_value=365,
                value=7,
                key="decomp_period"
            )
        
        if st.button("🔍 اجرای تجزیه", use_container_width=True):
            with st.spinner("در حال تجزیه سری زمانی..."):
                series = df.set_index(date_col)[value_col]
                
                if decomp_method == "Seasonal Decompose":
                    from src.decomposition.seasonal_decompose import SeasonalDecomposer
                    decomposer = SeasonalDecomposer()
                    decomposition = decomposer.decompose(series, period=period)
                    components = decomposer.get_components()
                    strengths = decomposer.get_component_strength()
                else:
                    from src.decomposition.stl_decompose import STLDecomposer
                    decomposer = STLDecomposer()
                    decomposition = decomposer.decompose(series, period=period)
                    components = decomposer.get_components()
                    strengths = decomposer.get_component_strength()
                
                st.session_state.decomposition = components
                
                # نمایش قدرت اجزا
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("قدرت فصلی", f"{strengths['seasonal_strength']:.3f}")
                with col2:
                    st.metric("قدرت روند", f"{strengths['trend_strength']:.3f}")
                with col3:
                    st.metric("نسبت باقیمانده", f"{strengths['residual_ratio']:.3f}")
        
        if st.session_state.decomposition is not None:
            # نمودار تجزیه
            fig_decomp = st.session_state.visualizer.plot_decomposition(
                st.session_state.decomposition,
                pd.to_datetime(df[date_col])
            )
            st.plotly_chart(fig_decomp, use_container_width=True)
    
    # ========================================
    # تب 3: آموزش مدل‌ها
    # ========================================
    
    with tab3:
        st.header("🤖 آموزش مدل‌های پیش‌بینی")
        
        if st.session_state.get('run_forecast', False):
            with st.spinner("🚀 در حال آموزش مدل‌ها..."):
                
                series = df.set_index(date_col)[value_col]
                
                # تقسیم داده
                train_size = int(len(series) * 0.8)
                train = series[:train_size]
                test = series[train_size:]
                
                results = {}
                
                # ========================================
                # مدل‌های پایه
                # ========================================
                
                if "Baseline" in model_type:
                    st.info("📊 در حال آموزش مدل‌های پایه...")
                    baseline = BaselineModels()
                    
                    # پیش‌بینی با مدل‌های پایه
                    baseline_forecasts = baseline.ensemble_baseline(
                        train, seasonal_period=seasonal_period, steps=len(test)
                    )
                    
                    for name, forecast in baseline_forecasts.items():
                        metrics = st.session_state.metrics.generate_full_report(
                            test.values[:len(forecast)],
                            forecast[:len(test)],
                            train.values
                        )
                        results[f"Baseline_{name}"] = {
                            'forecast': forecast,
                            'metrics': metrics['metrics'],
                            'model': baseline
                        }
                
                # ========================================
                # مدل ARIMA
                # ========================================
                
                if "ARIMA" in model_type:
                    st.info("📈 در حال آموزش مدل ARIMA...")
                    arima = ARIMAModel()
                    arima.train(train)
                    forecast = arima.predict(steps=len(test))
                    
                    metrics = st.session_state.metrics.generate_full_report(
                        test.values,
                        forecast['forecast'][:len(test)],
                        train.values
                    )
                    
                    results['ARIMA'] = {
                        'forecast': forecast['forecast'],
                        'metrics': metrics['metrics'],
                        'model': arima
                    }
                
                # ========================================
                # مدل SARIMA
                # ========================================
                
                if "SARIMA" in model_type:
                    st.info("📈 در حال آموزش مدل SARIMA...")
                    sarima = SARIMAModel()
                    sarima.auto_sarima(train, seasonal_period=seasonal_period)
                    forecast = sarima.predict(steps=len(test))
                    
                    metrics = st.session_state.metrics.generate_full_report(
                        test.values,
                        forecast['forecast'][:len(test)],
                        train.values
                    )
                    
                    results['SARIMA'] = {
                        'forecast': forecast['forecast'],
                        'metrics': metrics['metrics'],
                        'model': sarima
                    }
                
                # ========================================
                # مدل Prophet
                # ========================================
                
                if "Prophet" in model_type:
                    st.info("📈 در حال آموزش مدل Prophet...")
                    
                    # آماده‌سازی داده برای Prophet
                    prophet_df = pd.DataFrame({
                        'ds': train.index,
                        'y': train.values
                    })
                    
                    prophet = ProphetModel()
                    prophet.train(prophet_df, date_col='ds', value_col='y')
                    
                    # پیش‌بینی
                    future = prophet.model.make_future_dataframe(periods=len(test))
                    forecast_df = prophet.model.predict(future)
                    forecast_values = forecast_df['yhat'].values[-len(test):]
                    
                    metrics = st.session_state.metrics.generate_full_report(
                        test.values,
                        forecast_values,
                        train.values
                    )
                    
                    # پیش‌بینی با فاصله اطمینان
                    forecast_with_ci = {
                        'forecast': forecast_values,
                        'lower_bound': forecast_df['yhat_lower'].values[-len(test):],
                        'upper_bound': forecast_df['yhat_upper'].values[-len(test):]
                    }
                    
                    results['Prophet'] = {
                        'forecast': forecast_with_ci,
                        'metrics': metrics['metrics'],
                        'model': prophet
                    }
                
                st.session_state.forecast_results = results
                st.session_state.test_data = test
                st.session_state.train_data = train
                
                st.success(f"✅ {len(results)} مدل با موفقیت آموزش دیدند!")
                st.balloons()
            
            # نمایش نتایج
            if st.session_state.forecast_results:
                
                # جدول مقایسه
                comparison_data = []
                for name, result in st.session_state.forecast_results.items():
                    comparison_data.append({
                        'Model': name,
                        'RMSE': result['metrics'].get('rmse', 0),
                        'MAE': result['metrics'].get('mae', 0),
                        'MAPE': result['metrics'].get('mape', 0),
                        'R²': result['metrics'].get('r2', 0)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('RMSE')
                
                st.subheader("📊 مقایسه مدل‌ها")
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # بهترین مدل
                best_model = comparison_df.iloc[0]['Model']
                st.markdown(f"""
                <div class="insight-box">
                    <h4 style="margin-top: 0;">🏆 بهترین مدل: {best_model}</h4>
                    <p>RMSE: {comparison_df.iloc[0]['RMSE']:.2f}</p>
                    <p>MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================
    # تب 4: پیش‌بینی
    # ========================================
    
    with tab4:
        st.header("📈 پیش‌بینی مقادیر آینده")
        
        if st.session_state.forecast_results:
            
            # انتخاب مدل برای پیش‌بینی
            model_names = list(st.session_state.forecast_results.keys())
            selected_model = st.selectbox(
                "انتخاب مدل برای پیش‌بینی:",
                model_names,
                index=0
            )
            
            if selected_model:
                result = st.session_state.forecast_results[selected_model]
                
                # پیش‌بینی برای افق جدید
                n_future = st.slider(
                    "تعداد روزهای آینده:",
                    min_value=7,
                    max_value=90,
                    value=forecast_horizon,
                    key="future_days"
                )
                
                if st.button("🔮 اجرای پیش‌بینی", use_container_width=True):
                    
                    with st.spinner("در حال پیش‌بینی..."):
                        
                        if selected_model.startswith('Baseline'):
                            # مدل‌های پایه
                            baseline = result['model']
                            name = selected_model.replace('Baseline_', '')
                            
                            if name == 'naive':
                                forecast = baseline.naive_forecast(st.session_state.train_data, n_future)
                            elif name == 'mean':
                                forecast = baseline.mean_forecast(st.session_state.train_data, n_future)
                            elif name == 'moving_average':
                                forecast = baseline.moving_average_forecast(
                                    st.session_state.train_data, 
                                    window=seasonal_period, 
                                    steps=n_future
                                )
                            elif name == 'seasonal_naive':
                                forecast = baseline.seasonal_naive_forecast(
                                    st.session_state.train_data,
                                    seasonal_period=seasonal_period,
                                    steps=n_future
                                )
                            else:
                                forecast = baseline.weighted_moving_average(
                                    st.session_state.train_data,
                                    window=seasonal_period,
                                    steps=n_future
                                )
                            
                            forecast_dict = {'forecast': forecast}
                            
                        elif selected_model == 'ARIMA':
                            model = result['model']
                            forecast_dict = model.predict(steps=n_future, return_conf_int=True)
                            
                        elif selected_model == 'SARIMA':
                            model = result['model']
                            forecast_dict = model.predict(steps=n_future, return_conf_int=True)
                            
                        elif selected_model == 'Prophet':
                            model = result['model']
                            prophet_result = model.predict(periods=n_future)
                            
                            # استخراج پیش‌بینی و فاصله اطمینان
                            forecast_dict = {
                                'forecast': prophet_result['yhat'].values[-n_future:],
                                'lower_bound': prophet_result['yhat_lower'].values[-n_future:],
                                'upper_bound': prophet_result['yhat_upper'].values[-n_future:]
                            }
                        
                        # نمایش نمودار پیش‌بینی
                        fig_forecast = st.session_state.visualizer.plot_forecast(
                            df,
                            forecast_dict,
                            date_col,
                            value_col,
                            title=f"پیش‌بینی با مدل {selected_model}"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # نمایش مقادیر پیش‌بینی
                        st.subheader("📋 مقادیر پیش‌بینی شده")
                        
                        forecast_dates = pd.date_range(
                            start=pd.to_datetime(df[date_col].iloc[-1]) + pd.Timedelta(days=1),
                            periods=n_future
                        )
                        
                        forecast_df = pd.DataFrame({
                            'تاریخ': forecast_dates,
                            'پیش‌بینی': forecast_dict['forecast']
                        })
                        
                        if 'lower_bound' in forecast_dict:
                            forecast_df['حد پایین'] = forecast_dict['lower_bound']
                            forecast_df['حد بالا'] = forecast_dict['upper_bound']
                        
                        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("👈 ابتدا مدل‌ها را در تب 'آموزش مدل‌ها' آموزش دهید.")
    
    # ========================================
    # تب 5: گزارش
    # ========================================
    
    with tab5:
        st.header("📋 گزارش نهایی")
        
        if st.session_state.forecast_results:
            
            # خلاصه گزارش
            st.subheader("📊 خلاصه عملکرد مدل‌ها")
            
            comparison_data = []
            for name, result in st.session_state.forecast_results.items():
                comparison_data.append({
                    'مدل': name,
                    'RMSE': result['metrics'].get('rmse', 0),
                    'MAE': result['metrics'].get('mae', 0),
                    'MAPE': f"{result['metrics'].get('mape', 0):.2f}%",
                    'R²': result['metrics'].get('r2', 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('RMSE')
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # بهترین مدل
            best_model = comparison_df.iloc[0]['مدل']
            best_rmse = comparison_df.iloc[0]['RMSE']
            best_mape = comparison_df.iloc[0]['MAPE']
            
            st.markdown(f"""
            <div class="insight-box">
                <h4 style="margin-top: 0;">🏆 بهترین مدل</h4>
                <p style="font-size: 20px; font-weight: bold; color: #4fc3f7;">{best_model}</p>
                <p>RMSE: {best_rmse:.2f}</p>
                <p>MAPE: {best_mape}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # تحلیل خطا
            st.subheader("📉 تحلیل خطا")
            
            if 'test_data' in st.session_state and st.session_state.forecast_results:
                test_data = st.session_state.test_data
                best_result = st.session_state.forecast_results[best_model.split('_')[0] if 'Baseline' in best_model else best_model]
                
                if isinstance(best_result['forecast'], dict):
                    forecast_values = best_result['forecast']['forecast'][:len(test_data)]
                else:
                    forecast_values = best_result['forecast'][:len(test_data)]
                
                fig_residuals = st.session_state.visualizer.plot_residuals(
                    test_data.values,
                    forecast_values,
                    test_data.index
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        else:
            st.info("👈 ابتدا مدل‌ها را در تب 'آموزش مدل‌ها' آموزش دهید.")

# ============================================
# فوتر
# ============================================

st.markdown("""
<div class="footer">
    <p>📈 پیش‌بینی سری‌های زمانی - پروژه علم داده</p>
    <p>توسعه داده شده با ❤️ برای جامعه داده کاوی ایران</p>
    <p style="font-size: 12px; margin-top: 1rem;">نسخه ۱.۰.۰</p>
</div>
""", unsafe_allow_html=True)