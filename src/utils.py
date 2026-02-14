"""
📁 ماژول توابع کمکی
این ماژول شامل توابع کاربردی برای کل پروژه پیش‌بینی سری‌های زمانی است
"""

import os
import yaml
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

# ============================================
# تنظیم سیستم لاگ‌گیری
# ============================================

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    🎯 تنظیم سیستم لاگ‌گیری
    
    پارامترها:
        name: نام logger
        log_file: مسیر فایل لاگ
        level: سطح لاگ‌گیری
    
    بازگشت:
        logger: شیء logger پیکربندی شده
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # هندلر کنسول
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # هندلر فایل
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# ============================================
# بارگذاری تنظیمات
# ============================================

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    📋 بارگذاری فایل تنظیمات
    
    پارامترها:
        config_path: مسیر فایل تنظیمات
    
    بازگشت:
        config: دیکشنری تنظیمات
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"⚠️ فایل تنظیمات یافت نشد. استفاده از تنظیمات پیش‌فرض...")
        return get_default_config()
    except Exception as e:
        print(f"⚠️ خطا در بارگذاری تنظیمات: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    🔧 تنظیمات پیش‌فرض
    
    بازگشت:
        config: دیکشنری تنظیمات پیش‌فرض
    """
    return {
        'data': {
            'synthetic': {
                'sales': {
                    'n_points': 1000,
                    'start_date': '2020-01-01',
                    'freq': 'D'
                }
            },
            'test_size': 0.2
        },
        'models': {
            'baseline': {'enabled': True},
            'arima': {'enabled': True},
            'prophet': {'enabled': True}
        },
        'forecasting': {
            'forecast_horizon': 30
        }
    }

# ============================================
# ذخیره و بارگذاری مدل
# ============================================

def save_model(model, model_name: str, metadata: Optional[Dict] = None) -> str:
    """
    💾 ذخیره مدل آموزش دیده
    
    پارامترها:
        model: مدل آموزش دیده
        model_name: نام مدل
        metadata: ابرداده مدل
    
    بازگشت:
        save_path: مسیر فایل ذخیره شده
    """
    model_dir = Path('models/saved_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f"{model_name}_{timestamp}.joblib"
    
    joblib.dump(model, model_path)
    
    if metadata:
        metadata_path = model_dir / f"{model_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    return str(model_path)

def load_model(model_path: str):
    """
    📂 بارگذاری مدل ذخیره شده
    
    پارامترها:
        model_path: مسیر فایل مدل
    
    بازگشت:
        model: مدل بارگذاری شده
    """
    return joblib.load(model_path)

# ============================================
# توابع متریک‌های سری زمانی
# ============================================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    📊 محاسبه RMSE (ریشه میانگین مربعات خطا)
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    بازگشت:
        rmse: مقدار خطا
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    📊 محاسبه MAE (میانگین قدر مطلق خطا)
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    بازگشت:
        mae: مقدار خطا
    """
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    📊 محاسبه MAPE (درصد میانگین قدر مطلق خطا)
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    بازگشت:
        mape: درصد خطا
    """
    # جلوگیری از تقسیم بر صفر
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    📊 محاسبه MSE (میانگین مربعات خطا)
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    بازگشت:
        mse: مقدار خطا
    """
    return float(np.mean((y_true - y_pred) ** 2))

# ============================================
# توابع کمکی سری زمانی
# ============================================

def create_date_range(start_date: str, periods: int, freq: str = 'D') -> pd.DatetimeIndex:
    """
    📅 ایجاد بازه زمانی
    
    پارامترها:
        start_date: تاریخ شروع
        periods: تعداد دوره‌ها
        freq: فرکانس (D=day, H=hour, W=week, M=month)
    
    بازگشت:
        date_range: بازه زمانی
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)

def add_holiday_effect(dates: pd.DatetimeIndex, amplitude: float = 1.5) -> np.ndarray:
    """
    🎉 اضافه کردن اثر تعطیلات
    
    پارامترها:
        dates: تاریخ‌ها
        amplitude: دامنه اثر تعطیلات
    
    بازگشت:
        holiday_effect: اثر تعطیلات
    """
    # تعطیلات ثابت (مثال)
    holidays = [
        '2020-01-01',  # سال نو
        '2020-03-20',  # نوروز
        '2020-04-01',  # 1 آوریل
        '2020-12-25',  # کریسمس
    ]
    
    holiday_dates = pd.to_datetime(holidays)
    holiday_effect = np.zeros(len(dates))
    
    for i, date in enumerate(dates):
        if date in holiday_dates:
            holiday_effect[i] = amplitude
        # تعطیلات آخر هفته
        elif date.dayofweek >= 5:  # 5=شنبه, 6=یکشنبه
            holiday_effect[i] = 0.3
    
    return holiday_effect

# ============================================
# توابع فرمت‌دهی
# ============================================

def format_date(date: Union[str, datetime, pd.Timestamp], format: str = '%Y-%m-%d') -> str:
    """
    📅 فرمت کردن تاریخ
    
    پارامترها:
        date: تاریخ ورودی
        format: فرمت خروجی
    
    بازگشت:
        formatted_date: تاریخ فرمت شده
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date.strftime(format)

def format_number(num: float, decimals: int = 2) -> str:
    """
    🔢 فرمت کردن اعداد
    
    پارامترها:
        num: عدد ورودی
        decimals: تعداد اعشار
    
    بازگشت:
        formatted: عدد فرمت شده
    """
    if isinstance(num, (int, float)):
        if abs(num) >= 1_000_000:
            return f"{num/1_000_000:.{decimals}f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.{decimals}f}K"
        else:
            return f"{num:.{decimals}f}"
    return str(num)