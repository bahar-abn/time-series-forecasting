"""
📊 ماژول تولید داده‌های مصنوعی سری زمانی
این ماژول داده‌های فروش و مصرف انرژی را با الگوهای فصلی و روند تولید می‌کند
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

from src.utils import setup_logger, load_config, create_date_range, add_holiday_effect

class TimeSeriesGenerator:
    """
    🎯 کلاس تولید داده‌های مصنوعی سری زمانی
    
    این کلاس داده‌های واقع‌گرایانه از سری‌های زمانی را با ویژگی‌های زیر تولید می‌کند:
    - روند (Trend)
    - فصلی بودن (Seasonality) - روزانه، هفتگی، سالانه
    - نویز (Noise)
    - اثر تعطیلات (Holiday Effect)
    - ناهنجاری‌ها (Anomalies)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        
        پارامترها:
            config_path: مسیر فایل تنظیمات
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.data_config = self.config.get('data', {}).get('synthetic', {})
        
        self.logger = setup_logger(
            'time_series_generator',
            log_file='outputs/logs/time_series_generator.log'
        )
        
        self.logger.info("✅ TimeSeriesGenerator initialized")
    
    # ============================================
    # تولید داده فروش
    # ============================================
    
    def generate_sales_data(self, save: bool = True) -> pd.DataFrame:
        """
        🏪 تولید داده فروش با الگوهای فصلی
        
        پارامترها:
            save: ذخیره فایل CSV
        
        بازگشت:
            df: دیتافریم فروش
        """
        self.logger.info("🚀 شروع تولید داده فروش...")
        
        config = self.data_config.get('sales', {})
        
        n_points = config.get('n_points', 1000)
        start_date = config.get('start_date', '2020-01-01')
        freq = config.get('freq', 'D')
        trend = config.get('trend', 0.1)
        seasonality_config = config.get('seasonality', {})
        noise_level = config.get('noise', 20)
        random_state = config.get('random_state', 42)
        
        np.random.seed(random_state)
        
        # ایجاد بازه زمانی
        dates = create_date_range(start_date, n_points, freq)
        
        # ========================================
        # 1. روند خطی (Trend)
        # ========================================
        t = np.arange(n_points)
        trend_component = trend * t
        
        # ========================================
        # 2. فصلی بودن (Seasonality)
        # ========================================
        seasonal_component = np.zeros(n_points)
        amplitude = seasonality_config.get('amplitude', 100)
        
        # فصلی سالانه (365 روز)
        if seasonality_config.get('yearly', True):
            seasonal_component += amplitude * np.sin(2 * np.pi * t / 365)
        
        # فصلی هفتگی (7 روز)
        if seasonality_config.get('weekly', True):
            seasonal_component += amplitude * 0.5 * np.sin(2 * np.pi * t / 7)
        
        # ========================================
        # 3. اثر روز هفته
        # ========================================
        dayofweek_effect = np.zeros(n_points)
        for i, date in enumerate(dates):
            # فروش بیشتر در آخر هفته
            if date.dayofweek == 5:  # شنبه
                dayofweek_effect[i] = 50
            elif date.dayofweek == 6:  # یکشنبه
                dayofweek_effect[i] = 80
            elif date.dayofweek == 0:  # دوشنبه
                dayofweek_effect[i] = -30
            elif date.dayofweek == 4:  # پنجشنبه
                dayofweek_effect[i] = 30
        
        # ========================================
        # 4. اثر تعطیلات
        # ========================================
        holiday_effect = add_holiday_effect(dates) * 100
        
        # ========================================
        # 5. نویز
        # ========================================
        noise = np.random.normal(0, noise_level, n_points)
        
        # ========================================
        # 6. ترکیب همه اجزا
        # ========================================
        base_sales = 500  # پایه فروش
        sales = (base_sales + 
                trend_component + 
                seasonal_component + 
                dayofweek_effect + 
                holiday_effect + 
                noise)
        
        # اطمینان از مثبت بودن فروش
        sales = np.maximum(sales, 0)
        
        # ========================================
        # 7. ایجاد دیتافریم
        # ========================================
        df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'trend': trend_component,
            'seasonal': seasonal_component,
            'dayofweek_effect': dayofweek_effect,
            'holiday_effect': holiday_effect,
            'noise': noise
        })
        
        # اضافه کردن ویژگی‌های زمانی
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        self.logger.info(f"✅ داده فروش با {len(df)} رکورد تولید شد")
        self.logger.info(f"   - محدوده فروش: {df['sales'].min():.0f} - {df['sales'].max():.0f}")
        self.logger.info(f"   - میانگین فروش: {df['sales'].mean():.0f}")
        
        # ذخیره فایل
        if save:
            output_path = Path('data/raw/synthetic_sales.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"💾 داده فروش در {output_path} ذخیره شد")
        
        return df
    
    # ============================================
    # تولید داده مصرف انرژی
    # ============================================
    
    def generate_energy_data(self, save: bool = True) -> pd.DataFrame:
        """
        ⚡ تولید داده مصرف انرژی با الگوهای ساعتی
        
        پارامترها:
            save: ذخیره فایل CSV
        
        بازگشت:
            df: دیتافریم مصرف انرژی
        """
        self.logger.info("🚀 شروع تولید داده مصرف انرژی...")
        
        config = self.data_config.get('energy', {})
        
        n_points = config.get('n_points', 2000)
        start_date = config.get('start_date', '2019-01-01')
        freq = config.get('freq', 'H')
        trend = config.get('trend', 0.05)
        seasonality_config = config.get('seasonality', {})
        noise_level = config.get('noise', 10)
        random_state = config.get('random_state', 42)
        
        np.random.seed(random_state)
        
        # ایجاد بازه زمانی ساعتی
        dates = create_date_range(start_date, n_points, freq)
        
        # ========================================
        # 1. روند خطی (Trend)
        # ========================================
        t = np.arange(n_points)
        trend_component = trend * t
        
        # ========================================
        # 2. فصلی بودن (Seasonality)
        # ========================================
        seasonal_component = np.zeros(n_points)
        amplitude = seasonality_config.get('amplitude', 50)
        
        # فصلی سالانه (8760 ساعت در سال)
        if seasonality_config.get('yearly', True):
            seasonal_component += amplitude * np.sin(2 * np.pi * t / 8760)
        
        # فصلی هفتگی (168 ساعت در هفته)
        if seasonality_config.get('weekly', True):
            seasonal_component += amplitude * 0.7 * np.sin(2 * np.pi * t / 168)
        
        # فصلی روزانه (24 ساعت)
        if seasonality_config.get('daily', True):
            seasonal_component += amplitude * 0.5 * np.sin(2 * np.pi * t / 24)
        
        # ========================================
        # 3. اثر ساعت روز
        # ========================================
        hourofday_effect = np.zeros(n_points)
        for i, date in enumerate(dates):
            hour = date.hour
            # مصرف بیشتر در ساعات اوج (شب)
            if 18 <= hour <= 22:
                hourofday_effect[i] = 40
            elif 8 <= hour <= 12:  # صبح
                hourofday_effect[i] = 20
            elif 0 <= hour <= 5:  # نیمه شب
                hourofday_effect[i] = -20
        
        # ========================================
        # 4. اثر روز هفته
        # ========================================
        dayofweek_effect = np.zeros(n_points)
        for i, date in enumerate(dates):
            # مصرف کمتر در آخر هفته
            if date.dayofweek >= 5:
                dayofweek_effect[i] = -15
        
        # ========================================
        # 5. اثر تعطیلات
        # ========================================
        holiday_effect = add_holiday_effect(dates) * 30
        
        # ========================================
        # 6. نویز
        # ========================================
        noise = np.random.normal(0, noise_level, n_points)
        
        # ========================================
        # 7. ترکیب همه اجزا
        # ========================================
        base_consumption = 200  # پایه مصرف
        consumption = (base_consumption + 
                      trend_component + 
                      seasonal_component + 
                      hourofday_effect + 
                      dayofweek_effect + 
                      holiday_effect + 
                      noise)
        
        # اطمینان از مثبت بودن مصرف
        consumption = np.maximum(consumption, 0)
        
        # ========================================
        # 8. ایجاد دیتافریم
        # ========================================
        df = pd.DataFrame({
            'timestamp': dates,
            'energy_consumption': consumption,
            'trend': trend_component,
            'seasonal': seasonal_component,
            'hourofday_effect': hourofday_effect,
            'dayofweek_effect': dayofweek_effect,
            'holiday_effect': holiday_effect,
            'noise': noise
        })
        
        # اضافه کردن ویژگی‌های زمانی
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['quarter'] = df['timestamp'].dt.quarter
        df['weekofyear'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        self.logger.info(f"✅ داده مصرف انرژی با {len(df)} رکورد تولید شد")
        self.logger.info(f"   - محدوده مصرف: {df['energy_consumption'].min():.0f} - {df['energy_consumption'].max():.0f}")
        self.logger.info(f"   - میانگین مصرف: {df['energy_consumption'].mean():.0f}")
        
        # ذخیره فایل
        if save:
            output_path = Path('data/raw/synthetic_energy.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"💾 داده مصرف انرژی در {output_path} ذخیره شد")
        
        return df
    
    # ============================================
    # اضافه کردن ناهنجاری
    # ============================================
    
    def add_anomalies(self, df: pd.DataFrame, value_col: str, 
                     n_anomalies: int = 10, factor: float = 3) -> pd.DataFrame:
        """
        ⚠️ اضافه کردن ناهنجاری به داده‌ها
        
        پارامترها:
            df: دیتافریم
            value_col: ستون مقدار
            n_anomalies: تعداد ناهنجاری
            factor: ضریب ناهنجاری
        
        بازگشت:
            df: دیتافریم با ناهنجاری
        """
        df = df.copy()
        
        # انتخاب نقاط تصادفی
        indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        # ایجاد ناهنجاری (افزایش یا کاهش ناگهانی)
        for idx in indices:
            if np.random.random() > 0.5:
                df.loc[idx, value_col] *= factor  # افزایش
            else:
                df.loc[idx, value_col] /= factor  # کاهش
        
        self.logger.info(f"⚠️ {n_anomalies} ناهنجاری به داده اضافه شد")
        
        return df

# ============================================
# تابع کمکی برای بارگذاری سریع داده
# ============================================

def load_or_generate_time_series(data_type: str = 'sales', 
                                 force_generate: bool = False) -> pd.DataFrame:
    """
    📂 بارگذاری داده موجود یا تولید داده جدید
    
    پارامترها:
        data_type: نوع داده ('sales' یا 'energy')
        force_generate: اجبار به تولید داده جدید
    
    بازگشت:
        df: دیتافریم سری زمانی
    """
    if data_type == 'sales':
        data_path = Path('data/raw/synthetic_sales.csv')
    else:
        data_path = Path('data/raw/synthetic_energy.csv')
    
    generator = TimeSeriesGenerator()
    
    if data_path.exists() and not force_generate:
        df = pd.read_csv(data_path)
        df['date' if data_type == 'sales' else 'timestamp'] = pd.to_datetime(
            df['date' if data_type == 'sales' else 'timestamp']
        )
        print(f"✅ داده {data_type} از فایل موجود بارگذاری شد: {len(df)} رکورد")
        return df
    else:
        if data_type == 'sales':
            df = generator.generate_sales_data(save=True)
        else:
            df = generator.generate_energy_data(save=True)
        return df