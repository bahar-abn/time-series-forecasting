"""
🛠 ماژول پیش‌پردازش داده برای سری‌های زمانی
این ماژول شامل توابع پیش‌پردازش و مهندسی ویژگی‌های زمانی است
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging

from src.utils import setup_logger, load_config

class TimeSeriesPreprocessor:
    """
    🎯 کلاس پیش‌پردازش داده‌های سری زمانی
    
    این کلاس شامل:
    - مدیریت مقادیر گمشده
    - تشخیص و حذف داده‌های پرت
    - مهندسی ویژگی‌های زمانی
    - ایجاد ویژگی‌های تاخیری (Lags)
    - ایجاد ویژگی‌های میانگین متحرک (Rolling)
    - ایجاد ویژگی‌های فوریه (Fourier features)
    - نرمال‌سازی (در صورت نیاز)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.preprocess_config = self.config.get('preprocessing', {})
        self.feat_eng_config = self.config.get('feature_engineering', {})
        
        self.logger = setup_logger(
            'time_series_preprocessor',
            log_file='outputs/logs/time_series_preprocessor.log'
        )
        
        self.date_col = None
        self.value_col = None
        self.scaler = None
        
        self.logger.info("✅ TimeSeriesPreprocessor initialized")
    
    # ============================================
    # مدیریت مقادیر گمشده
    # ============================================
    
    def handle_missing_values(self, df: pd.DataFrame, 
                              date_col: str, value_col: str) -> pd.DataFrame:
        """
        🧹 مدیریت مقادیر گمشده در سری زمانی
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
        
        بازگشت:
            df_clean: دیتافریم بدون مقادیر گمشده
        """
        self.logger.info("🔍 بررسی مقادیر گمشده...")
        
        df_clean = df.copy()
        
        missing_count = df_clean[value_col].isnull().sum()
        missing_percent = missing_count / len(df_clean) * 100
        
        self.logger.info(f"   - تعداد مقادیر گمشده: {missing_count} ({missing_percent:.2f}%)")
        
        if missing_count > 0:
            method = self.preprocess_config.get('missing_values', {}).get('method', 'interpolate')
            
            if method == 'interpolate':
                # درون‌یابی (مناسب برای سری زمانی)
                order = self.preprocess_config.get('missing_values', {}).get('order', 3)
                df_clean[value_col] = df_clean[value_col].interpolate(
                    method='polynomial', order=order
                )
                self.logger.info(f"   ✅ درون‌یابی با مرتبه {order} انجام شد")
            
            elif method == 'ffill':
                # پر کردن با مقدار قبلی
                df_clean[value_col] = df_clean[value_col].fillna(method='ffill')
                self.logger.info("   ✅ پر کردن با مقدار قبلی انجام شد")
            
            elif method == 'bfill':
                # پر کردن با مقدار بعدی
                df_clean[value_col] = df_clean[value_col].fillna(method='bfill')
                self.logger.info("   ✅ پر کردن با مقدار بعدی انجام شد")
            
            elif method == 'drop':
                # حذف رکوردهای دارای مقدار گمشده
                df_clean = df_clean.dropna(subset=[value_col])
                self.logger.info(f"   ✅ {missing_count} رکورد دارای مقدار گمشده حذف شد")
        
        return df_clean
    
    # ============================================
    # تشخیص و مدیریت داده‌های پرت
    # ============================================
    
    def detect_outliers(self, series: pd.Series) -> np.ndarray:
        """
        🔍 تشخیص داده‌های پرت با روش IQR
        
        پارامترها:
            series: سری زمانی
        
        بازگشت:
            outlier_mask: ماسک داده‌های پرت
        """
        outlier_config = self.preprocess_config.get('outlier_detection', {})
        method = outlier_config.get('method', 'iqr')
        multiplier = outlier_config.get('iqr_multiplier', 3)
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            outlier_mask = z_scores > multiplier
        
        else:
            outlier_mask = pd.Series(False, index=series.index)
        
        return outlier_mask
    
    def handle_outliers(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        📉 مدیریت داده‌های پرت
        
        پارامترها:
            df: دیتافریم
            value_col: ستون مقدار
        
        بازگشت:
            df_clean: دیتافریم بدون داده‌های پرت
        """
        outlier_config = self.preprocess_config.get('outlier_detection', {})
        action = outlier_config.get('action', 'winsorize')
        
        df_clean = df.copy()
        
        outlier_mask = self.detect_outliers(df_clean[value_col])
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            self.logger.info(f"⚠️ {n_outliers} داده پرت شناسایی شد")
            
            if action == 'remove':
                # حذف داده‌های پرت
                df_clean = df_clean[~outlier_mask]
                self.logger.info(f"   ✅ {n_outliers} داده پرت حذف شد")
            
            elif action == 'cap':
                # محدود کردن به کران‌ها
                Q1 = df_clean[value_col].quantile(0.25)
                Q3 = df_clean[value_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                df_clean.loc[outlier_mask, value_col] = np.where(
                    df_clean.loc[outlier_mask, value_col] < lower_bound,
                    lower_bound,
                    upper_bound
                )
                self.logger.info(f"   ✅ داده‌های پرت محدود شدند")
            
            elif action == 'winsorize':
                # winsorizing - جایگزینی با نزدیک‌ترین مقدار غیر پرت
                from scipy.stats.mstats import winsorize
                df_clean[value_col] = winsorize(
                    df_clean[value_col], 
                    limits=[0.01, 0.01]  # 1% در هر طرف
                )
                self.logger.info(f"   ✅ winsorize انجام شد")
        
        return df_clean
    
    # ============================================
    # مهندسی ویژگی‌های زمانی
    # ============================================
    
    def add_datetime_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        📅 اضافه کردن ویژگی‌های مبتنی بر تاریخ
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
        
        بازگشت:
            df: دیتافریم با ویژگی‌های زمانی
        """
        df = df.copy()
        
        # اطمینان از نوع datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        datetime_config = self.feat_eng_config.get('datetime_features', [])
        
        feature_mapping = {
            'hour': df[date_col].dt.hour,
            'dayofweek': df[date_col].dt.dayofweek,
            'dayofmonth': df[date_col].dt.day,
            'month': df[date_col].dt.month,
            'quarter': df[date_col].dt.quarter,
            'year': df[date_col].dt.year,
            'weekofyear': df[date_col].dt.isocalendar().week,
            'is_weekend': (df[date_col].dt.dayofweek >= 5).astype(int)
        }
        
        for feature in datetime_config:
            if feature in feature_mapping:
                df[feature] = feature_mapping[feature]
                self.logger.info(f"   ✅ ویژگی {feature} اضافه شد")
        
        # سینوس و کسینوس برای ویژگی‌های دوره‌ای
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if 'dayofweek' in df.columns:
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df
    
    # ============================================
    # ایجاد ویژگی‌های تاخیری (Lags)
    # ============================================
    
    def add_lag_features(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        ⏪ اضافه کردن ویژگی‌های تاخیری
        
        پارامترها:
            df: دیتافریم
            value_col: ستون مقدار
        
        بازگشت:
            df: دیتافریم با ویژگی‌های تاخیری
        """
        df = df.copy()
        
        lag_config = self.feat_eng_config.get('lag_features', {})
        lags = lag_config.get('lags', [1, 2, 3, 7, 14, 30])
        
        for lag in lags:
            df[f'lag_{lag}'] = df[value_col].shift(lag)
            self.logger.info(f"   ✅ ویژگی lag_{lag} اضافه شد")
        
        return df
    
    # ============================================
    # ایجاد ویژگی‌های میانگین متحرک (Rolling)
    # ============================================
    
    def add_rolling_features(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        📊 اضافه کردن ویژگی‌های میانگین متحرک
        
        پارامترها:
            df: دیتافریم
            value_col: ستون مقدار
        
        بازگشت:
            df: دیتافریم با ویژگی‌های میانگین متحرک
        """
        df = df.copy()
        
        rolling_config = self.feat_eng_config.get('rolling_features', {})
        windows = rolling_config.get('windows', [7, 14, 30])
        functions = rolling_config.get('functions', ['mean', 'std', 'min', 'max'])
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df[f'rolling_{window}_mean'] = df[value_col].rolling(window=window).mean()
                elif func == 'std':
                    df[f'rolling_{window}_std'] = df[value_col].rolling(window=window).std()
                elif func == 'min':
                    df[f'rolling_{window}_min'] = df[value_col].rolling(window=window).min()
                elif func == 'max':
                    df[f'rolling_{window}_max'] = df[value_col].rolling(window=window).max()
                
                self.logger.info(f"   ✅ ویژگی rolling_{window}_{func} اضافه شد")
        
        return df
    
    # ============================================
    # ایجاد ویژگی‌های فوریه (Fourier)
    # ============================================
    
    def add_fourier_features(self, df: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
        """
        📈 اضافه کردن ویژگی‌های فوریه برای فصلی بودن
        
        پارامترها:
            df: دیتافریم
            t: اندیس زمانی
        
        بازگشت:
            df: دیتافریم با ویژگی‌های فوریه
        """
        df = df.copy()
        
        fourier_config = self.feat_eng_config.get('fourier_features', {})
        
        if not fourier_config.get('enabled', False):
            return df
        
        k = fourier_config.get('k', 5)
        periods = fourier_config.get('period', [7, 30, 365])
        
        for period in periods:
            for i in range(1, k + 1):
                df[f'fourier_{period}_sin_{i}'] = np.sin(2 * np.pi * i * t / period)
                df[f'fourier_{period}_cos_{i}'] = np.cos(2 * np.pi * i * t / period)
            
            self.logger.info(f"   ✅ {k} هارمونیک فوریه برای دوره {period} اضافه شد")
        
        return df
    
    # ============================================
    # نرمال‌سازی
    # ============================================
    
    def scale_features(self, df: pd.DataFrame, 
                       value_col: str, fit: bool = True) -> pd.DataFrame:
        """
        📏 نرمال‌سازی ویژگی‌ها (اختیاری)
        
        پارامترها:
            df: دیتافریم
            value_col: ستون مقدار
            fit: آیا مدل را آموزش دهیم؟
        
        بازگشت:
            df: دیتافریم نرمال‌سازی شده
        """
        scaling_config = self.preprocess_config.get('scaling', {})
        
        if not scaling_config.get('enabled', False):
            return df
        
        df = df.copy()
        method = scaling_config.get('method', 'standard')
        
        # انتخاب ویژگی‌های عددی (به جز ستون هدف و تاریخ)
        exclude_cols = [value_col, 'date', 'timestamp']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if len(feature_cols) > 0:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            
            if fit:
                df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
                self.logger.info(f"   ✅ {len(feature_cols)} ویژگی نرمال‌سازی شدند")
            else:
                if self.scaler is not None:
                    df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    # ============================================
    # پایپلاین کامل پیش‌پردازش
    # ============================================
    
    def preprocess(self, df: pd.DataFrame, date_col: str, value_col: str,
                  target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        🔄 پایپلاین کامل پیش‌پردازش
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
            target_col: ستون هدف (برای پیش‌بینی)
        
        بازگشت:
            processed_data: دیکشنری داده‌های پردازش شده
        """
        self.logger.info("🚀 شروع پایپلاین پیش‌پردازش...")
        
        self.date_col = date_col
        self.value_col = value_col
        
        df_processed = df.copy()
        
        # 1. مرتب‌سازی بر اساس تاریخ
        df_processed = df_processed.sort_values(date_col)
        self.logger.info("✅ داده‌ها بر اساس تاریخ مرتب شدند")
        
        # 2. مدیریت مقادیر گمشده
        df_processed = self.handle_missing_values(df_processed, date_col, value_col)
        
        # 3. مدیریت داده‌های پرت
        df_processed = self.handle_outliers(df_processed, value_col)
        
        # 4. اضافه کردن ویژگی‌های زمانی
        df_processed = self.add_datetime_features(df_processed, date_col)
        
        # 5. ایجاد اندیس زمانی
        t = np.arange(len(df_processed))
        
        # 6. اضافه کردن ویژگی‌های فوریه
        df_processed = self.add_fourier_features(df_processed, t)
        
        # 7. اضافه کردن ویژگی‌های تاخیری
        df_processed = self.add_lag_features(df_processed, value_col)
        
        # 8. اضافه کردن ویژگی‌های میانگین متحرک
        df_processed = self.add_rolling_features(df_processed, value_col)
        
        # 9. حذف رکوردهای دارای NaN (به دلیل lag و rolling)
        initial_len = len(df_processed)
        df_processed = df_processed.dropna()
        final_len = len(df_processed)
        self.logger.info(f"✅ {initial_len - final_len} رکورد با NaN حذف شد")
        
        # 10. نرمال‌سازی
        df_processed = self.scale_features(df_processed, value_col, fit=True)
        
        # 11. جدا کردن ویژگی‌ها و هدف
        if target_col is None:
            target_col = value_col
        
        feature_cols = [col for col in df_processed.columns 
                       if col not in [date_col, target_col]]
        
        X = df_processed[feature_cols]
        y = df_processed[target_col]
        dates = df_processed[date_col]
        
        self.logger.info(f"✅ پیش‌پردازش کامل شد:")
        self.logger.info(f"   - تعداد نمونه: {len(X)}")
        self.logger.info(f"   - تعداد ویژگی‌ها: {len(feature_cols)}")
        self.logger.info(f"   - بازه زمانی: {dates.min()} تا {dates.max()}")
        
        return {
            'X': X,
            'y': y,
            'dates': dates,
            'feature_names': feature_cols,
            'date_col': date_col,
            'value_col': value_col,
            'preprocessor': self
        }