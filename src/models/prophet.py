"""
📈 ماژول مدل Prophet برای پیش‌بینی سری‌های زمانی
این ماژول شامل پیاده‌سازی مدل Prophet از فیسبوک است
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, save_model, calculate_rmse, calculate_mae, calculate_mape

class ProphetModel:
    """
    🎯 کلاس مدل Prophet برای پیش‌بینی سری‌های زمانی
    
    Prophet مدلی از فیسبوک است که برای پیش‌بینی سری‌های زمانی با
    الگوهای فصلی متعدد و اثر تعطیلات طراحی شده است.
    
    ویژگی‌ها:
    - روند خطی یا لجستیک
    - فصلی بودن سالانه، هفتگی، روزانه
    - اثر تعطیلات
    - تغییرپذیری در روند (changepoints)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        from src.utils import load_config
        self.config = load_config(config_path) if config_path else load_config()
        self.prophet_config = self.config.get('models', {}).get('prophet', {})
        
        self.logger = setup_logger(
            'prophet_model',
            log_file='outputs/logs/prophet_model.log'
        )
        
        self.model = None
        self.forecast = None
        
        self.logger.info("✅ ProphetModel initialized")
    
    # ============================================
    # آماده‌سازی داده
    # ============================================
    
    def prepare_data(self, df: pd.DataFrame, 
                    date_col: str, value_col: str) -> pd.DataFrame:
        """
        📋 آماده‌سازی داده برای Prophet
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
        
        بازگشت:
            prophet_df: دیتافریم با ستون‌های ds و y
        """
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[value_col]
        })
        
        # حذف مقادیر NaN
        prophet_df = prophet_df.dropna()
        
        self.logger.info(f"✅ داده برای Prophet آماده شد: {len(prophet_df)} رکورد")
        
        return prophet_df
    
    # ============================================
    # آموزش مدل
    # ============================================
    
    def train(self, df: pd.DataFrame, 
             date_col: str, value_col: str,
             **kwargs) -> Prophet:
        """
        📚 آموزش مدل Prophet
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
            **kwargs: پارامترهای اضافی
        
        بازگشت:
            model: مدل آموزش دیده
        """
        self.logger.info("🚀 شروع آموزش مدل Prophet...")
        
        # آماده‌سازی داده
        prophet_df = self.prepare_data(df, date_col, value_col)
        
        # تنظیم پارامترها
        params = {
            'yearly_seasonality': self.prophet_config.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': self.prophet_config.get('weekly_seasonality', 'auto'),
            'daily_seasonality': self.prophet_config.get('daily_seasonality', 'auto'),
            'seasonality_mode': self.prophet_config.get('seasonality_mode', 'additive'),
            'changepoint_prior_scale': self.prophet_config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': self.prophet_config.get('seasonality_prior_scale', 10),
            'holidays_prior_scale': self.prophet_config.get('holidays_prior_scale', 10),
            'uncertainty_samples': self.prophet_config.get('uncertainty_samples', 1000),
            **kwargs
        }
        
        # ایجاد و آموزش مدل
        self.model = Prophet(**params)
        self.model.fit(prophet_df)
        
        self.logger.info(f"✅ مدل Prophet با موفقیت آموزش دید")
        
        return self.model
    
    # ============================================
    # پیش‌بینی
    # ============================================
    
    def predict(self, periods: int = 30, 
               freq: str = 'D',
               include_history: bool = True) -> pd.DataFrame:
        """
        🔮 پیش‌بینی مقادیر آینده
        
        پارامترها:
            periods: تعداد دوره‌های پیش‌بینی
            freq: فرکانس (D=day, H=hour, W=week)
            include_history: شامل کردن داده‌های تاریخی
        
        بازگشت:
            forecast: دیتافریم پیش‌بینی
        """
        if self.model is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        # ایجاد future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        
        # پیش‌بینی
        self.forecast = self.model.predict(future)
        
        self.logger.info(f"✅ پیش‌بینی برای {periods} دوره آینده انجام شد")
        
        return self.forecast
    
    # ============================================
    # ارزیابی
    # ============================================
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        📊 ارزیابی عملکرد مدل
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            metrics: متریک‌های ارزیابی
        """
        metrics = {
            'rmse': calculate_rmse(y_true, y_pred),
            'mae': calculate_mae(y_true, y_pred),
            'mape': calculate_mape(y_true, y_pred),
            'mse': np.mean((y_true - y_pred) ** 2)
        }
        
        self.logger.info(f"📊 نتایج ارزیابی Prophet:")
        self.logger.info(f"   - RMSE: {metrics['rmse']:.2f}")
        self.logger.info(f"   - MAE: {metrics['mae']:.2f}")
        self.logger.info(f"   - MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    # ============================================
    # استخراج اجزا
    # ============================================
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        📊 دریافت اجزای مدل (روند، فصلی)
        
        بازگشت:
            components: اجزای مدل
        """
        if self.forecast is None:
            raise ValueError("❌ ابتدا predict را اجرا کنید")
        
        components = {
            'trend': self.forecast['trend'].values,
            'yearly': self.forecast['yearly'].values if 'yearly' in self.forecast.columns else None,
            'weekly': self.forecast['weekly'].values if 'weekly' in self.forecast.columns else None,
            'daily': self.forecast['daily'].values if 'daily' in self.forecast.columns else None,
            'holidays': None
        }
        
        # جمع کردن اثر تعطیلات
        holiday_cols = [col for col in self.forecast.columns if col.startswith('holidays')]
        if holiday_cols:
            components['holidays'] = self.forecast[holiday_cols].sum(axis=1).values
        
        return components
    
    # ============================================
    # تشخیص changepoints
    # ============================================
    
    def get_changepoints(self) -> pd.DataFrame:
        """
        📍 دریافت نقاط تغییر روند
        
        بازگشت:
            changepoints: نقاط تغییر
        """
        if self.model is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        # نقاط تغییر پیش‌فرض
        changepoints = self.model.changepoints
        
        # میزان تغییر
        deltas = self.model.params['delta'].mean(axis=0)
        
        changepoints_df = pd.DataFrame({
            'changepoint': changepoints,
            'delta': deltas
        })
        
        return changepoints_df
    
    # ============================================
    # اعتبارسنجی متقابل
    # ============================================
    
    def cross_validation(self, initial: str = '730 days', 
                        period: str = '180 days',
                        horizon: str = '365 days') -> pd.DataFrame:
        """
        🔄 اعتبارسنجی متقابل برای مدل Prophet
        
        پارامترها:
            initial: دوره آموزش اولیه
            period: فاصله بین پیش‌بینی‌ها
            horizon: افق پیش‌بینی
        
        بازگشت:
            cv_results: نتایج اعتبارسنجی
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        if self.model is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        self.logger.info("🚀 شروع اعتبارسنجی متقابل...")
        
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        df_perf = performance_metrics(df_cv)
        
        self.logger.info(f"✅ اعتبارسنجی متقابل کامل شد")
        
        return df_perf
    
    # ============================================
    # ذخیره و بارگذاری
    # ============================================
    
    def save(self, metadata: Optional[Dict] = None) -> str:
        """
        💾 ذخیره مدل
        
        پارامترها:
            metadata: ابرداده مدل
        
        بازگشت:
            save_path: مسیر فایل ذخیره شده
        """
        if self.model is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_type': 'Prophet',
            'params': str(self.model.__dict__)
        })
        
        save_path = save_model(self.model, 'prophet', metadata)
        return save_path