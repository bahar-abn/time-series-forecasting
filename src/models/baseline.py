"""
📈 ماژول مدل‌های پایه برای پیش‌بینی سری‌های زمانی
این ماژول شامل مدل‌های ساده برای مقایسه و ارزیابی است
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, save_model, calculate_rmse, calculate_mae, calculate_mape

class BaselineModels:
    """
    🎯 کلاس مدل‌های پایه برای پیش‌بینی سری‌های زمانی
    
    این مدل‌ها برای مقایسه با مدل‌های پیچیده‌تر استفاده می‌شوند:
    - Naive Forecast: آخرین مقدار مشاهده شده
    - Seasonal Naive: مقدار متناظر از فصل قبل
    - Moving Average: میانگین متحرک
    - Simple Exponential Smoothing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        from src.utils import load_config
        self.config = load_config(config_path) if config_path else load_config()
        self.baseline_config = self.config.get('models', {}).get('baseline', {})
        
        self.logger = setup_logger(
            'baseline_models',
            log_file='outputs/logs/baseline_models.log'
        )
        
        self.logger.info("✅ BaselineModels initialized")
    
    # ============================================
    # پیش‌بینی Naive
    # ============================================
    
    def naive_forecast(self, series: pd.Series, steps: int = 30) -> np.ndarray:
        """
        📊 پیش‌بینی ساده با آخرین مقدار
        
        پارامترها:
            series: سری زمانی
            steps: تعداد گام‌های پیش‌بینی
        
        بازگشت:
            forecast: مقادیر پیش‌بینی شده
        """
        self.logger.info(f"🚀 اجرای پیش‌بینی Naive برای {steps} گام...")
        
        last_value = series.iloc[-1]
        forecast = np.full(steps, last_value)
        
        self.logger.info(f"   ✅ آخرین مقدار: {last_value:.2f}")
        
        return forecast
    
    # ============================================
    # پیش‌بینی Seasonal Naive
    # ============================================
    
    def seasonal_naive_forecast(self, series: pd.Series, 
                               seasonal_period: int = 7,
                               steps: int = 30) -> np.ndarray:
        """
        📊 پیش‌بینی فصلی ساده
        
        پارامترها:
            series: سری زمانی
            seasonal_period: دوره فصلی
            steps: تعداد گام‌های پیش‌بینی
        
        بازگشت:
            forecast: مقادیر پیش‌بینی شده
        """
        self.logger.info(f"🚀 اجرای پیش‌بینی Seasonal Naive با دوره {seasonal_period}...")
        
        forecast = []
        
        for i in range(steps):
            # مقدار متناظر از دوره قبل
            idx = -seasonal_period + i % seasonal_period
            if abs(idx) <= len(series):
                val = series.iloc[idx]
            else:
                val = series.iloc[-1]
            
            forecast.append(val)
        
        forecast = np.array(forecast)
        
        self.logger.info(f"   ✅ پیش‌بینی فصلی با {steps} گام انجام شد")
        
        return forecast
    
    # ============================================
    # پیش‌بینی میانگین متحرک
    # ============================================
    
    def moving_average_forecast(self, series: pd.Series,
                               window: int = 7,
                               steps: int = 30) -> np.ndarray:
        """
        📊 پیش‌بینی با میانگین متحرک
        
        پارامترها:
            series: سری زمانی
            window: اندازه پنجره
            steps: تعداد گام‌های پیش‌بینی
        
        بازگشت:
            forecast: مقادیر پیش‌بینی شده
        """
        self.logger.info(f"🚀 اجرای پیش‌بینی Moving Average با پنجره {window}...")
        
        # میانگین متحرک آخرین پنجره
        ma = series.iloc[-window:].mean()
        forecast = np.full(steps, ma)
        
        self.logger.info(f"   ✅ میانگین آخرین {window} روز: {ma:.2f}")
        
        return forecast
    
    # ============================================
    # پیش‌بینی با میانگین کلی
    # ============================================
    
    def mean_forecast(self, series: pd.Series, steps: int = 30) -> np.ndarray:
        """
        📊 پیش‌بینی با میانگین کل داده‌ها
        
        پارامترها:
            series: سری زمانی
            steps: تعداد گام‌های پیش‌بینی
        
        بازگشت:
            forecast: مقادیر پیش‌بینی شده
        """
        self.logger.info(f"🚀 اجرای پیش‌بینی Mean Forecast...")
        
        mean_value = series.mean()
        forecast = np.full(steps, mean_value)
        
        self.logger.info(f"   ✅ میانگین کل داده‌ها: {mean_value:.2f}")
        
        return forecast
    
    # ============================================
    # پیش‌بینی با میانگین متحرک وزنی
    # ============================================
    
    def weighted_moving_average(self, series: pd.Series,
                               window: int = 7,
                               steps: int = 30) -> np.ndarray:
        """
        📊 پیش‌بینی با میانگین متحرک وزنی (وزن‌های بیشتر برای داده‌های جدیدتر)
        
        پارامترها:
            series: سری زمانی
            window: اندازه پنجره
            steps: تعداد گام‌های پیش‌بینی
        
        بازگشت:
            forecast: مقادیر پیش‌بینی شده
        """
        self.logger.info(f"🚀 اجرای پیش‌بینی Weighted Moving Average...")
        
        # ایجاد وزن‌های نمایی
        weights = np.exp(np.linspace(0, 1, window))
        weights = weights / weights.sum()
        
        # محاسبه میانگین وزنی آخرین پنجره
        last_values = series.iloc[-window:].values
        wma = np.sum(last_values * weights)
        
        forecast = np.full(steps, wma)
        
        self.logger.info(f"   ✅ میانگین وزنی: {wma:.2f}")
        
        return forecast
    
    # ============================================
    # پیش‌بینی ترکیبی از همه مدل‌های پایه
    # ============================================
    
    def ensemble_baseline(self, series: pd.Series,
                         seasonal_period: int = 7,
                         steps: int = 30) -> Dict[str, np.ndarray]:
        """
        🔀 اجرای همه مدل‌های پایه
        
        پارامترها:
            series: سری زمانی
            seasonal_period: دوره فصلی
            steps: تعداد گام‌های پیش‌بینی
        
        بازگشت:
            all_forecasts: دیکشنری همه پیش‌بینی‌ها
        """
        self.logger.info(f"🚀 اجرای همه مدل‌های پایه برای {steps} گام...")
        
        forecasts = {
            'naive': self.naive_forecast(series, steps),
            'mean': self.mean_forecast(series, steps),
            'moving_average': self.moving_average_forecast(series, 
                                                          window=seasonal_period, 
                                                          steps=steps),
            'weighted_ma': self.weighted_moving_average(series, 
                                                        window=seasonal_period, 
                                                        steps=steps),
            'seasonal_naive': self.seasonal_naive_forecast(series, 
                                                          seasonal_period=seasonal_period,
                                                          steps=steps)
        }
        
        self.logger.info(f"✅ {len(forecasts)} مدل پایه اجرا شد")
        
        return forecasts
    
    # ============================================
    # ارزیابی مدل‌های پایه
    # ============================================
    
    def evaluate_baselines(self, y_true: np.ndarray, 
                          forecasts: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        📊 ارزیابی همه مدل‌های پایه
        
        پارامترها:
            y_true: مقادیر واقعی
            forecasts: دیکشنری پیش‌بینی‌ها
        
        بازگشت:
            results: دیتافریم نتایج
        """
        self.logger.info("📊 ارزیابی مدل‌های پایه...")
        
        results = []
        
        for name, forecast in forecasts.items():
            # اطمینان از هم اندازه بودن
            min_len = min(len(y_true), len(forecast))
            y_true_trimmed = y_true[:min_len]
            forecast_trimmed = forecast[:min_len]
            
            metrics = {
                'model': name,
                'rmse': calculate_rmse(y_true_trimmed, forecast_trimmed),
                'mae': calculate_mae(y_true_trimmed, forecast_trimmed),
                'mape': calculate_mape(y_true_trimmed, forecast_trimmed)
            }
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('rmse')
        
        self.logger.info(f"✅ بهترین مدل پایه: {results_df.iloc[0]['model']} "
                        f"(RMSE: {results_df.iloc[0]['rmse']:.2f})")
        
        return results_df