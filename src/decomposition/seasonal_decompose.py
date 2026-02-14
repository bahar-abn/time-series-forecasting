"""
📊 ماژول تجزیه فصلی سری‌های زمانی با روش کلاسیک
این ماژول شامل توابع تجزیه سری زمانی به اجزای روند، فصلی و باقیمانده است
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, load_config

class SeasonalDecomposer:
    """
    🎯 کلاس تجزیه فصلی سری‌های زمانی با روش کلاسیک
    
    این کلاس شامل:
    - تجزیه افزایشی (Additive Decomposition)
    - تجزیه ضربی (Multiplicative Decomposition)
    - مصورسازی اجزا
    - تحلیل قدرت فصلی
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.decomp_config = self.config.get('decomposition', {}).get('seasonal_decompose', {})
        
        self.logger = setup_logger(
            'seasonal_decomposer',
            log_file='outputs/logs/seasonal_decomposer.log'
        )
        
        self.decomposition = None
        self.period = None
        self.model_type = None
        
        self.logger.info("✅ SeasonalDecomposer initialized")
    
    # ============================================
    # تجزیه سری زمانی
    # ============================================
    
    def decompose(self, series: pd.Series, 
                 period: Optional[int] = None,
                 model: str = 'additive') -> Any:
        """
        📊 تجزیه سری زمانی به اجزا
        
        پارامترها:
            series: سری زمانی
            period: دوره فصلی (مثلاً 7 برای هفتگی، 12 برای ماهانه)
            model: نوع مدل ('additive' یا 'multiplicative')
        
        بازگشت:
            decomposition: نتیجه تجزیه
        """
        self.logger.info(f"🚀 شروع تجزیه سری زمانی با مدل {model}...")
        
        if period is None:
            period = self.decomp_config.get('period', 7)
        
        self.period = period
        self.model_type = model
        
        try:
            self.decomposition = seasonal_decompose(
                series, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            self.logger.info(f"✅ تجزیه سری زمانی با موفقیت انجام شد")
            self.logger.info(f"   - دوره فصلی: {period}")
            self.logger.info(f"   - مدل: {model}")
            
        except Exception as e:
            self.logger.error(f"❌ خطا در تجزیه سری زمانی: {e}")
            raise
        
        return self.decomposition
    
    # ============================================
    # استخراج اجزا
    # ============================================
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        📋 دریافت اجزای سری زمانی
        
        بازگشت:
            components: دیکشنری اجزا
        """
        if self.decomposition is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        components = {
            'observed': self.decomposition.observed,
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'resid': self.decomposition.resid
        }
        
        return components
    
    def get_component_strength(self) -> Dict[str, float]:
        """
        📈 محاسبه قدرت هر جزء
        
        بازگشت:
            strengths: قدرت اجزا
        """
        if self.decomposition is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        # حذف مقادیر NaN
        seasonal = self.decomposition.seasonal
        resid = self.decomposition.resid
        trend = self.decomposition.trend
        
        # محاسبه واریانس
        var_seasonal = np.nanvar(seasonal)
        var_resid = np.nanvar(resid)
        var_trend = np.nanvar(trend)
        var_total = np.nanvar(self.decomposition.observed)
        
        # قدرت فصلی
        seasonal_strength = max(0, 1 - var_resid / (var_seasonal + var_resid))
        
        # قدرت روند
        trend_strength = max(0, 1 - var_resid / (var_trend + var_resid))
        
        strengths = {
            'seasonal_strength': float(seasonal_strength),
            'trend_strength': float(trend_strength),
            'residual_ratio': float(var_resid / var_total)
        }
        
        self.logger.info(f"📊 قدرت فصلی: {seasonal_strength:.3f}")
        self.logger.info(f"📊 قدرت روند: {trend_strength:.3f}")
        
        return strengths
    
    # ============================================
    # تشخیص نوع فصلی بودن
    # ============================================
    
    def detect_seasonality(self, series: pd.Series, 
                          periods: List[int] = [7, 12, 24, 30, 365]) -> Dict[str, Any]:
        """
        🔍 تشخیص بهترین دوره فصلی
        
        پارامترها:
            series: سری زمانی
            periods: لیست دوره‌های احتمالی
        
        بازگشت:
            best_period: بهترین دوره فصلی
        """
        self.logger.info("🔍 تشخیص بهترین دوره فصلی...")
        
        results = []
        
        for period in periods:
            if period >= len(series) / 3:  # دوره نباید خیلی بزرگ باشد
                continue
            
            try:
                decomp = seasonal_decompose(series, model='additive', period=period)
                seasonal_strength = 1 - np.nanvar(decomp.resid) / np.nanvar(decomp.seasonal + decomp.resid)
                
                results.append({
                    'period': period,
                    'strength': float(seasonal_strength)
                })
                
            except Exception as e:
                self.logger.warning(f"⚠️ دوره {period} قابل محاسبه نیست: {e}")
        
        if not results:
            return {'best_period': None, 'strength': 0}
        
        # انتخاب بهترین دوره
        results.sort(key=lambda x: x['strength'], reverse=True)
        best = results[0]
        
        self.logger.info(f"✅ بهترین دوره فصلی: {best['period']} (قدرت: {best['strength']:.3f})")
        
        return {
            'best_period': best['period'],
            'strength': best['strength'],
            'all_results': results
        }
    
    # ============================================
    # تشخیص مدل مناسب (افزایشی یا ضربی)
    # ============================================
    
    def detect_model_type(self, series: pd.Series, period: int) -> str:
        """
        🔍 تشخیص نوع مدل مناسب (additive یا multiplicative)
        
        پارامترها:
            series: سری زمانی
            period: دوره فصلی
        
        بازگشت:
            model_type: نوع مدل
        """
        self.logger.info("🔍 تشخیص نوع مدل مناسب...")
        
        # آزمون هر دو مدل
        try:
            decomp_add = seasonal_decompose(series, model='additive', period=period)
            decomp_mul = seasonal_decompose(series, model='multiplicative', period=period)
            
            # مقایسه واریانس باقیمانده
            var_resid_add = np.nanvar(decomp_add.resid)
            var_resid_mul = np.nanvar(decomp_mul.resid)
            
            if var_resid_add < var_resid_mul:
                model_type = 'additive'
                self.logger.info(f"✅ مدل افزایشی مناسب‌تر است")
            else:
                model_type = 'multiplicative'
                self.logger.info(f"✅ مدل ضربی مناسب‌تر است")
            
            return model_type
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطا در تشخیص مدل: {e}")
            return 'additive'
    
    # ============================================
    # مصورسازی اجزا
    # ============================================
    
    def plot_decomposition(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        📈 رسم اجزای سری زمانی
        
        پارامترها:
            figsize: اندازه شکل
        
        بازگشت:
            fig: شکل matplotlib
        """
        if self.decomposition is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Observed
        axes[0].plot(self.decomposition.observed, color='black', linewidth=1)
        axes[0].set_ylabel('Observed')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(self.decomposition.trend, color='blue', linewidth=2)
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(self.decomposition.seasonal, color='green', linewidth=1)
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(self.decomposition.resid, color='red', linewidth=0.5, marker='o', markersize=2)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(f'Time Series Decomposition (Period={self.period}, Model={self.model_type})')
        plt.tight_layout()
        
        return fig
    
    def plot_seasonal_pattern(self, figsize: Tuple[int, int] = (10, 4)) -> plt.Figure:
        """
        📊 رسم الگوی فصلی
        
        پارامترها:
            figsize: اندازه شکل
        
        بازگشت:
            fig: شکل matplotlib
        """
        if self.decomposition is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        # استخراج الگوی فصلی
        seasonal = self.decomposition.seasonal
        seasonal_pattern = seasonal[:self.period]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(self.period)
        ax.plot(x, seasonal_pattern, 'b-o', linewidth=2, markersize=6)
        ax.fill_between(x, seasonal_pattern, alpha=0.2)
        
        ax.set_xlabel(f'Period (1-{self.period})')
        ax.set_ylabel('Seasonal Effect')
        ax.set_title(f'Seasonal Pattern (Period={self.period})')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        return fig
    
    # ============================================
    # ذخیره و بارگذاری
    # ============================================
    
    def save_decomposition(self, filepath: str) -> None:
        """
        💾 ذخیره نتایج تجزیه
        
        پارامترها:
            filepath: مسیر فایل
        """
        if self.decomposition is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.decomposition, f)
        
        self.logger.info(f"💾 نتایج تجزیه در {filepath} ذخیره شد")
    
    def load_decomposition(self, filepath: str) -> Any:
        """
        📂 بارگذاری نتایج تجزیه
        
        پارامترها:
            filepath: مسیر فایل
        
        بازگشت:
            decomposition: نتایج تجزیه
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.decomposition = pickle.load(f)
        
        self.logger.info(f"📂 نتایج تجزیه از {filepath} بارگذاری شد")
        
        return self.decomposition