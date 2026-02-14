"""
📊 ماژول تجزیه فصلی سری‌های زمانی با روش STL
این ماژول شامل پیاده‌سازی تجزیه STL (Seasonal-Trend decomposition using LOESS) است
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, load_config

class STLDecomposer:
    """
    🎯 کلاس تجزیه سری‌های زمانی با روش STL
    
    STL (Seasonal-Trend decomposition using LOESS) روشی مقاوم و انعطاف‌پذیر
    برای تجزیه سری‌های زمانی است که می‌تواند هر نوع فصلی را مدل کند.
    
    مزایای STL:
    - مقاوم در برابر داده‌های پرت
    - می‌تواند فصلی متغیر با زمان را مدل کند
    - کنترل دقیق بر نرم‌کنندگی اجزا
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.stl_config = self.config.get('decomposition', {}).get('stl', {})
        
        self.logger = setup_logger(
            'stl_decomposer',
            log_file='outputs/logs/stl_decomposer.log'
        )
        
        self.result = None
        self.period = None
        
        self.logger.info("✅ STLDecomposer initialized")
    
    # ============================================
    # تجزیه STL
    # ============================================
    
    def decompose(self, series: pd.Series, 
                 period: Optional[int] = None,
                 robust: bool = True,
                 seasonal: int = 13,
                 trend: Optional[int] = None) -> Any:
        """
        📊 تجزیه سری زمانی با روش STL
        
        پارامترها:
            series: سری زمانی
            period: دوره فصلی
            robust: استفاده از روش مقاوم
            seasonal: طول پنجره فصلی (باید فرد باشد)
            trend: طول پنجره روند
        
        بازگشت:
            result: نتیجه تجزیه
        """
        self.logger.info("🚀 شروع تجزیه سری زمانی با STL...")
        
        if period is None:
            period = self.stl_config.get('period', 7)
        
        if trend is None:
            trend = self.stl_config.get('trend', 21)
        
        if seasonal is None:
            seasonal = self.stl_config.get('seasonal', 13)
        
        self.period = period
        
        # اطمینان از فرد بودن seasonal
        if seasonal % 2 == 0:
            seasonal += 1
        
        try:
            stl = STL(
                series,
                period=period,
                seasonal=seasonal,
                trend=trend,
                robust=robust
            )
            
            self.result = stl.fit()
            
            self.logger.info(f"✅ تجزیه STL با موفقیت انجام شد")
            self.logger.info(f"   - دوره فصلی: {period}")
            self.logger.info(f"   - روش مقاوم: {robust}")
            self.logger.info(f"   - پنجره فصلی: {seasonal}")
            self.logger.info(f"   - پنجره روند: {trend}")
            
        except Exception as e:
            self.logger.error(f"❌ خطا در تجزیه STL: {e}")
            raise
        
        return self.result
    
    # ============================================
    # استخراج اجزا
    # ============================================
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        📋 دریافت اجزای سری زمانی
        
        بازگشت:
            components: دیکشنری اجزا
        """
        if self.result is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        components = {
            'observed': self.result.observed,
            'trend': self.result.trend,
            'seasonal': self.result.seasonal,
            'resid': self.result.resid
        }
        
        return components
    
    def get_component_strength(self) -> Dict[str, float]:
        """
        📈 محاسبه قدرت هر جزء
        
        بازگشت:
            strengths: قدرت اجزا
        """
        if self.result is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        seasonal = self.result.seasonal
        resid = self.result.resid
        trend = self.result.trend
        
        # محاسبه واریانس
        var_seasonal = np.nanvar(seasonal)
        var_resid = np.nanvar(resid)
        var_trend = np.nanvar(trend)
        var_total = np.nanvar(self.result.observed)
        
        # قدرت فصلی
        seasonal_strength = max(0, 1 - var_resid / (var_seasonal + var_resid))
        
        # قدرت روند
        trend_strength = max(0, 1 - var_resid / (var_trend + var_resid))
        
        strengths = {
            'seasonal_strength': float(seasonal_strength),
            'trend_strength': float(trend_strength),
            'residual_ratio': float(var_resid / var_total)
        }
        
        self.logger.info(f"📊 قدرت فصلی (STL): {seasonal_strength:.3f}")
        self.logger.info(f"📊 قدرت روند (STL): {trend_strength:.3f}")
        
        return strengths
    
    # ============================================
    # تحلیل باقیمانده
    # ============================================
    
    def analyze_residuals(self) -> Dict[str, Any]:
        """
        📉 تحلیل باقیمانده‌ها
        
        بازگشت:
            residual_analysis: تحلیل باقیمانده‌ها
        """
        if self.result is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        resid = self.result.resid.dropna()
        
        from scipy import stats
        
        analysis = {
            'mean': float(np.mean(resid)),
            'std': float(np.std(resid)),
            'skewness': float(stats.skew(resid)),
            'kurtosis': float(stats.kurtosis(resid)),
            'normality_test': stats.normaltest(resid).pvalue,
            'is_normal': stats.normaltest(resid).pvalue > 0.05
        }
        
        # آزمون Ljung-Box برای خودهمبستگی
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
        analysis['ljung_box_pvalue'] = float(lb_test['lb_pvalue'].iloc[0])
        analysis['has_autocorrelation'] = analysis['ljung_box_pvalue'] < 0.05
        
        self.logger.info(f"📊 میانگین باقیمانده: {analysis['mean']:.4f}")
        self.logger.info(f"📊 نرمال بودن: {analysis['is_normal']}")
        self.logger.info(f"📊 خودهمبستگی: {analysis['has_autocorrelation']}")
        
        return analysis
    
    # ============================================
    # مصورسازی
    # ============================================
    
    def plot_decomposition(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        📈 رسم اجزای سری زمانی
        
        پارامترها:
            figsize: اندازه شکل
        
        بازگشت:
            fig: شکل matplotlib
        """
        if self.result is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Observed
        axes[0].plot(self.result.observed, color='black', linewidth=1)
        axes[0].set_ylabel('Observed')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(self.result.trend, color='blue', linewidth=2)
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(self.result.seasonal, color='green', linewidth=1)
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(self.result.resid, color='red', linewidth=0.5, marker='o', markersize=2)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'STL Decomposition (Period={self.period})')
        plt.tight_layout()
        
        return fig
    
    def plot_seasonal_subseries(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        📊 رسم زیرسری‌های فصلی
        
        پارامترها:
            figsize: اندازه شکل
        
        بازگشت:
            fig: شکل matplotlib
        """
        if self.result is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        # ایجاد زیرسری‌های فصلی
        seasonal = self.result.seasonal
        n_periods = len(seasonal) // self.period
        
        fig, axes = plt.subplots(self.period, 1, figsize=figsize, sharex=True)
        
        for i in range(self.period):
            subseries = seasonal[i::self.period][:n_periods]
            axes[i].plot(subseries, 'o-', markersize=3, linewidth=1)
            axes[i].set_ylabel(f'Period {i+1}')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel('Seasonal Cycle')
        plt.suptitle('Seasonal Subseries Plot')
        plt.tight_layout()
        
        return fig
    
    def plot_residual_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        📈 نمودارهای تشخیصی باقیمانده
        
        پارامترها:
            figsize: اندازه شکل
        
        بازگشت:
            fig: شکل matplotlib
        """
        if self.result is None:
            raise ValueError("❌ ابتدا decompose را اجرا کنید")
        
        resid = self.result.resid.dropna()
        
        from scipy import stats
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig = plt.figure(figsize=figsize)
        
        # هیستوگرام
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.hist(resid, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title('Histogram of Residuals')
        ax1.set_xlabel('Residual')
        ax1.set_ylabel('Frequency')
        
        # Q-Q plot
        ax2 = fig.add_subplot(2, 3, 2)
        stats.probplot(resid, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        # Residuals vs Time
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(resid.index, resid.values, 'o-', markersize=2, linewidth=0.5)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_title('Residuals vs Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Residual')
        
        # ACF
        ax4 = fig.add_subplot(2, 3, 4)
        plot_acf(resid, ax=ax4, lags=40)
        ax4.set_title('Autocorrelation Function')
        
        # PACF
        ax5 = fig.add_subplot(2, 3, 5)
        plot_pacf(resid, ax=ax5, lags=40)
        ax5.set_title('Partial Autocorrelation Function')
        
        # Box plot
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.boxplot(resid)
        ax6.set_title('Box Plot of Residuals')
        ax6.set_ylabel('Residual')
        
        plt.suptitle('Residual Diagnostics', fontsize=14)
        plt.tight_layout()
        
        return fig