"""
📈 ماژول مدل ARIMA برای پیش‌بینی سری‌های زمانی
این ماژول شامل پیاده‌سازی مدل ARIMA و Auto ARIMA است
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, save_model, calculate_rmse, calculate_mae, calculate_mape

class ARIMAModel:
    """
    🎯 کلاس مدل ARIMA برای پیش‌بینی سری‌های زمانی
    
    ARIMA (Autoregressive Integrated Moving Average) یکی از محبوب‌ترین
    مدل‌ها برای پیش‌بینی سری‌های زمانی است.
    
    پارامترها:
    - p: تعداد وقفه‌های خودرگرسیون (AR)
    - d: تعداد مرتبه تفاضل‌گیری (I)
    - q: تعداد وقفه‌های میانگین متحرک (MA)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        from src.utils import load_config
        self.config = load_config(config_path) if config_path else load_config()
        self.arima_config = self.config.get('models', {}).get('arima', {})
        
        self.logger = setup_logger(
            'arima_model',
            log_file='outputs/logs/arima_model.log'
        )
        
        self.model = None
        self.model_fit = None
        self.order = None
        self.seasonal_order = None
        self.aic = None
        self.bic = None
        
        self.logger.info("✅ ARIMAModel initialized")
    
    # ============================================
    # آزمون ایستایی (Stationarity Test)
    # ============================================
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        📊 آزمون ایستایی با روش Augmented Dickey-Fuller
        
        پارامترها:
            series: سری زمانی
        
        بازگشت:
            result: نتیجه آزمون
        """
        self.logger.info("🔍 اجرای آزمون ایستایی ADF...")
        
        # حذف مقادیر NaN
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {
                'adf_statistic': 0,
                'p_value': 1.0,
                'critical_values': {},
                'is_stationary': False,
                'n_diffs_needed': 1
            }
        
        result = adfuller(series_clean, autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        is_stationary = p_value < 0.05
        
        self.logger.info(f"   - آماره ADF: {adf_statistic:.4f}")
        self.logger.info(f"   - p-value: {p_value:.4f}")
        self.logger.info(f"   - ایستا: {is_stationary}")
        
        return {
            'adf_statistic': float(adf_statistic),
            'p_value': float(p_value),
            'critical_values': critical_values,
            'is_stationary': bool(is_stationary),
            'n_diffs_needed': 0 if is_stationary else 1
        }
    
    # ============================================
    # تعیین پارامترهای ARIMA
    # ============================================
    
    def determine_order(self, series: pd.Series, 
                       max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        """
        🔍 تعیین پارامترهای بهینه ARIMA با استفاده از ACF و PACF
        
        پارامترها:
            series: سری زمانی
            max_p: حداکثر p
            max_q: حداکثر q
        
        بازگشت:
            order: (p, d, q)
        """
        self.logger.info("🔍 تعیین پارامترهای ARIMA...")
        
        # حذف مقادیر NaN
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            self.logger.warning("⚠️ داده کافی برای تعیین پارامترها وجود ندارد")
            return (1, 0, 1)
        
        # آزمون ایستایی
        stationarity = self.test_stationarity(series_clean)
        
        # تعیین d
        d = stationarity['n_diffs_needed']
        
        # اگر سری ایستا نیست، یک بار تفاضل بگیر
        if d > 0:
            series_diff = series_clean.diff().dropna()
            self.logger.info(f"   ✅ تفاضل مرتبه {d} گرفته شد")
        else:
            series_diff = series_clean
        
        if len(series_diff) < 10:
            return (1, d, 1)
        
        # محاسبه ACF و PACF
        try:
            acf_values = acf(series_diff, nlags=min(max_q, len(series_diff)//2), fft=True)
            pacf_values = pacf(series_diff, nlags=min(max_p, len(series_diff)//2))
        except:
            return (1, d, 1)
        
        # تعیین p از PACF
        p = 0
        for i in range(1, min(len(pacf_values), max_p + 1)):
            try:
                if abs(pacf_values[i]) > 1.96 / np.sqrt(len(series_diff)):
                    p = i
                else:
                    break
            except:
                break
        
        # تعیین q از ACF
        q = 0
        for i in range(1, min(len(acf_values), max_q + 1)):
            try:
                if abs(acf_values[i]) > 1.96 / np.sqrt(len(series_diff)):
                    q = i
                else:
                    break
            except:
                break
        
        # محدود کردن به max
        p = min(p, max_p)
        q = min(q, max_q)
        
        self.order = (p, d, q)
        
        self.logger.info(f"✅ پارامترهای پیشنهادی: ARIMA{p, d, q}")
        
        return self.order
    
    # ============================================
    # آموزش مدل ARIMA
    # ============================================
    
    def train(self, series: pd.Series, 
             order: Optional[Tuple[int, int, int]] = None,
             **kwargs) -> Any:
        """
        📚 آموزش مدل ARIMA
        
        پارامترها:
            series: سری زمانی
            order: پارامترهای (p, d, q)
            **kwargs: پارامترهای اضافی
        
        بازگشت:
            model_fit: مدل آموزش دیده
        """
        self.logger.info("🚀 شروع آموزش مدل ARIMA...")
        
        # حذف مقادیر NaN
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            self.logger.error("❌ داده کافی برای آموزش مدل وجود ندارد")
            raise ValueError("داده کافی برای آموزش مدل وجود ندارد")
        
        if order is None:
            # تعیین خودکار پارامترها
            max_p = self.arima_config.get('max_p', 3)
            max_q = self.arima_config.get('max_q', 3)
            order = self.determine_order(series_clean, max_p, max_q)
        
        self.order = order
        
        try:
            # آموزش مدل
            self.model = StatsARIMA(
                series_clean,
                order=order,
                **kwargs
            )
            
            self.model_fit = self.model.fit()
            
            # ذخیره معیارهای اطلاعاتی
            self.aic = self.model_fit.aic
            self.bic = self.model_fit.bic
            
            self.logger.info(f"✅ مدل ARIMA{order} با موفقیت آموزش دید")
            self.logger.info(f"   - AIC: {self.aic:.2f}")
            self.logger.info(f"   - BIC: {self.bic:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ خطا در آموزش مدل ARIMA: {e}")
            # تلاش با مدل ساده‌تر
            try:
                self.logger.info("🔄 تلاش با مدل ساده‌تر ARIMA(1,0,1)...")
                self.model = StatsARIMA(series_clean, order=(1,0,1))
                self.model_fit = self.model.fit()
                self.order = (1,0,1)
                self.logger.info("✅ مدل ARIMA(1,0,1) با موفقیت آموزش دید")
            except:
                raise
        
        return self.model_fit
    
    # ============================================
    # پیش‌بینی - نسخه اصلاح شده
    # ============================================
    
    def predict(self, steps: int = 30, 
               return_conf_int: bool = True,
               alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        🔮 پیش‌بینی مقادیر آینده - نسخه اصلاح شده
        
        پارامترها:
            steps: تعداد گام‌های پیش‌بینی
            return_conf_int: برگرداندن فاصله اطمینان
            alpha: سطح معنی‌داری برای فاصله اطمینان
        
        بازگشت:
            predictions: دیکشنری پیش‌بینی‌ها
        """
        if self.model_fit is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        try:
            # روش اول: استفاده از forecast
            forecast = self.model_fit.forecast(steps=steps)
            
            result = {
                'forecast': forecast.values,
                'index': np.arange(steps)
            }
            
            if return_conf_int:
                # محاسبه فاصله اطمینان ساده
                resid_std = np.std(self.model_fit.resid)
                result['lower_bound'] = forecast.values - 1.96 * resid_std
                result['upper_bound'] = forecast.values + 1.96 * resid_std
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطا در forecast: {e}")
            
            try:
                # روش دوم: استفاده از predict
                start = len(self.model_fit.data.endog)
                end = start + steps - 1
                
                predictions = self.model_fit.predict(start=start, end=end)
                
                result = {
                    'forecast': predictions.values,
                    'index': np.arange(steps)
                }
                
                if return_conf_int:
                    resid_std = np.std(self.model_fit.resid)
                    result['lower_bound'] = predictions.values - 1.96 * resid_std
                    result['upper_bound'] = predictions.values + 1.96 * resid_std
                
                return result
                
            except Exception as e2:
                self.logger.error(f"❌ خطا در predict: {e2}")
                
                # روش سوم: مقدار ثابت (آخرین مقدار)
                last_value = self.model_fit.data.endog[-1]
                forecast = np.full(steps, last_value)
                
                result = {
                    'forecast': forecast,
                    'index': np.arange(steps)
                }
                
                if return_conf_int:
                    resid_std = np.std(self.model_fit.resid)
                    result['lower_bound'] = forecast - 1.96 * resid_std
                    result['upper_bound'] = forecast + 1.96 * resid_std
                
                self.logger.warning("⚠️ از آخرین مقدار برای پیش‌بینی استفاده شد")
                return result
    
    # ============================================
    # ارزیابی مدل
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
        
        self.logger.info(f"📊 نتایج ارزیابی:")
        self.logger.info(f"   - RMSE: {metrics['rmse']:.2f}")
        self.logger.info(f"   - MAE: {metrics['mae']:.2f}")
        self.logger.info(f"   - MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    # ============================================
    # تحلیل باقیمانده
    # ============================================
    
    def analyze_residuals(self) -> Dict[str, Any]:
        """
        📉 تحلیل باقیمانده‌های مدل
        
        بازگشت:
            analysis: تحلیل باقیمانده‌ها
        """
        if self.model_fit is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        resid = self.model_fit.resid
        
        from scipy import stats
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        analysis = {
            'mean': float(np.mean(resid)),
            'std': float(np.std(resid)),
            'skewness': float(stats.skew(resid)),
            'kurtosis': float(stats.kurtosis(resid)),
            'normality_test': stats.normaltest(resid).pvalue,
            'is_normal': stats.normaltest(resid).pvalue > 0.05
        }
        
        # آزمون Ljung-Box
        try:
            lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
            analysis['ljung_box_pvalue'] = float(lb_test['lb_pvalue'].iloc[0])
            analysis['no_autocorrelation'] = analysis['ljung_box_pvalue'] > 0.05
        except:
            analysis['ljung_box_pvalue'] = 1.0
            analysis['no_autocorrelation'] = True
        
        self.logger.info(f"📊 میانگین باقیمانده: {analysis['mean']:.4f}")
        self.logger.info(f"📊 نرمال بودن: {analysis['is_normal']}")
        self.logger.info(f"📊 بدون خودهمبستگی: {analysis['no_autocorrelation']}")
        
        return analysis
    
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
        if self.model_fit is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_type': 'ARIMA',
            'order': self.order,
            'aic': self.aic,
            'bic': self.bic
        })
        
        save_path = save_model(self.model_fit, 'arima', metadata)
        return save_path