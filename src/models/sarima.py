"""
📈 ماژول مدل SARIMA برای پیش‌بینی سری‌های زمانی
این ماژول شامل پیاده‌سازی مدل SARIMA (فصلی) است
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, save_model, calculate_rmse, calculate_mae, calculate_mape

class SARIMAModel:
    """
    🎯 کلاس مدل SARIMA برای پیش‌بینی سری‌های زمانی
    
    SARIMA (Seasonal ARIMA) نسخه فصلی ARIMA است که برای سری‌های
    زمانی با الگوهای فصلی مناسب است.
    
    پارامترها:
    - p, d, q: پارامترهای غیرفصلی
    - P, D, Q, m: پارامترهای فصلی
    - m: دوره فصلی (مثلاً 7 برای هفتگی، 12 برای ماهانه)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        🏗 سازنده کلاس
        """
        from src.utils import load_config
        self.config = load_config(config_path) if config_path else load_config()
        self.sarima_config = self.config.get('models', {}).get('sarima', {})
        
        self.logger = setup_logger(
            'sarima_model',
            log_file='outputs/logs/sarima_model.log'
        )
        
        self.model = None
        self.model_fit = None
        self.order = None
        self.seasonal_order = None
        self.aic = None
        self.bic = None
        
        self.logger.info("✅ SARIMAModel initialized")
    
    # ============================================
    # آموزش خودکار با Auto ARIMA
    # ============================================
    
    def auto_sarima(self, series: pd.Series, 
                   seasonal_period: int = 7,
                   **kwargs) -> Any:
        """
        🤖 آموزش خودکار SARIMA با pmdarima
        
        پارامترها:
            series: سری زمانی
            seasonal_period: دوره فصلی
            **kwargs: پارامترهای اضافی
        
        بازگشت:
            model: مدل آموزش دیده
        """
        self.logger.info(f"🚀 شروع آموزش خودکار SARIMA با دوره فصلی {seasonal_period}...")
        
        # تنظیم پارامترها
        max_p = self.sarima_config.get('max_p', 3)
        max_d = self.sarima_config.get('max_d', 1)
        max_q = self.sarima_config.get('max_q', 3)
        max_P = self.sarima_config.get('max_P', 2)
        max_D = self.sarima_config.get('max_D', 1)
        max_Q = self.sarima_config.get('max_Q', 2)
        
        try:
            # آموزش خودکار
            self.model = pm.auto_arima(
                series,
                start_p=0, max_p=max_p,
                start_d=0, max_d=max_d,
                start_q=0, max_q=max_q,
                start_P=0, max_P=max_P,
                start_D=0, max_D=max_D,
                start_Q=0, max_Q=max_Q,
                m=seasonal_period,
                seasonal=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                information_criterion='aic',
                **kwargs
            )
            
            self.order = self.model.order
            self.seasonal_order = self.model.seasonal_order
            self.aic = self.model.aic()
            self.bic = self.model.bic()
            
            self.logger.info(f"✅ مدل SARIMA با موفقیت آموزش دید")
            self.logger.info(f"   - order: {self.order}")
            self.logger.info(f"   - seasonal_order: {self.seasonal_order}")
            self.logger.info(f"   - AIC: {self.aic:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ خطا در آموزش خودکار SARIMA: {e}")
            raise
        
        return self.model
    
    # ============================================
    # آموزش با پارامترهای مشخص
    # ============================================
    
    def train(self, series: pd.Series,
             order: Tuple[int, int, int],
             seasonal_order: Tuple[int, int, int, int],
             **kwargs) -> Any:
        """
        📚 آموزش مدل SARIMA با پارامترهای مشخص
        
        پارامترها:
            series: سری زمانی
            order: (p, d, q)
            seasonal_order: (P, D, Q, m)
            **kwargs: پارامترهای اضافی
        
        بازگشت:
            model_fit: مدل آموزش دیده
        """
        self.logger.info(f"🚀 شروع آموزش SARIMA با پارامترهای مشخص...")
        
        self.order = order
        self.seasonal_order = seasonal_order
        
        try:
            self.model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                **kwargs
            )
            
            self.model_fit = self.model.fit(disp=False)
            
            self.aic = self.model_fit.aic
            self.bic = self.model_fit.bic
            
            self.logger.info(f"✅ مدل SARIMA{order}x{seasonal_order} آموزش دید")
            self.logger.info(f"   - AIC: {self.aic:.2f}")
            self.logger.info(f"   - BIC: {self.bic:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ خطا در آموزش SARIMA: {e}")
            raise
        
        return self.model_fit
    
    # ============================================
    # پیش‌بینی
    # ============================================
    
    def predict(self, steps: int = 30,
               return_conf_int: bool = True,
               alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        🔮 پیش‌بینی مقادیر آینده
        
        پارامترها:
            steps: تعداد گام‌های پیش‌بینی
            return_conf_int: برگرداندن فاصله اطمینان
            alpha: سطح معنی‌داری برای فاصله اطمینان
        
        بازگشت:
            predictions: دیکشنری پیش‌بینی‌ها
        """
        if self.model is None and self.model_fit is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        # اگر با auto_arima آموزش داده شده
        if self.model is not None and hasattr(self.model, 'predict'):
            predictions = self.model.predict(n_periods=steps)
            
            result = {
                'forecast': predictions.values if hasattr(predictions, 'values') else predictions,
                'index': np.arange(steps)
            }
            
            return result
        
        # اگر با SARIMAX آموزش داده شده
        elif self.model_fit is not None:
            forecast_result = self.model_fit.get_forecast(steps=steps)
            
            predictions = forecast_result.predicted_mean
            
            result = {
                'forecast': predictions.values,
                'index': predictions.index if hasattr(predictions, 'index') else np.arange(steps)
            }
            
            if return_conf_int:
                conf_int = forecast_result.conf_int(alpha=alpha)
                result['lower_bound'] = conf_int.iloc[:, 0].values
                result['upper_bound'] = conf_int.iloc[:, 1].values
            
            return result
        
        else:
            raise ValueError("❌ مدل معتبر یافت نشد")
    
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
        
        self.logger.info(f"📊 نتایج ارزیابی SARIMA:")
        self.logger.info(f"   - RMSE: {metrics['rmse']:.2f}")
        self.logger.info(f"   - MAE: {metrics['mae']:.2f}")
        self.logger.info(f"   - MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
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
        if self.model is None and self.model_fit is None:
            raise ValueError("❌ مدل هنوز آموزش ندیده است!")
        
        if metadata is None:
            metadata = {}
        
        model_to_save = self.model if self.model is not None else self.model_fit
        
        metadata.update({
            'model_type': 'SARIMA',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.aic,
            'bic': self.bic
        })
        
        save_path = save_model(model_to_save, 'sarima', metadata)
        return save_path