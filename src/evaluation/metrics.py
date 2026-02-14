"""
📊 ماژول متریک‌های ارزیابی برای پیش‌بینی سری‌های زمانی
این ماژول شامل توابع محاسبه متریک‌های مختلف برای ارزیابی مدل‌های پیش‌بینی است
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import logging

from src.utils import setup_logger

class TimeSeriesMetrics:
    """
    🎯 کلاس ارزیابی متریک‌های پیش‌بینی سری‌های زمانی
    
    این کلاس شامل:
    - متریک‌های پایه (RMSE, MAE, MAPE, MSE)
    - متریک‌های پیشرفته (MASE, SMAPE)
    - تحلیل خطا
    - مقایسه مدل‌ها
    """
    
    def __init__(self):
        """
        🏗 سازنده کلاس
        """
        self.logger = setup_logger(
            'time_series_metrics',
            log_file='outputs/logs/time_series_metrics.log'
        )
        
        self.logger.info("✅ TimeSeriesMetrics initialized")
    
    # ============================================
    # متریک‌های پایه
    # ============================================
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه RMSE (ریشه میانگین مربعات خطا)
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            rmse: مقدار خطا
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه MAE (میانگین قدر مطلق خطا)
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            mae: مقدار خطا
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه MAPE (درصد میانگین قدر مطلق خطا)
        
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
    
    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه MSE (میانگین مربعات خطا)
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            mse: مقدار خطا
        """
        return float(mean_squared_error(y_true, y_pred))
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه R² (ضریب تعیین)
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            r2: ضریب تعیین
        """
        return float(r2_score(y_true, y_pred))
    
    # ============================================
    # متریک‌های پیشرفته
    # ============================================
    
    def calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_train: np.ndarray, seasonal_period: int = 1) -> float:
        """
        📈 محاسبه MASE (Mean Absolute Scaled Error)
        
        این متریک خطا را نسبت به یک پیش‌بینی ساده Naive مقایسه می‌کند
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
            y_train: داده‌های آموزش
            seasonal_period: دوره فصلی
        
        بازگشت:
            mase: مقدار MASE
        """
        n = len(y_train)
        
        if seasonal_period == 1:
            # Naive پیش‌بینی با آخرین مقدار
            naive_forecast = y_train[:-1]
            naive_actual = y_train[1:]
        else:
            # Seasonal Naive
            naive_forecast = y_train[:-seasonal_period]
            naive_actual = y_train[seasonal_period:]
        
        naive_mae = np.mean(np.abs(naive_actual - naive_forecast))
        
        if naive_mae == 0:
            return float('inf')
        
        forecast_mae = np.mean(np.abs(y_true - y_pred))
        mase = forecast_mae / naive_mae
        
        return float(mase)
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه SMAPE (Symmetric Mean Absolute Percentage Error)
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            smape: درصد خطای متقارن
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)
    
    def calculate_rmse_percentage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        📈 محاسبه RMSE درصدی
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
        
        بازگشت:
            rmse_pct: درصد RMSE نسبت به میانگین
        """
        rmse = self.calculate_rmse(y_true, y_pred)
        mean_y = np.mean(y_true)
        
        if mean_y == 0:
            return float('inf')
        
        return float((rmse / mean_y) * 100)
    
    # ============================================
    # تحلیل خطا
    # ============================================
    
    def error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                      dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        🔍 تحلیل خطاهای پیش‌بینی
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
            dates: تاریخ‌ها
        
        بازگشت:
            analysis: تحلیل خطاها
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors) / y_true * 100
        
        analysis = {
            'error_stats': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'median': float(np.median(errors)),
                'q1': float(np.percentile(errors, 25)),
                'q3': float(np.percentile(errors, 75))
            },
            'absolute_error_stats': {
                'mean': float(np.mean(abs_errors)),
                'std': float(np.std(abs_errors)),
                'min': float(np.min(abs_errors)),
                'max': float(np.max(abs_errors))
            },
            'percentage_error_stats': {
                'mean': float(np.mean(pct_errors)),
                'std': float(np.std(pct_errors)),
                'min': float(np.min(pct_errors)),
                'max': float(np.max(pct_errors))
            }
        }
        
        # اگر تاریخ وجود داشت، تحلیل زمانی خطاها
        if dates is not None:
            # خطا بر اساس ماه
            if hasattr(dates, 'month'):
                df_errors = pd.DataFrame({
                    'date': dates,
                    'error': errors,
                    'abs_error': abs_errors
                })
                
                df_errors['month'] = df_errors['date'].dt.month
                df_errors['year'] = df_errors['date'].dt.year
                df_errors['dayofweek'] = df_errors['date'].dt.dayofweek
                
                monthly_errors = df_errors.groupby('month')['abs_error'].mean().to_dict()
                yearly_errors = df_errors.groupby('year')['abs_error'].mean().to_dict()
                dow_errors = df_errors.groupby('dayofweek')['abs_error'].mean().to_dict()
                
                analysis['temporal_errors'] = {
                    'by_month': {int(k): float(v) for k, v in monthly_errors.items()},
                    'by_year': {int(k): float(v) for k, v in yearly_errors.items()},
                    'by_dayofweek': {int(k): float(v) for k, v in dow_errors.items()}
                }
        
        return analysis
    
    # ============================================
    # گزارش کامل ارزیابی
    # ============================================
    
    def generate_full_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_train: Optional[np.ndarray] = None,
                            seasonal_period: int = 7,
                            model_name: str = "Model") -> Dict[str, Any]:
        """
        📋 گزارش کامل ارزیابی
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
            y_train: داده‌های آموزش (برای MASE)
            seasonal_period: دوره فصلی
            model_name: نام مدل
        
        بازگشت:
            report: گزارش کامل
        """
        self.logger.info(f"🚀 تولید گزارش کامل برای {model_name}...")
        
        # متریک‌های پایه
        metrics = {
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mae': self.calculate_mae(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred),
            'mse': self.calculate_mse(y_true, y_pred),
            'r2': self.calculate_r2(y_true, y_pred),
            'smape': self.calculate_smape(y_true, y_pred),
            'rmse_pct': self.calculate_rmse_percentage(y_true, y_pred)
        }
        
        # متریک MASE اگر داده آموزش موجود باشد
        if y_train is not None:
            metrics['mase'] = self.calculate_mase(y_true, y_pred, y_train, seasonal_period)
        
        # تحلیل خطا
        error_analysis = self.error_analysis(y_true, y_pred)
        
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'error_analysis': error_analysis,
            'n_observations': len(y_true)
        }
        
        # امتیاز کلی
        report['overall_score'] = self._calculate_overall_score(metrics)
        report['performance_grade'] = self._get_performance_grade(report['overall_score'])
        
        self.logger.info(f"✅ گزارش کامل برای {model_name} تولید شد")
        self.logger.info(f"   - RMSE: {metrics['rmse']:.2f}")
        self.logger.info(f"   - MAPE: {metrics['mape']:.2f}%")
        self.logger.info(f"   - R²: {metrics['r2']:.3f}")
        
        return report
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        🎯 محاسبه امتیاز کلی مدل
        """
        # نرمال‌سازی متریک‌ها
        weights = {
            'r2': 0.3,
            'mape': 0.25,
            'rmse_pct': 0.25,
            'smape': 0.2
        }
        
        score = 0
        if 'r2' in metrics:
            score += max(0, metrics['r2']) * weights['r2'] * 100
        
        if 'mape' in metrics:
            score += max(0, 100 - metrics['mape']) * weights['mape']
        
        if 'rmse_pct' in metrics:
            score += max(0, 100 - metrics['rmse_pct']) * weights['rmse_pct']
        
        if 'smape' in metrics:
            score += max(0, 100 - metrics['smape']) * weights['smape']
        
        return min(score, 100)
    
    def _get_performance_grade(self, score: float) -> str:
        """
        🏆 درجه‌بندی عملکرد
        """
        if score >= 90:
            return "A+ (عالی) 🏆"
        elif score >= 80:
            return "A (بسیار خوب) ⭐"
        elif score >= 70:
            return "B (خوب) ✅"
        elif score >= 60:
            return "C (قابل قبول) ⚠️"
        elif score >= 50:
            return "D (ضعیف) ❌"
        else:
            return "F (غیرقابل قبول) 🔴"
    
    # ============================================
    # مقایسه مدل‌ها
    # ============================================
    
    def compare_models(self, models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        🤝 مقایسه چند مدل
        
        پارامترها:
            models_results: دیکشنری نتایج مدل‌ها
        
        بازگشت:
            comparison_df: دیتافریم مقایسه
        """
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            row = {
                'Model': model_name,
                'RMSE': metrics.get('rmse', 0),
                'MAE': metrics.get('mae', 0),
                'MAPE': metrics.get('mape', 0),
                'R²': metrics.get('r2', 0),
                'SMAPE': metrics.get('smape', 0),
                'MASE': metrics.get('mase', 0)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # مرتب‌سازی بر اساس RMSE (کمتر بهتر است)
        comparison_df = comparison_df.sort_values('RMSE', ascending=True)
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df