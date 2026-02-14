"""
🔄 ماژول اعتبارسنجی برای پیش‌بینی سری‌های زمانی
این ماژول شامل توابع اعتبارسنجی متقابل مخصوص سری‌های زمانی است
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from sklearn.model_selection import TimeSeriesSplit

from src.utils import setup_logger
from src.evaluation.metrics import TimeSeriesMetrics

class TimeSeriesBacktester:
    """
    🎯 کلاس اعتبارسنجی برای مدل‌های سری زمانی
    
    این کلاس شامل:
    - اعتبارسنجی با پنجره متحرک (Rolling Window)
    - اعتبارسنجی با پنجره افزایشی (Expanding Window)
    - اعتبارسنجی با پنجره لغزان (Sliding Window)
    - محاسبه متریک‌ها روی هر پنجره
    """
    
    def __init__(self):
        """
        🏗 سازنده کلاس
        """
        self.logger = setup_logger(
            'time_series_backtester',
            log_file='outputs/logs/time_series_backtester.log'
        )
        
        self.metrics = TimeSeriesMetrics()
        
        self.logger.info("✅ TimeSeriesBacktester initialized")
    
    # ============================================
    # اعتبارسنجی با پنجره متحرک (Rolling Window)
    # ============================================
    
    def rolling_window_cv(self, series: pd.Series,
                          model_func: Callable,
                          window_size: int = 100,
                          step_size: int = 10,
                          forecast_horizon: int = 30,
                          **model_params) -> Dict[str, Any]:
        """
        🔄 اعتبارسنجی با پنجره متحرک
        
        پارامترها:
            series: سری زمانی
            model_func: تابع آموزش مدل
            window_size: اندازه پنجره آموزش
            step_size: گام حرکت پنجره
            forecast_horizon: افق پیش‌بینی
            **model_params: پارامترهای مدل
        
        بازگشت:
            results: نتایج اعتبارسنجی
        """
        self.logger.info(f"🚀 شروع اعتبارسنجی با پنجره متحرک...")
        
        n = len(series)
        n_splits = (n - window_size - forecast_horizon) // step_size + 1
        
        results = {
            'forecasts': [],
            'actuals': [],
            'metrics': [],
            'split_indices': []
        }
        
        for i in range(n_splits):
            start_idx = i * step_size
            train_end = start_idx + window_size
            test_start = train_end
            test_end = min(test_start + forecast_horizon, n)
            
            if test_end <= test_start:
                break
            
            # داده‌های آموزش و تست
            train_data = series.iloc[start_idx:train_end]
            test_data = series.iloc[test_start:test_end]
            
            try:
                # آموزش مدل
                model = model_func(train_data, **model_params)
                
                # پیش‌بینی
                if hasattr(model, 'predict'):
                    forecast = model.predict(steps=len(test_data))
                    if isinstance(forecast, dict):
                        forecast = forecast['forecast']
                elif hasattr(model, 'forecast'):
                    forecast = model.forecast(len(test_data))
                else:
                    forecast = model(len(test_data))
                
                # محاسبه متریک‌ها
                metrics = self.metrics.generate_full_report(
                    test_data.values,
                    forecast[:len(test_data)],
                    train_data.values
                )
                
                results['forecasts'].append(forecast)
                results['actuals'].append(test_data.values)
                results['metrics'].append(metrics['metrics'])
                results['split_indices'].append({
                    'train': (start_idx, train_end),
                    'test': (test_start, test_end)
                })
                
                self.logger.info(f"   ✅ Split {i+1}/{n_splits} - "
                               f"RMSE: {metrics['metrics']['rmse']:.2f}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ خطا در split {i+1}: {e}")
        
        # محاسبه میانگین متریک‌ها
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            values = [m[metric] for m in results['metrics'] if metric in m]
            if values:
                avg_metrics[f'avg_{metric}'] = float(np.mean(values))
                avg_metrics[f'std_{metric}'] = float(np.std(values))
        
        results['average_metrics'] = avg_metrics
        
        self.logger.info(f"✅ اعتبارسنجی با {n_splits} split کامل شد")
        self.logger.info(f"   - میانگین RMSE: {avg_metrics.get('avg_rmse', 0):.2f}")
        self.logger.info(f"   - میانگین MAPE: {avg_metrics.get('avg_mape', 0):.2f}%")
        
        return results
    
    # ============================================
    # اعتبارسنجی با پنجره افزایشی (Expanding Window)
    # ============================================
    
    def expanding_window_cv(self, series: pd.Series,
                           model_func: Callable,
                           min_train_size: int = 50,
                           step_size: int = 10,
                           forecast_horizon: int = 30,
                           **model_params) -> Dict[str, Any]:
        """
        🔄 اعتبارسنجی با پنجره افزایشی
        
        پارامترها:
            series: سری زمانی
            model_func: تابع آموزش مدل
            min_train_size: حداقل اندازه آموزش
            step_size: گام افزایش
            forecast_horizon: افق پیش‌بینی
            **model_params: پارامترهای مدل
        
        بازگشت:
            results: نتایج اعتبارسنجی
        """
        self.logger.info(f"🚀 شروع اعتبارسنجی با پنجره افزایشی...")
        
        n = len(series)
        n_splits = (n - min_train_size - forecast_horizon) // step_size + 1
        
        results = {
            'forecasts': [],
            'actuals': [],
            'metrics': [],
            'split_indices': []
        }
        
        train_size = min_train_size
        
        for i in range(n_splits):
            test_start = train_size
            test_end = min(test_start + forecast_horizon, n)
            
            if test_end <= test_start:
                break
            
            # داده‌های آموزش و تست
            train_data = series.iloc[:train_size]
            test_data = series.iloc[test_start:test_end]
            
            try:
                # آموزش مدل
                model = model_func(train_data, **model_params)
                
                # پیش‌بینی
                if hasattr(model, 'predict'):
                    forecast = model.predict(steps=len(test_data))
                    if isinstance(forecast, dict):
                        forecast = forecast['forecast']
                elif hasattr(model, 'forecast'):
                    forecast = model.forecast(len(test_data))
                else:
                    forecast = model(len(test_data))
                
                # محاسبه متریک‌ها
                metrics = self.metrics.generate_full_report(
                    test_data.values,
                    forecast[:len(test_data)],
                    train_data.values
                )
                
                results['forecasts'].append(forecast)
                results['actuals'].append(test_data.values)
                results['metrics'].append(metrics['metrics'])
                results['split_indices'].append({
                    'train': (0, train_size),
                    'test': (test_start, test_end)
                })
                
                self.logger.info(f"   ✅ Split {i+1}/{n_splits} - "
                               f"RMSE: {metrics['metrics']['rmse']:.2f}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ خطا در split {i+1}: {e}")
            
            train_size += step_size
        
        # محاسبه میانگین متریک‌ها
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            values = [m[metric] for m in results['metrics'] if metric in m]
            if values:
                avg_metrics[f'avg_{metric}'] = float(np.mean(values))
                avg_metrics[f'std_{metric}'] = float(np.std(values))
        
        results['average_metrics'] = avg_metrics
        
        self.logger.info(f"✅ اعتبارسنجی با {n_splits} split کامل شد")
        self.logger.info(f"   - میانگین RMSE: {avg_metrics.get('avg_rmse', 0):.2f}")
        
        return results
    
    # ============================================
    # اعتبارسنجی با scikit-learn TimeSeriesSplit
    # ============================================
    
    def time_series_split_cv(self, series: pd.Series,
                            model_func: Callable,
                            n_splits: int = 5,
                            forecast_horizon: int = 30,
                            **model_params) -> Dict[str, Any]:
        """
        🔄 اعتبارسنجی با TimeSeriesSplit از scikit-learn
        
        پارامترها:
            series: سری زمانی
            model_func: تابع آموزش مدل
            n_splits: تعداد splits
            forecast_horizon: افق پیش‌بینی
            **model_params: پارامترهای مدل
        
        بازگشت:
            results: نتایج اعتبارسنجی
        """
        self.logger.info(f"🚀 شروع اعتبارسنجی با TimeSeriesSplit...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = np.arange(len(series)).reshape(-1, 1)
        
        results = {
            'forecasts': [],
            'actuals': [],
            'metrics': [],
            'split_indices': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # محدود کردن اندازه تست
            test_idx = test_idx[:forecast_horizon]
            
            train_data = series.iloc[train_idx]
            test_data = series.iloc[test_idx]
            
            try:
                # آموزش مدل
                model = model_func(train_data, **model_params)
                
                # پیش‌بینی
                if hasattr(model, 'predict'):
                    forecast = model.predict(steps=len(test_data))
                    if isinstance(forecast, dict):
                        forecast = forecast['forecast']
                elif hasattr(model, 'forecast'):
                    forecast = model.forecast(len(test_data))
                else:
                    forecast = model(len(test_data))
                
                # محاسبه متریک‌ها
                metrics = self.metrics.generate_full_report(
                    test_data.values,
                    forecast[:len(test_data)],
                    train_data.values
                )
                
                results['forecasts'].append(forecast)
                results['actuals'].append(test_data.values)
                results['metrics'].append(metrics['metrics'])
                results['split_indices'].append({
                    'train': (train_idx[0], train_idx[-1]),
                    'test': (test_idx[0], test_idx[-1])
                })
                
                self.logger.info(f"   ✅ Fold {fold+1}/{n_splits} - "
                               f"RMSE: {metrics['metrics']['rmse']:.2f}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ خطا در fold {fold+1}: {e}")
        
        # محاسبه میانگین متریک‌ها
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            values = [m[metric] for m in results['metrics'] if metric in m]
            if values:
                avg_metrics[f'avg_{metric}'] = float(np.mean(values))
                avg_metrics[f'std_{metric}'] = float(np.std(values))
        
        results['average_metrics'] = avg_metrics
        
        self.logger.info(f"✅ اعتبارسنجی با {n_splits} fold کامل شد")
        self.logger.info(f"   - میانگین RMSE: {avg_metrics.get('avg_rmse', 0):.2f}")
        
        return results
    
    # ============================================
    # پیش‌بینی out-of-sample
    # ============================================
    
    def out_of_sample_test(self, series: pd.Series,
                          model_func: Callable,
                          train_size: float = 0.8,
                          forecast_horizon: Optional[int] = None,
                          **model_params) -> Dict[str, Any]:
        """
        📊 آزمون out-of-sample ساده
        
        پارامترها:
            series: سری زمانی
            model_func: تابع آموزش مدل
            train_size: نسبت داده آموزش
            forecast_horizon: افق پیش‌بینی
            **model_params: پارامترهای مدل
        
        بازگشت:
            results: نتایج آزمون
        """
        self.logger.info(f"🚀 شروع آزمون out-of-sample...")
        
        n = len(series)
        split_idx = int(n * train_size)
        
        if forecast_horizon is None:
            forecast_horizon = n - split_idx
        
        train_data = series.iloc[:split_idx]
        test_data = series.iloc[split_idx:split_idx + forecast_horizon]
        
        # آموزش مدل
        model = model_func(train_data, **model_params)
        
        # پیش‌بینی
        if hasattr(model, 'predict'):
            forecast = model.predict(steps=len(test_data))
            if isinstance(forecast, dict):
                forecast = forecast['forecast']
        elif hasattr(model, 'forecast'):
            forecast = model.forecast(len(test_data))
        else:
            forecast = model(len(test_data))
        
        # محاسبه متریک‌ها
        metrics = self.metrics.generate_full_report(
            test_data.values,
            forecast[:len(test_data)],
            train_data.values
        )
        
        results = {
            'forecast': forecast,
            'actual': test_data.values,
            'metrics': metrics['metrics'],
            'train_size': len(train_data),
            'test_size': len(test_data),
            'split_idx': split_idx
        }
        
        self.logger.info(f"✅ آزمون out-of-sample کامل شد")
        self.logger.info(f"   - RMSE: {metrics['metrics']['rmse']:.2f}")
        self.logger.info(f"   - MAPE: {metrics['metrics']['mape']:.2f}%")
        
        return results