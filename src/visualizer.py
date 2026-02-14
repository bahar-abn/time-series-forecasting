"""
🎨 ماژول مصورسازی برای پیش‌بینی سری‌های زمانی
این ماژول شامل توابع رسم نمودارهای مختلف برای تحلیل و پیش‌بینی سری‌های زمانی است
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from src.utils import setup_logger, format_date, format_number

class TimeSeriesVisualizer:
    """
    🎯 کلاس مصورسازی سری‌های زمانی
    
    این کلاس شامل:
    - نمودارهای سری زمانی
    - نمودارهای تجزیه
    - نمودارهای پیش‌بینی
    - نمودارهای ارزیابی
    - داشبوردهای تعاملی
    """
    
    def __init__(self):
        """
        🏗 سازنده کلاس
        """
        self.logger = setup_logger(
            'time_series_visualizer',
            log_file='outputs/logs/time_series_visualizer.log'
        )
        
        # تنظیم استایل
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info("✅ TimeSeriesVisualizer initialized")
    
    # ============================================
    # نمودارهای سری زمانی پایه
    # ============================================
    
    def plot_time_series(self, df: pd.DataFrame,
                        date_col: str, value_col: str,
                        title: str = "سری زمانی",
                        color: str = '#636EFA') -> go.Figure:
        """
        📈 نمودار سری زمانی ساده
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
            title: عنوان نمودار
            color: رنگ خط
        
        بازگشت:
            fig: نمودار Plotly
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode='lines',
            name=value_col,
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="تاریخ",
            yaxis_title="مقدار",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    def plot_multiple_series(self, df: pd.DataFrame,
                            date_col: str, value_cols: List[str],
                            title: str = "مقایسه سری‌های زمانی") -> go.Figure:
        """
        📈 نمودار چند سری زمانی با هم
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_cols: لیست ستون‌های مقدار
            title: عنوان نمودار
        
        بازگشت:
            fig: نمودار Plotly
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="تاریخ",
            yaxis_title="مقدار",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    # ============================================
    # نمودارهای فصلی و الگوها
    # ============================================
    
    def plot_seasonal_patterns(self, df: pd.DataFrame,
                               date_col: str, value_col: str,
                               period: str = 'month') -> go.Figure:
        """
        📊 نمودار الگوهای فصلی
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
            period: دوره ('month', 'dayofweek', 'hour')
        
        بازگشت:
            fig: نمودار Plotly
        """
        df_plot = df.copy()
        df_plot['date'] = pd.to_datetime(df_plot[date_col])
        
        # تعریف نام‌های دوره‌ها
        if period == 'month':
            df_plot['period'] = df_plot['date'].dt.month
            period_names = {
                1: 'فروردین', 2: 'اردیبهشت', 3: 'خرداد', 4: 'تیر',
                5: 'مرداد', 6: 'شهریور', 7: 'مهر', 8: 'آبان',
                9: 'آذر', 10: 'دی', 11: 'بهمن', 12: 'اسفند'
            }
            x_title = "ماه"
            
        elif period == 'dayofweek':
            df_plot['period'] = df_plot['date'].dt.dayofweek
            period_names = {
                0: 'دوشنبه', 1: 'سه‌شنبه', 2: 'چهارشنبه', 
                3: 'پنجشنبه', 4: 'جمعه', 5: 'شنبه', 6: 'یکشنبه'
            }
            x_title = "روز هفته"
            
        elif period == 'hour':
            df_plot['period'] = df_plot['date'].dt.hour
            period_names = {i: f'{i}:00' for i in range(24)}
            x_title = "ساعت"
            
        else:
            df_plot['period'] = df_plot['date'].dt.month
            period_names = {i: str(i) for i in range(1, 13)}
            x_title = "دوره"
        
        # محاسبه آمار
        stats = df_plot.groupby('period')[value_col].agg(['mean', 'std', 'min', 'max']).reset_index()
        stats['period_name'] = stats['period'].map(lambda x: period_names.get(x, str(x)))
        
        # مرتب‌سازی بر اساس period
        stats = stats.sort_values('period')
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['میانگین و انحراف معیار', 'باکس پلات'],
            specs=[[{'type': 'bar'}, {'type': 'box'}]]
        )
        
        # نمودار میله‌ای
        fig.add_trace(
            go.Bar(
                x=stats['period_name'],
                y=stats['mean'],
                error_y=dict(type='data', array=stats['std']),
                marker_color='#636EFA',
                name='میانگین',
                text=[f"{v:.1f}" for v in stats['mean']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # باکس پلات
        for p in sorted(df_plot['period'].unique()):
            data = df_plot[df_plot['period'] == p][value_col]
            period_name = period_names.get(p, str(p))
            
            if len(data) > 0:
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=period_name,
                        boxmean='sd',
                        marker_color='#636EFA'
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            title=f'الگوهای فصلی - {x_title}',
            height=500,
            showlegend=False,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        fig.update_xaxes(title_text=x_title, row=1, col=1)
        fig.update_xaxes(title_text=x_title, row=1, col=2)
        
        return fig
    
    def plot_seasonal_subseries(self, df: pd.DataFrame,
                                date_col: str, value_col: str,
                                period: int = 7) -> go.Figure:
        """
        📊 نمودار زیرسری‌های فصلی
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
            period: دوره فصلی
        
        بازگشت:
            fig: نمودار Plotly
        """
        df_plot = df.copy()
        df_plot['date'] = pd.to_datetime(df_plot[date_col])
        df_plot['seasonal_idx'] = np.arange(len(df_plot)) % period
        
        fig = make_subplots(
            rows=period, cols=1,
            subplot_titles=[f'دوره {i+1}' for i in range(period)],
            shared_xaxes=True,
            vertical_spacing=0.02
        )
        
        for i in range(period):
            subset = df_plot[df_plot['seasonal_idx'] == i]
            if len(subset) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=subset['date'],
                        y=subset[value_col],
                        mode='lines+markers',
                        name=f'دوره {i+1}',
                        line=dict(width=2, color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]),
                        marker=dict(size=4)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=f'زیرسری‌های فصلی (دوره {period})',
            height=200 * period,
            showlegend=False,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    # ============================================
    # نمودارهای تجزیه
    # ============================================
    
    def plot_decomposition(self, components: Dict[str, np.ndarray],
                          dates: pd.DatetimeIndex,
                          title: str = "تجزیه سری زمانی") -> go.Figure:
        """
        📊 نمودار اجزای سری زمانی
        
        پارامترها:
            components: دیکشنری اجزا
            dates: تاریخ‌ها
            title: عنوان نمودار
        
        بازگشت:
            fig: نمودار Plotly
        """
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Original
        if 'observed' in components and components['observed'] is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=components['observed'],
                    mode='lines',
                    name='Observed',
                    line=dict(color='black', width=1)
                ),
                row=1, col=1
            )
        
        # Trend
        if components.get('trend') is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=components['trend'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # Seasonal
        if components.get('seasonal') is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=components['seasonal'],
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='green', width=1)
                ),
                row=3, col=1
            )
        
        # Residual
        if components.get('resid') is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=components['resid'],
                    mode='markers+lines',
                    name='Residual',
                    line=dict(color='red', width=0.5),
                    marker=dict(size=2)
                ),
                row=4, col=1
            )
            # خط صفر
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    # ============================================
    # نمودارهای پیش‌بینی - نسخه اصلاح شده
    # ============================================
    
    def plot_forecast(self, historical: pd.DataFrame,
                     forecast: Dict[str, np.ndarray],
                     date_col: str, value_col: str,
                     title: str = "پیش‌بینی سری زمانی") -> go.Figure:
        """
        🔮 نمودار پیش‌بینی - نسخه اصلاح شده برای رفع خطای Timestamp
        
        پارامترها:
            historical: داده‌های تاریخی
            forecast: دیکشنری پیش‌بینی (شامل forecast, lower_bound, upper_bound)
            date_col: ستون تاریخ
            value_col: ستون مقدار
            title: عنوان نمودار
        
        بازگشت:
            fig: نمودار Plotly
        """
        fig = go.Figure()
        
        # داده‌های تاریخی
        fig.add_trace(go.Scatter(
            x=historical[date_col],
            y=historical[value_col],
            mode='lines',
            name='Historical',
            line=dict(color='#636EFA', width=2)
        ))
        
        # آخرین تاریخ
        last_date = pd.to_datetime(historical[date_col].iloc[-1])
        
        # پیش‌بینی - استفاده از تاریخ‌های شاخص عددی برای جلوگیری از خطا
        forecast_indices = np.arange(len(historical), len(historical) + len(forecast['forecast']))
        
        # ذخیره نقشه ایندکس به تاریخ برای نمایش
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(forecast['forecast'])
        )
        
        # پیش‌بینی
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#EF553B', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # فاصله اطمینان
        if 'lower_bound' in forecast and 'upper_bound' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(239, 85, 59, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        # خط جداکننده - استفاده از عدد به جای Timestamp برای جلوگیری از خطا
        fig.add_vline(
            x=len(historical) - 0.5,  # استفاده از ایندکس عددی
            line_dash="dash",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top right"
        )
        
        # تنظیم محور x به صورت تاریخ
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickangle=45
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="تاریخ",
            yaxis_title="مقدار",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    # ============================================
    # نمودارهای ارزیابی
    # ============================================
    
    def plot_forecast_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                               dates: Optional[pd.DatetimeIndex] = None,
                               title: str = "پیش‌بینی vs واقعی") -> go.Figure:
        """
        📊 نمودار مقایسه پیش‌بینی و واقعی
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
            dates: تاریخ‌ها
            title: عنوان نمودار
        
        بازگشت:
            fig: نمودار Plotly
        """
        if dates is None:
            dates = np.arange(len(y_true))
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['مقایسه زمانی', 'پراکندگی'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # مقایسه زمانی
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_true,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#00CC96', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_pred,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#EF553B', width=2, dash='dash'),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # نمودار پراکندگی
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(
                    size=8,
                    color=np.abs(y_true - y_pred),
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Error")
                )
            ),
            row=1, col=2
        )
        
        # خط ایده‌آل
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Ideal',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=500,
            showlegend=True,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        fig.update_xaxes(title_text="تاریخ", row=1, col=1)
        fig.update_yaxes(title_text="مقدار", row=1, col=1)
        fig.update_xaxes(title_text="مقادیر واقعی", row=1, col=2)
        fig.update_yaxes(title_text="مقادیر پیش‌بینی", row=1, col=2)
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      dates: Optional[pd.DatetimeIndex] = None,
                      title: str = "تحلیل باقیمانده‌ها") -> go.Figure:
        """
        📉 نمودار تحلیل باقیمانده‌ها
        
        پارامترها:
            y_true: مقادیر واقعی
            y_pred: مقادیر پیش‌بینی شده
            dates: تاریخ‌ها
            title: عنوان نمودار
        
        بازگشت:
            fig: نمودار Plotly
        """
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        if dates is None:
            dates = np.arange(len(y_true))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'باقیمانده‌ها vs زمان',
                'هیستوگرام باقیمانده‌ها',
                'Q-Q Plot',
                'ACF باقیمانده‌ها'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # باقیمانده‌ها vs زمان
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=residuals,
                mode='markers+lines',
                name='Residuals',
                line=dict(color='red', width=0.5),
                marker=dict(size=3)
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # هیستوگرام
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='#636EFA',
                name='Distribution'
            ),
            row=1, col=2
        )
        
        # Q-Q Plot
        from scipy import stats
        qq = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name='Q-Q',
                marker=dict(color='#00CC96')
            ),
            row=2, col=1
        )
        
        # خط نرمال
        x_line = np.array([qq[0][0].min(), qq[0][0].max()])
        y_line = qq[1][1] + qq[1][0] * x_line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # ACF
        try:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(residuals, nlags=min(40, len(residuals)//2), fft=True)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(acf_values))),
                    y=acf_values,
                    marker_color='#EF553B',
                    name='ACF'
                ),
                row=2, col=2
            )
            
            # خطوط اطمینان
            conf_level = 1.96 / np.sqrt(len(residuals))
            fig.add_hline(y=conf_level, line_dash="dash", line_color="gray", row=2, col=2)
            fig.add_hline(y=-conf_level, line_dash="dash", line_color="gray", row=2, col=2)
        except:
            pass
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    # ============================================
    # داشبورد جامع
    # ============================================
    
    def create_dashboard(self, df: pd.DataFrame,
                        date_col: str, value_col: str,
                        decomposition: Optional[Dict] = None,
                        forecast: Optional[Dict] = None,
                        title: str = "داشبورد تحلیل سری زمانی") -> go.Figure:
        """
        📊 داشبورد جامع تحلیل سری زمانی
        
        پارامترها:
            df: دیتافریم
            date_col: ستون تاریخ
            value_col: ستون مقدار
            decomposition: نتایج تجزیه
            forecast: نتایج پیش‌بینی
            title: عنوان داشبورد
        
        بازگشت:
            fig: داشبورد Plotly
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'سری زمانی',
                'توزیع مقادیر',
                'الگوی ماهانه',
                'روند',
                'فصلی',
                'باقیمانده',
                'ACF',
                'PACF',
                'پیش‌بینی'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'box'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. سری زمانی
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[value_col],
                mode='lines',
                line=dict(color='#636EFA', width=2)
            ),
            row=1, col=1
        )
        
        # 2. هیستوگرام
        fig.add_trace(
            go.Histogram(
                x=df[value_col],
                nbinsx=30,
                marker_color='#00CC96'
            ),
            row=1, col=2
        )
        
        # 3. باکس پلات ماهانه
        df_temp = df.copy()
        df_temp['month'] = pd.to_datetime(df_temp[date_col]).dt.month
        month_names = {1: 'فر', 2: 'ار', 3: 'خر', 4: 'تی', 5: 'مر', 6: 'شه',
                      7: 'مه', 8: 'آب', 9: 'آذ', 10: 'دی', 11: 'به', 12: 'اس'}
        
        for month in range(1, 13):
            month_data = df_temp[df_temp['month'] == month][value_col]
            if len(month_data) > 0:
                fig.add_trace(
                    go.Box(
                        y=month_data,
                        name=month_names[month],
                        boxmean='sd'
                    ),
                    row=1, col=3
                )
        
        # 4-6. اجزای تجزیه
        if decomposition is not None:
            # روند
            if 'trend' in decomposition and decomposition['trend'] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=decomposition['trend'],
                        mode='lines',
                        line=dict(color='#EF553B', width=2)
                    ),
                    row=2, col=1
                )
            
            # فصلی
            if 'seasonal' in decomposition and decomposition['seasonal'] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=decomposition['seasonal'],
                        mode='lines',
                        line=dict(color='#AB63FA', width=1)
                    ),
                    row=2, col=2
                )
            
            # باقیمانده
            if 'resid' in decomposition and decomposition['resid'] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=decomposition['resid'],
                        mode='markers',
                        marker=dict(color='#FFA15A', size=3)
                    ),
                    row=2, col=3
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=3)
        
        # 7. ACF
        try:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(df[value_col].dropna(), nlags=min(40, len(df)//2), fft=True)
            fig.add_trace(
                go.Bar(
                    x=list(range(len(acf_values))),
                    y=acf_values,
                    marker_color='#19D3F3'
                ),
                row=3, col=1
            )
        except:
            pass
        
        # 8. PACF
        try:
            from statsmodels.tsa.stattools import pacf
            pacf_values = pacf(df[value_col].dropna(), nlags=min(40, len(df)//2))
            fig.add_trace(
                go.Bar(
                    x=list(range(len(pacf_values))),
                    y=pacf_values,
                    marker_color='#FF6692'
                ),
                row=3, col=2
            )
        except:
            pass
        
        # 9. پیش‌بینی
        if forecast is not None and 'forecast' in forecast:
            last_date = pd.to_datetime(df[date_col].iloc[-1])
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(forecast['forecast'])
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast['forecast'],
                    mode='lines',
                    line=dict(color='#EF553B', width=2, dash='dash'),
                    name='Forecast'
                ),
                row=3, col=3
            )
            
            if 'lower_bound' in forecast and 'upper_bound' in forecast:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                        y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(239, 85, 59, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence'
                    ),
                    row=3, col=3
                )
        
        fig.update_layout(
            title_text=title,
            height=900,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig