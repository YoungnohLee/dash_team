# %%
import pathlib
import re
import json
from datetime import datetime
import flask
import dash
import dash_table
import matplotlib.colors as mcolors
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
from sklearn.manifold import TSNE
import sy_database
import sy_class
import bw_class
import bw_database
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# %%
import plotly.express as px

# Plotly Express를 사용하여 바 차트 그리기
def barplot1():
    train_df = bw_database.making_dataframe_train_db('train_table')
    train = bw_database.making_dataframe_train_db('train_table')
    sy = bw_class.bw_preprocessing(train)
    sy.apply_my_function()
    sy_df = sy.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_sy = sy_df.merge(cluster, on = '고객ID', how = 'left')
    CategoryAnalysis = sy_class.CustomerCategoryAnalysis(train_sy)
    CategoryAnalysis.calculate_repurchase_periods()
    category = CategoryAnalysis.create_category_dataframe()

    # 기본 바 차트 생성
    fig = px.bar(category, 
                 x='평균 재구매 주기(일)',
                 y='제품카테고리', 
                 title='카테고리별 평균 재구매 주기(일)',
                 orientation='h',
                 color='제품카테고리',  # 카테고리에 따라 색상 지정
                 text='평균 재구매 주기(일)')  # 막대에 텍스트 라벨 추가
    
    # 차트 레이아웃 꾸미기
    fig.update_layout(
        title={
            'text': "카테고리별 평균 재구매 주기(일)",
            'y':0.9,
            'x':0.3,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="평균 재구매 주기(일)",
        yaxis_title="제품카테고리",
        legend_title="제품카테고리",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # 막대 색상 꾸미기
    fig.update_traces(
        marker=dict(line=dict(color='#000000', width=1)),
        texttemplate='%{text:.2f}', textposition='outside'
    )
    
    return fig


# %%
# Plotly Express를 사용하여 바 차트 그리기
def barplot2():
    train_df = bw_database.making_dataframe_train_db('train_table')
    train = bw_database.making_dataframe_train_db('train_table')
    sy = bw_class.bw_preprocessing(train)
    sy.apply_my_function()
    sy_df = sy.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_sy = sy_df.merge(cluster, on = '고객ID', how = 'left')
    CategoryAnalysis = sy_class.CustomerCategoryAnalysis(train_sy)
    CategoryAnalysis.calculate_repurchase_periods()
    category = CategoryAnalysis.create_category_dataframe()
    
    # 기본 바 차트 생성
    fig = px.bar(category, 
                 x='재구매율',
                 y='제품카테고리',  
                 title='재구매율',
                 color='제품카테고리',  # 카테고리에 따라 색상 지정
                 orientation='h',
                 text='재구매율')  # 막대에 텍스트 라벨 추가
    
    # 차트 레이아웃 꾸미기
    fig.update_layout(
        title={
            'text': "재구매율",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="재구매율",
        yaxis_title="제품카테고리",
        legend_title="제품카테고리",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # 막대 색상 꾸미기
    fig.update_traces(
        marker=dict(line=dict(color='#000000', width=1)),
        texttemplate='%{text:.2f}', textposition='outside'
    )
    
    return fig


# %%
def heatmap():
    train_df = bw_database.making_dataframe_train_db('train_table')
    train = bw_database.making_dataframe_train_db('train_table')
    sy = bw_class.bw_preprocessing(train)
    sy.apply_my_function()
    sy_df = sy.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_sy = sy_df.merge(cluster, on = '고객ID', how = 'left')
    cohort_analysis = sy_class.CohortAnalysis(train_sy)
    cohort = cohort_analysis.calculate_cohort()
    retention_matrix = cohort_analysis.calculate_retention_rate(cohort)
     
    fig = go.Figure(data=go.Heatmap(
        z=retention_matrix.values,
        x=retention_matrix.columns,
        y=retention_matrix.index,
        colorscale='Blues',
        text=retention_matrix.values,
        texttemplate="%{text:.2%}",  # 비율 형식으로 표시
        textfont={"size": 10, "color": "black"}
    ))
    
    # 레이아웃 업데이트
    fig.update_layout(
        title='코호트 분석 - Retention Rates',
        title_font_size=20,
        xaxis_title='Months After First Purchase',
        yaxis_title='Cohort Group',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor='white',
        margin=dict(l=60, r=20, t=50, b=50),
        width=800,
        height=600
    )
    
    # 축 레이블 및 타이틀 스타일링
    fig.update_xaxes(
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )
    
    # 색상바(컬러바) 스타일링
    fig.update_coloraxes(colorbar=dict(
        title="재구매율",
        titleside="right",
        titlefont=dict(size=14),
        tickfont=dict(size=12)
    ))
    return fig