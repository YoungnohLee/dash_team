from login import app as flask_app
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from flask_login import login_required, current_user, logout_user
from dash.dependencies import State
from openais import chat_completion_request
import json
from waitress import serve

# from arppu_page import layout as arppu_layout
# from churn_prediction_page2 import layout as churn_layout

import pandas as pd
import threading
import time
# from shoot_row import shoot_row, csv2db
from shoot_row_connection import shoot_row, csv2db # Connection Pool for Threading shoot_row.py
from flask import Flask

# mj page
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx
import dash_mantine_components as dmc
import database_bw
from bw_class import RFMProcessor
import mj_class
import warnings
warnings.filterwarnings('ignore')
# dh page
import pickle as pkl
import dh_class as dc
import database_bw
# bw page
import pathlib
import re
import json
from datetime import datetime
import flask
import dash
import dash_table
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.express as px

from dateutil import relativedelta
import bw_class
import bw_database
# hi page
import hi_database
import hi_class

# Code Starts

## Connection Pool for threading connection
import sqlite3
import queue

test_db = "TEST.DB"
train_db = "TRAIN.DB"

MAX_CONNECTIONS = 5

class ConnectionPool:
    def __init__(self, database):
        self._connections = queue.Queue(maxsize=MAX_CONNECTIONS)
        self._lock = threading.Lock()
        for _ in range(MAX_CONNECTIONS):
            self._connections.put(sqlite3.connect(database, check_same_thread=False))
    
    def get_connection(self):
        return self._connections.get()

    def release_connection(self, conn):
        self._connections.put(conn)

train_conn_pool = ConnectionPool(train_db)
test_conn_pool = ConnectionPool(test_db)


def run_shoot_row():
    csv2db()
    while True:
        time.sleep(0.01)
        shoot_row()

# Create a Dash instance within the Flask app with a unique name
app = Flask(__name__)
dash_app = dash.Dash(server=flask_app, name="uniqueDashboard", url_base_pathname="/dashboard/", external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app.title = 'Admin Dashboard'

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Admin \n Dashboard", className="display-5"),
        html.Hr(),
        html.P("관리자를 위한 단 하나의 대시보드.", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/dashboard/", active="exact"),
                dbc.NavLink("고객 이탈률", href="/dashboard/page-1", active="exact"),
                dbc.NavLink("재무", href="/dashboard/page-2", active="exact"),
                dbc.NavLink("마케팅", href="/dashboard/page-3", active="exact"),
                dbc.NavLink("고객 충성도 (재구매)", href="/dashboard/page-4", active="exact"),
                dbc.NavLink("성과지표 (ARPPU)", href="/dashboard/page-5", active="exact"),
                dbc.NavLink("로그아웃", href="/logout", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

dash_app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    dcc.Interval(id='interval-component', interval=100*1000, n_intervals=0)
])

# layout for each role

## mj layout (arppu)
mj_layout = html.Div(
    style= {'overflow-x':'hidden'},
    children=[
        dmc.Group(
            direction = 'column',
            grow = True,
            position = 'center',
            spacing = 'sm',
            children=[
                dmc.Title(children='ARPPU Analysis', order=3, style={'font-family':'IntegralCF-ExtraBold', 'text-align':'center', 'color':'slategray'}),
                # dmc.Divider(label='Overview', labelPosition='center', size='xl'),
                dmc.Paper(
                    shadow = 'md',
                    m = 'sm',
                    p = 'md',
                    #style = {'width':'90%'},
                    withBorder = True,
                    children=[
                        dmc.Stack(
                            children=[
                                dmc.Stack(
                                    children=[
                                        dmc.Select(
                                            id='arppu-select',
                                            label='Select ARPPU Analysis Type',
                                            style={'width': '50%', 'margin': 'auto'},
                                            data=[
                                                {'label': 'Cluster Analysis', 'value': 'cluster'},
                                                {'label': 'Monthly ARPPU', 'value': 'monthly'},
                                                {'label': 'Area Analysis', 'value': 'area'},
                                                {'label': 'Subscription Period Analysis', 'value': 'subscription'},
                                                {'label': 'Area Analysis(map)', 'value': 'area(map)'}
                                                
                                            ],
                                            value='monthly'
                                        ),
                                    ]
                                ),
                                dcc.Graph(id='arppu-graph'),
                                dmc.Divider(),
                                html.Div(children=[
                                    html.H3("연관분석"),
                                    html.P("ARPPU를 증가시키기 위해서는 한번에 결제를 더 많이 할 수 있도록 유도해야합니다. "
                                           "고객이 구매할 때 다른 제품 및 서비스도 제안할 수 있는 Cross-Selling 방법을 생각해보아야합니다. "
                                           "아래의 표를 통해 A 제품을 구매한 고객이 B제품도 구매했는 지 알아 볼 수 있습니다."),
                                    html.P("지지도: 두 제품을 모두 구매한 고객 수의 비율"),
                                    html.P("신뢰도: A를 구매한 고객 중 B를 구매한 고객의 비율"),
                                    html.P("향상도: 마케팅 효과 증가율"),
                                    html.Div(id='apriori-results')
                                ])
                            ]
                        )
                    ]
                ),
                dmc.Space(h=50)
            ]
        )
    ]
)
## dh layout
dh_layout = dmc.Container(
    fluid=True,
    style={"padding": "20px"},
    children=[
        html.H1("고객 이탈률 예측", style={"textAlign": "center"}),
        dmc.Group(
            spacing="lg",
            grow=True,
            style={"marginBottom": "20px"},
            children=[
                dmc.Paper(
                    children=[
                        html.P("전체 고객수", style={"fontWeight": "500"}),
                        html.P(id="total_customers", style={"fontWeight": "700", "fontSize": "20px"})
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="xs",
                    style={"textAlign": "center"},
                ),
                dmc.Paper(
                    children=[
                        html.P("이탈 위험 고객 수", style={"fontWeight": "500"}),
                        html.P(id="churned_customers", style={"fontWeight": "700", "fontSize": "20px"})
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="xs",
                    style={"textAlign": "center"},
                ),
                dmc.Paper(
                    children=[
                        html.P("이탈 위험 고객 비율", style={"fontWeight": "500"}),
                        html.P(id="churned_ratio", style={"fontWeight": "700", "fontSize": "20px"})
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="xs",
                    style={"textAlign": "center"},
                )
            ]
        ),
        dmc.SimpleGrid(
            cols=4,
            spacing="lg",
            children=[
                dmc.Paper(
                    children=[
                        html.H2("고객 성별 분포", style={"textAlign": "center"}),
                        dcc.Graph(id='gender_pie_chart')
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="lg",
                    style={"padding": "20px"}
                ),
                dmc.Paper(
                    children=[
                        html.H2("고객 지역 분포", style={"textAlign": "center"}),
                        dcc.Graph(id='location_pie_chart')
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="lg",
                    style={"padding": "20px"}
                ),
                dmc.Paper(
                    children=[
                        html.H2("고객 별 평균 구매 금액", style={"textAlign": "center"}),
                        dcc.Graph(id='purchase_amount_chart')
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="lg",
                    style={"padding": "20px"}
                ),
                dmc.Paper(
                    children=[
                        html.H2("고객 별 선호 제품군", style={"textAlign": "center"}),
                        dcc.Graph(id='category_chart')
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="lg",
                    style={"padding": "20px"}
                ),
            ]
        ),
        dmc.SimpleGrid(
            cols=2,
            spacing="lg",
            children=[
                dmc.Paper(
                    children=[
                        html.H2("지역 별 이탈 위험 고객 수", style={"textAlign": "center"}),
                        dcc.Graph(id='mapbox_chart')
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="lg",
                    style={"padding": "20px"}
                ),
                dmc.Paper(
                    children=[
                        html.H2("월 별 구매 고객 수", style={"textAlign": "center"}),
                        dcc.Graph(id='month_chart')
                    ],
                    withBorder=True,
                    shadow="sm",
                    p="lg",
                    style={"padding": "20px"}
                )
            ]
        ),
        dmc.Paper(
            style={"border": "1px solid #ddd"},  # 수정된 부분
            shadow="sm",
            p="lg",
            mb="lg",
            children=[
                html.H2("90일 이후의 예측 이탈률 상위 고객 정보", style={"textAlign": "center"}),
                dcc.Dropdown(
                    id='row-dropdown',
                    options=[
                        {"label": "10명", "value": 10},
                        {"label": "15명", "value": 15},
                        {"label": "20명", "value": 20},
                        {"label": "25명", "value": 25},
                        {"label": "30명", "value": 30}
                    ],
                    value=10,
                    clearable=False,
                    style={'width': '50%', 'margin': 'auto', 'marginTop': '10px'}
                ),
                dmc.Table(
                    id='customer-table',
                    children={"head": [], "body": []},
                    striped=True,
                    highlightOnHover=True,
                    style={"border": "1px solid #ddd"}  # 추가된 부분
                )
            ]
        )
    ]
)
## bw layout

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# %%
NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("고객 매출 분석", className="ml-2")
                    ),
                ],
                align="center",
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

# %%
def donut_chart1():
    custom_sort = {'VIP고객': 0, '우수고객': 1, '관심고객': 2, '잠재고객': 3, '이탈고객': 4}
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID', 'Recency', 'Frequency', 'Monetary', '고객분류']]
    train_bw = bw_df.merge(cluster, on='고객ID', how='left')
    rfm_clusters = train_bw[['고객ID', 'Recency', 'Frequency', 'Monetary', '고객분류']].drop_duplicates(subset=['고객ID'])
    analysis = bw_class.first_dash(rfm_clusters)
    rfm_clusters_final = analysis.get_final_dataframe()
    rfm_clusters_final.reset_index(drop=True, inplace=True)

    sorted_categories = sorted(rfm_clusters_final['고객분류'], key=lambda x: custom_sort[x])

    fig = px.pie(
        rfm_clusters_final,
        title="고객비율",
        values="고객수",
        names="고객분류",
        template="plotly_white",
        hover_data=["고객수 비율"],
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={"고객분류": sorted_categories},
        hole=0.5
    )

    fig.update_layout(legend=dict(x=-1, y=1.1), legend_orientation="v")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig

def donut_chart2():
    custom_sort = {'VIP고객': 0, '우수고객': 1, '관심고객': 2, '잠재고객': 3, '이탈고객': 4}
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID', 'Recency', 'Frequency', 'Monetary', '고객분류']]
    train_bw = bw_df.merge(cluster, on='고객ID', how='left')
    rfm_clusters = train_bw[['고객ID', 'Recency', 'Frequency', 'Monetary', '고객분류']].drop_duplicates(subset=['고객ID'])
    analysis = bw_class.first_dash(rfm_clusters)
    rfm_clusters_final = analysis.get_final_dataframe()
    rfm_clusters_final.reset_index(drop=True, inplace=True)

    sorted_categories = sorted(rfm_clusters_final['고객분류'], key=lambda x: custom_sort[x])

    fig = px.pie(
        rfm_clusters_final,
        title="매출비율",
        values="매출",
        names="고객분류",
        template="plotly_white",
        hover_data=["매출 비율"],
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={"고객분류": sorted_categories},
        hole=0.5
    )

    fig.update_layout(legend=dict(x=-1, y=1.1), legend_orientation="v")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig

CLUSTER1 = dbc.Card(
    [
        dbc.CardHeader(html.H5("Comparison of each cluster")),
        dbc.CardBody(
            [
                dcc.Loading(
                    id="loading-cluster-comps1",
                    children=[
                        dbc.Alert(
                            "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                            id="cluster-alert1",
                            color="warning",
                            style={"display": "none"},
                        ),
                        dcc.Graph(id="cluster-comps1"),
                    ],
                    type="default",
                )
            ],
            style={"marginTop": 0, "marginBottom": 0},
        ),
    ]
)

CLUSTER2 = dbc.Card(
    [
        dbc.CardHeader(html.H5("Comparison of each cluster")),
        dbc.CardBody(
            [
                dcc.Loading(
                    id="loading-cluster-comps2",
                    children=[
                        dbc.Alert(
                            "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                            id="cluster-alert2",
                            color="warning",
                            style={"display": "none"},
                        ),
                        dcc.Graph(id="cluster-comps2"),
                    ],
                    type="default",
                )
            ],
            style={"marginTop": 0, "marginBottom": 0},
        ),
    ]
)

TOP_BIGRAM_COMPS = dbc.Card(
    [
        dbc.CardHeader(html.H5("월별 매출액")),
        dbc.CardBody(
            [
                dcc.Loading(
                    id="loading-bigrams-comps",
                    children=[
                        dbc.Alert(
                            "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                            id="bigrams-alert",  # ID 변경
                            color="warning",
                            style={"display": "none"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.P("Choose you want to know group:"), md=12),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            id="bigrams-comp_1",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in ["VIP고객","우수고객","관심고객","잠재고객","이탈고객"]
                                            ],
                                            value="VIP고객",
                                        )
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            id="bigrams-comp_2",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in ['매출','이탈률','마케팅비용','재구매율','ROI,ARPPU']
                                            ],
                                            value="매출",
                                        )
                                    ],
                                    md=6,
                                ),
                            ]
                        ),
                        dcc.Graph(id="bigrams-comps"),
                    ],
                    type="default",
                )
            ],
            style={"marginTop": 0, "marginBottom": 0},
        ),
    ]
)

CLUSTER3 = dbc.Card(
    [
        dbc.CardHeader(html.H5("Comparison of each cluster")),
        dbc.CardBody(
            [
                dcc.Loading(
                    id="loading-cluster3-comps",
                    children=[
                        dbc.Alert(
                            "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                            id="cluster-alert",
                            color="warning",
                            style={"display": "none"},
                        ),
                        dcc.Dropdown(
                            id="customer-type-dropdown",
                            options=[
                                {"label": "VIP고객", "value": "VIP고객"},
                                {"label": "관심고객", "value": "관심고객"},
                                {"label": "우수고객", "value": "우수고객"},
                                {"label": "이탈고객", "value": "이탈고객"},
                                {"label": "잠재고객", "value": "잠재고객"}
                            ],
                            value="VIP고객",
                            clearable=False,
                            style={"marginBottom": 50, "font-size": 12},
                        ),
                        dcc.Graph(id="customer-cluster-graph"),
                        dcc.Graph(id="monetary-cluster-graph"),
                    ],
                    type="default",
                )
            ],
            style={"marginTop": 0, "marginBottom": 0},
        ),
    ]
)

def make_options_coupon_drop(coupon_states):
    return [{'label': state, 'value': state} for state in coupon_states]


LEFT_COLUMN = dbc.Card(
    [
        html.H4(children="Select Month & Coupon", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select Month", className="lead"),
        dcc.Slider(
            id="month-selection-slider",
            min=1,
            max=12,
            step=1,
            marks={i: f"{i}월" for i in range(1, 13)},
            value=1,  # 기본값을 1로 설정
        ),
        html.Label("Select a coupon status", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can select a coupon status)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="coupon-drop", 
            options=make_options_coupon_drop(["Not Used","Clicked","Used"]), 
            clearable=False, 
            value="Clicked",  # 초기값 설정
            style={"marginBottom": 50, "font-size": 12}
        ),
    ]
)

BAR_PLOT = [
    dbc.CardHeader(html.H5("제품별 매출액")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bar-plot",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-bar",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="bar-plot"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

def make_options_customer_type(customer_types):
    return [{'label': ctype, 'value': ctype} for ctype in customer_types]

TREEMAP = dbc.Card(
    [
        dbc.CardHeader(html.H4("쿠폰별 매출액")),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id='customer-type-dropdown1',
                                options=make_options_customer_type(["VIP고객","우수고객","관심고객","잠재고객","이탈고객"]),
                                value="VIP고객",  # 초기값 설정
                                clearable=False,
                            ),
                            width=3,
                        ),
                    ],
                    style={"marginTop": 30},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="bank-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Bar plot",
                                        children=[
                                            dcc.Loading(
                                                id="loading-bar-plot",
                                                children=[
                                                    dcc.Graph(id="bar-chart")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                            width=12,
                        ),
                    ],
                    style={"marginTop": 30},
                ),
            ]
        ),
    ]
)



# %%
BODY = dbc.Container(
    [
        dbc.Row(
            [   
                dbc.Col(CLUSTER1, align="center"),
                dbc.Col(CLUSTER2, align="center")
            ],
            
            style={"marginTop": 30}
        ),
        dbc.Row(
        [
            dbc.Col(TOP_BIGRAM_COMPS, align="center"),  
        ]
        ),
        dbc.Row([dbc.Col(CLUSTER3),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(BAR_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        TREEMAP
                    ]
                ),
            ],
            style={"marginTop": 30},
        ),
        
        
        
    ],
    className="mt-12",
)
bw_layout = html.Div(children=[NAVBAR, BODY])

## hi layout
hi_layout = html.Div(
    style={'overflow-x': 'hidden'},
    children=[
        html.Div(
            style={'textAlign': 'center', 'padding': '20px'},
            children=[
                html.H1('Marketing Cost Prediction Dashboard', style={'font-family': 'IntegralCF-ExtraBold', 'color': 'slategray'}),
                dcc.Tabs(id='tabs', value='overall', children=[
                    dcc.Tab(label='Overall Data', value='overall'),
                    dcc.Tab(label='VIP 고객', value='cluster-0'),
                    dcc.Tab(label='관심 고객', value='cluster-1'),
                    dcc.Tab(label='우수 고객', value='cluster-2'),
                    dcc.Tab(label='이탈 고객', value='cluster-3'),
                    dcc.Tab(label='잠재 고객', value='cluster-4'),
                ]),
                dcc.Graph(id='marketing-cost-graph')
            ]
        )
    ]
)


@dash_app.callback(
    Output("page-content", "children"), 
    [Input("url", "pathname"), Input('interval-component', 'n_intervals')]
)
@login_required
def render_page_content(pathname, n_intervals):
    if pathname == "/dashboard/":
        return html.Div([
            html.P("홈페이지"),
            html.Div([
                html.H1("Ask me Anything"),
                dcc.Textarea(id='input-box', value='무엇을 도와드릴까요?', style={'width': '100%', 'height': 200}),
                html.Button('전송', id='button', n_clicks=0),
                html.Div(id='output-container-button', children=[]),
            ])
        ])
    elif pathname == "/dashboard/page-1":
        return html.Div([
            dcc.Location(pathname="/dashboard/page-1", id="redirect-page-1"),
            dh_layout
        ])
    elif pathname == "/dashboard/page-2":
        return html.Div([
            dcc.Location(pathname="/dashboard/page-2", id="redirect-page-2"),
            bw_layout
        ])
    elif pathname == "/dashboard/page-3":
        return html.Div([
            dcc.Location(pathname="/dashboard/page-3", id="redirect-page-3"),
            hi_layout
        ])
    elif pathname == "/dashboard/page-4":
        return html.Div([
            dcc.Location(pathname="/dashboard/page-4", id="redirect-page-4"),
            html.P("소영 파트")
        ])
    elif pathname == "/dashboard/page-5":
        return html.Div([
            dcc.Location(pathname="/dashboard/page-5", id="redirect-page-5"),
            mj_layout
        ])
    return dcc.Location(pathname="/login", id="redirect-login")
# def render_page_content(pathname, n_intervals):
#     if pathname == "/dashboard/":
#         return html.Div([
#             html.P("홈페이지"),
#             html.Div([
#                 html.H1("Ask me Anything"),
#                 dcc.Textarea(id='input-box', value='무엇을 도와드릴까요?', style={'width': '100%', 'height': 200}),
#                 html.Button('전송', id='button', n_clicks=0),
#                 html.Div(id='output-container-button', children=[]),
#             ])
#         ])
#     elif pathname == "/dashboard/page-1":
#         return dh_layout 
#     elif pathname == "/dashboard/page-2":
#         return html.P("페이지 2")
#     elif pathname == "/dashboard/page-3":
#         return html.P("페이지 3")
#     elif pathname == "/dashboard/page-4":
#         return html.P("ee")
#     elif pathname == "/dashboard/page-5":
#         return mj_layout
#     return dcc.Location(pathname="/login", id="redirect-login")

@dash_app.callback(Output("redirect-login", "pathname"), [Input("url", "pathname")])
def logout_on_click(pathname):
    if pathname == "/logout":
        logout_user()
        return "/login"
    return pathname

tools = [
    {
        "type": "function",
        "function": {
            "name": "force_category",
            "description": "Use this function to (1) Capture which category does the user's intention belongs to. (2) Redirect to the page which belongs to that category",
            "parameters": {
                "type": "object",
                "properties": {
                    "intention": {
                        "type": "string",
                        "description": "Intention(Category). Possible category is 6. 고객 이탈률, 재무, 마케팅, 재구매 및 기존고객 관리, 성과지표(ROI, APPRU), 해당없음. After the Category is fixed, redirect to the belonging category."
                    }
                },
                "required": ["intention"]
            }
        }
    }
]

@dash_app.callback(
    Output('output-container-button', 'children'),
    [Input('button', 'n_clicks')],
    [State('input-box', 'value')]
)
@login_required
def update_output(n_clicks, input_value):
    if n_clicks > 0:
        messages = [{"role": "system", "content": "You infer about which category(intention) the user wants to know about. Then, you use tool to redirect the user with user's intention."}]
        messages.append({"role": "user", "content": input_value})
        response = chat_completion_request(messages, tools=tools)
        if 'conversation_stack' not in globals():
            global conversation_stack
            conversation_stack = []
        for choice in response.choices:
            if choice.message.role == "user":
                conversation_stack.append(html.P(choice.message.content))
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    function_arguments = json.loads(tool_call.function.arguments)
                    if function_name == "force_category":
                        category = function_arguments["intention"]
                        if category == "고객 이탈률":
                            return dcc.Location(pathname="/dashboard/page-1", id="redirect-page-1")
                        elif category == "재무":
                            return dcc.Location(pathname="/dashboard/page-2", id="redirect-page-2")
                        elif category == "마케팅":
                            return dcc.Location(pathname="/dashboard/page-3", id="redirect-page-3")
                        elif category == "재구매 및 기존고객 관리":
                            return dcc.Location(pathname="/dashboard/page-4", id="redirect-page-4")
                        elif category == "성과지표 (ARPPU)":
                            return dcc.Location(pathname="/dashboard/page-5", id="redirect-page-5")
                        else:
                            return html.P("요구사항이 명확하지 않습니다. 원하시는 사안을 더 자세하게 말씀해주세요.")
    else:
        return html.P("무엇을 도와드릴까요?")


# layout callback for each role
## mj
@dash_app.callback(
    [Output('arppu-graph', 'figure'),Output('apriori-results', 'children')],
    Input('arppu-select', 'value')
)

def update_arppu_chart(selected_analysis):
    df = database_bw.making_dataframe_train_db('train_table')
    df = mj_class.mj_preprocessing(df)
    df.apply_my_function()
    df = df.return_dataframe()

    processor = RFMProcessor(df)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    rfm = processor.predict(df)

    viz = mj_class.mj_visualization(df, rfm)

    apriori_analyzer = mj_class.mj_apriori(min_support=0.6, min_confidence=0.005, min_lift=1, top_n=5)
    apriori_results = apriori_analyzer.apriori_analysis(df)

    if selected_analysis == 'cluster':
        graph_figure = viz.cluster_calculate_and_plot_arppu()
    elif selected_analysis == 'monthly':
        graph_figure = viz.month_calculate_and_plot_arppu()
    elif selected_analysis == 'area':
        graph_figure = viz.area_calculate_and_plot_arppu()
    elif selected_analysis == 'subscription':
        graph_figure = viz.calculate_and_plot_arppu_by_subscription_period_grouped()
    elif selected_analysis == 'area(map)':
        graph_figure = viz.area_calculate_and_plot_mapbox()

    apriori_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in apriori_results.columns],
        data=apriori_results.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'height': 'auto',
            'minWidth': '0px', 'maxWidth': '180px',
            'whiteSpace': 'normal'
        }
    )

    return graph_figure, apriori_table
## dh
@dash_app.callback(
    [Output('total_customers', 'children'),
     Output('churned_customers', 'children'),
     Output('churned_ratio', 'children'),
     Output('gender_pie_chart', 'figure'),
     Output('location_pie_chart', 'figure'),
     Output('mapbox_chart', 'figure'),
     Output('purchase_amount_chart', 'figure'),
     Output('category_chart', 'figure'),
     Output('month_chart', 'figure'),
     Output('customer-table', 'children')],
    [Input('row-dropdown', 'value')]
)
def update_dashboard(num_rows):
    
    new_df = database_bw.making_dataframe_train_db("train_table")
    
    prep = dc.dh_preprocessing(new_df)
    prep.apply_my_function()
    new_X, _ = prep.return_X_y()
    
    pipe = pkl.load(open("dh_pipeline.pkl", "rb"))
    cp = dc.churn_prediction()
    cp.pipeline = pipe
    cp.predict_churn_rate(new_X, pipe)
    result_table = cp.to_result_table(new_X)
    
    viz = dc.dh_visualization(result_table)
    
    display_df = viz.result_table.head(num_rows)

    total_customers = viz.caculate_churned_ratio()["total_customer"]
    churned_customers = viz.caculate_churned_ratio()["churned_customer"]
    churned_ratio = f"{viz.caculate_churned_ratio()['churned_ratio']}%"

    gender_pie_chart = viz.plot_gender_ratio()
    location_pie_chart = viz.plot_location_ratio()
    mapbox_chart = viz.plot_mapbox()
    purchase_amount_chart = viz.plot_purchase_amount()
    category_chart = viz.plot_category()
    month_chart = viz.plot_month()

    table_data = {"head": display_df.columns.tolist(), "body": display_df.values.tolist()}

    return total_customers, churned_customers, churned_ratio, gender_pie_chart, location_pie_chart, mapbox_chart, purchase_amount_chart, category_chart, month_chart, table_data

## bw

@dash_app.callback(
    Output('cluster-comps1', 'figure'),
    Input('cluster-comps1', 'id')
)
def update_cluster_comps1(_):
    return donut_chart1()

@dash_app.callback(
    Output('cluster-comps2', 'figure'),
    Input('cluster-comps2', 'id')
)
def update_cluster_comps2(_):
    return donut_chart2()



@dash_app.callback(
    Output("bigrams-comps", "figure"),
    [Input("bigrams-comp_1", "value"), Input("bigrams-comp_2", "value")],
)

def comp_bigram_comparisons(comp2, comp1):
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_bw = bw_df.merge(cluster, on = '고객ID', how = 'left')
    joined_df = train_bw[['고객ID',	'거래ID',	'거래날짜',	'제품ID',	'제품카테고리',	'수량',	'평균금액',	'배송료',	'쿠폰상태',	'성별',	'고객지역',	'가입기간',	'월',	'쿠폰코드',	'할인율',	'GST',	'고객소비액',	'매출',	'고객분류']]
    temp_df = joined_df.groupby(['월', '고객분류'])[comp1].sum().reset_index()
    temp_df = temp_df.loc[temp_df.고객분류 == comp2, ]

    # 숫자형 월을 문자형으로 변경
    temp_df['월'] = temp_df['월'].apply(lambda x: str(x) + '월')

    fig = px.bar(
        temp_df,
        title="Comparison: " + comp2 + " | " + comp1,
        x="월",
        y="매출",
        color="월",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Plotly,  # 도넛 차트와 같은 색상 팔레트 적용
        labels={"월": "Month:", "매출": "Revenue"},
        hover_data=[],
    )
    fig.update_layout(legend=dict(x=0.1, y=1.1), legend_orientation="h")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig
comp_bigram_comparisons("VIP고객", "매출")



@dash_app.callback(
    [Output("customer-cluster-graph", "figure"),
     Output("monetary-cluster-graph", "figure")],
    [Input("customer-type-dropdown", "value")]
)   



def update_graphs(customer_type):
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_bw = bw_df.merge(cluster, on = '고객ID', how = 'left')

    clustered_summary = bw_class.thrid_dash(train_bw)
    clustered_summary.create_clustered_summary()

    monthly_clustered_customers = clustered_summary.get_monthly_clustered_customers()
    monthly_clustered_monetary = clustered_summary.get_monthly_clustered_monetary()

    monthly_clustered_customers.reset_index(drop=True, inplace=True)
    monthly_clustered_monetary.reset_index(drop=True, inplace=True)

    fig_customers = px.line(
        monthly_clustered_customers,
        x="month",
        y=customer_type,
        title=f"{customer_type} 고객의 월별 거래량",
        labels={"month": "월", customer_type: "거래량"},
    )
    fig_customers.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='black'),  
    title_font_color="navy",  
    xaxis=dict(linecolor='gray'), 
    yaxis=dict(linecolor='gray', visible=True),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor="gray", borderwidth=1),  # 범례 배경색을 투명으로 설정하고 테두리 색상과 너비 설정
    )
    
    
    fig_monetary = px.line(
        monthly_clustered_monetary,
        x="month",
        y=customer_type,
        title=f"{customer_type} 고객의 월별 거래금액",
        labels={"month": "월", customer_type: "거래금액"},
    )
    
    fig_monetary.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='black'),  
    title_font_color="navy",  
    xaxis=dict(linecolor='gray'), 
    yaxis=dict(linecolor='gray', visible=True),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor="gray", borderwidth=1),  # 범례 배경색을 투명으로 설정하고 테두리 색상과 너비 설정
    )
    
    return fig_customers, fig_monetary



@dash_app.callback(
    Output('bar-plot', 'figure'),
    [
        Input('month-selection-slider', 'value'),
        Input('coupon-drop', 'value')
    ]
)
     
    
def update_bar_plot(selected_month, selected_coupon):
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_bw = bw_df.merge(cluster, on = '고객ID', how = 'left')
    grouped_df_processor = bw_class.fourth_dash(train_bw)
    grouped_df_processor.preprocess()
    grouped_df_processor.group_by_columns(['제품카테고리', '월', '쿠폰상태'], '매출')
    grouped_df = grouped_df_processor.get_grouped_df()
    grouped_df.reset_index(drop=True, inplace=True)
    df = grouped_df

    filtered_df = df[(df['월'] == selected_month) & (df['쿠폰상태'] == selected_coupon)]
    
    if filtered_df.empty:
        fig = px.bar(title="No data available", labels={"제품카테고리": "제품 카테고리", "매출": "매출액"})
    else:
        fig = px.bar(filtered_df, x='제품카테고리', y='매출', color='제품카테고리', title="제품별 매출액", labels={"제품카테고리": "제품 카테고리", "매출": "매출액"})

    # 그래프 꾸미기
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # 배경색을 투명으로 설정
        paper_bgcolor='rgba(0,0,0,0)',  # 그래프 영역의 배경색을 투명으로 설정
        font=dict(color='black'),  # 폰트 색상을 검은색으로 설정
        title_font_color="navy",  # 제목 색상을 navy로 설정
        xaxis=dict(linecolor='gray'),  # x축 라인 색상 설정
        yaxis=dict(linecolor='gray'),  # y축 라인 색상 설정
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor="gray", borderwidth=1),  # 범례 배경색을 투명으로 설정하고 테두리 색상과 너비 설정
    )

    return fig

@dash_app.callback(
    Output('bar-chart', 'figure'),
    [Input('customer-type-dropdown1', 'value')]
)

def update_bar_chart(customer_type):
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_bw = bw_df.merge(cluster, on = '고객ID', how = 'left')
    coupon_sales_processor = bw_class.fifth_dash(train_bw)
    coupon_sales_processor.preprocess()
    coupon_sales_processor.calculate_coupon_sales()

    coupon_sales = coupon_sales_processor.get_coupon_sales()
    coupon_sales.reset_index(drop=True, inplace=True)

    filtered_df = coupon_sales[coupon_sales['고객분류'] == customer_type]
    filtered_df.drop(columns=['쿠폰코드'], inplace=True)
    bar_fig = px.bar(filtered_df, x='제품카테고리', y='매출',
                     title=f"{customer_type} 고객의 제품카테고리별 매출액",
                     color='제품카테고리',
                     color_continuous_scale=px.colors.sequential.Reds)
    bar_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        title_font_color="navy",
        xaxis=dict(linecolor='gray', title_font=dict(size=14)),
        yaxis=dict(linecolor='gray', title_font=dict(size=14)),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor="gray", borderwidth=1)
    )
    return bar_fig


@dash_app.callback(
    Output('bank-treemap', 'figure'),
    [Input('customer-type-dropdown1', 'value')]
)



def update_treemap(customer_type):
    train = bw_database.making_dataframe_train_db('train_table')
    bw = bw_class.bw_preprocessing(train)
    bw.apply_my_function()
    bw_df = bw.return_dataframe()
    processor = bw_class.RFMProcessor(train)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor = bw_class.RFMProcessor(train) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    new_data_predictions = processor.predict(train)
    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_bw = bw_df.merge(cluster, on = '고객ID', how = 'left')
    coupon_sales_processor = bw_class.fifth_dash(train_bw)
    coupon_sales_processor.preprocess()
    coupon_sales_processor.calculate_coupon_sales()

    coupon_sales = coupon_sales_processor.get_coupon_sales()
    coupon_sales.reset_index(drop=True, inplace=True)

    filtered_df = coupon_sales[coupon_sales['고객분류'] == customer_type]
    fig = px.treemap(filtered_df, path=['쿠폰코드', '제품카테고리'], values='매출', title=f"{customer_type} 고객의 쿠폰별 매출액",
                     color='매출', color_continuous_scale=px.colors.sequential.Reds)
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        title_font_color="navy",
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor="gray", borderwidth=1)
    )
    return fig

## hi callback

@dash_app.callback(
    Output('marketing-cost-graph', 'figure'),
    Input('tabs', 'value')
)
def update_graph(tab_value):
    # Load data and perform RFM clustering
    df = hi_database.db_to_df(db_name="TRAIN.DB", table_name="train_table")
    processor = bw_class.RFMProcessor(df)
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=5)
    new_data_predictions = processor.predict(df)

    cluster_data = bw_class.mapping_cluster(new_data_predictions)
    cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
    train_hi = df.merge(cluster, on='고객ID', how='left')

    # Auto ARIMA processing
    hi_pipeline = hi_class.AutoArimaPipeline()
    df_arima = hi_pipeline.hi_preprocessing(train_hi)
    out = hi_pipeline.split_data_by_cluster(df_arima)
    train0, test0 = hi_pipeline.create_train_test_by_cluster(train_hi, out)

    hi_pipeline.fit(train0)
    predictions = hi_pipeline.predict(test0)

    # Visualizer
    visualizer = hi_class.hi_visualizer(data=train0, clusters=predictions)

    if tab_value == "overall":
        figure = visualizer.visualize_overall()
    else:
        cluster_number = int(tab_value.split('-')[-1])
        figure = visualizer.visualize_cluster(cluster_number)

    return figure

if __name__ == "__main__":
    thread = threading.Thread(target=run_shoot_row)
    thread.start()
    
    # serve(app, host='0.0.0.0', port=8888)
    
    flask_app.run(port=8888, debug=False)
