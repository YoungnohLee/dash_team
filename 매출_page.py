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
from dash import dcc
from dash import html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
import bw_database
import bw_class
# import bw_preprocessing
# import bw_dataframe

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

# %%
from app import app

layout = html.Div(children=[NAVBAR, BODY])

@app.callback(
    Output('cluster-comps1', 'figure'),
    Input('cluster-comps1', 'id')
)
def update_cluster_comps1(_):
    return donut_chart1()

@app.callback(
    Output('cluster-comps2', 'figure'),
    Input('cluster-comps2', 'id')
)
def update_cluster_comps2(_):
    return donut_chart2()



@app.callback(
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



@app.callback(
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



@app.callback(
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

@app.callback(
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


@app.callback(
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


