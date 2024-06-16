# -*- coding: utf-8 -*-
import dash
from dash import Dash, html, dcc, Input, Output
import dash_mantine_components as dmc
import pandas as pd
import pickle as pkl
import dh_class as dc
import database_bw 

dh_layout = dmc.Container(
    fluid=True,
    style={"padding": "20px"},
    children=[
        html.H1("고객 이탈률 예측", style={"textAlign": "center", "marginBottom": "20px"}),
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
            style={"marginBottom": "20px"},
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
            style={"marginBottom": "20px"},
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
            style={"border": "1px solid #ddd", "marginBottom": "20px"},
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
                    striped=True,
                    highlightOnHover=True,
                    withBorder=True,
                    withColumnBorders=True
                )
            ]
        )
    ]
)

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

    table_header = [html.Thead(html.Tr([html.Th(col) for col in display_df.columns]))]
    table_body = [html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in display_df.values])]

    table_children = table_header + table_body

    return total_customers, churned_customers, churned_ratio, gender_pie_chart, location_pie_chart, mapbox_chart, purchase_amount_chart, category_chart, month_chart, table_children