# -*- coding: utf-8 -*-
import dash
from dash import Dash, html, dcc, Input, Output
import dash_mantine_components as dmc
import pandas as pd
import dh_class as dc
import database_bw as db
from app import app

dash._dash_renderer._set_react_version("18.2.0")

layout = dmc.MantineProvider(
    children=[
        dmc.Container(
            fluid=True,
            style={"padding": "20px"},
            children=[
                html.H1("고객 이탈률 예측", style={"textAlign": "center"}),
                html.H2("Today", style={"textAlign": "center"}),
                html.H2(id = "last_transaction_date", style = {"textAlign": "center"}),
                dmc.Group(
                    gap="lg",
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
                    cols=2,
                    spacing="lg",
                    mb="lg",
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
                    withBorder=True,
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
                            data={"head": [], "body": []},
                            striped=True,
                            highlightOnHover=True,
                            withTableBorder=True,
                            withColumnBorders=True
                        )
                    ]
                )
            ]
        )
    ]
)

@app.callback(
    [Output('total_customers', 'children'),
     Output('churned_customers', 'children'),
     Output('churned_ratio', 'children'),
     Output('gender_pie_chart', 'figure'),
     Output('location_pie_chart', 'figure'),
     Output('mapbox_chart', 'figure'),
     Output('purchase_amount_chart', 'figure'),
     Output('category_chart', 'figure'),
     Output('month_chart', 'figure'),
     Output('customer-table', 'data')],
    [Input('row-dropdown', 'value')]
)
# check_transaction_date() 함수를 이용해서 자동으로 날짜 변화가 감지되면,
# 대시보드가 업데이트되어야 되는데 로직 구현이 잘 안됩니다.
# 그래서 수정이 필요할 것 같아요.
def update_dashboard(num_rows):

    viz = dc.dh_visualization()
    
    if num_rows == "모두 표시":
        display_df = viz.result_table
    else:
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



#  http://127.0.0.1:8050/

# step 1. train.db에 데이터가 shoot_row() 함수에 의해 적재됨
# step 2. 데이터가 적재될 때마다 check_transaction_date() 함수가 하루가 지났는지 여부 판단
# step 3-1. 하루가 지나지 않았을 경우, 현 상태 유지
# step 3-2. 하루가 지났을 경우, train.db에 지금까지 쌓인 데이터를 result_table로 변환하고, OUR_DATABASE의 churn_prediction_table로 업데이트
# step 4. 대시보드 또한 업데이트한 churn_prediction_table의 정보를 반영하여 업데이트
 

