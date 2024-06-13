# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
import pickle as pkl
import sqlite3
import database_bw as db
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
import plotly.express as px

warnings.filterwarnings("ignore")

# 로그 데이터를 Random Survival Forest에 맞는 데이터로 변환
class dh_preprocessing:
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df["거래날짜"] = pd.to_datetime(self.df["거래날짜"])
        self.df = self.df.drop(["IND", "거래ID", "제품ID", "배송료", "월", "가입기간", "쿠폰코드", 
                                "할인율", "GST", "오프라인비용", "온라인비용", "마케팅비용"], axis = 1)
        
    def generate_new_features(self):
        
        # 고객 별로 기존 변수 추출
        self.gender = self.df.groupby("고객ID")["성별"].first()
        self.location = self.df.groupby("고객ID")["고객지역"].first()
        
        ## 새로운 변수 만들기
        # 고객 별 평균 금액(구매 건수 대비)
        self.mean_price = self.df.groupby("고객ID")["평균금액"].mean()
        # 고객 별 평균 구매 수량
        self.mean_quantity = self.df.groupby("고객ID")["수량"].mean()
        # 고객 별 가장 많이 구매한 제품 카테고리
        self.most_purchased_categories = self.df.groupby("고객ID")["제품카테고리"]\
            .value_counts().unstack(fill_value = 0).idxmax(axis = 1)
        # 고객 별 월 구매 횟수 
        self.months = self.df.groupby(["고객ID", self.df["거래날짜"].dt.to_period("M")])\
            .size().unstack(fill_value = 0)
        # 고객 별 쿠폰 사용 여부
        self.coupon = self.df.groupby("고객ID")["쿠폰상태"].value_counts().unstack(fill_value = 0)\
            .rename(columns = {"Clicked" : "클릭함", "Not Used" : "사용 안함", "Used" : "사용함"})
    
    def generate_y(self):
        
        # 고객 별 이탈 여부 결정
        date = self.df["거래날짜"].max() + pd.Timedelta(1, "days")
        end_date = self.df.groupby("고객ID")["거래날짜"].max()
        start_date = self.df.groupby("고객ID")["거래날짜"].min()
        # 고객의 이탈 정보가 데이터에는 없기 때문에 데이터의 가장 최신 날짜에서
        # 고객 별 최신 거래 날짜의 차이가 90일 이상인 경우 이탈했다고 판단
        churned = (((date - end_date).dt.days >= 90).astype(int))
        time = ((end_date - start_date) + pd.Timedelta(1, "days")).dt.days
        # y 변수를 scikit-survival에서 요구하는 structured array로 변경
        self.y = Surv.from_arrays(churned, time, "이탈 여부", "생존 시간")
    
    def apply_my_function(self):
        
        self.generate_new_features()
        self.generate_y()
        
    def return_X_y(self):
        
        # 기존 변수와 새로운 변수를 결합한 데이터프레임 생성
        X = pd.DataFrame({"성별" : self.gender, "지역" : self.location, 
                          "평균 금액" : self.mean_price, "평균 수량" : self.mean_quantity, 
                          "선호 제품군" : self.most_purchased_categories})
        X = pd.concat([X, self.months, self.coupon], axis = 1)
        X.columns = X.columns.astype(str)
        
        return X, self.y

#########################################################################################
# # Note)
# 1. shoot_row() 함수에 의해 데이터가 test.db에서 train.db로 적재되는 상황
# 2. 초기 train.csv 파일에는 10월 31일까지의 데이터가 담겨 있음. 따라서 오늘은 11월 1일
# 3. 10월 31일까지의 데이터를 train 데이터로 해서 random survival forest 모델을 fitting
# 4. test.csv 파일에는 11월 1일부터 12월 31일까지의 데이터가 담겨 있음.
# 5. 이 데이터만 가지고 예측을 수행하려 하면 해당 기간에 구매 기록이 있는 고객들만 예측됨
# 6. 즉 1월 ~ 10월까지 구매 기록이 있으나, 11월 ~ 12월 구매 기록이 없는 고객들은 예측되지 않음
# 7. 따라서 test 데이터를 만들 때에는 1월부터 12월까지의 모든 구매 기록 정보를 합쳐야 됨
# # example
# train_df = pd.read_csv("./train.csv")
# test_df = pd.read_csv("./test.csv")
# merged = pd.concat([train_df, test_df], axis = 0) # 1월부터 12월까지의 모든 구매 기록 정보를 합침
# dh = dh_preprocessing(merged)
# dh.apply_my_function()
# X_test, y_test = dh.return_X_y()

# 고객별 이탈률 예측
class churn_prediction:
    
    def __init__(self):
        
        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(drop = "if_binary", sparse_output = False, handle_unknown = "ignore"))
            ]) 

        num_pipe = Pipeline([
            ("scaling", StandardScaler())
            ])

        trans = ColumnTransformer([
            ("cat", cat_pipe, make_column_selector(dtype_include = "object")),
            ("num", num_pipe, make_column_selector(dtype_include = ["int64", "float64"]))
            ])

        self.pipe = Pipeline([
            ("transform", trans),
            ("rsf", RandomSurvivalForest(random_state = 42))
            ])
        
    def fit(self, X, y):
        
        self.columns = X.columns.tolist()
        self.pipe.fit(X, y)
        
        return self
    
    # 고객 별로 n일 이후의 이탈률을 예측
    def predict_churn_rate(self, X_test, pipeline = None, n = 90):
        
        pipeline = self.pipe if pipeline is None else pipeline
        
        surv_funcs = pipeline.predict_survival_function(X_test)
        prob_predict = []
        
        for fn in surv_funcs:
            prob = (1 - fn(n)).round(4)
            prob_predict.append(prob)
            
        self.predict = np.array(prob_predict)
        
        return self.predict

    # def feature_importances(self, X_train, y_train):
        
    #     # 시간이 약간 걸림
    #     imp = permutation_importance(self.pipe, X_train, y_train, n_repeats = 5, random_state = 42)
    #     imp_table = pd.DataFrame(imp["importances_mean"], 
    #                              index = self.columns, columns = ["features"])

    #     return imp_table
    
    # 모델의 성능 평가 (cumulative/dynamic AUC, concordance index ipcw)
    def return_metrics(self, X_test, y_train, y_test, times = np.arange(7, 92, 7), tau = 90):
        
        chf_funcs = self.pipe.predict_cumulative_hazard_function(X_test)
        risk_scores = np.row_stack([chf(times) for chf in chf_funcs])
        risk_score_at_tau = [chf(tau) for chf in chf_funcs]
        self.mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)[1]
        self.cindex_ipcw = concordance_index_ipcw(y_train, y_test, risk_score_at_tau, tau)[0]
        
        return self.mean_auc, self.cindex_ipcw
    
    # pipeline을 pickle로 저장
    def to_pkl(self, pipeline_file_name = "dh_pipeline.pkl"):
        
        with open(pipeline_file_name, "wb") as f: 
            pkl.dump(self.pipe, f)
    
    # 최종 결과 테이블을 만들고 OUR_DATABASE.db에 적재 -> 대시보드에 출력 시 db에서 가져와 사용
    def to_result_table(self, X_test, save = False):
        
        self.result_table = pd.DataFrame(
            np.column_stack((X_test, self.predict)),
            columns = X_test.columns.tolist() + ["예측 이탈률"], 
            index = X_test.index
            )
        
        most_purchased_month = self.result_table.filter(like = "2019-").idxmax(axis = 1)
        self.result_table = self.result_table.loc[:, ~self.result_table.columns.str.startswith("2019-")]
        self.result_table.insert(5, "최다 구매 월", most_purchased_month)
        self.result_table = self.result_table.sort_values("예측 이탈률", ascending = False)
        
        if save == True:
            db.create_new_table(self.result_table, "churn_prediction_table")
            
        return self.result_table
    
# # example
# model = churn_prediction()
# model.fit(X_train, y_train)
# churn_predict = model.predict_churn_rate(X_test, 90)
# model.return_metrics(X_test, y_train, y_test)
# model.to_pkl()
# model.to_result_table(X_test)

# 대시보드 시각화 -> 글자 폰트 같은 디자인 요소들은 다른 대시보드와 통일할 예정
class dh_visualization:
    
    def __init__(self, result_table):
        
        self.result_table = result_table
    
    ## 대시보드에 출력할 정보와 그래프 만들기
    # 전체 고객수, 이탈 위험 고객 수, 이탈 위험 고객 비율 계산
    def caculate_churned_ratio(self, threshold = 0.5):
        
        # self.result_table = df
        total_customer = len(self.result_table)
        churned_customer = (self.result_table["예측 이탈률"] >= threshold).sum()
        churned_ratio = round(((churned_customer / total_customer) * 100), 1)
        
        return {
            "total_customer": total_customer,
            "churned_customer": churned_customer,
            "churned_ratio": churned_ratio
            }
        
    # 고객의 남,여 비율을 파이 차트로 시각화
    def plot_gender_ratio(self):
        
        gender_summary = self.result_table["성별"].value_counts().reset_index()
        gender_pie_plot = px.pie(gender_summary, names = "성별", 
                                 values = "count", hole = 0.5)
        gender_pie_plot.update_layout(legend_title = "성별")
        gender_pie_plot.update_traces(hovertemplate = "<b>%{label}</b><br>고객 수: %{value}<extra></extra>")

        return gender_pie_plot
    
    # 고객의 지역 비율을 파이 차트로 시각화
    def plot_location_ratio(self):
        
        self.location_summary = self.result_table["지역"].value_counts().reset_index()
        location_pie_plot = px.pie(self.location_summary, names = "지역", 
                                   labels = "지역", values = "count")
        location_pie_plot.update_layout(legend_title = "지역")
        location_pie_plot.update_traces(hovertemplate = "<b>%{label}</b><br>고객 수: %{value}<extra></extra>")
        
        return location_pie_plot
    
    # 각 지역 별 이탈 위험 고객 수를 mapbox로 시각화
    def plot_mapbox(self):
        
        coordinates = {
            'Chicago': {'lat': 41.8781, 'long': -87.6298},
            'California': {'lat': 36.7783, 'long': -119.4179},
            'New York': {'lat': 40.7128, 'long': -74.0060},
            'New Jersey': {'lat': 40.0583, 'long': -74.4057},
            'Washington DC': {'lat': 38.9072, 'long': -77.0369}
        }

        self.location_summary['lat'] = self.location_summary["지역"].map(lambda x: coordinates[x]['lat'])
        self.location_summary['long'] = self.location_summary["지역"].map(lambda x: coordinates[x]['long'])
        mapbox = px.scatter_mapbox(
            self.location_summary, lat = "lat", lon = "long", size = "count", color = "count", 
            hover_name = "지역", labels={"count": "이탈 위험 고객 수"}, 
            hover_data = {"lat": False, "long": False}, size_max = 40, zoom = 3,
            color_continuous_scale = px.colors.cyclical.IceFire, 
            mapbox_style = "carto-positron"
            )
        mapbox.update_layout(mapbox = dict(center = dict(lat = 37.0902, lon = -95.7129)))
        mapbox.update_traces(hovertemplate = "<b>%{hovertext}</b><br>이탈 위험 고객 수 : %{marker.size}<extra></extra>")

        return mapbox
    
    # 고객 별 평균 구매 금액을 히스토그램으로 시각화
    def plot_purchase_amount(self):
        
        hist_plot = px.histogram(self.result_table, x = "평균 금액")
        hist_plot.update_yaxes(title_text = "고객 수")
        hist_plot.update_traces(hovertemplate = "<b>%{x}</b><br>고객 수 : %{y}<extra></extra>")
        
        return hist_plot
    
    # 고객 별 선호 제품군을 막대 그래프로 시각화
    def plot_category(self):
        
        category_summary = self.result_table["선호 제품군"].value_counts().reset_index()
        bar_plot = px.bar(category_summary, x = "선호 제품군", y = "count")
        bar_plot.update_yaxes(title_text = "고객 수")
        bar_plot.update_traces(hovertemplate = "<b>%{x}</b><br>고객 수 : %{y}<extra></extra>")
        
        return bar_plot
    
    # 고객 별 최다 구매 월을 선 그래프로 시각화
    def plot_month(self):
        
        month_summary = self.result_table["최다 구매 월"].value_counts().sort_index().reset_index()
        line_plot = px.line(month_summary, x = "최다 구매 월", y = "count")
        line_plot.update_yaxes(title_text = "고객 수")
        line_plot.update_traces(hovertemplate = "<b>%{x}</b><br>고객 수 : %{y}<extra></extra>")
        
        return line_plot