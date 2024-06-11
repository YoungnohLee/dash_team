#!/usr/bin/env python
# coding: utf-8

# 코호트 분석 관련 class

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class CohortAnalysis:
    def __init__(self, total):
        self.total = total

    def calculate_cohort(self):
        # '거래날짜' 컬럼을 datetime 형식으로 변환
        self.total['거래날짜'] = pd.to_datetime(self.total['거래날짜'])

        # 고객별 처음거래날짜와 마지막거래날짜 계산
        result = self.total.groupby('고객ID').agg(
            처음거래날짜=('거래날짜', 'min'),
            마지막거래날짜=('거래날짜', 'max')
        ).reset_index()

        # 전체기간 계산
        result['전체기간'] = result['마지막거래날짜'] - result['처음거래날짜']

        # 재구매여부 계산
        result['재구매여부'] = result['전체기간'].apply(lambda x: 0 if x == pd.Timedelta(days=0) else 1)

        # 고객별 구매횟수 계산
        p_count = self.total.groupby('고객ID').agg(구매횟수=('거래날짜', 'count')).reset_index()
        result = pd.merge(result, p_count, how='left', on='고객ID')

        # 필요한 열만 추출하여 self.total에 병합
        result_subset = result[['고객ID', '전체기간', '재구매여부', '구매횟수']]
        self.total = pd.merge(self.total, result_subset, how='left', on='고객ID')

        # 처음거래날짜와 마지막거래날짜를 datetime 형식으로 변환
        self.total['처음거래날짜'] = pd.to_datetime(self.total['처음거래날짜'])
        self.total['마지막거래날짜'] = pd.to_datetime(self.total['마지막거래날짜'])

        # 최초구매_월과 구매_월 계산
        self.total['최초구매_월'] = self.total['처음거래날짜'].dt.to_period('M').dt.to_timestamp()
        self.total['구매_월'] = self.total['마지막거래날짜'].dt.to_period('M').dt.to_timestamp()

        # 코호트별 고객 수 계산
        cohort = self.total.groupby(['최초구매_월', '구매_월']).agg(n_customers=('고객ID', 'nunique')).reset_index()

        # 코호트 기간 계산
        cohort['코호트_기간'] = (cohort['구매_월'].dt.to_period('M') - cohort['최초구매_월'].dt.to_period('M')).apply(lambda x: x.n)

        # 코호트 크기 계산
        cohort_size = cohort[cohort['코호트_기간'] == 0][['최초구매_월', 'n_customers']].rename(columns={'n_customers': '코호트_크기'})
        cohort = pd.merge(cohort, cohort_size, on='최초구매_월')

        return cohort
        
    def calculate_retention_rate(self, cohort):
        cohort['재구매율'] = cohort['n_customers'] / cohort['코호트_크기']
        cohort['최초구매_월'] = cohort['최초구매_월'].dt.strftime('%Y-%m')

        retention_matrix = cohort.pivot_table(index='최초구매_월', columns='코호트_기간', values='재구매율')
        return retention_matrix




# 카테고리별 재구매 관련 class

import plotly.express as px

class CustomerCategoryAnalysis:
    def __init__(self, online_sales):
        self.online_sales = online_sales

    def calculate_repurchase_periods(self):
        self.online_sales['거래날짜'] = pd.to_datetime(self.online_sales['거래날짜'])
        
        customer_category_groups = self.online_sales.groupby(['고객ID', '제품카테고리'])
        category_repurchase_periods = {}
    
        for name, group in customer_category_groups:
            group = group.sort_values(by='거래날짜')
            purchase_gaps = group['거래날짜'].diff().dt.days
            purchase_gaps = purchase_gaps.iloc[1:]
            category = name[1]

            if category not in category_repurchase_periods:
                category_repurchase_periods[category] = []
            category_repurchase_periods[category].extend(purchase_gaps)

        average_repurchase_periods = {}
        for category, periods in category_repurchase_periods.items():
            average_repurchase_periods[category] = pd.Series(periods).mean()

        self.average_repurchase_periods = average_repurchase_periods

    def create_category_dataframe(self):
        # 재구매 주기 계산
        self.calculate_repurchase_periods()
        
        # 평균 재구매 주기 데이터프레임 생성
        result_df = pd.DataFrame(list(self.average_repurchase_periods.items()), columns=['제품카테고리', '평균 재구매 주기(일)'])

        # 재구매율 계산
        klk = self.online_sales.groupby(['고객ID', '거래날짜', '제품카테고리']).size().reset_index(name='n')
        klk = klk.groupby(['고객ID', '제품카테고리']).size().reset_index(name='n2')
        klk = klk[klk['n2'] != 1]
        klk['n2'] = klk['n2'] - 1

        ka = klk.groupby('제품카테고리').agg(
            n=('고객ID', 'size'),
            sum=('n2', 'sum')
        ).reset_index()
        ka['재구매율'] = ka['n'] / ka['sum']

        # ka와 result_df 병합
        category = pd.merge(ka, result_df, on='제품카테고리')
        
        return category

    def visualize_repurchase_periods(self, x_col, category_df):
        fig = px.bar(category_df, 
                     x=x_col,
                     y='제품카테고리',  
                     title=f'제품 카테고리별 {x_col}',
                     color='제품카테고리',  
                     orientation='h',
                     text=category_df['재구매율'].map(lambda x: f'{x:.3f}'))  
        
        return fig


# 재구매 여부 예측 관련 class

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle as pkl
import sqlite3

class RebuyPredictionModel:
    def __init__(self, data, target_column):
        self.data = data[['제품카테고리', '평균금액', '배송료', '쿠폰상태', '성별', '고객지역', '가입기간',  '할인율', '마케팅비용', '고객소비액', '매출', '재방문여부', '고객분류']]
        self.target_column = target_column
        self.pipeline = None
        self.numeric_features = None
        self.categorical_features = None

    def train_test_split(self, test_size=0.3):
        train, test = train_test_split(self.data, test_size=test_size)
        return train, test

    def create_pipeline(self, train):
        X_cols = train.drop(columns=[self.target_column]).columns
        self.numeric_features = train[X_cols].select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = train[X_cols].select_dtypes(exclude=np.number).columns.tolist()

        trans = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(), self.categorical_features)
            ])
        
        pipeline = Pipeline(steps=[
            ('transform', trans),
            ('model', LogisticRegression())
        ])
        
        return pipeline

    def run_pipeline(self, train):
        self.pipeline = self.create_pipeline(train)
        X_train = train.drop(columns=[self.target_column])
        y_train = train[self.target_column]
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def save_model_and_transformation(self, model_file="model.pkl", trans_file="transform.pkl"):
        with open(model_file, "wb") as f:
            pkl.dump(self.pipeline.named_steps['model'], f)
        with open(trans_file, "wb") as f:
            pkl.dump(self.pipeline.named_steps['transform'], f)

    def evaluate_model(self, test):
        X_test = test.drop(columns=[self.target_column])
        y_test = test[self.target_column]
        pred = self.pipeline.predict(X_test)
        accuracy = np.mean(pred == y_test)
        confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
        
        print("Accuracy for test set:", accuracy)
        print("Confusion matrix for test set:")
        print(confusion_matrix)
        return accuracy, confusion_matrix

    def get_feature_names(self):
        num_features = self.numeric_features
        cat_features = self.pipeline.named_steps['transform'].named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        return list(num_features) + list(cat_features)
     
    def inference(self, test_data):
        model = pkl.load(open("model.pkl","rb"))
        trans = pkl.load(open("transform.pkl","rb"))

        ## x variables preprocessing 
        X_cols = test_data.loc[:, [i for i in list(test_data.columns) if i not in ['재방문여부']]].columns
        X_test = trans.transform(test_data[X_cols])

        ## make prediction for testset
        pred = model.predict(X_test)
        
        return pred
    
    def save_predictions_to_db(self, test_data, pred, db_file="sy_db.db", table_name="predict"):
        # 데이터베이스 연결
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # test_data와 pred를 하나의 데이터프레임으로 결합
        results_df = test_data.copy()
        results_df['prediction'] = pred

        # 결과 데이터프레임을 데이터베이스에 저장
        results_df.to_sql(table_name, conn, if_exists='replace', index=False)

        # 연결 닫기
        conn.commit()
        conn.close()

        print(f"Predictions saved to {table_name} table in {db_file}")


