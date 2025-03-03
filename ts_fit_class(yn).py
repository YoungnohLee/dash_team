import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pmdarima as pm
import plotly.graph_objs as go
from scipy.spatial.distance import jensenshannon 




def data_preprocessing(marketing_info, onlinesales_info, customer_info, discount_info):
  # 마케팅비용 = 오프라인비용 + 온라인비용
  marketing_info['마케팅비용'] = marketing_info['오프라인비용'] + marketing_info['온라인비용']

  # 총구매금액 = 수량 * 평균금액
  onlinesales_info['총구매금액'] = onlinesales_info['수량'] * onlinesales_info['평균금액']

  # 쿠폰상태 1: 사용, 0: 나머지
  onlinesales_info['쿠폰상태'] = onlinesales_info['쿠폰상태'].map({'Used': 1, 'Not Used': 0, 'Clicked': 0})

  # 고객지역: Chicago = 1, California = 2, New York = 3, New Jersey = 4, Washington DC = 5
  region_mapping = {'Chicago': 1, 'California': 2, 'New York': 3, 'New Jersey': 4, 'Washington DC': 5}
  customer_info['고객지역'] = customer_info['고객지역'].map(region_mapping)

  # 남자 = 1, 여자 = 0
  customer_info['성별'] = customer_info['성별'].map({'남': 1, '여': 0})

  # 고객정보, 할인정보, 마케팅정보, 온라인판매정보 병합
  data = pd.merge(customer_info, onlinesales_info, on='고객ID')

  marketing_info = marketing_info.rename(columns = {"날짜" : "거래날짜"})
  data = pd.merge(data, marketing_info, on='거래날짜')

  data['월'] = pd.to_datetime(data['거래날짜']).dt.month
  month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
  discount_info['월'] = discount_info['월'].map(month_mapping)
  data = pd.merge(data, discount_info, on=['월', '제품카테고리'])
  data = data.drop(['월', '쿠폰코드'], axis = 1)

  # 필요없는 문자 제거
  data['고객ID'] = data['고객ID'].str.replace("USER_", "")
  data['거래ID'] = data['거래ID'].str.replace("Transaction_", "")
  data['제품ID'] = data['제품ID'].str.replace("Product_", "")

  # integer to string
  data = data.astype({"성별": 'str', "고객지역": 'str', "쿠폰상태": 'str', "할인율": 'str'})

  return data


### js-div ###
def check_distribution_and_retrain(training_data, realtime_data, model, feature='마케팅비용', threshold=0.1):
    """

    """
    hist_train, bins = np.histogram(training_data[feature], bins=30, density=True)
    hist_realtime, _ = np.histogram(realtime_data[feature], bins=bins, density=True)
    
    eps = 1e-10
    hist_train = hist_train + eps
    hist_realtime = hist_realtime + eps
    
    hist_train = hist_train / hist_train.sum()
    hist_realtime = hist_realtime / hist_realtime.sum()
    
    jsd = jensenshannon(hist_train, hist_realtime)
    print(f"Feature '{feature}''s js-div': {jsd:.4f}")
    
    # Retraining Rule
    if jsd > threshold:
        print("Distributional Change Detected in the real-time data. Automatically Re-train the model.")
        if isinstance(model, AutoArimaPipeline):
            
            model.fit([realtime_data])
        else:
            model.fit(realtime_data)
        print("Model Re-training Finished.")
    else:
        print("No Distributional Change Detected.")


class numeric_filtering(BaseEstimator, TransformerMixin):
    def __init__(self, check_const_col=True, check_id_col=True):
        self.check_const_col = check_const_col
        self.check_id_col = check_id_col

    def fit(self, X, y=None):
        if self.check_const_col:
            self.constant_col = [i for i in range(X.shape[1]) if X[:,i].std() == 0]
        else:
            self.constant_col = []

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(np.diff(X[:,i]))) == 1]
        else:
            self.id_col = []

        self.rm_cols = self.constant_col + self.id_col
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        return X[:, self.final_cols]

class categorical_filtering(BaseEstimator, TransformerMixin):
    def __init__(self, check_const_col=True, check_id_col=True, check_cardinality=True):
        self.check_const_col = check_const_col
        self.check_id_col = check_id_col
        self.check_cardinality = check_cardinality

    def fit(self, X, y=None):
        if self.check_const_col:
            self.constant_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) == 1]
        else:
            self.constant_col = []

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) == X.shape[0]]
        else:
            self.id_col = []

        if self.check_cardinality:
            self.cardinality = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) > 50]
        else:
            self.cardinality = []

        self.rm_cols = self.constant_col + self.id_col + self.cardinality
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        return X[:, self.final_cols]



class RFMClusteringPipeline:
    def __init__(self):
        self.pipeline = self.create_pipeline()
        self.rfm_scaled = None
        self.optimal_clusters = None
        self.kmeans_model = None


    def rfm_data(self, data):
        # 거래날짜를 datetime 형태로 변환
        data['거래날짜'] = pd.to_datetime(data['거래날짜'])
        data['총구매금액'] = data['수량'] * data['평균금액']
        data['고객ID'] = data['고객ID'].str.replace("USER_", "")
        data['성별'] = data['성별'].str.replace("남", "1").str.replace("여", "0")
        data['거래ID'] = data['거래ID'].str.replace("Transaction_", "")
        data['제품ID'] = data['제품ID'].str.replace("Product_", "")

        data = data.astype({"성별": 'str', "고객지역": 'str', "거래ID": 'str', "제품ID": 'str'})

        # RFM 데이터 생성
        rfm_data = data.groupby('고객ID').agg({
            '거래날짜': lambda x: (data['거래날짜'].max() - x.max()).days,
            '거래ID': 'nunique',
            '총구매금액': 'sum'
        }).rename(columns={'거래날짜': 'Recency', '거래ID': 'Frequency', '총구매금액': 'MonetaryValue'})

        rfm_data.reset_index(inplace=True)
        rfm_data = pd.merge(rfm_data, data, on='고객ID')

        return rfm_data


    def create_pipeline(self):

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) == X.shape[0]] # Assuming 'X' is defined within the method
        else:
            self.id_col = []


        num_pipeline = Pipeline(steps=[
            ('step1',   SimpleImputer(strategy="mean") ),
            ('step2',   numeric_filtering()  ),
            ('step3',   StandardScaler()  ),
        ])

        cat_pipeline = Pipeline(steps=[
            ('step1',   SimpleImputer(strategy="most_frequent") ),
            ('step2',   categorical_filtering()  ),
            ('step3',   OneHotEncoder()  ),
        ])

        transformer = ColumnTransformer(transformers=[
            ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipeline, make_column_selector(dtype_exclude=np.number))
        ])

        return transformer

    def fit_transform(self, X):
        self.pipeline.fit(X[['Recency', 'Frequency', 'MonetaryValue']])
        self.rfm_scaled = self.pipeline.transform(X[['Recency', 'Frequency', 'MonetaryValue']])
        return self.rfm_scaled

    def elbow_method(self):
        sse = {}
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=1)
            kmeans.fit(self.rfm_scaled)
            sse[k] = kmeans.inertia_

        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()), marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE")
        plt.title("Elbow Method for Optimal Clusters")
        plt.show()

        return self.find_elbow_point(sse)

    def find_elbow_point(self, sse):
        keys = list(sse.keys())
        for i in range(len(keys) - 1):
            value1 = sse[keys[i]]
            value2 = sse[keys[i + 1]]
            if abs(value1 - value2) >= 250:
                return keys[i + 1]
        return keys[-1]

    def silhouette_method(self):
        silhouette_scores = {}
        for k in range(3, 8):
            kmeans = KMeans(n_clusters=k, random_state=1)
            kmeans.fit(self.rfm_scaled)
            score = silhouette_score(self.rfm_scaled, kmeans.labels_)
            silhouette_scores[k] = score

        plt.figure()
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Scores for Different Number of Clusters")
        plt.show()

        return max(silhouette_scores, key=silhouette_scores.get)

    def fit_kmeans(self, method='silhouette'):
        if self.rfm_scaled is None:
            raise ValueError("Data has not been fit and transformed. Call fit_transform first.")

        if method == 'silhouette':
            self.optimal_clusters = self.silhouette_method()
        elif method == 'elbow':
            self.optimal_clusters = self.elbow_method()
        else:
            raise ValueError("Invalid method. Choose 'silhouette' or 'elbow'.")

        self.kmeans_model = KMeans(n_clusters=self.optimal_clusters, random_state=1)
        clusters = self.kmeans_model.fit_predict(self.rfm_scaled)
        return clusters

    def add_cluster(self, data, clusters, on='고객ID'):
        clusters_df = pd.DataFrame({'고객ID': data['고객ID'], 'Cluster': clusters})
        merged_data = pd.merge(data, clusters_df, on=on)
        return merged_data


# Usage:
# rfm_pipeline = RFMClusteringPipeline()
# rfm_data = rfm_pipeline.rfm_data(df)
# rfm_scaled = rfm_pipeline.fit_transform(rfm_data)
# rfm_data['Cluster'] = rfm_pipeline.fit_kmeans(method='silhouette')
# print(rfm_data.head())




class AutoArimaPipeline:
    def __init__(self):
        pipe1 = Pipeline([
            ('step1', SimpleImputer(strategy="mean")),
            ('step2', StandardScaler()),
        ])

        pipe2 = Pipeline([
            ('step1', SimpleImputer(strategy="most_frequent")),
            ('step2', OneHotEncoder()),
        ])

        self.transform = ColumnTransformer([
            ('num', pipe1, make_column_selector(dtype_include=np.number)),
            ('cat', pipe2, make_column_selector(dtype_exclude=np.number)),
        ])

    def fit(self, train_data):
        self.models = []
        for data in train_data:
            auto_arima_model = pm.auto_arima(data['마케팅비용'])
            self.models.append(auto_arima_model)

    def predict(self, test_data):
        predictions = []
        for i, data in enumerate(test_data):
            marketing_pred = []
            pred_upper = []
            pred_lower = []

            for new_ob in data['마케팅비용']:
                # 새로운 데이터를 이용하여 모델 업데이트
                self.models[i].update(new_ob)

                # 예측
                fc, conf = self.models[i].predict(n_periods=1, return_conf_int=True)
                fc = pd.Series(fc)

                marketing_pred.append(fc.values[0])
                pred_upper.append(conf[0][1])
                pred_lower.append(conf[0][0])

            marketing_pred = pd.Series(marketing_pred, index=data.index)
            marketing_pred = pd.DataFrame(marketing_pred).rename(columns={0: 'pred'})

            marketing_test_pred = pd.concat([data, marketing_pred], axis=1)
            predictions.append(marketing_test_pred)

        return predictions

    def hi_preprocessing(self, data):
      out = data[['거래날짜', '마케팅비용', '고객분류']]
      out.rename(columns = {'고객분류':'Cluster'}, inplace = True)
      return out

    # 시계열 데이터 전처리 함수
    def split_data_by_cluster(self, data):
        data.rename(columns={'고객분류': 'Cluster'})
        clusters = np.sort(data['Cluster'].unique())
        data['거래날짜'] = pd.to_datetime(data['거래날짜'], format='%Y-%m-%d')
        out = []
        for cluster in clusters:
            cluster_data = data[data['Cluster'] == cluster][['거래날짜', '마케팅비용', 'Cluster']]
            cluster_data = cluster_data.drop_duplicates(subset=['거래날짜']).sort_values('거래날짜')
            out.append(cluster_data)

        out.append(data[['거래날짜', '마케팅비용', 'Cluster']].drop_duplicates(subset=['거래날짜']).sort_values('거래날짜'))
        return out


    def time_train_test_split(self, data, test_size=0.2):
        test_size = 1 - test_size
        data.set_index('거래날짜', inplace=True)

        train = data[:int(test_size * len(data))]
        test = data[int(test_size * len(data)):]
        return train, test

    def create_train_test_by_cluster(self, data, out):
        data.rename(columns={'고객분류': 'Cluster'}, inplace=True)
        clusters = data['Cluster'].unique()
        numeric_clusters = [int(c) for c in clusters if c.isdigit()]
        if numeric_clusters:
            max_cluster = max(numeric_clusters)
        else:
            max_cluster = 0
        clusters = np.append(clusters, str(max_cluster + 1))

        train = []
        test = []

        for cluster_name, cluster_data in zip(clusters, out):
            cluster_train, cluster_test = self.time_train_test_split(cluster_data)
            train.append(cluster_train)
            test.append(cluster_test)

        return train, test

    def run(self, data, test_size=0.2):
        out = self.split_data_by_cluster(data)
        train, test = self.create_train_test_by_cluster(data, out)

        self.fit(train)
        predictions = self.predict(test)

        for i, prediction in enumerate(predictions):
            print(f"Predictions for Cluster {i}:")
            print(prediction)

# Usage 1:
# auto_arima_pipeline = AutoArimaPipeline()

# auto_arima_pipeline.fit(train)

# predictions = auto_arima_pipeline.predict(test)

# output
# for i, prediction in enumerate(predictions):
#     print(f"Predictions for Cluster {i}:")
#     print(prediction)

# Usage 2:
# auto_arima_pipeline = AutoArimaPipeline()

# auto_arima_pipeline.run(ecommerce_df)



class hi_visualizer:
    def __init__(self, data, clusters, marketing_cost_column='마케팅비용', prediction_column='pred'):
        """
        Initializes the visualizer with data and clusters.

        Parameters:
        - data: list of pd.DataFrame, list of datasets containing actual marketing costs (last element is the overall dataset)
        - clusters: list of pd.DataFrame, list of datasets containing predictions (last element is the overall dataset)
        - marketing_cost_column: str, the column name for the actual marketing costs in the data dataset
        - prediction_column: str, the column name for the predicted marketing costs in the clusters dataset
        """
        self.data = data
        self.clusters = clusters
        self.marketing_cost_column = marketing_cost_column
        self.prediction_column = prediction_column

        # Convert index to datetime if not already
        for df in self.data:
            df.index = pd.to_datetime(df.index)
        for df in self.clusters:
            df.index = pd.to_datetime(df.index)

    def visualize_overall(self):
        """
        Visualizes marketing costs and predictions over time for the entire dataset.

        Returns:
        - A plotly.graph_objs.Figure object for the overall data
        """
        # Initialize the plotly figure
        fig = go.Figure()

        # Add trace for overall actual marketing costs
        overall_data = self.data[-1]
        overall_clusters = self.clusters[-1]

        fig.add_trace(go.Scatter(
            x=overall_data.index,
            y=overall_data[self.marketing_cost_column],
            mode='lines+markers',
            name='Actual Marketing Cost (Overall)'
        ))

        # Add trace for overall predicted marketing costs
        fig.add_trace(go.Scatter(
            x=overall_clusters.index,
            y=overall_clusters[self.prediction_column],
            mode='lines+markers',
            name='Predicted Marketing Cost (Overall)'
        ))

        # Update layout
        fig.update_layout(
            title='Marketing Costs and Predictions Over Time (Overall)',
            xaxis_title='Transaction Date',
            yaxis_title='Marketing Cost',
            legend_title='Legend',
            hovermode='x unified'
        )

        return fig

    def visualize_cluster(self, cluster_number):
        """
        Visualizes marketing costs and predictions over time for a specific cluster.

        Parameters:
        - cluster_number: int, the cluster number to visualize

        Returns:
        - A plotly.graph_objs.Figure object for the specified cluster
        """
        if cluster_number >= len(self.data) - 1:
            raise ValueError("Invalid cluster number. It should be less than the total number of clusters.")

        # Initialize the plotly figure
        fig = go.Figure()

        # Add traces for the specified cluster
        cluster_data = self.data[cluster_number]
        cluster_predictions = self.clusters[cluster_number]

        # Actual marketing cost for the cluster
        fig.add_trace(go.Scatter(
            x=cluster_data.index,
            y=cluster_data[self.marketing_cost_column],
            mode='lines+markers',
            name=f'Actual Marketing Cost - Cluster {cluster_number}'
        ))

        # Predicted marketing cost for the cluster
        fig.add_trace(go.Scatter(
            x=cluster_predictions.index,
            y=cluster_predictions[self.prediction_column],
            mode='lines+markers',
            name=f'Predicted Marketing Cost - Cluster {cluster_number}'
        ))

        # Update layout
        fig.update_layout(
            title=f'Marketing Costs and Predictions Over Time (Cluster {cluster_number})',
            xaxis_title='Transaction Date',
            yaxis_title='Marketing Cost',
            legend_title='Legend',
            hovermode='x unified'
        )

        return fig


# Example usage:
# visualizer = hi_visualizer(train, predictions)

# Visualize overall data
# overall_fig = visualizer.visualize_overall()
# overall_fig.show()

# Visualize cluster 0
# cluster_fig = visualizer.visualize_cluster(0)
# cluster_fig.show()