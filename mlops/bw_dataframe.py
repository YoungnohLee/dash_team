# %% [markdown]
# # 1. DB에서 데이터 불러오기

# %%
import bw_database
import bw_class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# %%
bw_database.csv2db()

# %% [markdown]
# # 2. 대시보드를 위해 필요한 데이터프레임 만들기

# %%
train = bw_database.making_dataframe_train_db('train_table')
test = bw_database.making_dataframe_test_db('test_table')

# %%
# bw_class 적용1 -> rfm 변수를 제외한 필요한 변수 생성
bw = bw_class.bw_preprocessing(train)
bw.apply_my_function()
bw_df = bw.return_dataframe()
bw_df

# %%
# bw_class 적용1 -> rfm 변수를 제외한 필요한 변수 생성
bw = bw_class.bw_preprocessing(train)
bw.apply_my_function()
bw_df = bw.return_dataframe()

# %%
# bw_class 적용2 -> rfm 변수 및 고객분류 변수 생성
processor = bw_class.RFMProcessor(train)
rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()

# %%
# 적용
processor = bw_class.RFMProcessor(train) 
rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
processor.fit_clustering(X_scaled, n_clusters=4)
new_data_predictions = processor.predict(train)

# %% [markdown]
# ### 시각화 자료를 통해 bw_class로 이동하여 각 클러스터에 대한 고객분류명 변경가능

# %%
# 클러스터 시각화 -> 이후에 새로운 데이터가 들어왔을 때 계속 점검할 수 있게끔
visualization = bw_class.Visualization(new_data_predictions)
visualization.plot_clusters(new_data_predictions["Cluster"])
visualization.plot_boxplots()

# %%
# 최종 데이터에 join
cluster_data = bw_class.mapping_cluster(new_data_predictions)
cluster = cluster_data[['고객ID','Recency','Frequency','Monetary','고객분류']]
train_bw = bw_df.merge(cluster, on = '고객ID', how = 'left')
train_bw

# %%
# 만들어진 최종 데이터프레임 -> DB에 저장
bw_database.create_new_table(train_bw, 'train_bw')

# %% [markdown]
# # 3. 1번째 대시보드를 위한 데이터프레임 만들기

# %%
train_bw = bw_database.making_dataframe_our_db('train_bw')

# %%
# class 이용
rfm_clusters = train_bw[['고객ID', 'Recency', 'Frequency', 'Monetary', '고객분류']].drop_duplicates(subset=['고객ID'])
analysis = bw_class.first_dash(rfm_clusters)
rfm_clusters_final = analysis.get_final_dataframe()
rfm_clusters_final.reset_index(drop=True, inplace=True)
rfm_clusters_final

# %%
rfm_clusters

# %% [markdown]
# # 4. 두번째 대시보드를 위한 데이터프레임 만들기

# %%
train_bw

# %% [markdown]
# # 5. 3번째 대시보드를 위한 데이터프레임 만들기

# %%
# class 이용
clustered_summary = bw_class.thrid_dash(train_bw)
clustered_summary.create_clustered_summary()

monthly_clustered_customers = clustered_summary.get_monthly_clustered_customers()
monthly_clustered_monetary = clustered_summary.get_monthly_clustered_monetary()

monthly_clustered_customers.reset_index(drop=True, inplace=True)
monthly_clustered_monetary.reset_index(drop=True, inplace=True)
monthly_clustered_customers

# %%
monthly_clustered_monetary

# %% [markdown]
# # 6. 4번째 대시보드를 위한 데이터프레임 만들기

# %%
# class 이용
grouped_df_processor = bw_class.fourth_dash(train_bw)
grouped_df_processor.preprocess()
grouped_df_processor.group_by_columns(['제품카테고리', '월', '쿠폰상태'], '매출')

grouped_df = grouped_df_processor.get_grouped_df()
grouped_df.reset_index(drop=True, inplace=True)
grouped_df

# %% [markdown]
# # 7. 5번째 대시보드를 위한 데이터프레임 만들기

# %%
# 사용
coupon_sales_processor = bw_class.fifth_dash(train_bw)
coupon_sales_processor.preprocess()
coupon_sales_processor.calculate_coupon_sales()

coupon_sales = coupon_sales_processor.get_coupon_sales()
coupon_sales.reset_index(drop=True, inplace=True)
coupon_sales

# %%
# 만들어진 최종 데이터프레임 -> DB에 저장
bw_database.create_new_table(rfm_clusters_final, 'rfm_clusters_final')
bw_database.create_new_table(train_bw, 'train_bw')
bw_database.create_new_table(monthly_clustered_customers, 'monthly_clustered_customers')
bw_database.create_new_table(monthly_clustered_monetary, 'monthly_clustered_monetary')
bw_database.create_new_table(grouped_df, 'grouped_df')
bw_database.create_new_table(coupon_sales, 'coupon_sales')


