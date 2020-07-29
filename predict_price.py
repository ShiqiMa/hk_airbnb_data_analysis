import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from .listing import format_price

disparse_columns = [
    'host_is_superhost',
    'host_identity_verified',
    'neighbourhood_cleansed',
    'property_type',
    'room_type',
    'is_business_travel_ready'
]

def get_ml_listings():
    listings_detailed = format_price()
    ml_listings = listings_detailed[listings_detailed['price']<1000][[
        'host_is_superhost',
        'host_identity_verified',
        'neighbourhood_cleansed',
        'latitude',
        'longitude',
        'property_type',
        'room_type',
        'accommodates',
        'bathrooms',
        'bedrooms',
        'cleaning_fee',
        'minimum_nights',
        'maximum_nights',
        'availability_90',
        'number_of_reviews',
        'is_business_travel_ready',
        'n_amenities',
        'price'
    ]]
    return ml_listings


def clean_data():
    # 删除异常值
    ml_listings = get_ml_listings()
    ml_listings.dropna(axis=0, inplace=True)
    return ml_listings


def split_features_targets():
    # 分割特征值和目标值
    ml_listings = clean_data()
    features = ml_listings.drop(columns='price')
    targets = ml_listings['price']
    return features, targets


def get_disparse_features():
    # 针对离散型（字符串类型）进行one-hot编码
    features, _ = split_features_targets()
    disparse_features = features[disparse_columns]
    disparse_features = pd.get_dummies(disparse_features)
    return disparse_features


def get_standard_features():
    # 对连续型数据进行标准化
    features, _ = split_features_targets()
    continuouse_features = features.drop(columns=disparse_columns)
    scaler = StandardScaler()
    continuouse_features = scaler.fit_transform(continuouse_features)
    return continuouse_features


def get_processed_features():
    # 对处理后的特征进行组合
    disparse_features = get_disparse_features()
    continuouse_features = get_standard_features()
    feature_array = np.hstack([disperse_features.to_numpy(), continuouse_features])
    return feature_array


def predict_house_prices():
    feature_array = get_processed_features()
    _, targets = split_features_targets()
    X_train, X_test, y_train, y_test = train_test_split(feature_array, targets, test_size=0.25)
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    pres = np.vstack([y_test, y_predict])
    res_df = pd.DataFrame(res, index=['真实价格', '预测价格'],columns=y_test.index)
    print(res_df)
    print("平均误差：",mean_absolute_error(y_test, y_predict))
    print("R2评分：",r2_score(y_test, y_predict))
