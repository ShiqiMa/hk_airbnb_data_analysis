import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from .reviews import format_date


def predict_reviews_numbers():
    reviews = format_date()
    year_month_reviews = reviews.groupby(['year', 'month']).size().reset_index().rename(columns={0: "count"})
    features = year_month_reviews[['year', 'month']]
    targets = year_month_reviews['count']
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(features, targets)

    y_predict = regressor.predict([
        [2020, 4],
        [2020, 5],
        [2020, 6],
        [2020, 7],
        [2020, 8],
        [2020, 9],
        [2020, 10],
        [2020, 11],
        [2020, 12],
    ])
    predict_reviews = pd.DataFrame([[2020, 4+index, x] for index,x in enumerate(y_predict)], columns=['year', 'month', 'count'])
    return predict_reviews


def plot_final_reviews():
    predict_reviews = predict_reviews_numbers()
    final_reviews = pd.concat([year_month_reviews, predict_reviews]).reset_index()
    years = final_reviews['year'].unique()
    plt.figure(figsize=(10,5))
    for year in years:
        df = final_reviews[final_reviews['year']==year]
        plt.plot(df['month'], df['count'])
        plt.legend(years)
        plt.grid(True)
        plt.xticks(list(range(1, 13)))
