import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from .loader import load_reviews

def format_date():
    reviews = load_reviews()
    reviews['date'] = pd.to_datetime(reviews['date'])
    reviews['year'] = reviews['date'].dt.year
    reviews['month'] = reviews['date'].dt.month
    return review


def plot_reviews_by_date():
    reviews = format_date()
    n_reviews_year = reviews.groupby('year').size()
    sns.barplot(n_reviews_year.index, n_reviews_year.values)
    n_reviews_month = reviews.groupby("month").size()
    sns.barplot(n_reviews_month.index,n_reviews_month.values)


def count_year_month_reviews():
    reviews = format_date()
    year_month_reviews = reviews.groupby(['year', 'month']).size().unstack('month').fillna(0)
    return year_month_reviews


def plot_year_month_reviews():
    fig,ax = plt.subplots(figsize=(10,5))
    for index in year_month_reviews.index:
        series = year_month_reviews.loc[index]
        sns.lineplot(x=series.index,y=series.values,ax=ax)
    ax.legend(labels=year_month_reviews.index)
    ax.grid()
    _ = ax.set_xticks(list(range(1,13)))
