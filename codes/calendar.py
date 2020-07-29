import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from .loader import load_calendar

def head(calendar_df):
    return calendar_df.head()


def format_calendar_price():
    calendar = load_calendar()
    formated_price = calendar['price'].str.replace(r"[$,]", "", regex=True).astype('float32')
    calendar['price'] = formated_price
    formated_adjusted_price = calendar['adjusted_price'].str.replace(r"[$,]", "", regex=True).astype('float32')
    calendar['adjusted_price'] = formated_adjusted_price
    return calendar


def format_calendar_date():
    calendar = format_price()
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['weekday'] = calendar['date'].dt.weekday
    calendar['month'] = calendar['date'].dt.month
    return calendar


def get_mean_price(sorted='month'):
    if sorted not in ['month', 'weekday']:
        raise ValueError('sorted parameter must be month or weekday!')

    calendar = format_date()
    if sorted == 'month':
        month_price = calendar.groupby("month")['price'].mean()
        return month_price
    elif sorted == 'weekday':
        weekday_price = calendar.groupby("weekday")['price'].mean()
        return weekday_price


def filter_price(maxprice):
    if maxprice < 0:
        raise ValueError('maxprice must be a nonzero number')

    calendar = format_date()
    res = calendar[calendar['price'] < maxprice]
    return res


def plot_mean_price(sorted='month'):
    if sorted not in ['month', 'weekday']:
        raise ValueError('sorted parameter must be month or weekday!')

    price = get_mean_price(sorted)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    ax1 = sns.barplot(month_price.index, month_price.values, ax=axes[0])
    ax1.set_title("每月住房均价")
    ax2 = sns.barplot(weekday_price.index, weekday_price.values, ax=axes[1])
    ax2.set_title("每个工作日住房均价")
