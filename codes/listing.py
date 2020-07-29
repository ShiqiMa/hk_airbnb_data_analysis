import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from .loader import load_listing


def format_listing_price():
    listings_detailed = load_listing()
    listings_detailed['price'] = listings_detailed['price'].str.replace(r"[$,]", "", regex=True).astype("float32")
    listings_detailed['cleacning_fee'] = listings_detailed['cleaning_fee'].str.replace(r"[$,]", "", regex=True).astype("float32").fillna(0, inplace=True)
    return listings_detailed


def add_min_cost_column():
    listings_detailed = format_price()
    listings_detailed['minimum_cost'] = (listings_detailed['price']+listings_detailed['cleaning_fee'])*listings_detailed['minimum_nights']
    return listings_detailed


def add_room_type_column():
    # 根据可容纳的人数，添加一个新的列，用来表示类型：Single(1)、Couple(2)、Family(5)、Group(100)
    listings_detailed['accommodates_type'] = pd.cut(listings_detailed['accommodates'], bins=[1,2,3,6,100], include_lowest=True, right=False,
                                               labels=['Single', 'Couple', 'Family', 'Group'])
    return listings_detailed


def preprocess():
    listings_detailed = get_min_cost().add_room_type_column()
    listings_detailed_df = listings_detailed[['id','host_id','listing_url','room_type',
                                          'neighbourhood_cleansed','price','cleaning_fee','amenities','n_amenities',
                                         'accommodates','accommodates_type','minimum_nights','minimum_cost']]
    return listings_detailed_df


def plot_room_type():
    listings_detailed_df = preprocess()
    room_type_counts = listings_detailed_df['room_type'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].pie(room_type_counts.values, autopct="%.2f%%", labels=room_type_counts.index)
    sns.barplot(room_type_counts.index, room_type_counts.values, ax=axes[1])
    plt.tight_layout()


def plot_room_location():
    plt.figure(figsize=(10, 5))
    neighbourhood_counts = listings_detailed_df['neighbourhood_cleansed'].value_counts()
    sns.barplot(y=neighbourhood_counts.index, x=neighbourhood_counts.values, orient='h')


def count_room_distribution():
    listings_detailed_df = preprocess()
    neighbour_room_type = listings_detailed_df.groupby(['neighbourhood_cleansed', 'room_type']) \
        .size() \
        .unstack('room_type') \
        .fillna(0) \
        .apply(lambda row:row/sum(row), axis=1) \
        .sort_values('Entire home/apt', ascending=True)
    return neighbour_room_type


def plot_room_distribution():
    neighbour_room_type = count_room_distribution()
    columns = neighbour_room_type.columns
    plt.figure(figsize=(10,5))
    index = neighbour_room_type.index
    plt.barh(index,neighbour_room_type[columns[0]])
    left = neighbour_room_type[columns[0]].copy()
    plt.barh(index,neighbour_room_type[columns[1]],left=left)
    left += neighbour_room_type[columns[1]].copy()
    plt.barh(index,neighbour_room_type[columns[2]],left=left)
    left += neighbour_room_type[columns[2]].copy()
    plt.barh(index,neighbour_room_type[columns[3]],left=left)
    plt.legend(columns)


def plot_host_id_nums():
    host_id_number = listings_detailed_df.groupby('host_id').size()
    host_id_number = pd.cut(host_id_number, bins=[1,2,3,6,1000], right=False, include_lowest=True, labels=['1', '2', '3-5', '6+']).value_counts()
    plt.pie(host_id_number, autopct="%.2f%%", labels=host_id_number.index)
