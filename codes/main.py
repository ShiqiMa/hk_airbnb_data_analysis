import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from .loader import *
from .calendar import *
from .listing import *
from .reviews import *
from .predict_prices import *
from .predict_reviews import *

# load calendar.csv and print the first 5 lines
calendar = load_calendar()
print(calendar.head())

# format price and date in calendar.read_csv
df1 = format_calendar_price()
print(f"formated calendar price dataframe:\n{df1}")
df2 = format_calendar_date()
print(f"formated calendar date dataframe:\n{df2}")

# get mean price sorted by month or weekday
month_mean_price = get_mean_price(sorted=month)
weekday_mean_price = get_mean_price(sorted=weekday)
print(f"mean price sorted by month: {month_mean_price}")
print(f"mean price sorted by weekday: {weekday_mean_price}")

# given specified price to filter houses
df3 = filter_price(500)
print(f"price lower than HK$500 houses:\n{df3}")

# plot mean price sorted by month or weekday
plot_mean_price()

###################################################################
# load listings_detailed.csv and print the first 5 lines
listings_detailed = load_listing()
print(listings_detailed.head())

# add minimum cost column to the listings_detailedngs dataframe
df4 = add_min_cost_column()
print(f"add minimun cost column to the listings_detailed dataframe:\n{df4}")

# add room type column to the listings_detailed dataframe
df5 = add_room_type_column()
print(f"add room type column to the listings_detailed dataframe:\n{df5}")

# quickly preprocess listings_detaied dataframe and select certain colomns
listings_detailed_df = preprocess()

# plot room type distribution
plot_room_type()

# count room type and location distribution, and plot room location distribution
df6 = count_room_distribution()
print(f"room type and location distribution dataframe:\n{df6}")
plot_room_distribution()

###################################################################
# load reviews_detailed.csv and print the first 5 lines
reviews = load_reviews()
print(reviews.head())

# plot reviews sorted by date
plot_reviews_by_date()

# count each month reviews numbers in all years, and plot
df7 = count_year_month_reviews()
print(f"year-month reviews dataframe:\n{df7}")
plot_year_month_reviews()

###################################################################
# process features data, split training and testing data, and fit the model
predict_price_df = predict_house_prices()
print(f"predict testing dataseta houses prices and compare them with the true target prices:\n{predict_price_df}")

###################################################################
# predict latest months reviews number
predict_reviews_df = predict_reviews_numbers()
print(f"predict the future months reviews numbers:\n{predict_reviews_df}")

# plot original and pridicted reviews numbers
plot_final_reviews()
