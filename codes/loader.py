import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

BASEPATH = os.getcwd()
PATH1 = os.path.join(BASEPATH, 'dataset/calendar.csv')
PATH2 = os.path.join(BASEPATH, 'dataset/listings_detailed.csv')
PATH3 = os.path.join(BASEPATH, 'dataset/reviews_detailed.csv')

def load_calendar():
    """load calendar.csv"""

    df = pd.read_csv(PATH1)
    return df


def load_listing():
    """load listings_detailed.csv"""

    df = pd.read_csv(PATH2)
    return df


def load_reviews():
    """load reviews_detailed.csv"""
    df = pd.read_csv(PATH3)
    return df
