# competition associated with ML tutorial
# https://www.kaggle.com/c/home-data-for-ml-course/overview

from sklearn.tree import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

train_data = pd.read_csv('data/house-prices/train.csv')
test_data = pd.read_csv('data/house-prices/test.csv')

