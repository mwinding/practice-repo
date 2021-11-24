# %%
# competition associated with ML tutorial; model made using tensorflow
# https://www.kaggle.com/c/home-data-for-ml-course/overview

import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime

# load data
train_data = pd.read_csv('data/house-prices/train.csv')
test_data = pd.read_csv('data/house-prices/test.csv')

# combine data to normalize in same way
data = pd.concat([train_data, test_data])

# %%
# prep training and validation data

# prep data
# all possible features (with no nans)
'''
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
            'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
            'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
'''
features = train_data.columns.drop(['Id', 'SalePrice'])

X = data[features]
y = train_data['SalePrice']

# normalize data

preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False), make_column_selector(dtype_include=object))
)

X = preprocessor.fit_transform(X)
X = pd.DataFrame(X)
X = X.fillna(0)

# split submission data and training data
test_data_processed = X.loc[1460:]
X = X.loc[0:1459]

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# set up a couple model architectures

input_shape = [len(X.columns)]

model1 = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])

model2 = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dropout(0.35),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])

# compile models
model1.compile(
    optimizer = 'adam',
    loss = 'mae'
)

model2.compile(
    optimizer = 'adam',
    loss = 'mae'
)

# %%
# fit models
batch_size = 256
epochs = 400

early_stopping = EarlyStopping(
    min_delta=0.001, 
    patience=30, 
    restore_best_weights=True,
)

history1 = model1.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [early_stopping]
)

history2 = model2.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [early_stopping]
)

# plot loss for 
history1_df = pd.DataFrame(history1.history)
history2_df = pd.DataFrame(history2.history)

history1_df.loc[20:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history1_df['val_loss'].min()))

history2_df.loc[20:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history2_df['val_loss'].min()))

# %%
# make predictions with model

predictions = model2.predict(test_data_processed)

predictions = [x for sublist in predictions for x in sublist]

submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
today = datetime.datetime.today()
submission.to_csv(f'submissions/housing-prices_{today.year}-{today.month}-{today.day}.csv', index=False)

# %%
