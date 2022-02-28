# %%
# competition associated with ML tutorial; model made using tensorflow
# https://www.kaggle.com/c/home-data-for-ml-course/overview

# implemented pipelines

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime

# prep training and validation data

# load data
train_data = pd.read_csv('data/house-prices/train.csv')
test_data = pd.read_csv('data/house-prices/test.csv')

features = train_data.columns.drop(['Id', 'SalePrice'])

X = train_data[features]
y = train_data['SalePrice']

# split train and test data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# preprocessing steps

# Imputer fills empty values, standardscaler scales values from [0,1]
numerical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('StandardScaler', StandardScaler())
])

# Imputer fills empty values, OneHot converts categorical data to binary
categorical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('OneHot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

# combines these steps into preprocessor to add to final preprocessor/model pipeline
preprocessor = ColumnTransformer(
    transformers=[
    ('num', numerical_transformer, make_column_selector(dtype_include=[np.number, 'int64'])), #int64 not registering as np.number
    ('cat', categorical_transformer, make_column_selector(dtype_include='object'))
])

X_train_processed = pd.DataFrame(preprocessor.fit_transform(X_train), index = X_train.index)
X_valid_processed = pd.DataFrame(preprocessor.transform(X_valid), index = X_valid.index)
test_data_processed = pd.DataFrame(preprocessor.transform(test_data), index = test_data.index)
# %%
# set up model architecture

def generate_model():
    input_shape = [len(X_train_processed.columns)]

    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=input_shape),
        layers.Dropout(0.35),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.35),
        layers.Dense(512, activation='relu'),
        layers.Dense(1)
    ])

    # compile models
    model.compile(
        optimizer = 'adam',
        loss = 'mae'
    )

    return(model)

batch_size = 256
epochs = 400

early_stopping = EarlyStopping(
    min_delta=0.001, 
    patience=30, 
    restore_best_weights=True,
)


# %%
# set up pipeline and run
from sklearn.metrics import mean_absolute_error

## SOME ISSUE HERE
model = KerasRegressor(model=generate_model, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

keras_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

keras_pipeline.fit(X_train, y_train)
preds = keras_pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print(mae)

# %%
# fit model

model = generate_model()

history = model.fit(
    X_train_processed, y_train,
    validation_data = (X_valid_processed, y_valid),
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [early_stopping]
)

# plot loss
history_df = pd.DataFrame(history.history)

history_df.loc[20:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

# %%
# make predictions with model

predictions = model.predict(test_data_processed)

predictions = [x for sublist in predictions for x in sublist]

submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
today = datetime.datetime.today()
submission.to_csv(f'submissions/housing-prices_{today.year}-{today.month}-{today.day}.csv', index=False)

# %%
