d# %%
# competition associated with ML tutorial
# https://www.kaggle.com/c/home-data-for-ml-course/overview

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime

# load data
train_data = pd.read_csv('data/house-prices/train.csv')
test_data = pd.read_csv('data/house-prices/test.csv')

# %%
# manually test best number of max_leaf_nodes for random forest regressors
# example of model tuning

# function to calculate mean absolute error of prediction
def get_mae(train_data, features, target, max_leaf_nodes, val_test=True):
    # prep features and prediction target
    X = train_data[features]
    y = train_data[target]

    # split into training and evaluation sets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)

    # generate model
    rf_model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=42)
    rf_model.fit(train_X, train_y)

    # predictions and evalution
    # using 
    if(val_test):
        predictions = rf_model.predict(val_X)
        mae_value = mean_absolute_error(predictions, val_y)

    if(val_test==False):
        predictions = rf_model.predict(train_X)
        mae_value = mean_absolute_error(predictions, train_y)

    return(mae_value)


# all possible features (with no nans)
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
            'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
            'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2000]

mae_val = [[max_leaf_nodes, 'val_test', get_mae(train_data, features, target='SalePrice', max_leaf_nodes=max_leaf_nodes)] for max_leaf_nodes in candidate_max_leaf_nodes]
mae_train = [[max_leaf_nodes, 'train_test', get_mae(train_data, features, target='SalePrice', max_leaf_nodes=max_leaf_nodes, val_test=False)] for max_leaf_nodes in candidate_max_leaf_nodes]

mae_data = pd.DataFrame(mae_val + mae_train, columns = ['max_leaf_nodes', 'test_type', 'mae'])
print(mae_data)

# %%
# plot mean prediction errors for test data vs training data
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.lineplot(x=mae_data.max_leaf_nodes, y=mae_data.mae, hue=mae_data.test_type, ax=ax)

# max_leaf_nodes = 100 might be best compromise

# %%
# iterate over larger set of random parameters to tune model

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# test the parameters
X = train_data[features]
y = train_data['SalePrice']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_X, train_y)
rf_random.best_params_

best_random_model = rf_random.best_estimator_

# %%
# generate submission

X = train_data[features]
y = train_data['SalePrice']

best_random_model.fit(X, y) # retrain with all data
predictions = best_random_model.predict(test_data[features])

submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
today = datetime.datetime.today()
submission.to_csv(f'submissions/housing-prices_{today.year}-{today.month}-{today.day}.csv', index=False)

# %%
