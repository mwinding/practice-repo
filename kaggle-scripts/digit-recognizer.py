# %%
# practice computer vision competition
# https://www.kaggle.com/c/digit-recognizer/

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime


# load training and test data
train_data = pd.read_csv('data/handwritten-digits_MNIST/train.csv')
test_data = pd.read_csv('data/handwritten-digits_MNIST/test.csv')

# convert to 28x28 Tensors
X = train_data.drop('label', axis=1).to_numpy()
X = X.reshape(len(X[:, 0]), 28, 28)
#X = [tf.constant(image) for image in X]

y = train_data.loc[:, 'label']

# convert test data to 28x28 Tensors
X_test = test_data.to_numpy()
X_test = X_test.reshape(len(X_test[:, 0]), 28, 28)
#X_test = [tf.constant(image) for image in X_test]

# plot a few examples
nrows = 5
ncols = 5
plt.figure(figsize=(nrows,ncols))
for i in range(nrows*ncols):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(X[i], cmap='Greys')
    plt.axis('off')
    plt.text(14, 0, str(y[i]), horizontalalignment='center', verticalalignment='center') # plot label above image
plt.show()

y = pd.get_dummies(y).to_numpy() # one-hot encoded to be compatible with model

# split train and test data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# set up classifier

input_shape = X_train[0].shape

model = tf.keras.Sequential([

    # base CNN layers
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape = [28, 28, 1]),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    # head neural net layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(10, activation='softmax') # 10 required to account for [0,1,2,3,4,5,6,7,8,9] classes based on categorical_crossentropy
])

# compile models
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# %%
# fit models
epochs = 50

# I found that one has to monitor early-stopping
# if it stops after just a few epochs, the model is not well generalized and performs poorly
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max'
)

history = model.fit(
    X_train, y_train,
    validation_data = [X_valid, y_valid],
    epochs = epochs,
    callbacks = [early_stopping]
)

# %%
# plot loss and accuracy
history_df = pd.DataFrame(history.history)

history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

history_df.loc[0:, ['accuracy', 'val_accuracy']].plot()
print("Minimum validation loss: {}".format(history_df['val_accuracy'].max()))

 # %%
# make predictions with model

predictions = model.predict(X_test)
predictions = pd.DataFrame(predictions, index = test_data.index).round()

labels = []
for i in predictions.index:
    max_val = max(predictions.loc[i, :])
    num = np.where(predictions.loc[i, :]==max_val)[0][0]
    labels.append(num)

predictions = pd.DataFrame(predictions.index+1, columns=['ImageId'])
predictions['Label'] = labels

# convert one-hot encoded values to [0,1,2,3,4,5,6,7,8,9]
#predictions = pd.DataFrame(predictions.columns[np.where(predictions!=0)[1]], columns=['Label'])
#predictions['ImageId'] = predictions.index

# generate submission
submission = predictions.loc[:, ['ImageId', 'Label']]
today = datetime.datetime.today()
submission.to_csv(f'submissions/digit-recognizer_{today.year}-{today.month}-{today.day}.csv', index=False)

# %%
# demonstrate classification with plot

# plot a few examples of classified images
nrows = 5
ncols = 5
fig = plt.figure(figsize=(nrows,ncols))
for i in range(nrows*ncols):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(X_test[i], cmap='Greys')
    plt.axis('off')
    plt.text(14, 0, str(submission.loc[i, 'Label']), horizontalalignment='center', verticalalignment='center') # plot label above image
plt.show()
fig.savefig(f'submissions/digit-recognizer_classifier_{today.year}-{today.month}-{today.day}.pdf')

# %%
