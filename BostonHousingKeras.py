#!/usr/bin/env python
# coding: utf-8

import keras
from keras.optimizers import SGD
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy  as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd


house = pd.read_csv('BostonHousing.csv')
print(house.head(20))
#import pandas.to_numeric

house = house.astype(np.float64)
#house = house.to_numeric()
print(house.head(20))
y=np.array(house['medv'])
df1 = house.drop(labels='medv', axis = 1)
print(df1.head())
X = df1.as_matrix()
dim = X.shape[1]
print(dim)
print(house.head(10))


house.hist(column='medv', bins=50)
plt.show()

house.hist(bins=50)
plt.show()
import seaborn as sns
sns.distplot(house['medv']);
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_test.shape, y_test.shape, len(X_test), len(y_test))



np.random.seed(102)
from keras import optimizers
# define the keras model
model= Sequential()
model.add(Dense(26, activation='relu',input_shape=(13,)))
model.add(Dense(26, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#model = build_model()
early_stopping= keras.callbacks.EarlyStopping(monitor='loss',patience=3)
history=model.fit(X_train, y_train, batch_size = 1, epochs = 100, validation_data=(X_test, y_test),verbose=2,callbacks=[early_stopping])


score = model.evaluate(X_test, y_test, verbose=0)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])


test_pred = model.predict(X_test).flatten()

sns.regplot(y_test,test_pred)
plt.scatter(y_test, test_pred)
plt.show
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
#_=plt.plot([-100,100],[-100,100])


error = test_pred - y_test
plt.hist(-error,bins = 50)
plt.show()

datalist2 = np.array(house['medv'])
plt.hist(test_pred,bins=50,color='b',alpha=0.3,label='theoretical',histtype='stepfilled', normed=True)
plt.hist(datalist2,bins=20,alpha=0.5,color='g',label='experimental',histtype='stepfilled',normed=True)
plt.xlabel("Value")
plt.ylabel("Normalised Frequency")
plt.legend()
plt.show()
print(' mean of medv: ', house['medv'].mean(),'  mean of test_pred: ', test_pred.mean())

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(
    hidden_layer_sizes=(50,50,50),
    alpha = 0,
    activation='relu',
    batch_size=128,
    learning_rate_init = 1e-3,
    solver = 'adam',
    learning_rate = 'constant',
    verbose = False,
    n_iter_no_change = 1000,
    validation_fraction = 0.0,
    max_iter=1000)
model.fit(X_train, y_train)

py = model.predict(X_test)
err = y_test - py
mse = np.mean(err**2)
rmse = np.sqrt(mse)
print('sklearn rmse for test %g' % rmse)
err =test_pred - y_test
mse = np.mean(err*err)
print(' keras test mse =',np.sqrt(mse))
test_pred = model.predict(X_train).flatten()
err = test_pred - y_train
mse = np.mean(err*err)
print(' keras train mse =',np.sqrt(mse))

def plot_history(history):
    plt.figure
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),label='Val loss')
    plt.legend()
    plt.ylim([2,8])
    plt.show()

plot_history(history)



