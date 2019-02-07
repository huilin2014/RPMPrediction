import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.models import model_from_json

import h5py

from math import sqrt
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from numpy import array
import numpy as np
np.random.seed(10)
import random
random.seed(10)
import time
import csv

# Some pre-processing codes are adopted from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x)

# convert time series into supervised learning problem
def conversion(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def prepare_data(series, n_test, n_lag, n_seq):
    raw_values = series.values
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    supervised = conversion(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    # add one-layer lstm network
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, nb_epoch=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model

def forecast_lstm(model, X, n_batch):
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]

def make_forecasts(model, n_batch, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        forecast = forecast_lstm(model, X, n_batch)
        forecasts.append(forecast)
    return forecasts

def inverse_difference(last_ob, forecast):
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        forecast = forecast.reshape(1, len(forecast))
        inv_scale = inv_scale[0, :]
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        inverted.append(inv_diff)
    return inverted

def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    f = open('outputs/rmse_each_seq.csv','w')
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        f.write("t+%d RMSE: %f\n" % ((i+1), rmse))
    f.write('\n')

def plot_forecasts_LSTM(n_test, forecasts, series, ptId):
    ax = pyplot.gca()
    ax.relim()
    ax.autoscale_view()
    pyplot.figure()
    pyplot.plot(series.values[-n_test:], color = 'cornflowerblue', linewidth=3)
    forecasts_temp = list()
    for i in range(len(forecasts)):
        forecasts_temp.append((forecasts[i][0]))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_true_values = scaler.fit_transform(series.values[-(n_test-n_seq+1):])
    scaled_true_values = scaled_true_values.reshape(len(scaled_true_values), 1)
    scaled_pred_values = scaler.fit_transform(forecasts_temp)
    scaled_pred_values = scaled_pred_values.reshape(len(scaled_pred_values), 1)
    
    nameTrue = "outputs/scaled_true_pt_%i.csv" % ptId
    namePred = "outputs/scaled_pred_pt_%i.csv" % ptId
    f_t = open(nameTrue, 'w')
    f_p = open(namePred, 'w')
    for i in range(len(scaled_true_values)):
        f_t.write("%f\n" % (scaled_true_values[i]))
        f_p.write("%f\n" % (scaled_pred_values[i]))

    rmse_t = sqrt(mean_squared_error(scaled_true_values, scaled_pred_values))
    
    pyplot.plot(forecasts_temp, color='magenta', linewidth=1.5, linestyle="--")
    name = "outputs/lstm_test_%i.pdf" % ptId
    pyplot.savefig(name, format='pdf', bbox_inches='tight')

    return ('Total RMSE of patient %d: %f \n' % (ptId, rmse_t))


if __name__=="__main__":
    for d in ['/gpu:0']:
        with tf.device(d):
            N_PATIENTS = 100
          
            series = {}
            for i in range(N_PATIENTS):
              filename = "/home/LstmPrediction/inputs/series_pt%d.csv" % (i)
              series[i] = Series.from_csv(filename)          
            n_lag = 15 
            n_seq = 5 
            n_test = 10
            n_epochs = 15
            n_batch = 1
            n_neurons = 10
          
            scalers, trains, tests = [], [], []
            for i in range(N_PATIENTS):
              scaler, train, test = prepare_data(series[i], n_test, n_lag, n_seq)
              scalers.append(scaler)
              trains.append(train)
              tests.append(test)
              print ("patient %i preparation done." % i)
          
            train = np.concatenate(trains,axis=0) 
            test = np.concatenate(tests,axis=0) 
          
            nameTime = "outputs/Time.csv"
            f_time = open(nameTime, 'w')
                
            model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
        
            print('model training finished in %.3f seconds.' % elapsed)
            f_time.write("Model training finished in %.3f seconds.\n" % elapsed)
            
            model.save('breathModel.h5')

            nameTotalRmse = "outputs/Rmse.csv"
            f_rmse = open(nameTotalRmse, 'w')
            for i in range(N_PATIENTS):
                print ('fitting patient: ', i)
                forecasts = make_forecasts(model, n_batch, tests[i], n_lag, n_seq)
                print('model fitting for patient %i in %.3f seconds.' % (i, elapsed_f))
              
                forecasts = inverse_transform(series[i], forecasts, scalers[i], n_test+n_seq-1)
              
                actual = [row[n_lag:] for row in tests[i]]
                actual = inverse_transform(series[i], actual, scalers[i], n_test+n_seq-1)
                evaluate_forecasts(actual, forecasts, n_lag, n_seq)
        
                rmsePerPt = plot_forecasts_LSTM(n_test+n_seq-1, forecasts, series[i], i)
                f_rmse.write(rmsePerPt)