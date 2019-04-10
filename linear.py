from tensorflow.python.lib.io import file_io
import pandas
from pandas.compat import StringIO
import json
import math
import numpy as np
from sklearn.linear_model import LinearRegression

c_names =['customer_id', 'monetary_dnn', 'monetary_btyd', 'frequency_dnn',
       'frequency_btyd', 'recency', 'T', 'time_between', 'avg_basket_value',
       'avg_basket_size', 'cnt_returns', 'has_returned',
       'frequency_btyd_clipped', 'monetary_btyd_clipped',
       'target_monetary_clipped', 'target_monetary']


train_df = file_io.FileIO(
            'data/train.csv',
            mode='r').read()
train_df = pandas.read_csv(
            StringIO(train_df),
            header = None,
            names = c_names,
            delimiter=',',
            na_filter=True)

test_df = file_io.FileIO(
            'data/eval.csv',
            mode='r').read()
test_df = pandas.read_csv(
            StringIO(test_df),
            header = None,
            names = c_names,
            delimiter=',',
            na_filter=True)

reg = LinearRegression().fit(
    train_df.values[:, [1,3,5,6,7,8,9,10,11]],
    train_df.values[:, -1])

error = 0
i = 0
for p in reg.predict(test_df.values[:, [1,3,5,6,7,8,9,10,11]]):
    error = error + math.pow(p - test_df.values[i, -1], 2)
    i = i +1

print "RMSE = ", math.sqrt(error/test_df.values.shape[0])