# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core functions for Probabilistic (BTYD) models."""

from __future__ import print_function
from __future__ import absolute_import

from datetime import datetime
from lifetimes import BetaGeoFitter, ParetoNBDFitter, GammaGammaFitter
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from .model import PARETO, BGNBD

PENALIZER_COEF = 0.01
DISCOUNT_RATE = 0.01

TRAINING_DATA_FILE = 'btyd.csv'
OUTPUT_FILE = 'predictions.csv'


def load_data(datapath):
  """Loads data from CSV data file.

  Args:
    datapath: Location of the training file
  Returns:
    summary dataframe containing RFM data for btyd models
    actuals_df containing additional data columns for calculating error
  """
  # Does not used the summary_data_from_transaction_data from the Lifetimes
  # library as it wouldn't scale as well. The pre-processing done in BQ instead.
  tf.logging.info('Loading data...')

  ft_file = '{0}/{1}'.format(datapath, TRAINING_DATA_FILE)
#[START prob_selec]
  df_ft = pd.read_csv(ft_file)

  # Extracts relevant dataframes for RFM:
  # - summary has aggregated values before the threshold date
  # - actual_df has values of the overall period.
  summary = df_ft[['customer_id', 'frequency_btyd', 'recency', 'T',
                   'monetary_btyd']]
#[END prob_selec]
  summary.columns = ['customer_id', 'frequency', 'recency', 'T',
                     'monetary_value']
  summary = summary.set_index('customer_id')

  # additional columns needed for calculating error
  actual_df = df_ft[['customer_id', 'frequency_btyd', 'monetary_dnn',
                     'target_monetary']]
  actual_df.columns = ['customer_id', 'train_frequency', 'train_monetary',
                       'act_target_monetary']

  tf.logging.info('Data loaded.')

  return summary, actual_df


def bgnbd_model(summary):
  """Instantiate and fit a BG/NBD model.

  Args:
    summary: RFM transaction data
  Returns:
    bgnbd model fit to the data
  """
  bgf = BetaGeoFitter(penalizer_coef=PENALIZER_COEF)
  bgf.fit(summary['frequency'], summary['recency'], summary['T'])
  return bgf


def paretonbd_model(summary):
  """Instantiate and fit a Pareto/NBD model.

  Args:
    summary: RFM transaction data
  Returns:
    bgnbd model fit to the data
  """
  #[START run_btyd]
  paretof = ParetoNBDFitter(penalizer_coef=PENALIZER_COEF)
  paretof.fit(summary['frequency'], summary['recency'], summary['T'])
  return paretof
  #[END run_btyd]

def run_btyd(model_type, data_src, threshold_date, predict_end):
  """Run selected BTYD model on data files located in args.data_src.

  Args:
    model_type:                 model type (PARETO, BGNBD)
    data_src:                   path to data
    threshold_date:             end date for training data 'YYYY-mm-dd'
    predict_end:                end date for predictions 'YYYY-mm-dd'
  """
  train_end_date = datetime.strptime(threshold_date, '%Y-%m-%d')
  predict_end_date = datetime.strptime(predict_end, '%Y-%m-%d')

  # load training transaction data
  summary, actual_df = load_data(data_src)

  # train fitter for selected model
  tf.logging.info('Fitting model...')

  if model_type == PARETO:
    fitter = paretonbd_model(summary)
  elif model_type == BGNBD:
    fitter = bgnbd_model(summary)

  tf.logging.info('Done.')

  #
  # use trained fitter to compute actual vs predicted ltv for each user
  #

  # compute the number of days in the prediction period
  time_days = (predict_end_date - train_end_date).days
  time_months = int(math.ceil(time_days / 30.0))

  # fit gamma-gamma model
  tf.logging.info('Fitting GammaGamma model...')

  ggf = GammaGammaFitter(penalizer_coef=0)
  ggf.fit(summary['frequency'], summary['monetary_value'])

  tf.logging.info('Done.')

  ltv, rmse = predict_value(summary,
                            actual_df,
                            fitter,
                            ggf,
                            time_days,
                            time_months)

  # output results to csv
  output_file = os.path.join(data_src, OUTPUT_FILE)
  ltv.to_csv(output_file, index=False)

  # log results
  tf.logging.info('BTYD RMSE error for %s model: %.2f', model_type, rmse)
  print('RMSE prediction error: %.2f' % rmse)


def predict_value(summary, actual_df, fitter, ggf, time_days, time_months):
  """Predict lifetime values for customers.

  Args:
    summary:      RFM transaction data
    actual_df:    dataframe containing data fields for customer id,
                  actual customer values
    fitter:       lifetimes fitter, previously fit to data
    ggf:          lifetimes gamma/gamma fitter, already fit to data
    time_days:    time to predict purchases in days
    time_months:  time to predict value in months
  Returns:
    ltv:  dataframe with predicted values for each customer, along with actual
      values and error
    rmse: root mean squared error summed over all customers
  """
  # setup dataframe to hold results
  ltv = pd.DataFrame(data=np.zeros([actual_df.shape[0], 6]),
                     columns=['customer_id',
                              'actual_total',
                              'predicted_num_purchases',
                              'predicted_value',
                              'predicted_total',
                              'error'], dtype=np.float32)

  predicted_num_purchases = fitter.predict(time_days,
                                           summary['frequency'],
                                           summary['recency'],
                                           summary['T'])

  predicted_value = ggf.customer_lifetime_value(fitter,
                                                summary['frequency'],
                                                summary['recency'],
                                                summary['T'],
                                                summary['monetary_value'],
                                                time=time_months,
                                                discount_rate=DISCOUNT_RATE)

  ltv['customer_id'] = actual_df['customer_id']
  ltv['actual_total'] = actual_df['act_target_monetary']
  ltv['predicted_num_purchases'] = predicted_num_purchases.values
  ltv['predicted_value'] = predicted_value.values
  ltv['predicted_total'] = actual_df['train_monetary'] + ltv['predicted_value']
  ltv['error'] = ltv['actual_total'] - ltv['predicted_total']

  mse = pd.Series.sum(ltv['error'] * ltv['error']) / ltv.shape[0]
  rmse = math.sqrt(mse)

  return ltv, rmse
