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

"""DNN Estimator model code."""

from __future__ import print_function

import tensorflow as tf

from context import CLVFeatures

# Possible estimators:
# Canned: https://www.tensorflow.org/api_docs/python/tf/estimator or custom ones
CANNED_MODEL_TYPES = ['DNNRegressor']
MODEL_TYPES = CANNED_MODEL_TYPES[:] + ['dnn_model', 'paretonbd_model',
                                       'bgnbd_model']
CANNED_DEEP, DEEP, PARETO, BGNBD = MODEL_TYPES
PROBABILISTIC_MODEL_TYPES = [PARETO, BGNBD]

# Either a custom function or a canned estimator name
# Used as default it not passed as an argument when calling the task
MODEL_TYPE = DEEP

# Features
clvf = CLVFeatures(
    ignore_crosses=True, is_dnn=MODEL_TYPE not in PROBABILISTIC_MODEL_TYPES)


def parse_csv(csv_row):
  """Parse CSV data row.

  tf.data.Dataset.map takes a function as an input so need to call parse_fn
  using map(lamba x: parse_fn(x)) or do def parse_fn and return the function
  as we do here.
  Builds a pair (feature dictionary, label) tensor for each example.

  Args:
    csv_row: one example as a csv row coming from the Dataset.map()
  Returns:
    features and targets
  """
  columns = tf.decode_csv(csv_row, record_defaults=clvf.get_all_defaults())
  features = dict(zip(clvf.get_all_names(), columns))

  # Remove the headers that we don't use
  for column_name in clvf.get_unused():
    features.pop(column_name)

  target = features.pop(clvf.get_target_name())

  return features, target


def dataset_input_fn(data_folder, prefix=None, mode=None, params=None):
  """Creates a dataset reading example from filenames.

  Args:
    data_folder: Location of the files finishing with a '/'
    prefix: Start of the file names
    mode: tf.estimator.ModeKeys(TRAIN, EVAL)
    params: hyperparameters
  Returns:
    features and targets
  """
  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

  # Read CSV files into a Dataset
  filenames = tf.matching_files('{}{}*.csv'.format(data_folder, prefix))
  dataset = tf.data.TextLineDataset(filenames)

  # Parse the record into tensors.
  dataset = dataset.map(parse_csv)

  # Shuffle the dataset
  if shuffle:
    dataset = dataset.shuffle(buffer_size=params.buffer_size)

  # Repeat the input indefinitely
  dataset = dataset.repeat()

  # Generate batches
  dataset = dataset.batch(params.batch_size)

  # Create a one-shot iterator
  iterator = dataset.make_one_shot_iterator()

  # Get batch X and y
  features, target = iterator.get_next()

  return features, target


def read_train(data_folder, params):
  """Returns a shuffled dataset for training."""
  return dataset_input_fn(
      data_folder=data_folder,
      prefix='train',
      params=params,
      mode=tf.estimator.ModeKeys.TRAIN)


def read_eval(data_folder, params):
  """Returns a dataset for evaluation."""
  return dataset_input_fn(data_folder=data_folder,
                          prefix='eval',
                          params=params)


def read_test(data_folder, params):
  """Returns a dataset for test."""
  return dataset_input_fn(data_folder=data_folder, prefix='test', params=params)

#####################
# Model Definitions #
#####################
def dnn_model(features, mode, params):
  """Creates a DNN regressor model.

  Args:
    features: list of feature_columns
    mode: tf.estimator.ModeKeys(TRAIN, EVAL)
    params: hyperparameters

  Returns:
    output tensor
  """
  # Make features
  feature_columns = make_features()

  # Creates the input layers from the features.
  h = tf.feature_column.input_layer(features=features,
                                    feature_columns=feature_columns)

  # Loops through the layers.
  for size in params.hidden_units:
      h = tf.layers.dense(h, size, activation=tf.nn.relu)

  # Creates the logit layer
  logits = tf.layers.dense(h, 1, activation=None)
  return logits


def model_fn(features, labels, mode, params):
  """Model function for custom Estimator.

  Args:
    features: given by dataset_input_fn() tuple
    labels: given by dataset_input_fn() tuple
    mode: given when calling the estimator.train/predict/evaluate function
    params: hyperparameters
  Returns:
      EstimatorSpec that can be used by tf.estimator.Estimator.
  """
  # Build the dnn model and get output logits
  logits = dnn_model(features, mode, params)

  # Reshape output layer to 1-dim Tensor to return predictions
  output = tf.squeeze(logits)

  # Returns an estimator spec for PREDICT.
  # Get the output layer (logits) from the chosen model defined in MODEL_TYPE.
  logits = eval(MODEL_TYPE)(features, mode, params)

  # Reshape output layer to 1-dim Tensor to return predictions
  output = tf.squeeze(logits)

  # Returns an estimator spec for PREDICT.
  if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'scores': output
      }
      export_outputs = {
          'predictions': tf.estimator.export.PredictOutput(predictions)
      }

      return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=predictions,
                                        export_outputs=export_outputs)


  # Calculates loss using mean squared error between the given labels and the calculated output.
  loss = tf.losses.mean_squared_error(labels, output)

  # Creates Optimizer and its minimizing function (train operation).
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss=loss,
                                global_step=tf.train.get_global_step())

  # Root mean square error eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(labels, output)
  }

  # Returns an estimator spec for EVAL and TRAIN modes.
  return tf.estimator.EstimatorSpec(mode=mode,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops=eval_metric_ops)