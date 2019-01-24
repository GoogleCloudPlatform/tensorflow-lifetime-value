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
from __future__ import absolute_import

import tensorflow as tf

from .context import CLVFeatures

# Possible estimators:
# Canned: https://www.tensorflow.org/api_docs/python/tf/estimator or custom ones
CANNED_MODEL_TYPES = ['DNNRegressor', 'Linear']
MODEL_TYPES = CANNED_MODEL_TYPES[:] + ['dnn_model', 'paretonbd_model',
                                       'bgnbd_model']
CANNED_DEEP, LINEAR, DEEP, PARETO, BGNBD = MODEL_TYPES
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


def dataset_input_fn(data_folder, prefix=None, mode=None, params=None, count=None):
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

  # Repeat the input indefinitely if count is None
  dataset = dataset.repeat(count=count)

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
  return dataset_input_fn(data_folder=data_folder,
                          prefix='test',
                          params=params,
                          count=1)

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
  feature_columns = clvf.get_deep_features()

  # Creates the input layers from the features.
  h = tf.feature_column.input_layer(features=features,
                                    feature_columns=feature_columns)

  # Loops through the layers.
  for size in params.hidden_units:
    h = tf.layers.dense(h, size, activation=None)
    h = tf.layers.batch_normalization(h, training=(
        mode == tf.estimator.ModeKeys.TRAIN))
    h = tf.nn.relu(h)
    if (params.dropout is not None) and (mode == tf.estimator.ModeKeys.TRAIN):
      h = tf.layers.dropout(h, params.dropout)

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
  if mode == tf.estimator.ModeKeys.PREDICT:

    #[START prediction_output_format]
    predictions = {
        'customer_id': tf.squeeze(features[clvf.get_key()]),
        'predicted_monetary': output
    }
    export_outputs = {
        'predictions': tf.estimator.export.PredictOutput(predictions)
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      export_outputs=export_outputs)
    #[END prediction_output_format]

  # Calculates loss using mean squared error between the given labels
  # and the calculated output.
  loss = tf.losses.mean_squared_error(labels, output)

  # Create Optimizer and thhe train operation
  optimizer = get_optimizer(params)

  # add update ops for batch norm stats
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss=loss,
                                  global_step=tf.train.get_global_step())

  # Root mean square error eval metric
  eval_metric_ops = {
      'rmse': tf.metrics.root_mean_squared_error(labels, output)
  }

  # Returns an estimator spec for EVAL and TRAIN modes.
  return tf.estimator.EstimatorSpec(mode=mode,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops=eval_metric_ops)


def rmse_evaluator(labels, predictions):
  """Metric for RMSE.

  Args:
    labels: Truth provided by the estimator when adding the metric
    predictions: Predicted values. Provided by the estimator silently
  Returns:
    metric_fn that can be used to add the metrics to an existing Estimator
  """
  pred_values = predictions['predictions']
  return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}


def get_learning_rate(params):
  """Get learning rate given hyperparams.

  Args:
    params: hyperparameters

  Returns:
    learning_rate tensor if params.learning_rate_decay,
    else a constant.
  """
  if params.learning_rate_decay:
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate=params.learning_rate,
        global_step=global_step,
        decay_steps=params.checkpoint_steps,
        decay_rate=params.learning_decay_rate,
        staircase=True
    )
  else:
    learning_rate = params.learning_rate
  return learning_rate


def get_optimizer(params):
  """Get optimizer given hyperparams.

  Args:
    params: hyperparameters

  Returns:
    optimizer object

  Raises:
    ValueError: if params.optimizer is not supported.
  """
  if params.optimizer == 'ProximalAdagrad':
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=get_learning_rate(params),
        l1_regularization_strength=params.l1_regularization,
        l2_regularization_strength=params.l2_regularization
    )
  elif params.optimizer == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(get_learning_rate(params))
  elif params.optimizer == 'Adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=get_learning_rate(params))
  elif params.optimizer == 'RMSProp':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=get_learning_rate(params))
  else:
    raise ValueError('Invalid optimizer: %s' % params.optimizer)
  return optimizer


def get_estimator(estimator_name, config, params, model_dir):
  """Return one of the TF-provided canned estimators defined by MODEL_TYPE.

  Args:
    estimator_name:     estimator model type
    config:             run config
    params:             hyperparams
    model_dir:          model directory

  Returns:
    Estimator object
  """
  print('-- Running training with estimator {} --'.format(estimator_name))

  if estimator_name not in CANNED_MODEL_TYPES:
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=config,
                                       params=params,
                                       model_dir=model_dir)
  else:
    if estimator_name == CANNED_DEEP:
      estimator = tf.estimator.DNNRegressor(
          feature_columns=clvf.get_deep_features(),
          hidden_units=params.hidden_units,
          config=config,
          model_dir=model_dir,
          optimizer=lambda: get_optimizer(params),
          batch_norm=True,
          dropout=params.dropout)
    else:
      estimator = tf.estimator.LinearRegressor(
          feature_columns=clvf.get_wide_features(),
          config=config,
          model_dir=model_dir)

    # Add RMSE for metric for canned estimators
    estimator = tf.contrib.estimator.add_metrics(estimator, rmse_evaluator)
  return estimator
