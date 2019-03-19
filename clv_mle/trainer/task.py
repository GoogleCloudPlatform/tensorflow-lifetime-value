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

"""Entry point for CMLE jobs for CLV."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

import argparse
import json
import os
import shutil
import tensorflow as tf

from .btyd import run_btyd
from .context import CLVFeatures
from .model import get_estimator, read_train, read_eval, read_test
from .model import MODEL_TYPE, MODEL_TYPES, PROBABILISTIC_MODEL_TYPES

# Training defaults

# 100000 is the approximate size of our training set (to nearest 1000).
#[START hyperparams]
TRAIN_SIZE = 100000
NUM_EPOCHS = 70
BATCH_SIZE = 5
NUM_EVAL = 20

LEARNING_DECAY_RATE = 0.7
HIDDEN_UNITS = '128 64 32 16'
LEARNING_RATE = 0.00135
L1_REGULARIZATION = 0.0216647
L2_REGULARIZATION = 0.0673949
DROPOUT = 0.899732
SHUFFLE_BUFFER_SIZE = 10000
#[END hyperparams]
# TRAIN_SIZE = 100000
# NUM_EPOCHS = 70
# BATCH_SIZE = 20
# NUM_EVAL = 20
# HIDDEN_UNITS = '128 64 32 16'
# LEARNING_RATE = 0.096505
# L1_REGULARIZATION = 0.0026019
# L2_REGULARIZATION = 0.0102146
# DROPOUT = 0.843251
# SHUFFLE_BUFFER_SIZE = 10000


def create_parser():
  """Initialize command line parser using arparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_type',
                      help='Model type to train on',
                      choices=MODEL_TYPES,
                      default=MODEL_TYPE)

  parser.add_argument('--job-dir', type=str, required=True)
  parser.add_argument('--data-src', type=str, required=True)

  # The following parameters are required for BTYD.
  parser.add_argument('--predict_end', type=str, required=False,
                      help='Predict end date YYYY-mm-dd')
  parser.add_argument('--threshold_date', type=str, required=False,
                      help='Threshold date YYYY-mm-dd')

  # hyper params
  parser.add_argument('--hidden_units',
                      help='List of hidden units per fully connected layer.',
                      default=HIDDEN_UNITS,
                      type=str)
  parser.add_argument('--learning_rate',
                      help='Learning rate for the optimizer',
                      default=LEARNING_RATE,
                      type=float)
  parser.add_argument('--learning_rate_decay',
                      type=str,
                      help='Use learning rate decay [True|False]',
                      default='True')
  parser.add_argument('--learning_decay_rate',
                      help='Learning decay rate',
                      type=float,
                      default=LEARNING_DECAY_RATE)
  parser.add_argument('--train_size',
                      help='(Approximate) size of training set',
                      default=TRAIN_SIZE,
                      type=int)
  parser.add_argument('--batch_size',
                      help='Number of input records used per batch',
                      default=BATCH_SIZE,
                      type=int)
  parser.add_argument('--buffer_size',
                      help='Size of the buffer for training shuffle.',
                      default=SHUFFLE_BUFFER_SIZE,
                      type=float)
  parser.add_argument('--train_set_size',
                      help='Number of samples on the train dataset.',
                      type=int)
  parser.add_argument('--l1_regularization',
                      help='L1 Regularization (for ProximalAdagrad)',
                      type=float,
                      default=L1_REGULARIZATION)
  parser.add_argument('--l2_regularization',
                      help='L2 Regularization (for ProximalAdagrad)',
                      type=float,
                      default=L2_REGULARIZATION)
  parser.add_argument('--dropout',
                      help='Dropout probability, 0.0 = No dropout layer',
                      type=float,
                      default=DROPOUT)
  parser.add_argument('--hypertune',
                      action='store_true',
                      help='Perform hyperparam tuning',
                      default=False)
  parser.add_argument('--optimizer',
                      help='Optimizer: [Adam, ProximalAdagrad, SGD, RMSProp]',
                      type=str,
                      default='ProximalAdagrad')
  parser.add_argument('--num_epochs',
                      help='Number of epochs',
                      default=NUM_EPOCHS,
                      type=int)
  parser.add_argument('--ignore_crosses',
                      action='store_true',
                      default=False,
                      help='Whether to ignore crosses (linear model only).')
  parser.add_argument('--verbose-logging',
                      action='store_true',
                      default=False,
                      help='Turn on debug logging')
  parser.add_argument('--labels',
                      type=str,
                      default='',
                      help='Labels for job')
  parser.add_argument('--resume',
                      action='store_true',
                      default=False,
                      help='Resume training on saved model.')
  return parser


def csv_serving_input_fn():
  """Defines how the model gets exported and the required prediction inputs.

  Required to have a saved_model.pdtxt file that can be used for prediction.

  Returns:
    ServingInputReceiver for exporting model.
  """
  #[START csv_serving_fn]
  clvf = CLVFeatures(ignore_crosses=True,
                     is_dnn=MODEL_TYPE not in PROBABILISTIC_MODEL_TYPES)
  used_headers = clvf.get_used_headers(with_key=True, with_target=False)
  default_values = clvf.get_defaults(used_headers)

  rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None],
                                      name='csv_rows')
  receiver_tensor = {'csv_rows': rows_string_tensor}

  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=default_values)

  features = dict(zip(used_headers, columns))

  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
  #[END csv_serving_fn]


def main(argv=None):
  """Run the CLV model."""
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  # Set logging mode
  tf.logging.set_verbosity(tf.logging.INFO)

  # execute non-estimator models
  if args.model_type in PROBABILISTIC_MODEL_TYPES:
    run_btyd(args.model_type, args.data_src, args.threshold_date,
             args.predict_end)
    return

  if args.hypertune:
    # if tuning, join the trial number to the output path
    config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    trial = config.get('task', {}).get('trial', '')
    model_dir = os.path.join(args.job_dir, trial)
  else:
    model_dir = args.job_dir

  print('Running training with model {}'.format(args.model_type))

  # data path
  data_folder = '{}/'.format(args.data_src)

  # Calculate train steps and checkpoint steps based on approximate
  # training set size, batch size, and requested number of training
  # epochs.
  train_steps = (args.train_size/args.batch_size) * args.num_epochs
  checkpoint_steps = int((args.train_size/args.batch_size) * (
      args.num_epochs/NUM_EVAL))

  # create RunConfig
  config = tf.estimator.RunConfig(
      save_checkpoints_steps=checkpoint_steps
  )

  hidden_units = [int(n) for n in args.hidden_units.split()]

  # Hyperparameters
  params = tf.contrib.training.HParams(
      num_epochs=args.num_epochs,
      train_steps=train_steps,
      batch_size=args.batch_size,
      hidden_units=hidden_units,
      learning_rate=args.learning_rate,
      ignore_crosses=args.ignore_crosses,
      buffer_size=args.buffer_size,
      learning_rate_decay=(
          args.learning_rate_decay == 'True'),
      learning_decay_rate=args.learning_decay_rate,
      l1_regularization=args.l1_regularization,
      l2_regularization=args.l2_regularization,
      optimizer=args.optimizer,
      dropout=(
          None if args.dropout == 0.0 else args.dropout),
      checkpoint_steps=checkpoint_steps)

  print(params)
  print('')
  print('Dataset Size:', args.train_size)
  print('Batch Size:', args.batch_size)
  print('Steps per Epoch:', args.train_size/args.batch_size)
  print('Total Train Steps:', train_steps)
  print('Required Evaluation Steps:', NUM_EVAL)
  print('Perform evaluation step after each', args.num_epochs/NUM_EVAL,
        'epochs')
  print('Save Checkpoint After', checkpoint_steps, 'steps')
  print('**********************************************')

  # Creates the relevant estimator (canned or custom)
  estimator = None

  # get model estimator
  #[START choose_model]
  estimator = get_estimator(estimator_name=args.model_type,
                            config=config,
                            params=params,
                            model_dir=model_dir)
  #[END choose_model]
  # Creates the training and eval specs by reading the relevant datasets
  # Note that TrainSpec needs max_steps otherwise it runs forever.
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: read_train(data_folder, params),
      max_steps=train_steps)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: read_eval(data_folder, params),
      exporters=[
          tf.estimator.LatestExporter(
              name='estimate',
              serving_input_receiver_fn=csv_serving_input_fn,
              exports_to_keep=1,
              as_text=True
          )
      ],
      steps=1000,
      throttle_secs=1,
      start_delay_secs=1
  )

  if not args.resume:
    print('Removing previous trained model...')
    shutil.rmtree(model_dir, ignore_errors=True)
  else:
    print('Resuming training...')

  # Runs the training and evaluation using the chosen estimator.
  # Saves model data into export/estimate/1234567890/...
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # Evaluate the test set for final metrics
  estimator.evaluate(lambda: read_test(data_folder, params), name="Test Set")

if __name__ == '__main__':
  main()
