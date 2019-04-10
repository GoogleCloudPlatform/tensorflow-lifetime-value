# Copyright 2019 Google Inc. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
from google.cloud.automl_v1beta1 import AutoMlClient, PredictionServiceClient
import sys
import time

# parameter defaults
KEY_FILE = 'mykey.json'
LOCATION = 'us-central1'
BQ_DATASET = 'ltv_edu_auto'
BQ_TABLE = 'features_n_target'
AUTOML_DATASET = 'clv_solution'
TARGET_LABEL = 'target_monetary'
AUTOML_MODEL = 'clv_model'
BATCH_GCS_INPUT = 'gs://'
BATCH_GCS_OUTPUT = 'gs://'

def create_automl_model(client,
                        project_id,
                        location,
                        bq_dataset,
                        bq_table,
                        automl_dataset,
                        automl_model,
                        training_budget):
  """
  Create an AutoML Tables dataset based on the data in BigQuery.
  Create a model to predict CLV based on that dataset.

  Returns:
    The name of the created model.
  """
  location_path = client.location_path(project_id, location)
  dataset_display_name = automl_dataset

  # create dataset
  create_dataset_response = client.create_dataset(
      location_path,
      {'display_name': dataset_display_name, 'tables_dataset_metadata': {}})
  print("Creating AutoML Tables dataset...")
  dataset_name = create_dataset_response.name
  print("Done")

  # import data
  dataset_bq_input_uri = 'bq://{}.{}.{}'.format(project_id, bq_dataset, bq_table)
  input_config = {
      'bigquery_source': {
          'input_uri': dataset_bq_input_uri}}

  print("Importing data...")
  import_data_response = client.import_data(dataset_name, input_config)
  while import_data_response.done() is False:
    time.sleep(1)
  print("Done")

  # get column specs
  list_table_specs_response = client.list_table_specs(dataset_name)
  table_specs = [s for s in list_table_specs_response]
  table_spec_name = table_specs[0].name
  list_column_specs_response = client.list_column_specs(table_spec_name)
  column_specs = {s.display_name: s for s in list_column_specs_response}

  # update dataset to assign a label
  label_column_name = TARGET_LABEL
  label_column_spec = column_specs[label_column_name]
  label_column_id = label_column_spec.name.rsplit('/', 1)[-1]
  update_dataset_dict = {
      'name': dataset_name,
      'tables_dataset_metadata': {
          'target_column_spec_id': label_column_id
      }
  }
  print("Setting label...")
  update_dataset_response = client.update_dataset(update_dataset_dict)
  print("Done")

  # define the features used to train the model
  feat_list = list(column_specs.keys())
  feat_list.remove('target_monetary')
  feat_list.remove('customer_id')
  feat_list.remove('monetary_btyd')
  feat_list.remove('frequency_btyd')
  feat_list.remove('frequency_btyd_clipped')
  feat_list.remove('monetary_btyd_clipped')
  feat_list.remove('target_monetary_clipped')

  # create and train the model
  model_display_name = automl_model
  model_training_budget = training_budget * 1000
  model_dict = {
    'display_name': model_display_name,
    'dataset_id': dataset_name.rsplit('/', 1)[-1],
    'tables_model_metadata': {
        'target_column_spec': column_specs['target_monetary'],
        'input_feature_column_specs': [
            column_specs[x] for x in feat_list],
        'train_budget_milli_node_hours': model_training_budget,
        'optimization_objective': 'MINIMIZE_MAE'
    }
  }
  print("Creating AutoML Tables model...")
  create_model_response = client.create_model(location_path, model_dict)
  while create_model_response.done() is False:
    time.sleep(10)
  print("Done")

  create_model_result = create_model_response.result()
  model_name = create_model_result.name

  return model_name


def deploy_model(client, model_name):
  """
  Deploy model for predictions.
  """
  print("Deploying AutoML Tables model...")
  deploy_model_response = client.deploy_model(model_name)
  api = client.transport._operations_client
  while deploy_model_response.done is False:
    deploy_model_response = api.get_operation(deploy_model_response.name)
    time.sleep(10)
  print("Done")


def get_model_evaluation(client, model_name):
  """
  Get the evaluation stats for the model.
  """
  model_evaluations = [e for e in client.list_model_evaluations(model_name)]
  model_evaluation = model_evaluations[0]
  print("Model evaluation:")
  print(model_evaluation)
  return model_evaluation


def do_batch_prediction(prediction_client,
                        model_name,
                        gcs_input_uri,
                        gcs_output_uri_prefix):

  # Define input source.
  batch_prediction_input_source = {
    'gcs_source': {
      'input_uris': [gcs_input_uri]
    }
  }
  # Define output target.
  batch_prediction_output_target = {
      'gcs_destination': {
        'output_uri_prefix': gcs_output_uri_prefix
      }
  }

  # initiate batch predict
  print('Performing AutoML Tables batch predict...')
  batch_predict_response = prediction_client.batch_predict(
      model_name, batch_prediction_input_source, batch_prediction_output_target)

  # Wait until batch prediction is done.
  while batch_predict_response.done() is False:
    time.sleep(1)
  print('Done')

  batch_predict_result = batch_predict_response.result()
  return batch_predict_result


def create_parser():
  """Initialize command line parser using argparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()

  # required args
  parser.add_argument('--project_id',
                      help='Project id for project containing BQ data',
                      default=KEY_FILE,
                      type=str,
                      required=True)

  # data and model args
  parser.add_argument('--training_budget',
                      help='Training budget in hours',
                      default=1,
                      type=int)
  parser.add_argument('--key_file',
                      help='JSON key file for API access',
                      default=KEY_FILE,
                      type=str)
  parser.add_argument('--location',
                      help='GCP region to run',
                      default=LOCATION,
                      type=str)
  parser.add_argument('--automl_dataset',
                      help='Name of AutoML dataset',
                      default=AUTOML_DATASET,
                      type=str)
  parser.add_argument('--automl_model',
                      help='Name of AutoML model',
                      default=AUTOML_MODEL,
                      type=str)
  parser.add_argument('--bq_dataset',
                      help='BigQuery dataset to import from',
                      default=BQ_DATASET,
                      type=str)
  parser.add_argument('--bq_table',
                      help='BigQuery table to import from',
                      default=BQ_TABLE,
                      type=str)
  parser.add_argument('--batch_gcs_input',
                      help='GCS URI for batch predict CSV',
                      default=BATCH_GCS_INPUT,
                      type=str)
  parser.add_argument('--batch_gcs_output',
                      help='GCS URI for batch predict output',
                      default=BATCH_GCS_OUTPUT,
                      type=str)
  return parser


def main(argv=None):
  """Create and train the CLV model on AutoML Tables."""
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  # create and configure client
  keyfile_name = args.key_file
  client = AutoMlClient.from_service_account_file(keyfile_name)

  # create and deploy model
  model_name = create_automl_model(client,
                                   args.project_id,
                                   args.location,
                                   args.bq_dataset,
                                   args.bq_table,
                                   args.automl_dataset,
                                   args.automl_model,
                                   args.training_budget)

  # deploy model
  deploy_model(client, model_name)

  # get model evaluations
  model_evaluation = get_model_evaluation(client, model_name)

  # make predictions
  prediction_client = PredictionServiceClient.from_service_account_file(
      keyfile_name)
  do_batch_prediction(prediction_client,
                      model_name,
                      args.batch_gcs_input,
                      args.batch_gcs_output)

if __name__ == '__main__':
  main()
