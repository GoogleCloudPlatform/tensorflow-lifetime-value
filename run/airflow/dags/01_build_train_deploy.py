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

import datetime, json, re, logging
from airflow import models
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.hooks.base_hook import BaseHook
from airflow.contrib.operators import bigquery_operator
from airflow.contrib.operators import bigquery_get_data
from airflow.contrib.operators import gcs_to_bq
from airflow.contrib.operators import bigquery_to_gcs
from airflow.contrib.operators import mlengine_operator
from airflow.contrib.operators import mlengine_operator_utils
from airflow.contrib.hooks.gcp_mlengine_hook import MLEngineHook
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
from airflow.contrib.hooks.gcp_api_base_hook import GoogleCloudBaseHook
from airflow.operators import bash_operator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils import trigger_rule

from google.cloud.automl_v1beta1 import AutoMlClient
from clv_automl import clv_automl

def _get_project_id():
  """Get project ID from default GCP connection."""

  extras = BaseHook.get_connection('google_cloud_default').extra_dejson
  key = 'extra__google_cloud_platform__project'
  if key in extras:
    project_id = extras[key]
  else:
    raise ('Must configure project_id in google_cloud_default '
           'connection from Airflow Console')
  return project_id

PROJECT = _get_project_id()
REGION = models.Variable.get('region')
DATASET = models.Variable.get('dataset')
COMPOSER_BUCKET_NAME = models.Variable.get('composer_bucket_name')
GCS_SQL = 'sql'
DB_DUMP_FILENAME = 'db_dump.csv'

LOCATION_TRAINING_DATA = '{}/data'.format(COMPOSER_BUCKET_NAME)

PREFIX_JOBS_EXPORT = 'jobs/clv-composer'
PREFIX_FINAL_MODEL = '{}/final'.format(PREFIX_JOBS_EXPORT)

MODEL_PACKAGE_NAME = 'clv_ml_engine-0.1.tar.gz' # Matches name in setup.py

AUTOML_DATASET = models.Variable.get('automl_dataset')
AUTOML_MODEL = models.Variable.get('automl_model')
AUTOML_TRAINING_BUDGET = int(models.Variable.get('automl_training_budget'))


#[START dag_build_train_deploy]
default_dag_args = {
    'start_date': datetime.datetime(2050, 1, 1),
    'schedule_internal': None,
    'provide_context': True
}

dag = models.DAG(
    'build_train_deploy',
    default_args = default_dag_args)
#[END dag_build_train_deploy]

# instantiate Google Cloud base hook to get credentials and create automl clients
gcp_hook = GoogleCloudBaseHook(conn_id='google_cloud_default')
automl_client = AutoMlClient(credentials=gcp_hook._get_credentials())

# Loads the database dump from Cloud Storage to BigQuery
t1 = gcs_to_bq.GoogleCloudStorageToBigQueryOperator(
    task_id="db_dump_to_bigquery",
    bucket=COMPOSER_BUCKET_NAME,
    source_objects=[DB_DUMP_FILENAME],
    schema_object="schema_source.json",
    source_format="CSV",
    skip_leading_rows=1,
    destination_project_dataset_table="{}.{}.{}".format(PROJECT,
                                                        DATASET,
                                                        'data_source'),
    create_disposition="CREATE_IF_NEEDED",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)

# Clean the data from BigQuery to BigQuery
t2 = bigquery_operator.BigQueryOperator(
    task_id='bq_from_source_to_clean',
    bql='{}/common/clean.sql'.format(GCS_SQL),
    use_legacy_sql=False,
    allow_large_results=True,
    destination_dataset_table="{}.{}.{}".format(PROJECT,
                                                DATASET,
                                                'data_cleaned'),
    create_disposition="CREATE_IF_NEEDED",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)

# Creates split between features and targets and also aggregates both sides.
# The threshold date is passed as an arg when calling the Airflow job and
# dynamically understood within the .sql file.
# We should pass query_params but we run into various problems:
# - if using BigQueryOperator, we can not pass dag_run.conf['threshold_date']
# - if using hooks, run_query does not accept a .sql file and needs the string
# So the way is to add directly {{ dag_run.conf['threshold_date'] }} into the
# .sql file which Airflow can ping up when running the operator.
t3 = bigquery_operator.BigQueryOperator(
    task_id='bq_from_clean_to_features',
    bql='{}/common/features_n_target.sql'.format(GCS_SQL),
    use_legacy_sql=False,
    allow_large_results=True,
    destination_dataset_table="{}.{}.{}".format(PROJECT,
                                                DATASET,
                                                'features_n_target'),
    create_disposition="CREATE_IF_NEEDED",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)


def get_model_type(**kwargs):
  model_type = kwargs['dag_run'].conf.get('model_type')
  if model_type == 'automl':
    model_train_task = 'train_automl'
  else:
    model_train_task = 'train_ml_engine'
  return model_train_task

t4_train_cond = BranchPythonOperator(task_id='train_branch', dag=dag, python_callable=get_model_type)

#
# Train the model using AutoML
#
def do_train_automl(**kwargs):
    """
    Create, train and deploy automl model.
    """
    model_name = clv_automl.create_automl_model(automl_client,
                                                PROJECT,
                                                REGION,
                                                DATASET,
                                                'features_n_target',
                                                AUTOML_DATASET,
                                                AUTOML_MODEL,
                                                AUTOML_TRAINING_BUDGET)
    clv_automl.deploy_model(automl_client, model_name)

t4_automl = PythonOperator(
    task_id='train_automl', dag=dag, python_callable=do_train_automl)


t4_ml_engine = DummyOperator(task_id='train_ml_engine', dag=dag)

# Split the data into a training set and evaluation set within BigQuery
t4a = bigquery_operator.BigQueryOperator(
    task_id='bq_dnn_train',
    bql='{}/dnn/split_train.sql'.format(GCS_SQL),
    use_legacy_sql=False,
    allow_large_results=True,
    destination_dataset_table="{}.{}.{}".format(PROJECT,
                                                DATASET,
                                                'dnn_train'),
    create_disposition="CREATE_IF_NEEDED",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)

t4b = bigquery_operator.BigQueryOperator(
    task_id='bq_dnn_eval',
    bql='{}/dnn/split_eval.sql'.format(GCS_SQL),
    use_legacy_sql=False,
    allow_large_results=True,
    destination_dataset_table="{}.{}.{}".format(PROJECT,
                                                DATASET,
                                                'dnn_eval'),
    create_disposition="CREATE_IF_NEEDED",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)

t4c = bigquery_operator.BigQueryOperator(
    task_id='bq_dnn_test',
    bql='{}/dnn/split_test.sql'.format(GCS_SQL),
    use_legacy_sql=False,
    allow_large_results=True,
    destination_dataset_table="{}.{}.{}".format(PROJECT,
                                                DATASET,
                                                'dnn_test'),
    create_disposition="CREATE_IF_NEEDED",
    write_disposition="WRITE_TRUNCATE",
    dag=dag
)

# TODO: Currently all data steps are done whether BTYD or DNN are used. It would
# be better to have a condition to call only one task or the other using 'model_type'
data_btyd_location = ['gs://{}/{}'.format(LOCATION_TRAINING_DATA, 'btyd.csv')]
data_train_locations = ['gs://{}/{}'.format(LOCATION_TRAINING_DATA, 'train.csv')]
data_eval_locations = ['gs://{}/{}'.format(LOCATION_TRAINING_DATA, 'eval.csv')]
data_test_locations = ['gs://{}/{}'.format(LOCATION_TRAINING_DATA, 'test.csv')]

t5a = bigquery_to_gcs.BigQueryToCloudStorageOperator(
    task_id='bq_dnn_train_to_gcs',
    source_project_dataset_table="{}.{}.{}".format(PROJECT, DATASET, 'dnn_train'),
    destination_cloud_storage_uris=data_train_locations,
    print_header=False,
    dag=dag
)

t5b = bigquery_to_gcs.BigQueryToCloudStorageOperator(
    task_id='bq_dnn_eval_to_gcs',
    source_project_dataset_table="{}.{}.{}".format(PROJECT, DATASET, 'dnn_eval'),
    destination_cloud_storage_uris=data_eval_locations,
    print_header=False,
    dag=dag
)

t5c = bigquery_to_gcs.BigQueryToCloudStorageOperator(
    task_id='bq_dnn_test_to_gcs',
    source_project_dataset_table="{}.{}.{}".format(PROJECT, DATASET, 'dnn_test'),
    destination_cloud_storage_uris=data_test_locations,
    print_header=False,
    dag=dag
)

t5d = bigquery_to_gcs.BigQueryToCloudStorageOperator(
    task_id='bq_btyd_to_gcs',
    source_project_dataset_table="{}.{}.{}".format(PROJECT, DATASET, 'features_n_target'),
    destination_cloud_storage_uris=data_btyd_location,
    print_header=True,
    dag=dag
)


#
# Train the model using ML Engine (TensorFlow DNN or Lifetimes BTYD)
#
def do_train_ml_engine(**kwargs):
    """
    """
    job_id = 'clv-{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))

    mlengine_operator.MLEngineTrainingOperator(
        task_id='train_ml_engine_job',
        project_id=PROJECT,
        job_id=job_id,
        package_uris=['gs://{}/code/{}'.format(COMPOSER_BUCKET_NAME, MODEL_PACKAGE_NAME)],
        training_python_module='trainer.task',
        region=REGION,
        training_args=['--job-dir', 'gs://{}/{}/{}'.format(COMPOSER_BUCKET_NAME, PREFIX_JOBS_EXPORT, job_id),
                       '--data-src', 'gs://{}'.format(LOCATION_TRAINING_DATA),
                       '--model_type', kwargs['dag_run'].conf.get('model_type')],
        dag=dag
    ).execute(kwargs)

t6 = PythonOperator(
    task_id='train_ml_engine_task', dag=dag, python_callable=do_train_ml_engine)

#
# Copies the latest model to a consistent 'final' bucket
#
def do_copy_model_to_final(**kwargs):
    gcs = GoogleCloudStorageHook()

    # Returns all the objects within the bucket. All sub-buckets are considered
    # as prefix of the leaves. List does not differentiate files from subbuckets
    all_jobs_files = gcs.list(
        bucket=COMPOSER_BUCKET_NAME,
        prefix='{}/export/estimate'.format(PREFIX_JOBS_EXPORT)
    )

    # Extract the latest model bucket parent of variables/ and saved_model.pbtxt
    # The max() string contains the latest model folders in 1234567, we need to
    # extract that using regex
    # ex: jobs/clv-composer/export/estimate/1234567890/variables/variables.index
    # returns /1234567890/
    latest_model_bucket = re.findall(r'/\d+/', max(all_jobs_files))[0]

    # List all the files that needs to be copied (only files in the latest bucket
    # and skip the ones that are not files but sub buckets)
    for c in [f for f in all_jobs_files
              if latest_model_bucket in f and f[-1] != '/']:

        # The model used for training is saved into a 'final' sub bucket of the
        # export bucket.
        dest_object = c.split(latest_model_bucket)[1]
        dest_object = '{}/{}'.format(PREFIX_FINAL_MODEL, dest_object)

        logging.info("Copying {} to {} ...".format(dest_object, COMPOSER_BUCKET_NAME))

        gcs.copy(
            source_bucket=COMPOSER_BUCKET_NAME,
            source_object=c,
            destination_object=dest_object
        )

# Note that this could be done as well in Tensorflow using tf.gFile aftet the
# model is created but for reasons of flexibility, it was decided to do this in the
# wider workflow. This way, it is also possible pick other models.
t7 = PythonOperator(
    task_id='copy_model_to_final',
    python_callable=do_copy_model_to_final,
    dag=dag)

#
# Model Creation
#

def do_check_model(**kwargs):
    """ Check if a model with the name exists using Hooks instead of operators.
    Uses xcom_push to pass it to the next step. Could use return too if no key.
    """
    # pushes an XCom without a specific target, just by returning it
    mle = MLEngineHook()
    model_name = kwargs['dag_run'].conf.get('model_name')
    # return bool(mle.get_model(PROJECT, MODEL_DNN_NAME))
    project = mle.get_model(PROJECT, model_name)
    kwargs['ti'].xcom_push(key='is_project', value=bool(project))


def do_create_model(**kwargs):
    """ Creates a model only if one with the same name did not exist.
    It leverages the check from the previous task pushed using xcom.
    """
    model_params = {
      'name': kwargs['dag_run'].conf.get('model_name'),
      'description': 'A custom DNN regressor model',
      'regions': [REGION]
    }

    ti = kwargs['ti']

    is_model = ti.xcom_pull(key='is_project', task_ids='check_model')
    if not is_model:
        mle = MLEngineHook()
        mle.create_model(PROJECT, model_params)

# Checks if model exists using Hook instead of GCP operators due to conditional.
t8 = PythonOperator(
    task_id='check_model', dag=dag, python_callable=do_check_model)

# Creates model if it does not exist using Hook instead of GCP operators
t9 = PythonOperator(
    task_id='create_model', dag=dag, python_callable=do_create_model)

#
# Version Creation
#

def do_list_versions(**kwargs):
    """ Check if a version with the name exists using Hooks instead of operators.
    Uses xcom_push to pass it to the next step. Could use return too if no key.
    """
    mle = MLEngineHook()
    model_name = kwargs['dag_run'].conf.get('model_name')
    model_versions = mle.list_versions(PROJECT, model_name)
    kwargs['ti'].xcom_push(key='model_versions', value=model_versions)


def do_create_version(**kwargs):
    """ Creates a new version or overwrite if existing one. It leverages the
    check from the previous task pushed using xcom.
    """
    version_params = {
      "name": kwargs['dag_run'].conf.get('model_version'),
      "description": 'Version 1',
      "runtimeVersion": kwargs['dag_run'].conf.get('tf_version'),
      "deploymentUri": 'gs://{}/{}'.format(COMPOSER_BUCKET_NAME, PREFIX_FINAL_MODEL)
    }

    ti = kwargs['ti']

    mle = MLEngineHook()

    model_name = kwargs['dag_run'].conf.get('model_name')
    model_versions = ti.xcom_pull(key='model_versions', task_ids='list_versions')

    version_path = 'projects/{}/models/{}/versions/{}'.format(PROJECT,
                                                              model_name,
                                                              version_params['name'])

    if version_path in [v['name'] for v in model_versions]:
        logging.info("Delete previously version of the model to overwrite.")
        mle.delete_version(PROJECT, model_name, version_params['name'])

    mle.create_version(PROJECT, model_name, version_params)

# Checks if model exists using Hook instead of GCP operators due to conditional.
t10 = PythonOperator(
    task_id='list_versions', dag=dag, python_callable=do_list_versions)

# Creates model if it does not exist using Hook instead of GCP operators
t11 = PythonOperator(
    task_id='create_version', dag=dag, python_callable=do_create_version)

# Create task graph
t1.set_downstream(t2)
t2.set_downstream(t3)
t3.set_downstream(t4_train_cond)
t4_train_cond.set_downstream([t4_ml_engine, t4_automl])
t4_ml_engine.set_downstream([t4a, t4b, t4c])
t4_ml_engine.set_downstream(t5d)
t4a.set_downstream(t5a)
t4b.set_downstream(t5b)
t4c.set_downstream(t5c)
t6.set_upstream([t5a, t5b, t5c, t5d])
t6.set_downstream(t7)
t7.set_downstream(t8)
t9.set_upstream(t8)
t9.set_downstream(t10)
t10.set_downstream(t11)
