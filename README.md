This code supports the three-part solution [Predicting Customer Lifetime Value with Cloud ML Engine](https://cloud.google.com/solutions/machine-learning/clv-prediction-with-offline-training-intro) published on cloud.google.com.

This code is also used in the updated solution [Predicting Customer Lifetime Value with AutoML Tables ](https://cloud.google.com/solutions/machine-learning/clv-prediction-with-automl-tables) published on cloud.google.com.

# Customer Lifetime Value Prediction on GCP

This project shows how to use ML models to predict customer lifetime value in the following context:
- We apply the models using [this data set](http://archive.ics.uci.edu/ml/datasets/Online+Retail) [1].
- We provide an implementation using a TensorFlow DNN model with batch normalization and dropout.
- We provide an implementation, using the [Lifetimes library](https://github.com/CamDavidsonPilon/lifetimes) in Python, of [statistical models](https://rdrr.io/cran/BTYD/) commonly used in industry to perform lifetime value prediction.
- We also provide an implementation using [AutoML Tables](https://cloud.google.com/automl-tables).

The project also shows how to deploy a production-ready data processing pipeline for lifetime value prediction on Google Cloud Platform, using BigQuery and DataStore with orchestration provided by Cloud Composer.

## Install

### install Miniconda

The code works with python 2/3.  Using Miniconda2:

```
sudo apt-get install -y git bzip2
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh -b
export PATH=~/miniconda2/bin:$PATH
```

### Create dev environment

```
conda create -y -n clv
source activate clv
conda install -y -n clv python=2.7 pip
pip install -r requirements.txt
```

### Enable the required APIs in your GCP Project
Cloud Composer API
Machine Learning API (for TensorFlow / Lifetimes models)
Dataflow API
AutoML Tables API (for AutoML Tables models)


### Environment setup
Before running the training and Airflow scripts, you need some environment variables:

```
export PROJECT=$(gcloud config get-value project 2> /dev/null)
export BUCKET=gs://${PROJECT}_data_final
export REGION=us-central1
export DATASET_NAME=ltv

export COMPOSER_NAME="clv-final"
export COMPOSER_BUCKET_NAME=${PROJECT}_composer_final
export COMPOSER_BUCKET=gs://${COMPOSER_BUCKET_NAME}
export DF_STAGING=${COMPOSER_BUCKET}/dataflow_staging
export DF_ZONE=${REGION}-a
export SQL_MP_LOCATION="sql"

export LOCAL_FOLDER=$(pwd)
```


### Data setup
Creating the BigQuery workspace:

```
gsutil mb -l ${REGION} -p ${PROJECT} ${BUCKET}
gsutil mb -l ${REGION} -p ${PROJECT} ${COMPOSER_BUCKET}
bq --location=US mk --dataset ${PROJECT}:${DATASET_NAME}
```

Create a datastore database as detailed in the [Datastore documentation](https://cloud.google.com/datastore/docs/quickstart)


### Copy the raw dataset
```
gsutil cp gs://solutions-public-assets/ml-clv/db_dump.csv ${BUCKET}
gsutil cp ${BUCKET}/db_dump.csv ${COMPOSER_BUCKET}
```

### Copy the dataset to be predicted. Replace with your own.
```
gsutil cp clv_mle/to_predict.json ${BUCKET}/predictions/
gsutil cp ${BUCKET}/predictions/to_predict.json ${COMPOSER_BUCKET}/predictions/
gsutil cp clv_mle/to_predict.csv ${BUCKET}/predictions/
gsutil cp ${BUCKET}/predictions/to_predict.csv ${COMPOSER_BUCKET}/predictions/

```

### Create a service account
Creating a service account is important to make sure that your Cloud Composer instance can perform the required tasks within BigQuery, AutoML Tables, ML Engine, Dataflow, Cloud Storage and Datastore.  It is also needed to run training for AutoML locally.

The following creates a service account called composer@[YOUR-PROJECT-ID].iam.gserviceaccount.com. and assigns the required roles to the service account.

```
gcloud iam service-accounts create composer --display-name composer --project ${PROJECT}

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/composer.worker

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/bigquery.dataEditor

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/bigquery.jobUser

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/storage.admin

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/ml.developer

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/dataflow.developer

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/compute.viewer

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role roles/storage.objectAdmin

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:composer@${PROJECT}.iam.gserviceaccount.com \
--role='roles/automl.editor'
```

Wait until the service account has all the proper roles setup.


### Download API Key for AutoML Tables

[Create a service account API key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys) and download the json keyfile to run training for AutoML locally.


### Upload Machine Learning Engine packaged file
If using the TensorFlow or Lifetimes model, do this once.  If you make changes to any of the Python files in clv_mle, you need to repeat.

```
cd ${LOCAL_FOLDER}/clv_mle
rm -rf clv_ml_engine.egg-info/
rm -rf dist
python setup.py sdist
gsutil cp dist/* ${COMPOSER_BUCKET}/code/
```


## [Optional] launch Jupyter
The ```notebooks``` folder contains notebooks for data exploration and modeling with linear models and AutoML Tables.

```
jupyter notebook
```

If you are interested in using Jupyter with Datalab, you can do the following:

```
jupyter nbextension install --py datalab.notebook --sys-prefix
jupyter notebook
```

And enter in the first cell of your Notebook

```
%load_ext google.datalab.kernel
```

## Train and Tune Models

### AutoML Tables
You can train the model using the script clv_automl/clv_automl.py.  This takes several parameters.  See usage for full params and default values.

Make sure you have downloaded the json API key.  By default this is assumed to be in a file ```mykey.json``` in the same directory as the script.

For example:

```
cd ${LOCAL_FOLDER}/clv_automl
python clv_automl.py --project_id [YOUR_PROJECT]
```

### TensorFlow DNN/Lifetimes
To run training or hypertuning for the non-automl models you can use the mltrain.sh script.  It must be run from the top level directory, as in the examples below. For ML Engine jobs you must supply a bucket on GCS.  The job data folder will be gs://bucket/data and the job directory will be gs://bucket/jobs. So your data files must already be in gs://bucket/data.  If you use ${COMPOSER_BUCKET}, and the DAG has been run at least once, the data files will be present.  For DNN models the data should be named 'train.csv', 'eval.csv' and 'test.csv', for probablistic models the file must be 'btyd.csv'.

For example:

```
cd ${LOCAL_FOLDER}
gsutil -m cp -r ${COMPOSER_BUCKET}/data .
run/mltrain.sh local data
```

```
run/mltrain.sh train ${COMPOSER_BUCKET}
```

```
run/mltrain.sh tune gs://your-bucket
```

For statistical models:

```
run/mltrain.sh local data --model_type paretonbd_model --threshold_date 2011-08-08 --predict_end 2011-12-12
```


## Automation with AirFlow
This code is set to run automatically using Cloud Composer, a google-managed version of Airflow. The following steps describe how to go from your own copy of the data to a deployed model with results exported both in Datastore and BigQuery.

See [part three of the solution](https://cloud.google.com/solutions/machine-learning/clv-prediction-with-offline-training-deploy) for more details.


### Set up Cloud Composer

#### Create a composer instance with the service account
This will take a while.  This project assumes Airflow 1.9.0, which is the default for Cloud Composer as of March 2019.

```
gcloud composer environments create ${COMPOSER_NAME} \
--location ${REGION}  \
--zone ${REGION}-f \
--machine-type n1-standard-1 \
--service-account=composer@${PROJECT}.iam.gserviceaccount.com
```

#### Make SQL files available to the DAG
There are various ways of calling BigQuery queries. This solutions leverages BigQuery files directly. For them to be accessible by the DAGs, they need to be in the same folder.

The following command line, copies the entire sql folder as a subfolder in the Airflow dags folder.

```
cd ${LOCAL_FOLDER}/preparation

gcloud composer environments storage dags import \
--environment ${COMPOSER_NAME} \
--source  ${SQL_MP_LOCATION} \
--location ${REGION} \
--project ${PROJECT}
```

#### Other files
Some files are important when running the DAG. They can be saved in the `data` folder:

1 - The BigQuery schema file used to load data into BigQuery

```
cd ${LOCAL_FOLDER}
gsutil cp ./run/airflow/schema_source.json ${COMPOSER_BUCKET}
```

2 - A Javascript file used by the Dataflow template for processing.

```
gsutil cp ./run/airflow/gcs_datastore_transform.js ${COMPOSER_BUCKET}
```

#### Set Composer environment variables

Region where things happen

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set region ${REGION}
```

Staging location for Dataflow

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set df_temp_location ${DF_STAGING}
```

Zone where Dataflow should run

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set df_zone ${DF_ZONE}
```

BigQuery working dataset

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set dataset ${DATASET_NAME}
```

Composer bucket

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set composer_bucket_name ${COMPOSER_BUCKET_NAME}
```

#### (for AutoML Tables) Composer environment variables

AutoML Dataset name

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set automl_dataset "clv_solution"
```

AutoML Model name

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set automl_model "clv_model"
```

AutoML training budget

```
gcloud composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set automl_training_budget "1"
```

#### (for AutoML Tables) Import AutoML libraries
```
gcloud composer environments storage dags import \
--environment ${COMPOSER_NAME} \
--source clv_automl \
--location ${REGION} \
--project ${PROJECT}

gcloud composer environments update ${COMPOSER_NAME} \
--update-pypi-packages-from-file run/airflow/requirements.txt \
--location ${REGION} \
--project ${PROJECT}
```

#### Import DAGs
You need to run this for all your dag files. This solution has two DAGs located in the [run/airflow/dags](run/airflow/dags) folder.

```
gcloud composer environments storage dags import \
--environment ${COMPOSER_NAME} \
--source run/airflow/dags/01_build_train_deploy.py \
--location ${REGION} \
--project ${PROJECT}

gcloud composer environments storage dags import \
--environment ${COMPOSER_NAME} \
--source run/airflow/dags/02_predict_serve.py \
--location ${REGION} \
--project ${PROJECT}
```


### Run DAGs
You now should have both DAGs and the SQL files in the Cloud Composer's reserved bucket. Because you probably want to run training and prediction tasks independently, you can run the following script as needed. For more automatic runs (like daily for example, refer to the Airflow documentation to setup your DAGs accordingly.

Airflow can take various parameters as inputs.

The following are used within the .sql files through the syntax {{ dag_run.conf['PARAMETER-NAME'] }}

- project: Project ID where the data is located
- dataset: Dataset that is used to write and read the data
- predict_end: When is the final date of the whole sales dataset
- threshold_date: What is the data used to split the data

Other variables are important as they depend on your environment and are passed directly to the Operators:

- model_type: Name of the model that you want to use. Should be either 'automl' or one of the options from model.py
- project: Your project id
- dataset: Your dataset id
- threshold_date: Date that separates features from target
- predict_end: End date of the dataset
- model_name: Name of the model saved to AutoML Tables or Machine Learning Engine
- model_version: Name of the version of model_name save to Machine Learning Engine (not used for AutoML Tables)
- tf_version: Tensorflow version to be used
- max_monetary: Monetary cap to discard all customers that spend more than that amount

```
gcloud composer environments run ${COMPOSER_NAME} \
--project ${PROJECT} \
--location ${REGION} \
trigger_dag \
-- \
build_train_deploy \
--conf '{"model_type":"automl", "project":"'${PROJECT}'", "dataset":"'${DATASET_NAME}'", "threshold_date":"2011-08-08", "predict_end":"2011-12-12", "model_name":"clv_automl", "model_version":"v1", "tf_version":"1.10", "max_monetary":"15000"}'
```

```
gcloud composer environments run ${COMPOSER_NAME} \
--project ${PROJECT} \
--location ${REGION} \
trigger_dag \
-- \
predict_serve \
--conf '{"model_name":"clv_automl", "model_version":"v1", "dataset":"'${DATASET_NAME}'"}'
```


### Disclaimer: This is not an official Google product

[1]: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
