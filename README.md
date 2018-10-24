# Customer Lifetime Value Prediction with TensorFlow

This project shows how to use a TensorFlow model to predict customer lifetime value.  The model used is a DNN with batch normalization and dropout. We test the model using [this data set](https://www.kaggle.com/c/acquire-valued-shoppers-challenge) from Kaggle.  We also provide an implementation, using the [Lifetimes library](https://github.com/CamDavidsonPilon/lifetimes) in Python, of [probablistic models](https://rdrr.io/cran/BTYD/) commonly used in industry to perform lifetime value prediction.

The project also shows how to deploy a production-ready data processing pipeline for lifetime value prediction on Google Cloud Platform, using BigQuery and DataStore with orchestration provided by Cloud Composer.

## Install

### install Miniconda

The code works with python 2/3.  Using Miniconda2:

    sudo apt-get install -y git bzip2
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    bash Miniconda2-latest-Linux-x86_64.sh -b
    export PATH=~/miniconda2/bin:$PATH

### create environment

    conda create -n clv
    source activate clv
    conda install -n clv python pip
    pip install -r requirements.txt

### launch jupyter
If you are interested in using Jupyter with Datalab, you can do the following:

```
jupyter nbextension install --py datalab.notebook --sys-prefix
jupyter notebook
```

And enter in the first cell of your Notebook

```
%load_ext google.datalab.kernel
```


## Automation
This code is set to run automatically using Cloud Composer, a google-managed version of Airflow. The following steps describe how to go from your own copy of the data to a deployed model with results exported both in Datastore and BigQuery.

See this solution for more details.

### Setup
Before running the Airflow script, you need a couple of things to be set up:

1 - Setting variables

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

2 - Creating workspace

```
gsutil mb -l ${REGION} -p ${PROJECT} ${BUCKET}
gsutil mb -l ${REGION} -p ${PROJECT} ${COMPOSER_BUCKET}
bq --location=US mk --dataset ${PROJECT}:${DATASET_NAME}
```

Create a datastore database as detailed in the [Datastore documentation](https://cloud.google.com/datastore/docs/quickstart)

3 - Copying useful data

```
# Copy the raw dataset
gsutil cp gs://solutions-public-assets/ml-clv/db_dump.csv ${BUCKET}
gsutil cp ${BUCKET}/db_dump.csv ${COMPOSER_BUCKET}

# Copy the dataset to be predicted. Replace with your own.
gsutil cp clv_mle/to_predict.json ${BUCKET}/predictions/
gsutil cp ${BUCKET}/predictions/to_predict.json ${COMPOSER_BUCKET}/predictions/

```

3 - [Optional] Create Machine Learning Engine packaged file
If you make changes to any of the Python files in clv_mle, you need to recreate the packaged files usable by ML Engine.

```
cd ${LOCAL_FOLDER}/clv_mle
rm -rf clv_ml_engine.egg-info/
rm -rf dist
python setup.py sdist
gsutil cp dist/* ${COMPOSER_BUCKET}/code/
```

### Set up Cloud Composer

#### Enable the required APIs
Cloud Composer API
Machine Learning API
Dataflow API



#### Create a service account
Creating a service account is important to make sure that your Cloud Composer instance can perform the required tasks within BigQuery, ML Engine, Dataflow, Cloud Storage and Datastore.

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
```

Wait until the service account has all the proper roles setup.

#### Create a composer instance with the service account
This will take a while. The good thing is that Airflow gets all setup for you.

```
gcloud beta composer environments create ${COMPOSER_NAME} \
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

gcloud beta composer environments storage dags import \
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

#### Set environment variables

Region where things happen

```
gcloud beta composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set region ${REGION}
```

Staging location for Dataflow

```
gcloud beta composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set df_temp_location ${DF_STAGING}
```

Zone where Dataflow should run

```
gcloud beta composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set df_zone ${DF_ZONE}
```

BigQuery working dataset

```
gcloud beta composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set dataset ${DATASET_NAME}
```

Composer bucket

```
gcloud beta composer environments run ${COMPOSER_NAME} variables --location ${REGION} \
-- \
--set composer_bucket_name ${COMPOSER_BUCKET_NAME}
```

#### Import DAG
You need to run this for all your dag files. This solution only has two located in the [run/airflow/dags](run/airflow/dags) folder.

```
gcloud beta composer environments storage dags import \
--environment ${COMPOSER_NAME} \
--source run/airflow/dags/01_build_train_deploy.py \
--location ${REGION} \
--project ${PROJECT}

gcloud beta composer environments storage dags import \
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

- model_type: Name of the model that you want to use. Should match one of the options from model.py
- project: Your project id
- dataset: Your dataset id
- threshold_date: Date that separates features from target
- predict_end: End date of the dataset
- model_name: Name of the model saved to Machine Learning Engine
- model_version: Name of the version of model_name save to Machine Learning Engine
- tf_version: Tensorflow version to be used
- max_monetary: Monetary cap to discard all customers that spend more than that amount

```
gcloud beta composer environments run ${COMPOSER_NAME} \
--project ${PROJECT} \
--location ${REGION} \
trigger_dag \
-- \
build_train_deploy \
--conf '{"model_type":"dnn_model", "project":"'${PROJECT}'", "dataset":"'${DATASET_NAME}'", "threshold_date":"2013-01-31", "predict_end":"2013-07-31", "model_name":"dnn_airflow", "model_version":"v1", "tf_version":"1.10", "max_monetary":"20000"}'
```

```
gcloud beta composer environments run ${COMPOSER_NAME} \
--project ${PROJECT} \
--location ${REGION} \
trigger_dag \
-- \
predict_serve \
--conf '{"model_name":"dnn_airflow", "model_version":"v1", "dataset":"ltv"}'
```

## Train and Tune Models
To run training or hypertuning you can use the mltrain.sh script.  It must be run from the top level directory, as in the examples below. For ML Engine jobs you must supply a bucket on GCS.  The job data folder will be gs://bucket/data and the job directory will be gs://bucket/jobs. So your data files must already be in gs://bucket/data.  For DNN models the data should be named 'train.csv' and 'eval.csv', for probablistic models the file must be 'btyd.csv'.

For example:

```
gsutil -m cp -r ${COMPOSER_BUCKET}/data .
run/mltrain.sh local data
```

```
run/mltrain.sh train ${COMPOSER_BUCKET}
```

```
run/mltrain.sh tune gs://your-bucket
```

For probablistic models:

```
run/mltrain.sh local data --model_type paretonbd_model --threshold_date 2013-01-31 --predict_end 2013-07-31
```

### Disclaimer: This is not an official Google product
