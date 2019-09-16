#!/bin/bash

# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


usage () {
  echo "usage: run/mltrain.sh [local | train | tune] [gs://]data_folder_or_bucket [args]

Use 'local' to train locally with a local data folder, and 'train' and 'tune' to
run on ML Engine.

For ML Engine jobs you must supply a bucket on GCS.  The job data
folder will be gs://bucket/data and the job directory will be gs://bucket/jobs.
So your data files must already be in gs://bucket/data.  For DNN models the
data should be named 'train.csv', 'eval.csv' and 'test.csv, for probablistic
models the file must be 'btyd.csv'.

For probabilistic models, specify '--model_type paretonbd_model' and include
--threshold_date and --predict_end args.

Examples:

   # train locally
   run/mltrain.sh local data

   # train on ML Engine
   run/mltrain.sh train gs://your_bucket

   # tune hyperparams on ML Engine:
   run/mltrain.sh tune gs://your_bucket

   # train using btyd models
   run/mltrain.sh local data --model_type paretonbd_model --threshold_date 2011-08-08 --predict_end 2011-12-12
"

}

date

TIME=`date +"%Y%m%d_%H%M%S"`


if [[ $# == 0 || $# == 1 ]]; then
  usage
  exit 1
fi

# set job vars
TRAIN_JOB="$1"
DATA_DIR="$2"
BUCKET="$2"
JOB_NAME=clv_${TRAIN_JOB}_${TIME}
REGION=us-central1

# queue additional args
shift; shift

if [[ ${TRAIN_JOB} == "local" ]]; then

  ARGS="--data-src ${DATA_DIR} --model_type=dnn_model --job-dir=jobs/${JOB_NAME} --hidden_units=12 --learning_rate=0.1 --learning_decay_rate=0.001 --train_size=541910 --batch_size=1024 --buffer_size=102400 --train_set_size=500000 --l1_regularization=0.001 --l2_regularization=0.01 --dropout=0.4 --optimizer=ProximialAdagrad --num_epochs=20 --labels=22 --verbose-logging $@"

  mkdir -p jobs/${JOB_NAME}

  python3 -m clv_mle.trainer.task \
    ${ARGS}

elif [[ ${TRAIN_JOB} == "train" ]]; then

  ARGS="--data-src ${BUCKET}/data --verbose-logging $@"

  gcloud beta ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${BUCKET}/jobs/${JOB_NAME} \
    --region $REGION \
    --scale-tier=CUSTOM \
    --module-name trainer.task \
    --package-path clv_mle/trainer \
    --config clv_mle/config.yaml \
    --runtime-version 1.10 \
    -- \
    ${ARGS}

elif [[ $TRAIN_JOB == "tune" ]]; then

  ARGS="--data-src ${BUCKET}/data --verbose-logging $@"

  # set configuration for tuning
  CONFIG_TUNE="clv_mle/config_tune.json"

  gcloud beta ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${BUCKET}/jobs/${JOB_NAME} \
    --region ${REGION} \
    --scale-tier=CUSTOM \
    --module-name trainer.task \
    --package-path clv_mle/trainer \
    --config ${CONFIG_TUNE} \
    --runtime-version 1.10 \
    -- \
    --hypertune \
    ${ARGS}

else
  usage
fi

date
