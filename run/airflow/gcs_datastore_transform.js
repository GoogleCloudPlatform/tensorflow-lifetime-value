// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//            http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// To be copied to GCS so it can be uses with Dataflow template
function from_prediction_output_to_datastore_object(prediction_row, entity){
  /*
   * prediction_row looks like what is given by the export function in model.py
   * {"properties": 427.7606201171875, "key": ["'abc'"]}
   * entity should match https://cloud.google.com/datastore/docs/reference/data/rest/v1/Entity
   */

//[START row_to_ds]
  var prediction_object = JSON.parse(prediction_row);

  to_write = {
    key: {
      path: [{
        //id: prediction_object.key,
        kind: 'clv',
        name: prediction_object.customer_id
      }]
    },
    properties: {
      predicted_monetary: {doubleValue: prediction_object.predicted_monetary}
    }
  };

  return JSON.stringify(to_write);
//[END row_to_ds]
}