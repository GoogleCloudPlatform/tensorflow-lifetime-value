-- Copyright 2018 Google Inc. All Rights Reserved.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

--[START benchmark]
SELECT
  ROUND(SQRT( SUM(POW(predicted_monetary - target_monetary, 2)) / COUNT(1) ), 2) as rmse
FROM (
  SELECT
    tf.customer_id,
    avg_basket_value * ( cnt_orders * (1 + target_days/feature_days) ) AS predicted_monetary,
    ROUND(tt.target_monetary, 2) AS target_monetary
--[END benchmark]
  FROM (
      -- This SELECT takes records that are used for features later
    SELECT
      customer_id,
      AVG(order_value) avg_basket_value,
      COUNT(DISTINCT order_date) AS cnt_orders
    FROM
      `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_cleaned`
    WHERE
      order_date <= DATE('{{ dag_run.conf['threshold_date'] }}')
    GROUP BY
      customer_id) tf,
    (
      -- This SELECT takes records that are used for target later
    SELECT
      customer_id,
      SUM(order_value) target_monetary
    FROM
      `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_cleaned`
      --WHERE order_date > '2013-01-31'
    GROUP BY
      customer_id) tt,
    (
    SELECT
      DATE_DIFF(DATE('{{ dag_run.conf['threshold_date'] }}'), MIN(order_date), DAY) feature_days,
      DATE_DIFF(DATE('{{ dag_run.conf['predict_end'] }}'), DATE('{{ dag_run.conf['threshold_date'] }}'), DAY) target_days
    FROM
      `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_cleaned` ) AS days
  WHERE
    tf.customer_id = tt.customer_id )