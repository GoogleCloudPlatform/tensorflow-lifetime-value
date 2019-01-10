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

-- Keep all records before a threshold date for Features
-- And all records before a threshold date for Target
-- Threshold taken at {{ dag_run.conf['threshold_date'] }} ex: 2013-01-31
-- {{ dag_run.conf['threshold_date'] }} is understood by Airflow
SELECT
  tf.customer_id,
  -- For training period
  -- Copying the calculations from Lifetimes where first orders are ignored
  -- See https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/utils.py#L246
--[START features_target]
  tf.monetary_dnn,
  tf.monetary_btyd,
  tf.cnt_orders AS frequency_dnn,
  tf.cnt_orders - 1 AS frequency_btyd,
  tf.recency,
  tf.T,
  ROUND(tf.recency/cnt_orders, 2) AS time_between,
  ROUND(tf.avg_basket_value, 2) AS avg_basket_value,
  ROUND(tf.avg_basket_size, 2) AS avg_basket_size,
  tf.cnt_returns,
  (CASE
      WHEN tf.cnt_returns > 0 THEN 1
      ELSE 0 END) AS has_returned,

  -- Used by BTYD mainly, potentially DNN if clipped improve results
  (CASE
      WHEN tf.cnt_orders - 1 > 600 THEN 600
      ELSE tf.cnt_orders - 1 END) AS frequency_btyd_clipped,
  (CASE
      WHEN tf.monetary_btyd > 100000 THEN 100000
      ELSE ROUND(tf.monetary_btyd, 2) END) AS monetary_btyd_clipped,
  (CASE
      WHEN tt.target_monetary > 100000 THEN 100000
      ELSE ROUND(tt.target_monetary, 2) END) AS target_monetary_clipped,

  -- Target calculated for overall period
  ROUND(tt.target_monetary, 2) as target_monetary
--[END features_target]
FROM
  -- This SELECT uses only data before threshold to make features.
  (
    SELECT
      customer_id,
      SUM(order_value) AS monetary_dnn,
      (CASE
        WHEN COUNT(DISTINCT order_date) = 1 THEN 0
        ELSE SUM(order_value_btyd) / (COUNT(DISTINCT order_date) -1) END) AS monetary_btyd,
      DATE_DIFF(MAX(order_date), MIN(order_date), DAY) AS recency,
      DATE_DIFF(DATE('{{ dag_run.conf['threshold_date'] }}'), MIN(order_date), DAY) AS T,
      COUNT(DISTINCT order_date) AS cnt_orders,
      AVG(order_qty_articles) avg_basket_size,
      AVG(order_value) avg_basket_value,
      SUM(CASE
          WHEN order_value < 1 THEN 1
          ELSE 0 END) AS cnt_returns
    FROM
      -- Makes the order value = 0 if it is the first one
      (
        SELECT
          a.*,
          (CASE
              WHEN a.order_date = c.order_date_min THEN 0
              ELSE a.order_value END) AS order_value_btyd
--[START airflow_params]
        FROM
          `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_cleaned` a
--[END airflow_params]
        INNER JOIN (
          SELECT
            customer_id,
            MIN(order_date) AS order_date_min
          FROM
            `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_cleaned`
          GROUP BY
            customer_id) c
        ON
          c.customer_id = a.customer_id
      )
    WHERE
--[START threshold_date]
      order_date <= DATE('{{ dag_run.conf['threshold_date'] }}')
--[END threshold_date]
    GROUP BY
      customer_id) tf,

  -- This SELECT uses all records to calculate the target (could also use data after threshold )
  (
    SELECT
      customer_id,
      SUM(order_value) target_monetary
    FROM
      `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_cleaned`
      --WHERE order_date > DATE('{{ dag_run.conf['threshold_date'] }}')
    GROUP BY
      customer_id) tt
WHERE
  tf.customer_id = tt.customer_id
  AND tf.monetary_dnn > 0
  AND tf.monetary_dnn <= {{ dag_run.conf['max_monetary'] }}
  AND tf.monetary_btyd > 0