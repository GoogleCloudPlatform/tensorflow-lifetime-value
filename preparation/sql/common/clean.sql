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

-- Save to table 'data_cleaned'
-- We assume that only one order is passed per day so we agg at that level.
SELECT
  customer_id,
  order_date,
  order_value,
  order_qty_articles
FROM
(
  SELECT
    id AS customer_id,
    date AS order_date,
    ROUND(SUM(purchaseamount), 2) AS order_value,
    SUM(purchasequantity) AS order_qty_articles,
    (
      SELECT
        MAX(date)
      FROM
        `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_source` tl
      WHERE
        tl.id = t.id
    ) latest_order
  FROM
    `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_source` t
  GROUP BY
      id,
      date
) a

INNER JOIN (
  -- Only customers with more than one positive order values before threshold.
  SELECT
    id
  FROM (
    -- Customers and how many positive order values  before threshold.
    SELECT
      id,
      SUM(positive_value) cnt_positive_value
    FROM (
      -- Customer with whether order was positive or not at each date.
      SELECT
        id,
        (
          CASE
            WHEN SUM(purchaseamount) > 0 THEN 1
            ELSE 0
          END ) positive_value
      FROM
        `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_source`
      WHERE
        date < '{{ dag_run.conf['threshold_date'] }}'
      GROUP BY
        id,
        date )
    GROUP BY
      id )
  WHERE
    cnt_positive_value > 1
  ) b
ON
  a.customer_id = b.id
--[START common_clean]
WHERE
  -- Bought in the past 3 months
  DATE_DIFF(DATE('{{ dag_run.conf['predict_end'] }}'), latest_order, DAY) <= 90
  -- Make sure returns are consistent.
  AND (
    (order_qty_articles > 0 and order_Value > 0) OR
    (order_qty_articles < 0 and order_Value < 0)
  )
--[END common_clean]