SELECT
  customer_id,
  order_date,
  order_value,
  order_qty_articles
FROM
(
  SELECT
    CustomerID AS customer_id,
    PARSE_DATE("%m/%d/%y", SUBSTR(InvoiceDate, 0, 8)) AS order_date,
    ROUND(SUM(UnitPrice * Quantity), 2) AS order_value,
    SUM(Quantity) AS order_qty_articles,
    (
      SELECT
        MAX(PARSE_DATE("%m/%d/%y", SUBSTR(InvoiceDate, 0, 8)))
      FROM
        `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_source` tl
      WHERE
        tl.CustomerID = t.CustomerID
    ) latest_order
  FROM
    `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_source` t
  GROUP BY
      CustomerID,
      order_date
) a

INNER JOIN (
  -- Only customers with more than one positive order values before threshold.
  SELECT
    CustomerID
  FROM (
    -- Customers and how many positive order values  before threshold.
    SELECT
      CustomerID,
      SUM(positive_value) cnt_positive_value
    FROM (
      -- Customer with whether order was positive or not at each date.
      SELECT
        CustomerID,
        (
          CASE
            WHEN SUM(UnitPrice * Quantity) > 0 THEN 1
            ELSE 0
          END ) positive_value
      FROM
        `{{ dag_run.conf['project'] }}.{{ dag_run.conf['dataset'] }}.data_source`
      WHERE
        PARSE_DATE("%m/%d/%y", SUBSTR(InvoiceDate, 0, 8)) < DATE('{{ dag_run.conf['threshold_date'] }}')
      GROUP BY
        CustomerID,
        SUBSTR(InvoiceDate, 0, 8) )
    GROUP BY
      CustomerID )
  WHERE
    cnt_positive_value > 1
  ) b
ON
  a.customer_id = b. CustomerID
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