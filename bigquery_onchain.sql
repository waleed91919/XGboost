-- ==============================================================================
-- 🚀 Google BigQuery SQL - Bitcoin On-Chain 5-Minuten Aggregation
-- ==============================================================================
-- ANLEITUNG:
-- 1. Gehe zu https://console.cloud.google.com/bigquery
-- 2. Kopiere diesen SQL-Code in den Editor.
-- 3. Klicke auf "Ausführen" (Run).
-- 4. Klicke nach Abschluss auf "Ergebnisse speichern" (Save Results) -> "CSV (lokal)".
-- 5. Speichere die Datei als "bq_onchain_data.csv" in den Ordner "F:\server\BTC test\@xgboost\".
-- ==============================================================================

WITH aggregated_transactions AS (
  SELECT
    -- Gruppiere Zeitstempel in exakte 5-Minuten Intervalle
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(block_timestamp), 300) * 300) AS date,
    -- Gebühren in BTC umrechnen (Satoshi / 100M)
    SUM(fee) / 100000000 AS total_fees_btc,
    COUNT(`hash`) AS tx_count
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
  WHERE
    -- Partitions-Filter (spart drastisch Kosten & Zeit)
    block_timestamp_month >= '2017-09-01' 
    AND block_timestamp >= '2017-09-01 00:00:00'
  GROUP BY
    date
),

aggregated_blocks AS (
  SELECT
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(timestamp), 300) * 300) AS date,
    AVG(size) AS avg_block_size
  FROM
    `bigquery-public-data.crypto_bitcoin.blocks`
  WHERE
    timestamp_month >= '2017-09-01'
    AND timestamp >= '2017-09-01 00:00:00'
  GROUP BY
    date
)

-- Joine die aggregierten Tabellen (viel schneller und günstiger als Rohdaten-Join)
SELECT
  t.date,
  t.total_fees_btc,
  t.tx_count,
  b.avg_block_size
FROM
  aggregated_transactions t
LEFT JOIN
  aggregated_blocks b
ON
  t.date = b.date
ORDER BY
  t.date ASC;