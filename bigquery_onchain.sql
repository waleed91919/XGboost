-- ==============================================================================
-- 🚀 Google BigQuery SQL - Bitcoin On-Chain 5-Minuten Aggregation (Letzte 3 Monate)
-- ==============================================================================
-- ANLEITUNG:
-- 1. Gehe zu https://console.cloud.google.com/bigquery
-- 2. Kopiere diesen SQL-Code in den Editor.
-- 3. Klicke auf "Ausführen" (Run).
-- 4. Klicke nach Abschluss auf "Ergebnisse speichern" (Save Results) -> "CSV (lokal)".
-- 5. Speichere die Datei als "bigquery_onchain.csv".
-- ==============================================================================

WITH aggregated_transactions AS (
  SELECT
    -- Gruppiere Zeitstempel in exakte 5-Minuten Intervalle
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(block_timestamp), 300) * 300) AS date,
    -- Gebühren in BTC umrechnen (Satoshi / 100M)
    SUM(fee) / 100000000 AS total_fees_btc,
    COUNT(`hash`) AS transaction_count
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
  WHERE
    -- Filter für die letzten 3 Monate (90 Tage)
    block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
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
    timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  GROUP BY
    date
)

-- Joine die aggregierten Tabellen
SELECT
  t.date,
  t.total_fees_btc,
  t.transaction_count,
  b.avg_block_size
FROM
  aggregated_transactions t
LEFT JOIN
  aggregated_blocks b
ON
  t.date = b.date
ORDER BY
  t.date ASC;
