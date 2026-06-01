import pandas as pd
import numpy as np
import os

print("🔄 Zusammenführen von OHLCV und BigQuery On-Chain Daten...")

OHLCV_PATH = '../BTCUSDT_5m_2017-09-01_to_2025-09-23.csv'
BQ_PATH = 'bq_onchain_data.csv'
OUTPUT_PATH = 'BTCUSDT_5m_enriched.csv'

if not os.path.exists(BQ_PATH):
    print(f"❌ Fehler: {BQ_PATH} nicht gefunden.")
    print("Bitte lade die Daten aus BigQuery herunter und speichere sie als 'bq_onchain_data.csv' im @xgboost Ordner.")
    exit(1)

# 1. OHLCV laden
print(f"📥 Lade OHLCV Daten ({OHLCV_PATH})...")
df_price = pd.read_csv(OHLCV_PATH, parse_dates=['datetime'])
df_price.rename(columns={'datetime': 'date'}, inplace=True)
df_price.set_index('date', inplace=True)
df_price.columns = [c.lower() for c in df_price.columns]
# Standardisiere Zeitstempel
df_price.index = df_price.index.round('5min')

# 2. BigQuery Daten laden
print(f"📥 Lade BigQuery Daten ({BQ_PATH})...")
df_bq = pd.read_csv(BQ_PATH, parse_dates=['date'])
df_bq.set_index('date', inplace=True)
df_bq.index = df_bq.index.tz_localize(None) # Entferne Zeitzone für korrekten Join
df_bq.index = df_bq.index.round('5min')

# 3. Zusammenführen (Left Join auf OHLCV, um keine Preisdaten zu verlieren)
print("🔗 Führe Datensätze zusammen (Left Join)...")
df_merged = df_price.join(df_bq, how='left')

# Fehlende On-Chain Daten (z.B. keine Transaktionen in diesen 5 Minuten) mit 0 füllen
onchain_cols = ['total_fees_btc', 'tx_count', 'avg_block_size']
df_merged[onchain_cols] = df_merged[onchain_cols].fillna(0)

# Speichern
df_merged.to_csv(OUTPUT_PATH)
print(f"✅ Erfolgreich gespeichert als {OUTPUT_PATH}")
print(f"📊 Neues Dataset: {df_merged.shape}")
print(f"   Spalten: {list(df_merged.columns)}")
